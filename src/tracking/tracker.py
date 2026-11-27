from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

_TRACKERS = {"bytetrack": BYTETracker, "botsort": BOTSORT}


@dataclass
class TrackerConfig:
    weights: str = "yolov8x.pt"
    conf_threshold: float = 0.3
    ball_conf_threshold: float = 0.15
    iou_threshold: float = 0.5
    tracker: str = "bytetrack"  # bytetrack | botsort
    device: str = ""
    ball_class_id: int = 32  # COCO: 32  |  fine-tuned: 0
    player_class_ids: list[int] = field(
        default_factory=lambda: [0]
    )  # COCO: [0]  |  fine-tuned: [1, 2]
    referee_class_id: int | None = None  # fine-tuned: 3
    track_referee: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> TrackerConfig:
        return cls(
            weights=d.get("weights", "yolov8x.pt"),
            conf_threshold=d.get("conf_threshold", 0.3),
            ball_conf_threshold=d.get("ball_conf_threshold", 0.15),
            iou_threshold=d.get("iou_threshold", 0.5),
            tracker=d.get("tracker", "bytetrack"),
            device=d.get("device", ""),
            ball_class_id=d.get("ball_class_id", 32),
            player_class_ids=d.get("player_class_ids", [0]),
            referee_class_id=d.get("referee_class_id"),
            track_referee=d.get("track_referee", False),
        )

    @property
    def person_class_ids(self) -> list[int]:
        """Classes that are handed to the multi-object tracker."""
        ids = list(self.player_class_ids)
        if self.referee_class_id is not None and self.track_referee:
            ids.append(self.referee_class_id)
        return ids

    @property
    def all_class_ids(self) -> list[int]:
        return self.person_class_ids + [self.ball_class_id]


@dataclass
class TrackingResult:
    players: sv.Detections
    ball: sv.Detections
    referees: sv.Detections


class PlayerTracker:
    """
    YOLO detection + ByteTrack / BoT-SORT multi-object tracking.

    Works with both:
    - Pretrained COCO weights:  person (0), sports ball (32)
    - Fine-tuned football weights: ball (0), goalkeeper (1), player (2), referee (3)

    Configure the class IDs in config.yaml under the `model:` section.

    Why players and ball are handled differently
    --------------------------------------------
    YOLO runs once per frame.  Players (and optionally referees) are fed to
    the tracker, which assigns stable IDs.  The ball is *not* tracked: it is
    small and fast, so the IoU-based association almost never confirms a
    ball track, and Ultralytics' `model.track()` silently drops every
    detection that has no confirmed track.  Instead, the single most
    confident ball detection of each frame is used directly.

    Why ByteTrack over DeepSORT
    ---------------------------
    ByteTrack associates low-confidence detections rather than discarding
    them, which reduces ID switches during occlusions common when players
    cluster around the ball.
    """

    def __init__(self, config: TrackerConfig) -> None:
        if config.tracker not in _TRACKERS:
            raise ValueError(f"Unknown tracker '{config.tracker}', choose one of {list(_TRACKERS)}")
        weights = Path(config.weights)
        if weights.suffix == ".pt" and weights.parent != Path(".") and not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights}\n"
                "Train the model (see docs/training.md) or set model.weights in "
                "config.yaml to a pretrained model such as yolov8x.pt "
                "(and switch the class IDs back to the COCO values)."
            )

        self.config = config
        self.model = YOLO(config.weights)
        tracker_cfg = IterableSimpleNamespace(**YAML.load(check_yaml(f"{config.tracker}.yaml")))
        self._tracker = _TRACKERS[config.tracker](args=tracker_cfg)

    def track(self, frame: np.ndarray) -> TrackingResult:
        """
        Run detection + tracking on one frame.

        Returns a TrackingResult with separate Detections for:
          - players  (player + goalkeeper classes, with tracker_id)
          - ball     (at most one detection, no tracker_id)
          - referees (empty if track_referee=false or class not configured)
        """
        cfg = self.config
        result = self.model.predict(
            frame,
            conf=min(cfg.conf_threshold, cfg.ball_conf_threshold),
            iou=cfg.iou_threshold,
            classes=cfg.all_class_ids,
            device=cfg.device,
            verbose=False,
        )[0]

        boxes = result.boxes.cpu().numpy()
        cls = boxes.cls.astype(int)
        conf = boxes.conf

        # --- players / referees → tracker --------------------------------
        person_mask = np.isin(cls, cfg.person_class_ids) & (conf >= cfg.conf_threshold)
        tracks = self._tracker.update(boxes[person_mask], frame)
        tracked = self._tracks_to_detections(tracks)

        players = self._filter(tracked, cfg.player_class_ids)
        referees = sv.Detections.empty()
        if cfg.referee_class_id is not None and cfg.track_referee:
            referees = self._filter(tracked, [cfg.referee_class_id])

        # --- ball → best single detection --------------------------------
        ball = sv.Detections.empty()
        ball_idx = np.where((cls == cfg.ball_class_id) & (conf >= cfg.ball_conf_threshold))[0]
        if len(ball_idx) > 0:
            best = ball_idx[np.argmax(conf[ball_idx])]
            ball = sv.Detections(
                xyxy=boxes.xyxy[best : best + 1].astype(np.float32),
                confidence=conf[best : best + 1].astype(np.float32),
                class_id=cls[best : best + 1],
            )

        return TrackingResult(players=players, ball=ball, referees=referees)

    @staticmethod
    def _tracks_to_detections(tracks: np.ndarray) -> sv.Detections:
        # Ultralytics tracker rows: x1, y1, x2, y2, track_id, score, class, det_index
        if len(tracks) == 0:
            det = sv.Detections.empty()
            det.tracker_id = np.empty(0, dtype=int)
            return det
        return sv.Detections(
            xyxy=tracks[:, :4].astype(np.float32),
            confidence=tracks[:, 5].astype(np.float32),
            class_id=tracks[:, 6].astype(int),
            tracker_id=tracks[:, 4].astype(int),
        )

    @staticmethod
    def _filter(detections: sv.Detections, class_ids: list[int]) -> sv.Detections:
        if len(detections) == 0:
            return detections
        mask = np.isin(detections.class_id, class_ids)
        return detections[mask]
