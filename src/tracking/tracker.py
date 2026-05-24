from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from ultralytics import YOLO


@dataclass
class TrackerConfig:
    weights: str = "yolov8x.pt"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    tracker: str = "bytetrack"      # bytetrack | botsort
    device: str = ""
    ball_class_id: int = 32         # COCO: 32  |  fine-tuned: 0
    player_class_ids: list[int] = field(default_factory=lambda: [0])  # COCO: [0]  |  fine-tuned: [1, 2]
    referee_class_id: int | None = None   # fine-tuned: 3
    track_referee: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> TrackerConfig:
        return cls(
            weights=d.get("weights", "yolov8x.pt"),
            conf_threshold=d.get("conf_threshold", 0.3),
            iou_threshold=d.get("iou_threshold", 0.5),
            tracker=d.get("tracker", "bytetrack"),
            device=d.get("device", ""),
            ball_class_id=d.get("ball_class_id", 32),
            player_class_ids=d.get("player_class_ids", [0]),
            referee_class_id=d.get("referee_class_id"),
            track_referee=d.get("track_referee", False),
        )

    @property
    def all_class_ids(self) -> list[int]:
        ids = list(self.player_class_ids) + [self.ball_class_id]
        if self.referee_class_id is not None and self.track_referee:
            ids.append(self.referee_class_id)
        return ids


@dataclass
class TrackingResult:
    players: sv.Detections
    ball: sv.Detections
    referees: sv.Detections


class PlayerTracker:
    """
    Wraps YOLO's built-in multi-object tracking (ByteTrack / BoT-SORT).

    Works with both:
    - Pretrained COCO weights:  person (0), sports ball (32)
    - Fine-tuned football weights: ball (0), goalkeeper (1), player (2), referee (3)

    Configure the class IDs in config.yaml under the `model:` section.
    After fine-tuning, switch to:
        ball_class_id: 0
        player_class_ids: [1, 2]
        referee_class_id: 3

    Why ByteTrack over DeepSORT
    ---------------------------
    ByteTrack associates low-confidence detections rather than discarding
    them, which reduces ID switches during occlusions common when players
    cluster around the ball.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.model = YOLO(config.weights)

    def track(self, frame: np.ndarray) -> TrackingResult:
        """
        Run detection + tracking on one frame.

        Returns a TrackingResult with separate Detections for:
          - players  (player + goalkeeper classes)
          - ball
          - referees (empty if track_referee=false or class not configured)
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            tracker=f"{self.config.tracker}.yaml",
            classes=self.config.all_class_ids,
            device=self.config.device,
            verbose=False,
        )[0]

        all_det = sv.Detections.from_ultralytics(results)

        if results.boxes.id is not None:
            all_det.tracker_id = results.boxes.id.int().cpu().numpy()

        players = self._filter(all_det, self.config.player_class_ids)
        ball = self._filter(all_det, [self.config.ball_class_id])

        referees = sv.Detections.empty()
        if self.config.referee_class_id is not None and self.config.track_referee:
            referees = self._filter(all_det, [self.config.referee_class_id])

        return TrackingResult(players=players, ball=ball, referees=referees)

    @staticmethod
    def _filter(detections: sv.Detections, class_ids: list[int]) -> sv.Detections:
        mask = np.isin(detections.class_id, class_ids)
        return detections[mask]
