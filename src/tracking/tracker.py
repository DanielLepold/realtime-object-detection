from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from ultralytics import YOLO


@dataclass
class TrackerConfig:
    weights: str = "yolov8x.pt"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    tracker: str = "bytetrack"  # bytetrack | botsort
    device: str = ""
    person_class_id: int = 0
    ball_class_id: int = 32

    @classmethod
    def from_dict(cls, d: dict) -> TrackerConfig:
        return cls(
            weights=d.get("weights", "yolov8x.pt"),
            conf_threshold=d.get("conf_threshold", 0.3),
            iou_threshold=d.get("iou_threshold", 0.5),
            tracker=d.get("tracker", "bytetrack"),
            device=d.get("device", ""),
            person_class_id=d.get("person_class_id", 0),
            ball_class_id=d.get("ball_class_id", 32),
        )


class PlayerTracker:
    """
    Wraps YOLO's built-in multi-object tracking (ByteTrack / BoT-SORT).

    Returns separate Detections for players and the ball so downstream
    modules can handle them differently without branching logic here.

    Why ByteTrack over DeepSORT:
      ByteTrack associates low-confidence detections instead of discarding
      them, which greatly reduces ID switches during occlusions – common
      when players crowd together near the ball.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.model = YOLO(config.weights)
        self._classes = [config.person_class_id, config.ball_class_id]

    def track(self, frame: np.ndarray) -> tuple[sv.Detections, sv.Detections]:
        """
        Run tracking on one frame.

        Returns
        -------
        players : sv.Detections  – all detected persons with stable tracker_id
        ball    : sv.Detections  – detected sports balls (tracker_id may be None)
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            tracker=f"{self.config.tracker}.yaml",
            classes=self._classes,
            device=self.config.device,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        if results.boxes.id is not None:
            detections.tracker_id = results.boxes.id.int().cpu().numpy()

        players = detections[detections.class_id == self.config.person_class_id]
        ball = detections[detections.class_id == self.config.ball_class_id]

        return players, ball
