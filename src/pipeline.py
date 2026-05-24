from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from src.annotation.visualizer import FrameAnnotator, VisualizerConfig
from src.camera.stabilizer import CameraStabilizer, StabilizerConfig
from src.field.perspective import FieldConfig, PerspectiveTransformer
from src.metrics.speed_distance import MetricsConfig, SpeedDistanceTracker
from src.team.classifier import TeamClassifier, TeamClassifierConfig
from src.tracking.tracker import PlayerTracker, TrackerConfig, TrackingResult


@dataclass
class PipelineConfig:
    tracker: TrackerConfig
    stabilizer: StabilizerConfig
    field: FieldConfig
    team: TeamClassifierConfig
    metrics: MetricsConfig
    visualizer: VisualizerConfig
    fit_frames: int = 30
    video_codec: str = "mp4v"

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        return cls(
            tracker=TrackerConfig.from_dict(d.get("model", {})),
            stabilizer=StabilizerConfig.from_dict(d.get("camera", {})),
            field=FieldConfig.from_dict(d.get("field", {})),
            team=TeamClassifierConfig.from_dict(d.get("team", {})),
            metrics=MetricsConfig.from_dict(d.get("metrics", {})),
            visualizer=VisualizerConfig.from_dict(d.get("output", {})),
            fit_frames=d.get("fit_frames", 30),
            video_codec=d.get("output", {}).get("video_codec", "mp4v"),
        )


class AnalysisPipeline:
    """
    Orchestrates the full per-frame analysis loop.

    Frame flow
    ----------
    read frame
      → track (YOLO + ByteTrack)  →  players, ball
      → stabilize (optical flow)  →  cumulative camera offset
      → classify teams (KMeans)   →  team_id per tracker_id
      → measure speed / distance  →  metrics per tracker_id
      → annotate                  →  rendered frame
      → write to output video

    Team classifier fit
    -------------------
    The first `fit_frames` frames that contain at least one player are
    buffered.  Once the buffer is full, KMeans is fitted once on the
    batch.  All subsequent frames use the fitted cluster centers so team
    IDs stay consistent for the entire video.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._tracker = PlayerTracker(config.tracker)
        self._stabilizer = CameraStabilizer(config.stabilizer)
        self._team_clf = TeamClassifier(config.team)
        self._annotator = FrameAnnotator(config.visualizer)
        self._team_ids: dict[int, int] = {}
        self._transformer: PerspectiveTransformer | None = None
        self._speed_tracker: SpeedDistanceTracker | None = None
        self._fit_buffer: list[tuple[np.ndarray, np.ndarray]] = []

    def run(self, input_path: str, output_path: str) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._transformer = PerspectiveTransformer(self._config.field, (w, h))
        self._config.metrics.fps = fps
        self._speed_tracker = SpeedDistanceTracker(self._config.metrics, self._transformer)

        fourcc = cv2.VideoWriter_fourcc(*self._config.video_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        logger.info(f"Input : {input_path}  ({w}×{h} @ {fps:.1f} fps, {total} frames)")
        logger.info(f"Output: {output_path}")

        frame_idx = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated = self._process_frame(frame)
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"  {frame_idx}/{total} frames  |  {frame_idx / elapsed:.1f} fps processing"
                )

        cap.release()
        writer.release()
        elapsed = time.perf_counter() - t0
        logger.success(f"Finished {frame_idx} frames in {elapsed:.1f}s  →  {output_path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        result: TrackingResult = self._tracker.track(frame)
        camera_offset = self._stabilizer.update(frame)

        self._maybe_fit_team_classifier(frame, result.players)
        self._update_team_ids(frame, result.players)

        metrics: dict[int, tuple[float, float]] = {}
        if len(result.players) > 0 and result.players.tracker_id is not None:
            bottom_centers = np.stack(
                [
                    (result.players.xyxy[:, 0] + result.players.xyxy[:, 2]) / 2.0,
                    result.players.xyxy[:, 3],
                ],
                axis=1,
            )
            metrics = self._speed_tracker.update(
                result.players.tracker_id, bottom_centers, camera_offset
            )

        annotated = self._annotator.annotate(
            frame,
            result.players,
            result.ball,
            result.referees,
            self._team_ids,
            metrics,
        )

        if self._config.visualizer.show_minimap and self._transformer is not None:
            annotated = self._annotator.draw_minimap(
                annotated,
                result.players,
                result.ball,
                self._team_ids,
                self._transformer,
                camera_offset,
            )

        return annotated

    def _maybe_fit_team_classifier(self, frame: np.ndarray, players) -> None:
        if self._team_clf.is_fitted:
            return
        if len(players) > 0:
            self._fit_buffer.append((frame.copy(), players.xyxy.copy()))
        if len(self._fit_buffer) >= self._config.fit_frames:
            logger.info("Fitting team classifier…")
            frames, boxes = zip(*self._fit_buffer, strict=False)
            self._team_clf.fit(list(frames), list(boxes))
            self._fit_buffer.clear()
            logger.info("Team classifier ready.")

    def _update_team_ids(self, frame: np.ndarray, players) -> None:
        if not self._team_clf.is_fitted or players.tracker_id is None:
            return
        for tid, box in zip(players.tracker_id, players.xyxy, strict=False):
            tid_int = int(tid)
            if tid_int not in self._team_ids:
                self._team_ids[tid_int] = self._team_clf.predict(frame, box)
