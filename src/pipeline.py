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
      → detect (YOLO) + track     →  players (with IDs), ball
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

    Every frame, each visible player gets one team "vote" from the fitted
    classifier; the displayed team is the majority of that track's recent
    votes.  Players whose box overlaps another player's box do not vote
    (their crop would contain someone else's jersey), and old votes decay
    slowly, so a tracker ID switch between two players of different teams
    corrects itself within about a second.
    """

    _VOTE_DECAY = 0.95
    _MAX_OVERLAP = 0.1  # fraction of a box covered by another player's box

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._tracker = PlayerTracker(config.tracker)
        self._stabilizer = CameraStabilizer(config.stabilizer)
        self._team_clf = TeamClassifier(config.team)
        self._annotator = FrameAnnotator(config.visualizer)
        self._team_ids: dict[int, int] = {}
        self._team_votes: dict[int, np.ndarray] = {}
        self._frame_idx = 0
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

        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*self._config.video_codec), fps, (w, h)
        )
        if not writer.isOpened() and self._config.video_codec != "mp4v":
            logger.warning(
                f"Codec '{self._config.video_codec}' not available, falling back to 'mp4v'"
            )
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video writer for: {output_path}")

        logger.info(f"Input : {input_path}  ({w}×{h} @ {fps:.1f} fps, {total} frames)")
        logger.info(f"Output: {output_path}")

        frame_idx = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            self._frame_idx = frame_idx
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
            metrics = self._speed_tracker.update(
                result.players.tracker_id, result.players.xyxy, camera_offset, self._frame_idx
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
        isolated = players.xyxy[self._isolated_mask(players.xyxy)]
        if len(isolated) > 0:
            self._fit_buffer.append((frame.copy(), isolated.copy()))
        if len(self._fit_buffer) >= self._config.fit_frames:
            logger.info("Fitting team classifier…")
            frames, boxes = zip(*self._fit_buffer, strict=False)
            self._team_clf.fit(list(frames), list(boxes))
            self._fit_buffer.clear()
            logger.info("Team classifier ready.")

    def _update_team_ids(self, frame: np.ndarray, players) -> None:
        if not self._team_clf.is_fitted or players.tracker_id is None:
            return
        n_teams = self._config.team.kmeans_clusters
        isolated = self._isolated_mask(players.xyxy)
        for tid, box, ok in zip(players.tracker_id, players.xyxy, isolated, strict=True):
            if not ok:
                continue
            team = self._team_clf.predict(frame, box)
            if team is None:
                continue
            tid_int = int(tid)
            votes = self._team_votes.setdefault(tid_int, np.zeros(n_teams))
            votes *= self._VOTE_DECAY
            votes[team] += 1.0
            self._team_ids[tid_int] = int(np.argmax(votes))

    @classmethod
    def _isolated_mask(cls, xyxy: np.ndarray) -> np.ndarray:
        """True for boxes not significantly covered by any other box."""
        n = len(xyxy)
        if n < 2:
            return np.ones(n, dtype=bool)
        x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
        y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
        x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
        y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        np.fill_diagonal(inter, 0)
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        covered = inter.max(axis=1) / np.maximum(area, 1e-6)
        return covered <= cls._MAX_OVERLAP
