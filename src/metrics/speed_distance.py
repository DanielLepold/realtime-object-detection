from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsConfig:
    fps: float = 25.0
    speed_smoothing_window: int = 5
    distance_smoothing_window: int = 3
    max_speed_kmh: float = 40.0
    player_height_m: float = 1.8

    @classmethod
    def from_dict(cls, d: dict) -> MetricsConfig:
        return cls(
            fps=d.get("fps", 25.0),
            speed_smoothing_window=d.get("speed_smoothing_window", 5),
            distance_smoothing_window=d.get("distance_smoothing_window", 3),
            max_speed_kmh=d.get("max_speed_kmh", 40.0),
            player_height_m=d.get("player_height_m", 1.8),
        )


# (frame_idx, smoothed camera-corrected pixel position, smoothed box height in px)
_Sample = tuple[int, np.ndarray, float]
_DisplacementFn = Callable[[_Sample, _Sample], float]


class _PlayerMetrics:
    """
    Tracks speed and cumulative distance for one player.

    Noise handling
    --------------
    The bottom-centre of a bounding box jitters by a few pixels every frame
    even for a standing player.  Summing raw frame-to-frame steps would turn
    that jitter into phantom distance and speed, so:

    1. Positions are first smoothed with a rolling mean
       (`distance_smoothing_window` frames).
    2. Speed is the *net* displacement across the last
       `speed_smoothing_window` frames divided by the elapsed time, so
       back-and-forth jitter cancels out instead of accumulating.
    3. Distance is the time integral of that speed (speed × elapsed time,
       summed every frame), so the displayed speed and distance are always
       consistent and inherit the same noise suppression.
    4. Speeds above `max_speed_kmh` (tracker ID switches, detection
       glitches) are rejected; the previous speed is kept instead.
    5. Frame gaps (player temporarily lost by the tracker) are handled by
       using the real elapsed time; after a gap longer than one second the
       history is reset.
    """

    def __init__(self, config: MetricsConfig, displacement_m: _DisplacementFn) -> None:
        self._cfg = config
        self._displacement_m = displacement_m
        self._raw_pos: deque[np.ndarray] = deque(maxlen=max(1, config.distance_smoothing_window))
        self._raw_h: deque[float] = deque(maxlen=max(1, config.distance_smoothing_window))
        self._history: deque[_Sample] = deque(maxlen=max(1, config.speed_smoothing_window) + 1)
        self._total_m: float = 0.0
        self._speed_kmh: float = 0.0

    def update(self, frame_idx: int, pos_px: np.ndarray, height_px: float) -> None:
        fps = self._cfg.fps
        if self._history and frame_idx - self._history[-1][0] > fps:
            self._raw_pos.clear()
            self._raw_h.clear()
            self._history.clear()
            self._speed_kmh = 0.0

        self._raw_pos.append(pos_px.astype(float))
        self._raw_h.append(float(height_px))
        sample: _Sample = (
            frame_idx,
            np.mean(self._raw_pos, axis=0),
            float(np.mean(self._raw_h)),
        )

        prev_frame = self._history[-1][0] if self._history else None
        self._history.append(sample)

        if len(self._history) >= 2:
            first, last = self._history[0], self._history[-1]
            dt = (last[0] - first[0]) / fps
            if dt > 0:
                speed = self._displacement_m(first, last) / dt * 3.6
                if speed <= self._cfg.max_speed_kmh:
                    self._speed_kmh = speed

        if prev_frame is not None:
            self._total_m += self._speed_kmh / 3.6 * (frame_idx - prev_frame) / fps

    @property
    def speed_kmh(self) -> float:
        return self._speed_kmh

    @property
    def distance_m(self) -> float:
        return self._total_m


class SpeedDistanceTracker:
    """
    Manages per-player metrics across frames.

    Accepts raw bounding boxes and the cumulative camera offset, corrects
    positions by subtracting camera drift, and measures displacements in
    metres:

    - With a calibrated homography (`field.corners` set) displacements are
      measured between the projected real-world positions.
    - Without calibration the local pixel→metre scale is estimated from the
      player's own bounding-box height (≈ `player_height_m` tall).  This
      adapts automatically to perspective: players far from the camera
      appear smaller, so each of their pixels covers more ground.
    """

    def __init__(self, config: MetricsConfig, transformer) -> None:
        self._config = config
        self._transformer = transformer
        self._players: dict[int, _PlayerMetrics] = {}

    def update(
        self,
        tracker_ids: np.ndarray,
        xyxy: np.ndarray,
        camera_offset: np.ndarray,
        frame_idx: int,
    ) -> dict[int, tuple[float, float]]:
        """
        Parameters
        ----------
        tracker_ids   : shape (N,)   – stable IDs from the tracker
        xyxy          : shape (N, 4) – bounding boxes in pixels
        camera_offset : shape (2,)   – cumulative (dx, dy) from CameraStabilizer
        frame_idx     : index of the current frame (used for elapsed time)

        Returns
        -------
        dict mapping tracker_id → (speed_kmh, distance_m)
        """
        results: dict[int, tuple[float, float]] = {}

        bottom_centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2.0, xyxy[:, 3]], axis=1)
        heights = xyxy[:, 3] - xyxy[:, 1]

        for tid, px_pos, h in zip(tracker_ids, bottom_centers, heights, strict=True):
            tid_int = int(tid)
            if tid_int not in self._players:
                self._players[tid_int] = _PlayerMetrics(self._config, self._displacement_m)

            pm = self._players[tid_int]
            pm.update(frame_idx, px_pos - camera_offset, float(h))
            results[tid_int] = (pm.speed_kmh, pm.distance_m)

        return results

    def _displacement_m(self, a: _Sample, b: _Sample) -> float:
        _, pos_a, h_a = a
        _, pos_b, h_b = b
        if self._transformer.has_homography:
            pa = self._transformer.pixel_to_meters(pos_a)
            pb = self._transformer.pixel_to_meters(pos_b)
            return float(np.linalg.norm(pb - pa))

        mean_h = (h_a + h_b) / 2.0
        if mean_h <= 1.0:
            return 0.0
        metres_per_px = self._config.player_height_m / mean_h
        return float(np.linalg.norm(pos_b - pos_a) * metres_per_px)
