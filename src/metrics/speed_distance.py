from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsConfig:
    fps: float = 25.0
    speed_smoothing_window: int = 5
    distance_smoothing_window: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> MetricsConfig:
        return cls(
            fps=d.get("fps", 25.0),
            speed_smoothing_window=d.get("speed_smoothing_window", 5),
            distance_smoothing_window=d.get("distance_smoothing_window", 3),
        )


class _PlayerMetrics:
    """
    Tracks speed and cumulative distance for one player.

    Positions must be in real-world meters, already corrected for camera
    movement.  Speed is computed as the rolling average of per-frame
    displacements, converted from m/frame to km/h.
    """

    def __init__(self, fps: float, window: int) -> None:
        self._fps = fps
        self._positions: deque[np.ndarray] = deque(maxlen=window + 1)
        self._total_m: float = 0.0

    def update(self, pos_m: np.ndarray) -> None:
        if self._positions:
            self._total_m += float(np.linalg.norm(pos_m - self._positions[-1]))
        self._positions.append(pos_m.copy())

    @property
    def speed_kmh(self) -> float:
        if len(self._positions) < 2:
            return 0.0
        pts = list(self._positions)
        dists = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        return float(np.mean(dists) * self._fps * 3.6)

    @property
    def distance_m(self) -> float:
        return self._total_m


class SpeedDistanceTracker:
    """
    Manages per-player metrics across frames.

    Accepts raw pixel positions and the cumulative camera offset, corrects
    positions by subtracting camera drift, then converts to meters via the
    perspective transformer before updating each player's rolling metrics.
    """

    def __init__(self, config: MetricsConfig, transformer) -> None:
        self._config = config
        self._transformer = transformer
        self._players: dict[int, _PlayerMetrics] = {}

    def update(
        self,
        tracker_ids: np.ndarray,
        pixel_positions: np.ndarray,
        camera_offset: np.ndarray,
    ) -> dict[int, tuple[float, float]]:
        """
        Parameters
        ----------
        tracker_ids     : shape (N,)   – stable IDs from ByteTrack
        pixel_positions : shape (N, 2) – bottom-center of each bounding box
        camera_offset   : shape (2,)   – cumulative (dx, dy) from CameraStabilizer

        Returns
        -------
        dict mapping tracker_id → (speed_kmh, distance_m)
        """
        results: dict[int, tuple[float, float]] = {}

        for tid, px_pos in zip(tracker_ids, pixel_positions):
            corrected = px_pos - camera_offset
            pos_m = self._transformer.pixel_to_meters(corrected)

            tid_int = int(tid)
            if tid_int not in self._players:
                self._players[tid_int] = _PlayerMetrics(
                    self._config.fps, self._config.speed_smoothing_window
                )

            self._players[tid_int].update(pos_m)
            pm = self._players[tid_int]
            results[tid_int] = (pm.speed_kmh, pm.distance_m)

        return results
