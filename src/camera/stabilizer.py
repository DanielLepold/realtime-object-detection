from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StabilizerConfig:
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: float = 3.0
    block_size: int = 7

    @classmethod
    def from_dict(cls, d: dict) -> StabilizerConfig:
        return cls(
            max_corners=d.get("max_corners", 100),
            quality_level=d.get("quality_level", 0.3),
            min_distance=d.get("min_distance", 3.0),
            block_size=d.get("block_size", 7),
        )


class CameraStabilizer:
    """
    Estimates per-frame camera translation using sparse Lucas-Kanade optical flow.

    Algorithm
    ---------
    1. Detect Shi-Tomasi corners in the top and bottom strips of the frame
       (stands / sky areas that move rigidly with the camera but contain no
       players).
    2. Track those corners from the previous frame with LK optical flow.
    3. Average the displacement of successfully tracked points → (dx, dy).
    4. Accumulate into a running cumulative offset that represents total
       camera movement from frame 0.

    The cumulative offset is then subtracted from player pixel positions
    before converting to real-world coordinates, so speed/distance
    measurements are in field-space rather than camera-space.
    """

    _LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    def __init__(self, config: StabilizerConfig) -> None:
        self.config = config
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self.cumulative: np.ndarray = np.zeros(2, dtype=np.float64)

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame and return the cumulative camera offset (dx, dy) in pixels.

        Must be called exactly once per frame, in pipeline order.
        Returns a zero vector on the first frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is not None and self._prev_points is not None and len(self._prev_points) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_points, None, **self._LK_PARAMS
            )
            if new_pts is not None and status is not None:
                good_new = new_pts[status.ravel() == 1]
                good_old = self._prev_points[status.ravel() == 1]
                if len(good_new) >= 5:
                    self.cumulative += (good_new - good_old).mean(axis=0)

        self._prev_gray = gray
        self._prev_points = self._detect_corners(gray, frame.shape)
        return self.cumulative.copy()

    def _detect_corners(self, gray: np.ndarray, shape: tuple) -> np.ndarray:
        h = shape[0]
        mask = np.zeros(gray.shape, dtype=np.uint8)
        strip = h // 5
        mask[:strip] = 255        # top strip (stands / sky)
        mask[4 * strip:] = 255   # bottom strip (stands)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
            mask=mask,
        )
        return corners if corners is not None else np.empty((0, 1, 2), dtype=np.float32)
