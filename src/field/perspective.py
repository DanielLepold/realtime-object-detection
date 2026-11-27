from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FieldConfig:
    corners: list[list[int]] | None = None  # pixel: TL, TR, BR, BL
    real_width_m: float = 68.0
    real_length_m: float = 105.0
    minimap_scale: float = 0.22

    @classmethod
    def from_dict(cls, d: dict) -> FieldConfig:
        return cls(
            corners=d.get("corners"),
            real_width_m=d.get("real_width_m", 68.0),
            real_length_m=d.get("real_length_m", 105.0),
            minimap_scale=d.get("minimap_scale", 0.22),
        )


class PerspectiveTransformer:
    """
    Maps pixel coordinates ↔ real-world meter coordinates via homography.

    Setup
    -----
    Provide four pixel-space corners of the visible field region in
    config.corners (top-left, top-right, bottom-right, bottom-left).
    OpenCV's findHomography computes the 3×3 matrix H that projects those
    four points onto a top-down rectangle whose size matches the real
    pitch dimensions.

    Orientation
    -----------
    Real-world coordinates are (x along the pitch length, y along the
    width), matching a standard broadcast camera where the touchline runs
    horizontally across the image.  The minimap uses the same landscape
    orientation.

    Fallback
    --------
    When corners are not set (config.corners = null), pixel_to_meters
    falls back to stretching the frame over the pitch.  This is only used
    to place dots on the minimap; speed/distance switch to a player-height
    based scale instead (see SpeedDistanceTracker).
    """

    def __init__(self, config: FieldConfig, frame_size: tuple[int, int]) -> None:
        self.config = config
        self._frame_w, self._frame_h = frame_size
        self._H: np.ndarray | None = None

        if config.corners:
            src = np.array(config.corners, dtype=np.float32)
            length, width = config.real_length_m, config.real_width_m
            dst = np.array([[0, 0], [length, 0], [length, width], [0, width]], dtype=np.float32)
            self._H, _ = cv2.findHomography(src, dst)

    @property
    def has_homography(self) -> bool:
        return self._H is not None

    @property
    def minimap_aspect(self) -> float:
        """Minimap height / width ratio (landscape pitch)."""
        return self.config.real_width_m / self.config.real_length_m

    def pixel_to_meters(self, point: np.ndarray) -> np.ndarray:
        """Convert a single (x, y) pixel coordinate to (x_m, y_m) meters."""
        if self._H is not None:
            p = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
            return cv2.perspectiveTransform(p, self._H)[0][0]

        scale = np.array(
            [
                self.config.real_length_m / self._frame_w,
                self.config.real_width_m / self._frame_h,
            ]
        )
        return point.astype(float) * scale

    def meters_to_minimap_pixel(
        self, point_m: np.ndarray, minimap_wh: tuple[int, int]
    ) -> np.ndarray:
        """Convert real-world meters to pixel coordinates inside the minimap canvas."""
        mw, mh = minimap_wh
        x = int(np.clip(point_m[0] / self.config.real_length_m * mw, 0, mw - 1))
        y = int(np.clip(point_m[1] / self.config.real_width_m * mh, 0, mh - 1))
        return np.array([x, y])
