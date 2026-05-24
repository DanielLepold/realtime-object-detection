from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamClassifierConfig:
    kmeans_clusters: int = 2
    color_space: str = "LAB"  # LAB | HSV

    @classmethod
    def from_dict(cls, d: dict) -> TeamClassifierConfig:
        return cls(
            kmeans_clusters=d.get("kmeans_clusters", 2),
            color_space=d.get("color_space", "LAB"),
        )


class TeamClassifier:
    """
    Assigns players to one of two teams by clustering jersey colors.

    Why LAB over RGB
    ----------------
    The CIELAB color space is perceptually uniform – Euclidean distance
    between two LAB points corresponds much better to perceived color
    difference than in RGB.  This makes KMeans cluster centers more
    meaningful and reduces misclassification for similar-looking kits.

    Process
    -------
    fit()     – Called once on a batch of early frames.
                For each player crop it:
                  1. Takes the upper 60 % of the bounding box (jersey area).
                  2. Masks out green pixels (field grass) in HSV space.
                  3. Computes the dominant LAB color of the remaining pixels.
                Then fits KMeans(k=2) over all collected samples to discover
                the two jersey cluster centers.

    predict() – For a single player box, extracts its dominant jersey color
                and returns the index of the nearest KMeans cluster (0 or 1).
    """

    def __init__(self, config: TeamClassifierConfig) -> None:
        self.config = config
        self._kmeans: KMeans | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frames: list[np.ndarray], boxes_per_frame: list[np.ndarray]) -> None:
        samples: list[np.ndarray] = []

        for frame, boxes in zip(frames, boxes_per_frame):
            for box in boxes:
                color = self._jersey_color(frame, box)
                if color is not None:
                    samples.append(color)

        if len(samples) < self.config.kmeans_clusters:
            return

        km = KMeans(
            n_clusters=self.config.kmeans_clusters,
            random_state=42,
            n_init=10,
        )
        km.fit(np.array(samples))
        self._kmeans = km

    def predict(self, frame: np.ndarray, box: np.ndarray) -> int:
        if self._kmeans is None:
            return 0
        color = self._jersey_color(frame, box)
        if color is None:
            return 0
        return int(self._kmeans.predict(color.reshape(1, -1))[0])

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    @property
    def team_colors_bgr(self) -> list[tuple[int, int, int]] | None:
        if self._kmeans is None:
            return None
        result = []
        for center in self._kmeans.cluster_centers_:
            bgr = self._center_to_bgr(center)
            result.append(tuple(int(c) for c in bgr))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _jersey_color(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        h = y2 - y1
        crop = frame[y1 : y1 + max(1, int(h * 0.6)), x1:x2]
        if crop.size == 0:
            return None

        converted = self._to_color_space(crop).reshape(-1, 3).astype(np.float32)
        non_green = self._non_green_mask(crop).ravel().astype(bool)
        filtered = converted[non_green]
        if len(filtered) < 10:
            return None

        km = KMeans(n_clusters=1, n_init=1, random_state=0)
        km.fit(filtered)
        return km.cluster_centers_[0]

    def _to_color_space(self, bgr: np.ndarray) -> np.ndarray:
        if self.config.color_space == "LAB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    def _center_to_bgr(self, center: np.ndarray) -> np.ndarray:
        if self.config.color_space == "LAB":
            pixel = np.array([[[center[0], center[1], center[2]]]], dtype=np.float32)
            return (cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0][0] * 255).clip(0, 255)
        pixel = np.array([[[center[0], center[1], center[2]]]], dtype=np.uint8)
        return cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0].astype(float)

    @staticmethod
    def _non_green_mask(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        return (green_mask == 0).astype(np.uint8)
