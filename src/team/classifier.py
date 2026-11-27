from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamClassifierConfig:
    kmeans_clusters: int = 2
    color_space: str = "LAB"  # LAB | HSV
    lightness_weight: float = 0.5  # LAB only: down-weights L so shadows matter less

    @classmethod
    def from_dict(cls, d: dict) -> TeamClassifierConfig:
        return cls(
            kmeans_clusters=d.get("kmeans_clusters", 2),
            color_space=d.get("color_space", "LAB"),
            lightness_weight=d.get("lightness_weight", 0.5),
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

    LAB also separates lightness (L) from color (a, b).  The same jersey in
    sunlight and in shadow differs mostly in L, so L is multiplied by
    `lightness_weight` (default 0.5) before clustering.  Without this, KMeans
    tends to split "lit vs shadowed" players instead of "team A vs team B".

    Process
    -------
    fit()     – Called once on a batch of early frames.
                For each player crop it:
                  1. Takes the torso region of the bounding box (15–50 % of
                     the height, central 50 % of the width) – this avoids
                     the head, shorts, legs and most of the background.
                  2. Takes the per-channel median of the crop's pixels as the
                     jersey color (robust to the few remaining grass /
                     background pixels).  A fixed HSV "grass" mask is not
                     used on purpose: it also erases green and lime kits.
                Then fits KMeans(k=2) over all collected samples to discover
                the two jersey cluster centers.

    predict() – For a single player box, extracts its jersey color and
                returns the index of the nearest KMeans cluster (0 or 1),
                or None if no usable jersey pixels were found.  The pipeline
                accumulates these per-frame predictions as votes per track,
                so one occluded or blurred frame cannot flip a team.
    """

    def __init__(self, config: TeamClassifierConfig) -> None:
        self.config = config
        self._kmeans: KMeans | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frames: list[np.ndarray], boxes_per_frame: list[np.ndarray]) -> None:
        samples: list[np.ndarray] = []

        for frame, boxes in zip(frames, boxes_per_frame, strict=False):
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

    def predict(self, frame: np.ndarray, box: np.ndarray) -> int | None:
        if self._kmeans is None:
            return None
        color = self._jersey_color(frame, box)
        if color is None:
            return None
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

        h, w = y2 - y1, x2 - x1
        if h <= 0 or w <= 0:
            return None
        crop = frame[
            y1 + int(h * 0.15) : y1 + max(1, int(h * 0.5)),
            x1 + int(w * 0.25) : x1 + max(1, int(w * 0.75)),
        ]
        if crop.size == 0:
            return None

        pixels = self._to_color_space(crop).reshape(-1, 3)
        if len(pixels) < 10:
            return None
        return np.median(pixels, axis=0).astype(np.float64) * self._channel_weights

    @property
    def _channel_weights(self) -> np.ndarray:
        if self.config.color_space == "LAB":
            return np.array([self.config.lightness_weight, 1.0, 1.0])
        return np.ones(3)

    def _to_color_space(self, bgr: np.ndarray) -> np.ndarray:
        if self.config.color_space == "LAB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    def _center_to_bgr(self, center: np.ndarray) -> np.ndarray:
        # Centers live in OpenCV's 8-bit color scales (they were computed
        # from uint8 images), so convert back through uint8 as well.
        weights = np.where(self._channel_weights > 0, self._channel_weights, 1.0)
        pixel = np.clip(center / weights, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        code = cv2.COLOR_LAB2BGR if self.config.color_space == "LAB" else cv2.COLOR_HSV2BGR
        return cv2.cvtColor(pixel, code)[0][0].astype(float)
