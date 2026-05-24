import numpy as np
import pytest

from src.team.classifier import TeamClassifier, TeamClassifierConfig


def _make_frame(color_bgr: tuple[int, int, int], size: int = 100) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame


def _box(size: int = 100) -> np.ndarray:
    return np.array([0, 0, size, size], dtype=np.float32)


class TestTeamClassifier:
    def test_not_fitted_returns_zero(self):
        clf = TeamClassifier(TeamClassifierConfig())
        frame = _make_frame((255, 0, 0))
        assert clf.predict(frame, _box()) == 0
        assert not clf.is_fitted

    def test_fit_two_distinct_colors(self):
        clf = TeamClassifier(TeamClassifierConfig(color_space="LAB"))

        red_frame = _make_frame((0, 0, 200))    # BGR red
        blue_frame = _make_frame((200, 0, 0))   # BGR blue
        box = _box()

        frames = [red_frame] * 20 + [blue_frame] * 20
        boxes_per_frame = [[box]] * 40

        clf.fit(frames, boxes_per_frame)
        assert clf.is_fitted

        red_id = clf.predict(red_frame, box)
        blue_id = clf.predict(blue_frame, box)
        assert red_id != blue_id

    def test_team_colors_bgr_returns_two_colors(self):
        clf = TeamClassifier(TeamClassifierConfig())
        red = _make_frame((0, 0, 200))
        blue = _make_frame((200, 0, 0))
        box = _box()
        clf.fit([red] * 20 + [blue] * 20, [[box]] * 40)
        colors = clf.team_colors_bgr
        assert colors is not None
        assert len(colors) == 2

    def test_fit_requires_minimum_samples(self):
        clf = TeamClassifier(TeamClassifierConfig(kmeans_clusters=2))
        frame = _make_frame((255, 255, 255))
        clf.fit([frame], [[_box()]])
        assert not clf.is_fitted
