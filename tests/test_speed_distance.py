import numpy as np

from src.field.perspective import FieldConfig, PerspectiveTransformer
from src.metrics.speed_distance import MetricsConfig, SpeedDistanceTracker

FPS = 25.0
BOX_H = 90.0  # px → with player_height_m=1.8 one pixel is 2 cm


def _tracker() -> SpeedDistanceTracker:
    transformer = PerspectiveTransformer(FieldConfig(), (1920, 1080))
    return SpeedDistanceTracker(MetricsConfig(fps=FPS), transformer)


def _box(x: float, y: float) -> np.ndarray:
    return np.array([[x - 15, y - BOX_H, x + 15, y]], dtype=np.float32)


def _run(tracker: SpeedDistanceTracker, xs, ys) -> tuple[float, float]:
    out = (0.0, 0.0)
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        out = tracker.update(np.array([1]), _box(x, y), np.zeros(2), i)[1]
    return out


class TestSpeedDistance:
    def test_jitter_does_not_create_speed_or_distance(self):
        rng = np.random.default_rng(0)
        n = 100
        xs = 500 + rng.normal(0, 3, n)  # ±3 px box jitter, player standing still
        ys = 600 + rng.normal(0, 3, n)
        speed, dist = _run(_tracker(), xs, ys)
        assert speed < 3.0
        assert dist < 3.0

    def test_constant_speed_is_measured(self):
        # 5 m/s = 18 km/h → 0.2 m/frame → 10 px/frame at 2 cm/px
        n = 50
        xs = 100 + 10.0 * np.arange(n)
        ys = np.full(n, 600.0)
        speed, dist = _run(_tracker(), xs, ys)
        assert abs(speed - 18.0) < 0.5
        expected = 0.2 * (n - 1)
        assert abs(dist - expected) < 0.5

    def test_teleport_is_ignored(self):
        n = 20
        xs = np.full(n, 500.0)
        xs[10:] = 1500.0  # 20 m jump in one frame (e.g. ID switch)
        ys = np.full(n, 600.0)
        speed, dist = _run(_tracker(), xs, ys)
        assert speed <= MetricsConfig().max_speed_kmh
        assert dist < 1.0

    def test_camera_pan_is_compensated(self):
        tracker = _tracker()
        out = (0.0, 0.0)
        for i in range(50):
            pan = np.array([-8.0 * i, 0.0])  # camera pans, player stands still
            out = tracker.update(np.array([1]), _box(800 + pan[0], 600), pan, i)[1]
        speed, dist = out
        assert speed < 0.5
        assert dist < 0.5
