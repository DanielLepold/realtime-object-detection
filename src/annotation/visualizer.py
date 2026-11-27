from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

# Default team palette – overridden at runtime by discovered jersey colors.
_TEAM_COLORS = [
    sv.Color(r=230, g=57, b=70),  # team 0 – red
    sv.Color(r=69, g=123, b=157),  # team 1 – blue
    sv.Color(r=244, g=211, b=94),  # unknown / referee – yellow
]
_UNKNOWN = len(_TEAM_COLORS) - 1
_PALETTE = sv.ColorPalette(colors=_TEAM_COLORS)


@dataclass
class VisualizerConfig:
    show_minimap: bool = True
    show_speed: bool = True
    show_distance: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> VisualizerConfig:
        return cls(
            show_minimap=d.get("show_minimap", True),
            show_speed=d.get("show_speed", True),
            show_distance=d.get("show_distance", True),
        )


class FrameAnnotator:
    """
    Draws all visual overlays onto a frame using the supervision library.

    Overlays
    --------
    - Ellipse under each player (color = team)
    - Motion trail (last 30 frames)
    - Speed / distance label above each ellipse
    - White triangle marker above the ball
    - Bird's-eye minimap in the top-right corner (optional)

    Why supervision over raw cv2
    ----------------------------
    supervision provides consistent, composable annotators that handle
    edge cases (empty detections, out-of-frame boxes) internally, which
    keeps this module free of low-level drawing boilerplate.
    """

    def __init__(self, config: VisualizerConfig) -> None:
        self.config = config
        # All player annotators share one fixed palette; the per-detection
        # team index is passed as `custom_color_lookup`, so ellipse, trail
        # and label of a player always have that player's team color.
        self._ellipse = sv.EllipseAnnotator(color=_PALETTE)
        self._trace = sv.TraceAnnotator(color=_PALETTE, thickness=2, trace_length=30)
        self._label = sv.LabelAnnotator(color=_PALETTE, text_scale=0.4, text_thickness=1)
        self._triangle = sv.TriangleAnnotator(color=sv.Color.WHITE, base=20, height=16)

    # ------------------------------------------------------------------
    # Main annotation
    # ------------------------------------------------------------------

    def annotate(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        ball: sv.Detections,
        referees: sv.Detections,
        team_ids: dict[int, int],
        metrics: dict[int, tuple[float, float]],
    ) -> np.ndarray:
        out = frame.copy()

        if len(players) > 0 and players.tracker_id is not None:
            lookup = self._team_lookup(players, team_ids)
            out = self._ellipse.annotate(out, players, custom_color_lookup=lookup)
            out = self._trace.annotate(out, players, custom_color_lookup=lookup)
            labels = self._build_labels(players, metrics)
            out = self._label.annotate(out, players, labels=labels, custom_color_lookup=lookup)

        if len(referees) > 0:
            lookup = np.full(len(referees), _UNKNOWN, dtype=int)
            out = self._ellipse.annotate(out, referees, custom_color_lookup=lookup)

        if len(ball) > 0:
            out = self._triangle.annotate(out, ball)

        return out

    # ------------------------------------------------------------------
    # Minimap
    # ------------------------------------------------------------------

    def draw_minimap(
        self,
        frame: np.ndarray,
        players: sv.Detections,
        ball: sv.Detections,
        team_ids: dict[int, int],
        transformer,
        camera_offset: np.ndarray,
    ) -> np.ndarray:
        fh, fw = frame.shape[:2]
        pad = 10
        mw = int(fw * transformer.config.minimap_scale)
        mh = int(mw * transformer.minimap_aspect)
        if mh > fh - 2 * pad:  # keep the minimap inside the frame
            mh = fh - 2 * pad
            mw = int(mh / transformer.minimap_aspect)

        minimap = np.zeros((mh, mw, 3), dtype=np.uint8)
        self._draw_pitch_lines(minimap)

        if len(players) > 0 and players.tracker_id is not None:
            for tid, xyxy in zip(players.tracker_id, players.xyxy, strict=False):
                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = float(xyxy[3])
                pos_m = transformer.pixel_to_meters(np.array([cx, cy]) - camera_offset)
                mm_pt = transformer.meters_to_minimap_pixel(pos_m, (mw, mh))
                team = team_ids.get(int(tid), _UNKNOWN)
                color = _TEAM_COLORS[min(team, _UNKNOWN)]
                cv2.circle(minimap, tuple(mm_pt.tolist()), 5, color.as_bgr(), -1)

        if len(ball) > 0:
            bx1, by1, bx2, by2 = ball.xyxy[0]
            bx, by = (bx1 + bx2) / 2.0, float(by2)
            bpos_m = transformer.pixel_to_meters(np.array([bx, by]) - camera_offset)
            bmm = transformer.meters_to_minimap_pixel(bpos_m, (mw, mh))
            cv2.circle(minimap, tuple(bmm.tolist()), 6, (255, 255, 255), -1)

        frame[pad : pad + mh, fw - mw - pad : fw - pad] = minimap
        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _team_lookup(players: sv.Detections, team_ids: dict[int, int]) -> np.ndarray:
        """Palette index per detection: its team, or the 'unknown' color."""
        return np.array(
            [min(team_ids.get(int(tid), _UNKNOWN), _UNKNOWN) for tid in players.tracker_id],
            dtype=int,
        )

    def _build_labels(
        self,
        players: sv.Detections,
        metrics: dict[int, tuple[float, float]],
    ) -> list[str]:
        labels: list[str] = []
        for tid in players.tracker_id:
            tid_int = int(tid) if tid is not None else -1
            label = f"#{tid_int}"
            if tid_int in metrics:
                speed, dist = metrics[tid_int]
                if self.config.show_speed:
                    label += f"  {speed:.0f} km/h"
                if self.config.show_distance:
                    label += f"  {dist:.0f} m"
            labels.append(label)
        return labels

    @staticmethod
    def _draw_pitch_lines(minimap: np.ndarray) -> None:
        h, w = minimap.shape[:2]
        cv2.rectangle(minimap, (0, 0), (w - 1, h - 1), (34, 139, 34), -1)  # green fill
        cv2.rectangle(minimap, (2, 2), (w - 3, h - 3), (255, 255, 255), 1)  # boundary
        mid = w // 2
        cv2.line(minimap, (mid, 0), (mid, h), (255, 255, 255), 1)  # halfway line
        cv2.circle(minimap, (mid, h // 2), h // 7, (255, 255, 255), 1)  # centre circle
