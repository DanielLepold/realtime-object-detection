# Module Reference

## `src/tracking/tracker.py` – PlayerTracker

**What it does**
Detects players, referees and the ball with YOLOv8 and tracks players with
Ultralytics' ByteTrack (or BoT-SORT) implementation.

**Key class**: `PlayerTracker`
- `track(frame) → TrackingResult(players, ball, referees)`
  Runs YOLO once.  Player (and optionally referee) detections are fed to
  the tracker, which keeps state across frames and assigns stable integer
  `tracker_id` values.  The ball is *not* tracked – the most confident
  ball detection is used directly, because the tracker would drop it in
  most frames (small, fast object → track never confirmed).

**Config keys** (`model:` section in config.yaml)
| Key | Default | Description |
|-----|---------|-------------|
| `weights` | `yolov8x.pt` | YOLO model size (n/s/m/l/x) |
| `conf_threshold` | `0.3` | Minimum detection confidence (players / referees) |
| `ball_conf_threshold` | `0.15` | Minimum detection confidence for the ball |
| `iou_threshold` | `0.5` | NMS IoU threshold |
| `tracker` | `bytetrack` | `bytetrack` or `botsort` |
| `device` | `""` | `""` = auto, `cuda`, `mps`, `cpu` |

---

## `src/camera/stabilizer.py` – CameraStabilizer

**What it does**
Estimates how far the camera has panned between frames using sparse
optical flow.  Returns a cumulative `(dx, dy)` offset that is subtracted
from player positions before converting to real-world coordinates.

**Algorithm**
1. Detect Shi-Tomasi corners in the top and bottom 20 % of the frame.
2. Track them with Lucas-Kanade pyramid optical flow.
3. Average the displacement of successfully tracked points.
4. Accumulate into `self.cumulative`.

**Config keys** (`camera:` section)
| Key | Default | Description |
|-----|---------|-------------|
| `max_corners` | `100` | Feature points to track per frame |
| `quality_level` | `0.3` | Minimum corner quality (0–1) |
| `min_distance` | `3.0` | Minimum pixels between corners |

---

## `src/field/perspective.py` – PerspectiveTransformer

**What it does**
Converts pixel coordinates to real-world meters using a homography matrix
computed from four manually specified field corner points.

**Setup**
Set `field.corners` in config.yaml to four `[x, y]` pixel pairs:
top-left, top-right, bottom-right, bottom-left of the visible field area.

**Methods**
- `pixel_to_meters(point)` → real-world `[x_m, y_m]`
- `meters_to_minimap_pixel(point_m, minimap_wh)` → minimap pixel `[x, y]`

**Orientation**: x runs along the pitch length, y along the width
(landscape, like the broadcast image and the minimap).

**Fallback**: When `corners` is `null`, `pixel_to_meters` stretches the
frame over the pitch (minimap only, approximate);
`SpeedDistanceTracker` then uses a player-height based scale instead.

---

## `src/team/classifier.py` – TeamClassifier

**What it does**
Assigns each player to one of two teams by clustering the jersey color
using KMeans in the CIELAB color space.

**Two-phase usage**
1. `fit(frames, boxes_per_frame)` – called once on the first N frames.
   Collects one jersey color sample per (non-occluded) player, then fits
   `KMeans(k=2)` over all samples.
2. `predict(frame, box)` – returns team ID `0` or `1` for a single player
   (or `None` if the crop is unusable).  The pipeline turns these into
   per-track votes, see `AnalysisPipeline`.

**Why the jersey crop**
Only the torso is used: 15–50 % of the box height and the central 50 % of
the width, which excludes head, shorts, legs and most of the grass.  The
jersey color is the per-channel median of that crop.  A fixed HSV "green
= grass" mask is deliberately *not* used – it also deletes green/lime kits
(as in the sample video), leaving one team with no color at all.

**Why down-weight lightness**
The same kit in sun and shadow differs mainly in LAB's `L` channel.  With
full weight, KMeans splits "lit vs shadowed" instead of "team A vs team B";
`lightness_weight` (default 0.5) fixes that while still separating e.g.
white vs black kits.

**Config keys** (`team:` section)
| Key | Default | Description |
|-----|---------|-------------|
| `color_space` | `LAB` | `LAB` or `HSV` |
| `lightness_weight` | `0.5` | LAB only: weight of `L` relative to `a`/`b` |
| `kmeans_clusters` | `2` | Number of teams |

---

## `src/metrics/speed_distance.py` – SpeedDistanceTracker

**What it does**
Tracks speed (km/h) and cumulative distance (m) for each player across
the entire video.

**Process per frame**
1. Subtract camera offset from the player's bottom-centre pixel position.
2. Smooth the position with a rolling mean (`distance_smoothing_window`).
3. Speed = *net* displacement over the last `speed_smoothing_window`
   frames ÷ elapsed time × 3.6.  Using the net displacement (instead of
   the mean of frame-to-frame step lengths) makes box jitter cancel out
   rather than show up as phantom speed.  Displacements are measured in
   metres through the homography if `field.corners` is set, otherwise with
   the local scale `player_height_m / box_height_px`.
4. Speeds above `max_speed_kmh` (ID switch / glitch) are rejected and the
   previous speed is kept.
5. Distance += speed × elapsed time, i.e. distance is the integral of the
   displayed speed, so the two numbers are always consistent.
6. Elapsed time comes from frame indices, so frames where the tracker
   briefly lost a player are accounted for.

**Config keys** (`metrics:` section)
| Key | Default | Description |
|-----|---------|-------------|
| `speed_smoothing_window` | `5` | Frames over which net displacement is measured |
| `distance_smoothing_window` | `3` | Rolling mean of positions before measuring displacement |
| `max_speed_kmh` | `40` | Faster speeds are treated as glitches |
| `player_height_m` | `1.8` | Pixel→metre scale when no homography is set |

---

## `src/annotation/visualizer.py` – FrameAnnotator

**What it does**
Renders all visual overlays onto a frame using the `supervision` library.

**Overlays**
- `sv.EllipseAnnotator` – coloured ellipse under each player (team color)
- `sv.TraceAnnotator` – 30-frame motion trail (team color)
- `sv.LabelAnnotator` – speed and distance label
- `sv.TriangleAnnotator` – white triangle over the ball
- Custom minimap – bird's-eye pitch view with player dots
  (landscape, width = `field.minimap_scale` × frame width)

**Config keys** (`output:` section)
| Key | Default | Description |
|-----|---------|-------------|
| `show_minimap` | `true` | Bird's-eye minimap in top-right corner |
| `show_speed` | `true` | Speed label per player |
| `show_distance` | `true` | Cumulative distance label per player |

---

## `src/pipeline.py` – AnalysisPipeline

**What it does**
Orchestrates all modules.  Reads input video frame-by-frame, passes each
frame through the full processing chain, and writes the annotated result
to the output video.

**Team classifier warm-up**
The first `fit_frames` (default 30) frames containing players are buffered.
Once the buffer is full, `TeamClassifier.fit()` is called once and the
buffer is discarded.

**Team votes**
After fitting, every frame each player whose box is not covered by another
player's box (> 10 % overlap) casts one vote via `TeamClassifier.predict()`.
Votes decay by 5 % per appearance, and the displayed team is the argmax.
This keeps labels stable through brief occlusions while still recovering
after a tracker ID switch.
