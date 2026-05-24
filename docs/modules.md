# Module Reference

## `src/tracking/tracker.py` – PlayerTracker

**What it does**
Detects and tracks players and the ball in each frame using YOLOv8 with
ByteTrack multi-object tracking built in.

**Key class**: `PlayerTracker`
- `track(frame) → (players: sv.Detections, ball: sv.Detections)`
  Runs YOLO inference with `persist=True` so ByteTrack maintains track
  state across frames and assigns stable integer `tracker_id` values.

**Config keys** (`model:` section in config.yaml)
| Key | Default | Description |
|-----|---------|-------------|
| `weights` | `yolov8x.pt` | YOLO model size (n/s/m/l/x) |
| `conf_threshold` | `0.3` | Minimum detection confidence |
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

**Fallback**: When `corners` is `null`, uses a linear scale based on
`real_width_m` / `real_length_m` vs frame dimensions.

---

## `src/team/classifier.py` – TeamClassifier

**What it does**
Assigns each player to one of two teams by clustering the dominant jersey
color using KMeans in the CIELAB color space.

**Two-phase usage**
1. `fit(frames, boxes_per_frame)` – called once on the first N frames.
   Collects one dominant jersey color sample per player, then fits
   `KMeans(k=2)` over all samples.
2. `predict(frame, box)` – returns team ID `0` or `1` for a single player.

**Why the jersey crop**
The top 60 % of each bounding box is used (avoiding legs/feet) and green
pixels are masked out (HSV-based grass filter) before clustering.

**Config keys** (`team:` section)
| Key | Default | Description |
|-----|---------|-------------|
| `color_space` | `LAB` | `LAB` or `HSV` |
| `kmeans_clusters` | `2` | Number of teams |

---

## `src/metrics/speed_distance.py` – SpeedDistanceTracker

**What it does**
Tracks speed (km/h) and cumulative distance (m) for each player across
the entire video.

**Process per frame**
1. Subtract camera offset from the player's pixel position.
2. Convert corrected pixel position to meters via `PerspectiveTransformer`.
3. Update a per-player rolling buffer.
4. Compute speed as: `mean(frame-to-frame displacements) × fps × 3.6`.

**Config keys** (`metrics:` section)
| Key | Default | Description |
|-----|---------|-------------|
| `speed_smoothing_window` | `5` | Rolling average width in frames |

---

## `src/annotation/visualizer.py` – FrameAnnotator

**What it does**
Renders all visual overlays onto a frame using the `supervision` library.

**Overlays**
- `sv.EllipseAnnotator` – coloured ellipse under each player (team color)
- `sv.TraceAnnotator` – 30-frame motion trail
- `sv.LabelAnnotator` – speed and distance label
- `sv.TriangleAnnotator` – white triangle over the ball
- Custom minimap – bird's-eye pitch view with player dots

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
