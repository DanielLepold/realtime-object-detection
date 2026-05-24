# System Architecture

## Overview

The pipeline is a linear, frame-by-frame processor.  Each module has a
single responsibility and communicates only through plain data types
(`np.ndarray`, `sv.Detections`, Python dicts).  There is no shared mutable
state between modules except the `PipelineConfig` dataclass that is
constructed once at startup.

```
┌─────────────┐
│  main.py    │  parses CLI args, loads config.yaml, calls pipeline.run()
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│  AnalysisPipeline  (src/pipeline.py)                        │
│                                                             │
│  for each frame:                                            │
│    1. PlayerTracker.track()      → players, ball            │
│    2. CameraStabilizer.update()  → camera_offset (dx, dy)  │
│    3. TeamClassifier.predict()   → team_id per player       │
│    4. SpeedDistanceTracker.update() → speed, distance       │
│    5. FrameAnnotator.annotate()  → rendered frame           │
│    6. cv2.VideoWriter.write()                               │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
pipeline.py
  ├── tracking/tracker.py        (YOLO + ByteTrack)
  ├── camera/stabilizer.py       (Shi-Tomasi + LK optical flow)
  ├── field/perspective.py       (homography, pixel ↔ meters)
  ├── team/classifier.py         (KMeans jersey color)
  ├── metrics/speed_distance.py  (rolling speed, cumulative distance)
  └── annotation/visualizer.py   (supervision annotators + minimap)
```

## Data Flow per Frame

| Step | Input | Output |
|------|-------|--------|
| Track | `np.ndarray` frame | `sv.Detections` players, `sv.Detections` ball |
| Stabilize | `np.ndarray` frame | `np.ndarray` cumulative offset `[dx, dy]` |
| Classify teams | frame + player boxes | `dict[tracker_id → team_id]` |
| Measure | tracker IDs + pixel positions + camera offset | `dict[tracker_id → (speed_kmh, dist_m)]` |
| Annotate | frame + all above | annotated `np.ndarray` frame |

## Key Design Decisions

**ByteTrack over DeepSORT** – ByteTrack retains low-confidence detections
instead of discarding them, which reduces ID switches when players overlap.

**LAB color space for team classification** – CIELAB is perceptually
uniform; Euclidean distance between LAB values matches perceived color
difference better than RGB, making KMeans clusters more stable.

**Optical flow on background strips** – Camera movement is estimated only
from the top and bottom 20 % of each frame (stands/sky), which move
rigidly with the camera and contain no players.  This avoids player
motion contaminating the global motion estimate.

**Homography fallback** – When field corner coordinates are not provided
in the config, the transformer falls back to a linear pixel-to-meter
scale rather than crashing, so the pipeline runs on any video without
manual calibration.
