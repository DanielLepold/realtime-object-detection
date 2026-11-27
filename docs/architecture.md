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
│    1. PlayerTracker.track()      → players (IDs), ball      │
│    2. CameraStabilizer.update()  → camera_offset (dx, dy)  │
│    3. TeamClassifier.predict()   → team vote per player     │
│    4. SpeedDistanceTracker.update() → speed, distance       │
│    5. FrameAnnotator.annotate()  → rendered frame           │
│    6. cv2.VideoWriter.write()                               │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependency Graph

```
pipeline.py
  ├── tracking/tracker.py        (YOLO + ByteTrack / BoT-SORT)
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
| Measure | tracker IDs + boxes + camera offset + frame index | `dict[tracker_id → (speed_kmh, dist_m)]` |
| Annotate | frame + all above | annotated `np.ndarray` frame |

## Key Design Decisions

**ByteTrack over DeepSORT** – ByteTrack retains low-confidence detections
instead of discarding them, which reduces ID switches when players overlap.

**Ball is detected, not tracked** – The tracker only reports objects with
a confirmed track.  The ball is small and fast, so IoU-based association
rarely confirms it and it would be missing from most frames.  YOLO runs
once; players go through the tracker, and the most confident ball
detection of the frame is used directly.

**LAB color space for team classification** – CIELAB is perceptually
uniform; Euclidean distance between LAB values matches perceived color
difference better than RGB, making KMeans clusters more stable.  The
lightness channel is down-weighted so lit and shadowed players of the same
team are not split into different clusters.

**Team = majority of recent votes** – Each non-occluded appearance of a
player casts a vote; the displayed team is the (slowly decaying) majority.
A single bad crop cannot permanently mislabel a player, and an ID switch
between two players corrects itself.

**Noise-robust speed** – Box positions jitter every frame.  Speed is the
net displacement over a short window of smoothed positions (jitter
cancels out), distance is the time integral of that speed, and physically
impossible jumps (> `max_speed_kmh`, typically ID switches) are ignored.

**Optical flow on background strips** – Camera movement is estimated only
from the top and bottom 20 % of each frame (stands/sky), which move
rigidly with the camera and contain no players.  This avoids player
motion contaminating the global motion estimate.

**Homography fallback** – When field corner coordinates are not provided
in the config, speed and distance use each player's own bounding-box
height (≈ 1.8 m) as the local pixel-to-metre scale.  This adapts to
perspective (distant players are smaller) and gives realistic values
without manual calibration.  The minimap then only shows approximate
positions.
