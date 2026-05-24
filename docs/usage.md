# Usage Guide

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

YOLOv8 weights are downloaded automatically on first run.

---

## Running the pipeline

```bash
python main.py --input assets/videos/sample_match.mp4
```

Output is saved as `assets/videos/sample_match_analysed.mp4` by default.

### Custom output path

```bash
python main.py \
  --input  assets/videos/sample_match.mp4 \
  --output output/result.mp4
```

### Custom config

```bash
python main.py \
  --input  assets/videos/sample_match.mp4 \
  --config config.yaml
```

---

## Configuring field corners (recommended for accurate speed/distance)

Open `config.yaml` and set four pixel coordinates corresponding to the
visible corners of the football pitch in your video:

```yaml
field:
  corners:
    - [142, 98]    # top-left
    - [1138, 98]   # top-right
    - [1138, 620]  # bottom-right
    - [142, 620]   # bottom-left
  real_width_m: 68.0
  real_length_m: 105.0
```

To find the pixel coordinates, open the video in any image viewer that
shows cursor position, pause on a clear frame, and click each pitch corner.

If `corners` is left as `null`, the pipeline still runs but speed/distance
values will be approximate (linear pixel-to-metre scale).

---

## Choosing a model size

| Weight | Speed (CPU) | Accuracy | Use case |
|--------|-------------|----------|----------|
| `yolov8n.pt` | fastest | lowest | quick tests |
| `yolov8s.pt` | fast | good | laptops |
| `yolov8m.pt` | medium | better | recommended |
| `yolov8x.pt` | slowest | best | GPU server |

Edit `model.weights` in `config.yaml`.

---

## GPU acceleration

Set `model.device` in config.yaml:

```yaml
model:
  device: cuda   # NVIDIA GPU
  # device: mps  # Apple Silicon
  # device: cpu  # force CPU
  # device: ""   # auto-detect (default)
```

---

## Tracker choice

```yaml
model:
  tracker: bytetrack   # default – better under occlusion
  # tracker: botsort   # alternative – uses re-ID features
```
