# Model Training & Fine-Tuning

## Pretrained vs. Fine-tuned weights

### Pretrained (zero setup)

The default `yolov8x.pt` checkpoint was trained by Ultralytics on the
**COCO dataset** — roughly 130 million images covering 80 object classes
(person, car, dog, sports ball, etc.).

All learned knowledge is encoded in the model's **weights**: floating-point
matrices that parameterise each layer of the neural network.

This works out of the box, but has limitations for football:

- Detects a generic **"person"** — cannot distinguish player / goalkeeper / referee.
- No concept of pitch boundary — spectators and coaches trigger false positives.
- The `sports ball` class (COCO id 32) is generic and misses occluded or
  motion-blurred footballs.

### Fine-tuned (recommended)

**Transfer learning / fine-tuning** continues training a pretrained checkpoint
on a small, domain-specific dataset.  The model already knows how to extract
visual features (edges, textures, shapes, human body structure) from its
COCO pretraining — we only need to adapt the final classification layers to
the four football classes.

---

## How fine-tuning works internally

```
Pretrained YOLOv8 checkpoint
(backbone has learned: edges, gradients, textures, object parts …)
         │
         │  freeze=10  →  first 10 layers locked, gradients not computed
         │
         ▼
┌──────────────────────┐       ┌──────────────────────────┐
│      Backbone        │  ───► │     Detection Head       │
│  (frozen weights)    │       │   (weights updated)      │
│                      │       │                          │
│  feature extraction: │       │  learns to predict:      │
│  low-level edges,    │       │                          │
│  mid-level shapes,   │       │   class 0  –  ball       │
│  high-level parts    │       │   class 1  –  goalkeeper │
│                      │       │   class 2  –  player     │
│  (already optimal,   │       │   class 3  –  referee    │
│   leave unchanged)   │       │                          │
└──────────────────────┘       └──────────────────────────┘
```

**Backbone** — convolutional feature extractor.  Frozen so pretrained
representations are not destroyed by the small domain dataset.

**Detection head** — predicts bounding box coordinates (`box_loss`),
class probabilities (`cls_loss`), and distribution focal loss (`dfl_loss`).
This is the only part that receives gradient updates during fine-tuning.

### Training loop (per epoch)

```
for each mini-batch of images:
  1. Forward pass   — inference through backbone + head → raw predictions
  2. Loss computation — compare predictions to ground-truth annotations
                        (YOLO-format .txt label files)
  3. Backpropagation — compute gradients w.r.t. head weights only
  4. Optimiser step  — AdamW nudges weights in the direction that
                        reduces loss (backbone weights unchanged)

→ after each epoch: evaluate on validation split → save best.pt if mAP improved
```

One **epoch** = one full pass over all training images.
At 100 epochs with 612 images the head sees 61 200 annotated examples.

**mAP (mean Average Precision)** is the primary evaluation metric —
it measures detection accuracy across all classes and IoU thresholds.
`best.pt` is the checkpoint with the highest validation mAP.

---

## Why not train from scratch?

| | From scratch | Fine-tuning (transfer learning) |
|---|---|---|
| Training images needed | ~100 000+ | 612 |
| Training time (GPU) | Days | ~15–20 min |
| Risk with small data | High — underfitting / overfitting | Low — strong prior from COCO |
| Typical result | Unpredictable | Good domain accuracy |

---

## Running the fine-tuning

There are two options depending on available hardware.

---

### Option A — Google Colab (recommended, free T4 GPU)

Use the provided notebook: `notebooks/football_training_colab.ipynb`

**Hardware requirement:** T4 GPU runtime (free tier)
**Estimated time:** ~15–20 minutes for 100 epochs

#### Step-by-step

**1. Prepare the dataset archive locally**

```bash
cd realtime-object-detection/datasets
zip -r football-dataset.zip football-players-detection.v1i.yolov5pytorch
```

**2. Upload to Google Drive**

Create a folder called `football-analysis` in the root of your Drive and
upload `football-dataset.zip` into it.

**3. Open the notebook in Colab**

Go to [colab.research.google.com](https://colab.research.google.com) →
`File → Upload notebook` → select `notebooks/football_training_colab.ipynb`.

**4. Set the runtime to GPU**

`Runtime → Change runtime type → T4 GPU → Save`

Do this **before running any cell** — changing the runtime after execution
resets the environment.

**5. Run all cells in order**

The notebook will:
- Verify GPU availability (`nvidia-smi`)
- Install `ultralytics`
- Mount Google Drive and unzip the dataset
- Rewrite `data.yaml` with absolute paths (Roboflow exports use relative paths
  that break when the working directory changes)
- Run `model.train()` with `freeze=10`, 100 epochs, batch size 32
- Evaluate the best checkpoint on the test split (reports `mAP50`, `mAP50-95`)
- Copy `best.pt` to `MyDrive/football-analysis/best.pt`

**6. Download the weights**

After training, download `best.pt` from Google Drive and place it at:
```
realtime-object-detection/runs/football/weights/best.pt
```

> The repository already ships a trained `best.pt` at this path.  The
> `runs/` folder is gitignored, so after retraining add the new weights
> explicitly: `git add -f runs/football/weights/best.pt`.

---

### Option B — Local training (requires a GPU or significant patience on CPU)

```bash
python train.py \
  --dataset datasets/football-players-detection.v1i.yolov5pytorch \
  --base yolov8s.pt \
  --batch 8 \
  --epochs 50
```

Estimated runtimes:

| Hardware | ~time |
|----------|-------|
| NVIDIA GPU (CUDA) | 20–40 min |
| Apple Silicon (MPS) | 45–90 min |
| CPU only | 4–8 hours |

To resume an interrupted run:
```bash
python train.py --dataset datasets/... --resume
```

---

## Switching the pipeline to fine-tuned weights

After downloading `best.pt`, update `config.yaml`:

```yaml
model:
  weights: runs/football/weights/best.pt

  ball_class_id: 0
  player_class_ids: [1, 2]   # goalkeeper (1) + player (2)
  referee_class_id: 3
  track_referee: false        # set true to draw referee overlay
```

The rest of the pipeline — team classification, camera stabilisation,
speed/distance metrics, annotation — requires no changes.  It reads
class IDs from config and adapts automatically.

---

## Dataset

**Football Players Detection** — Roboflow Universe
License: CC BY 4.0
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

| Split | Images |
|-------|--------|
| Train | 612 |
| Validation | 38 |
| Test | 13 |

Format: YOLOv5 PyTorch (compatible with YOLOv8 without conversion).

Dataset files are excluded from this repository via `.gitignore`.
Download and place in `datasets/`.
