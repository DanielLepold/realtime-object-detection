#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on the football players detection dataset.

Dataset
-------
Roboflow – Football Players Detection (YOLOv5 PyTorch format)
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1

Classes (after fine-tuning)
---------------------------
  0  ball
  1  goalkeeper
  2  player
  3  referee

Usage
-----
    # Point to the downloaded dataset root
    python train.py --dataset /path/to/football-players-detection.v1i.yolov5pytorch

    # Lighter model for faster iteration
    python train.py --dataset /path/to/... --base yolov8s.pt --epochs 50

    # Resume interrupted training
    python train.py --dataset /path/to/... --resume
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml
from loguru import logger
from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on the football players dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Absolute path to the downloaded dataset root "
        "(the folder containing data.yaml, train/, valid/, test/)",
    )
    parser.add_argument(
        "--base",
        default="yolov8x.pt",
        help="Starting weights: yolov8n/s/m/l/x.pt  (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; use -1 for auto-batch (default: 16)",
    )
    parser.add_argument(
        "--device",
        default="",
        help='Device: "" (auto), "cuda", "mps", "cpu" (default: "")',
    )
    parser.add_argument(
        "--project",
        default="runs",
        help="Output directory for training runs (default: runs)",
    )
    parser.add_argument(
        "--name",
        default="football",
        help="Run name – weights saved to <project>/<name>/weights/ (default: football)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last interrupted training run",
    )
    return parser.parse_args()


def _build_data_yaml(dataset_root: Path) -> str:
    """
    Write a temporary data.yaml with absolute paths.

    The Roboflow-exported data.yaml uses relative paths (../train/images)
    which break when the training script is called from a different directory.
    We resolve them here against the dataset root.
    """
    original = dataset_root / "data.yaml"
    if not original.exists():
        raise FileNotFoundError(f"data.yaml not found in {dataset_root}")

    with original.open() as fh:
        meta = yaml.safe_load(fh)

    meta["train"] = str((dataset_root / "train" / "images").resolve())
    meta["val"] = str((dataset_root / "valid" / "images").resolve())
    meta["test"] = str((dataset_root / "test" / "images").resolve())

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="football_data_"
    )
    yaml.dump(meta, tmp)
    tmp.flush()
    return tmp.name


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset).resolve()

    if not dataset_root.exists():
        logger.error(f"Dataset not found: {dataset_root}")
        sys.exit(1)

    data_yaml = _build_data_yaml(dataset_root)
    logger.info(f"Dataset  : {dataset_root}")
    logger.info(f"Base model: {args.base}")
    logger.info(f"Epochs   : {args.epochs}  |  imgsz: {args.imgsz}  |  batch: {args.batch}")

    model = YOLO(args.base)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Freeze the first 10 layers (the backbone) for the whole run so the
        # pretrained features are preserved while the head adapts to the
        # football classes – also makes training faster on small datasets.
        freeze=10,
        # Augmentation – conservative to avoid degrading pitch-specific patterns
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # Newer Ultralytics versions nest relative projects under runs/detect/,
    # so ask the trainer where it actually saved instead of guessing.
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    logger.success(f"Training complete. Best weights: {best}")
    logger.info(
        f"Update config.yaml → model.weights: {best}\n"
        f"Also set the class IDs:\n"
        f"  ball_class_id:   0\n"
        f"  player_class_ids: [1, 2]   # goalkeeper + player\n"
        f"  referee_class_id: 3"
    )


if __name__ == "__main__":
    main()
