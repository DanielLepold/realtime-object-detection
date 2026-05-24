#!/usr/bin/env python3
"""
Football Analysis – entry point.

Usage
-----
    python main.py --input assets/videos/sample_match.mp4
    python main.py --input assets/videos/sample_match.mp4 --output output/result.mp4
    python main.py --input assets/videos/sample_match.mp4 --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

from src.pipeline import AnalysisPipeline, PipelineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time football player tracking and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output video (default: <input_stem>_analysed.mp4)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_analysed{input_path.suffix}")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    config = PipelineConfig.from_dict(raw)
    pipeline = AnalysisPipeline(config)
    pipeline.run(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
