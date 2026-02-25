#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import cv2

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _next_index(dir_path: Path) -> int:
    existing = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not existing:
        return 0
    nums = []
    for p in existing:
        stem = p.stem
        try:
            nums.append(int(stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else len(existing)


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Failed to write image: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy paired images in train/high and train/low.")
    parser.add_argument("--root", default="train", help="Root dir containing high/low subdirs")
    parser.add_argument("--count", type=int, default=100, help="Number of pairs to add")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    root = Path(args.root)
    high_dir = root / "high"
    low_dir = root / "low"

    rng = np.random.default_rng(args.seed)
    start_idx = max(_next_index(high_dir), _next_index(low_dir))

    for i in range(start_idx, start_idx + args.count):
        # low: random noise
        low = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
        # high: a lightly blurred version to keep them paired but distinct
        high = cv2.GaussianBlur(low, (5, 5), 0)

        _write_image(low_dir / f"dummy_{i:06d}.png", low)
        _write_image(high_dir / f"dummy_{i:06d}.png", high)

    print(f"Wrote {args.count} dummy pairs to {root}")


if __name__ == "__main__":
    main()
