from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def summer_score(path: Path) -> float:
    arr = load_rgb(path)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    green_excess = np.mean(g - (r + b) / 2.0)
    brightness = np.mean((r + g + b) / 3.0)
    sat = np.mean(np.max(arr, axis=2) - np.min(arr, axis=2))
    return float(green_excess * 2.0 + sat * 0.4 + brightness * 0.1)


def winter_score(path: Path) -> float:
    arr = load_rgb(path)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    brightness = (r + g + b) / 3.0
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)
    white_mask = (brightness > 180) & (sat < 35)
    cool_bias = np.mean(b - r)
    snow_ratio = float(np.mean(white_mask))
    stat = ImageStat.Stat(Image.fromarray(arr.astype(np.uint8)).convert("L"))
    contrast = stat.stddev[0]
    return float(snow_ratio * 600.0 + cool_bias * 2.0 + contrast * 0.5)


def curate(src: Path, dst: Path, mode: str, threshold: float) -> tuple[int, int]:
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.jpg"):
        old.unlink()

    scorer = summer_score if mode == "summer" else winter_score
    kept = 0
    removed = 0
    for idx, path in enumerate(sorted(src.glob("*.jpg")), start=1):
        score = scorer(path)
        if score >= threshold:
            shutil.copy2(path, dst / f"{idx:05d}.jpg")
            kept += 1
        else:
            removed += 1
    return kept, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summer-src", required=True)
    parser.add_argument("--winter-src", required=True)
    parser.add_argument("--summer-dst", required=True)
    parser.add_argument("--winter-dst", required=True)
    parser.add_argument("--summer-threshold", type=float, default=15.0)
    parser.add_argument("--winter-threshold", type=float, default=15.0)
    args = parser.parse_args()

    summer_kept, summer_removed = curate(
        Path(args.summer_src), Path(args.summer_dst), "summer", args.summer_threshold
    )
    winter_kept, winter_removed = curate(
        Path(args.winter_src), Path(args.winter_dst), "winter", args.winter_threshold
    )

    print(f"summer_kept={summer_kept}")
    print(f"summer_removed={summer_removed}")
    print(f"winter_kept={winter_kept}")
    print(f"winter_removed={winter_removed}")


if __name__ == "__main__":
    main()
