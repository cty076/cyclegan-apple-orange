from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


@dataclass
class ScoredImage:
    path: Path
    score: float


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


def score_folder(folder: Path, mode: str) -> list[ScoredImage]:
    scorer = summer_score if mode == "summer" else winter_score
    scored = [ScoredImage(path=p, score=scorer(p)) for p in sorted(folder.glob("*.jpg"))]
    return sorted(scored, key=lambda x: x.score, reverse=True)


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.glob("*"):
        if item.is_file():
            item.unlink()


def export_split(scored: list[ScoredImage], train_dir: Path, test_dir: Path, prefix: str, train_count: int, test_count: int) -> None:
    reset_dir(train_dir)
    reset_dir(test_dir)

    selected_test = scored[:test_count]
    selected_train = scored[test_count : test_count + train_count]

    for idx, item in enumerate(selected_train, start=1):
        shutil.copy2(item.path, train_dir / f"{prefix}_{idx:05d}.jpg")
    for idx, item in enumerate(selected_test, start=1):
        shutil.copy2(item.path, test_dir / f"{prefix}_{idx:05d}.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summer-src", required=True)
    parser.add_argument("--winter-src", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--test-count", type=int, default=18)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    summer = score_folder(Path(args.summer_src), "summer")
    winter = score_folder(Path(args.winter_src), "winter")

    export_split(
        summer,
        dataset_root / "trainA",
        dataset_root / "testA",
        "summer",
        args.train_count,
        args.test_count,
    )
    export_split(
        winter,
        dataset_root / "trainB",
        dataset_root / "testB",
        "winter",
        args.train_count,
        args.test_count,
    )

    print(f"summer_total={len(summer)}")
    print(f"winter_total={len(winter)}")
    print(f"train_count={args.train_count}")
    print(f"test_count={args.test_count}")


if __name__ == "__main__":
    main()
