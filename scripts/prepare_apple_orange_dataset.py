from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def collect_jpgs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"], key=lambda p: p.name)


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.glob("*"):
        if item.is_file():
            item.unlink()


def export_split(files: list[Path], train_dir: Path, test_dir: Path, prefix: str, test_count: int) -> tuple[int, int]:
    reset_dir(train_dir)
    reset_dir(test_dir)

    test_files = files[:test_count]
    train_files = files[test_count:]

    for idx, src in enumerate(train_files, start=1):
        shutil.copy2(src, train_dir / f"{prefix}_{idx:05d}.jpg")
    for idx, src in enumerate(test_files, start=1):
        shutil.copy2(src, test_dir / f"{prefix}_{idx:05d}.jpg")

    return len(train_files), len(test_files)


def validate_inputs(apples: list[Path], oranges: list[Path], test_count: int) -> None:
    if len(apples) <= test_count:
        raise ValueError(f"苹果图片数量不足，当前 {len(apples)} 张，至少需要大于 test_count={test_count}")
    if len(oranges) <= test_count:
        raise ValueError(f"橘子图片数量不足，当前 {len(oranges)} 张，至少需要大于 test_count={test_count}")


def count_split(dataset_root: Path) -> dict[str, int]:
    result = {}
    for split in ["trainA", "trainB", "testA", "testB"]:
        folder = dataset_root / split
        result[split] = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
    return result


def dataset_is_ready(dataset_root: Path, expected_train_a: int, expected_train_b: int, expected_test_a: int, expected_test_b: int) -> bool:
    counts = count_split(dataset_root)
    return (
        counts["trainA"] == expected_train_a
        and counts["trainB"] == expected_train_b
        and counts["testA"] == expected_test_a
        and counts["testB"] == expected_test_b
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apple-src", required=True)
    parser.add_argument("--orange-src", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Force rebuild even if dataset already matches expected counts")
    args = parser.parse_args()

    apple_src = Path(args.apple_src)
    orange_src = Path(args.orange_src)
    dataset_root = Path(args.dataset_root)

    train_a = dataset_root / "trainA"
    train_b = dataset_root / "trainB"
    test_a = dataset_root / "testA"
    test_b = dataset_root / "testB"

    apples = collect_jpgs(apple_src)
    oranges = collect_jpgs(orange_src)
    validate_inputs(apples, oranges, args.test_count)

    expected_train_a = len(apples) - args.test_count
    expected_train_b = len(oranges) - args.test_count
    expected_test_a = args.test_count
    expected_test_b = args.test_count

    if not args.force and dataset_is_ready(
        dataset_root,
        expected_train_a=expected_train_a,
        expected_train_b=expected_train_b,
        expected_test_a=expected_test_a,
        expected_test_b=expected_test_b,
    ):
        print("dataset_ready=1")
        print(f"trainA={expected_train_a}")
        print(f"trainB={expected_train_b}")
        print(f"testA={expected_test_a}")
        print(f"testB={expected_test_b}")
        return

    train_a_count, test_a_count = export_split(apples, train_a, test_a, "apple", args.test_count)
    train_b_count, test_b_count = export_split(oranges, train_b, test_b, "orange", args.test_count)

    print("dataset_ready=1")
    print(f"trainA={train_a_count}")
    print(f"trainB={train_b_count}")
    print(f"testA={test_a_count}")
    print(f"testB={test_b_count}")


if __name__ == "__main__":
    main()
