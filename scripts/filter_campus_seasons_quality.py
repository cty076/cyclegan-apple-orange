from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageStat
import pillow_avif  # noqa: F401


def should_keep(path: Path, min_bytes: int = 20_000) -> tuple[bool, str]:
    if path.stat().st_size < min_bytes:
        return False, "too_small_bytes"

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < 286:
                return False, "too_small_dims"

            gray = im.convert("L")
            stat = ImageStat.Stat(gray)
            stddev = stat.stddev[0]
            if stddev < 18:
                return False, "low_contrast"
    except Exception:
        return False, "unreadable"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--min-bytes", type=int, default=20000)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for idx, path in enumerate(sorted(src.glob("*.jpg")), start=1):
        ok, reason = should_keep(path, min_bytes=args.min_bytes)
        if ok:
            (dst / f"{idx:05d}.jpg").write_bytes(path.read_bytes())
            kept += 1
        else:
            skipped += 1

    print(f"kept={kept}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
