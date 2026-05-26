from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import pillow_avif  # noqa: F401


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def normalize_image(src: Path, dst: Path, size: int = 286) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            if min(w, h) < 256:
                return False
            scale = size / min(w, h)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst.with_suffix(".jpg"), quality=95)
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for idx, image_path in enumerate(iter_images(src), start=1):
        out_path = dst / f"{idx:05d}"
        if normalize_image(image_path, out_path):
            kept += 1
        else:
            skipped += 1

    print(f"kept={kept}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
