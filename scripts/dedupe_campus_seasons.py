from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()

    folder = Path(args.folder)
    seen: dict[str, Path] = {}
    removed = 0
    kept = 0
    for path in sorted(folder.glob("*.jpg")):
        digest = file_hash(path)
        if digest in seen:
            path.unlink()
            removed += 1
        else:
            seen[digest] = path
            kept += 1

    print(f"kept={kept}")
    print(f"removed={removed}")


if __name__ == "__main__":
    main()
