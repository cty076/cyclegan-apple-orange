from __future__ import annotations

import argparse
from pathlib import Path
import time

import requests


API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CampusSeasonsDatasetBot/1.0 (student project; local codex workspace)"


def search_images(query: str, limit: int) -> list[str]:
    headers = {"User-Agent": USER_AGENT}
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": 1600,
        "format": "json",
        "origin": "*",
    }

    collected: list[str] = []
    seen = set()
    offset = 0
    while len(collected) < limit:
        params["gsroffset"] = offset
        r = requests.get(API_URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break
        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            if info.get("mime") != "image/jpeg":
                continue
            url = info.get("thumburl") or info.get("url")
            if url and url not in seen:
                seen.add(url)
                collected.append(url)
            if len(collected) >= limit:
                break
        cont = data.get("continue", {})
        if "gsroffset" not in cont:
            break
        offset = cont["gsroffset"]
    return collected


def download(session: requests.Session, url: str, path: Path) -> bool:
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(4 + attempt * 2)
                continue
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(1.5)
            return True
        except Exception:
            time.sleep(2 + attempt)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(outdir.glob("wm_*.jpg"))
    start_idx = len(existing) + 1

    all_urls: list[str] = []
    seen = set()
    for q in args.query:
        for url in search_images(q, args.limit):
            if url not in seen:
                seen.add(url)
                all_urls.append(url)
            if len(all_urls) >= args.limit:
                break
        if len(all_urls) >= args.limit:
            break

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    saved = 0
    for idx, url in enumerate(all_urls, start=start_idx):
        if download(session, url, outdir / f"wm_{idx:05d}.jpg"):
            saved += 1

    print(f"collected_urls={len(all_urls)}")
    print(f"saved_images={saved}")


if __name__ == "__main__":
    main()
