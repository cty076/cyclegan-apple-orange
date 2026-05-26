from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlencode

import requests


SCRIPT_RE = re.compile(r'<script id="([^"]+)" type="application/json">(.*?)</script>', re.S)
IMG_RE = re.compile(r"https://images\.unsplash\.com/[^\" ]+")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
BASE = "https://unsplash.com"


def build_search_url(query: str) -> str:
    return f"{BASE}/s/photos/{quote(query)}"


def solve_hex_pow(random_data: str, difficulty: int) -> tuple[str, int, int]:
    start = time.time()
    nonce = 0
    prefix = "0" * difficulty
    while True:
        digest = hashlib.sha256((random_data + str(nonce)).encode()).hexdigest()
        if digest.startswith(prefix):
            elapsed_ms = int((time.time() - start) * 1000)
            return digest, nonce, elapsed_ms
        nonce += 1


def get_real_search_html(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=30)
    if r.status_code == 200 and "images.unsplash.com" in r.text:
        return r.text

    parts = dict(SCRIPT_RE.findall(r.text))
    if "anubis_challenge" not in parts:
        raise RuntimeError(f"Unexpected Unsplash response status={r.status_code}")

    challenge = json.loads(parts["anubis_challenge"])
    digest, nonce, elapsed_ms = solve_hex_pow(
        challenge["challenge"]["randomData"], challenge["rules"]["difficulty"]
    )

    params = {
        "id": challenge["challenge"]["id"],
        "response": digest,
        "nonce": nonce,
        "redir": url.replace(BASE, ""),
        "elapsedTime": elapsed_ms,
    }
    pass_url = f"{BASE}/.within.website/x/cmd/anubis/api/pass-challenge?{urlencode(params)}"
    pr = session.get(pass_url, allow_redirects=False, timeout=30)
    if pr.status_code not in (302, 303):
        raise RuntimeError(f"Failed to pass challenge: status={pr.status_code}")

    redirect = pr.headers.get("location")
    if not redirect:
        raise RuntimeError("Challenge passed but no redirect location returned")

    rr = session.get(BASE + redirect if redirect.startswith("/") else redirect, timeout=30)
    if rr.status_code != 200:
        raise RuntimeError(f"Redirect fetch failed: status={rr.status_code}")
    return rr.text


def extract_image_urls(html: str) -> list[str]:
    seen = set()
    urls: list[str] = []
    for match in IMG_RE.findall(html):
        if "plus.unsplash.com" in match:
            continue
        base = match.split("?")[0]
        final = f"{base}?w=1600&auto=format&fit=max&q=80"
        if final not in seen:
            seen.add(final)
            urls.append(final)
    return urls


def download_image(session: requests.Session, url: str, path: Path) -> bool:
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", required=True, help="Repeatable Unsplash search query")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(outdir.glob("candidate_*.jpg"))
    start_idx = len(existing) + 1

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )

    collected: list[str] = []
    seen = set()
    for query in args.query:
        html = get_real_search_html(session, build_search_url(query))
        for url in extract_image_urls(html):
            if url not in seen:
                seen.add(url)
                collected.append(url)
            if len(collected) >= args.limit:
                break
        if len(collected) >= args.limit:
            break

    saved = 0
    for idx, url in enumerate(collected, start=start_idx):
        target = outdir / f"candidate_{idx:05d}.jpg"
        if download_image(session, url, target):
            saved += 1

    print(f"collected_urls={len(collected)}")
    print(f"saved_images={saved}")


if __name__ == "__main__":
    main()
