"""Fetch MSMARCO-XI validation shards.

    uv run python -m vrag.download [--langs hin,ben,tam]

The HF dataset viewer and `datasets.load_dataset` both fail on this repo: each
shard is a single ~1.2GB parquet row group, which exceeds the viewer's limit and
makes the streaming loader materialise the whole group. Fetching the raw files
and streaming record batches out of them (see `vrag.corpus`) sidesteps both.

Downloads are resumable and skipped when the local file is already complete —
each shard is ~460MB and re-fetching them on every build is a waste of the
minutes it takes.
"""

from __future__ import annotations

import argparse
import sys

import httpx

from .config import settings

BASE = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation"

ALL_LANGS = (
    "asm", "ben", "guj", "hin", "kan", "mal", "mar",
    "nep", "ori", "pan", "san", "tam", "tel", "urd",
)


def _remote_size(client: httpx.Client, url: str) -> int:
    response = client.head(url, follow_redirects=True)
    response.raise_for_status()
    return int(response.headers.get("content-length", 0))


def download(lang: str, client: httpx.Client) -> None:
    url = f"{BASE}/{lang}val.parquet"
    dest = settings.raw_dir / f"{lang}val.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)

    expected = _remote_size(client, url)
    if dest.exists() and expected and dest.stat().st_size == expected:
        print(f"  {lang}: already complete ({expected / 1e6:.0f} MB)")
        return

    # Resume rather than restart: a shard interrupted at 400/460MB should cost
    # the remaining 60MB, not the whole file again.
    start = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    mode = "ab" if start else "wb"
    if start:
        print(f"  {lang}: resuming at {start / 1e6:.0f} MB")

    done = start
    with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        if start and response.status_code == 200:
            # Server ignored the range header; restart cleanly rather than
            # appending a second full copy onto the partial file.
            mode, done = "wb", 0
        response.raise_for_status()
        with open(dest, mode) as fh:
            for block in response.iter_bytes(chunk_size=1 << 20):
                fh.write(block)
                done += len(block)
                if expected:
                    pct = 100 * done / expected
                    print(f"\r  {lang}: {done / 1e6:7.0f} / {expected / 1e6:.0f} MB  ({pct:5.1f}%)",
                          end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(settings.languages),
                    help=f"comma-separated; available: {','.join(ALL_LANGS)}")
    args = ap.parse_args()

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    unknown = [lang for lang in langs if lang not in ALL_LANGS]
    if unknown:
        print(f"unknown language shard(s): {unknown}\navailable: {', '.join(ALL_LANGS)}")
        return 1

    print(f"downloading {len(langs)} shard(s) to {settings.raw_dir}")
    # Generous timeout: these are ~460MB files and a short read timeout turns a
    # slow-but-working connection into a spurious failure.
    with httpx.Client(timeout=httpx.Timeout(60.0, read=300.0)) as client:
        for lang in langs:
            try:
                download(lang, client)
            except httpx.HTTPError as exc:
                print(f"\n  {lang}: FAILED — {exc}")
                return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
