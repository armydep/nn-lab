"""Download a real training corpus for Chapters 9-12.

`tiny.txt` is four lines -- deliberately tiny so you can verify bigram counts by
hand in Chapters 6-7. It is far too small to train a Transformer on.

This fetches tiny-shakespeare (~1.1 MB, public domain), the standard corpus for
small character-level language models.

Usage:
    python3 sample-data/download_corpus.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/nanoGPT/master/data/shakespeare_char/input.txt",
]

DEST = Path(__file__).parent / "shakespeare.txt"
MIN_BYTES = 100_000


def download() -> int:
    if DEST.exists() and DEST.stat().st_size >= MIN_BYTES:
        print(f"Already present: {DEST} ({DEST.stat().st_size:,} bytes)")
        return 0

    for url in MIRRORS:
        try:
            print(f"Fetching {url} ...")
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  failed: {exc}")
            continue

        if len(data) < MIN_BYTES:
            print(f"  suspiciously small ({len(data)} bytes), trying next mirror")
            continue

        DEST.write_bytes(data)
        print(f"\nSaved {DEST} ({len(data):,} bytes)")
        print(f"Characters: {len(data.decode('utf-8')):,}")
        print(f"Vocabulary: {len(set(data.decode('utf-8')))} unique characters")
        return 0

    print(
        "\nAll mirrors failed. You do not need this exact text -- any plain-text file\n"
        "of roughly 1 MB or more will work. A public-domain book from Project Gutenberg\n"
        "is a fine substitute. Save it as sample-data/shakespeare.txt",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(download())
