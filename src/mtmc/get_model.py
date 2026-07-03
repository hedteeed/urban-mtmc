"""Fetch the pinned COCO-pretrained YOLOX-S ONNX model (CONTRACT.md §M1).

Run:  python -m mtmc.get_model

Downloads the official Megvii-BaseDetection/YOLOX release asset to
``models/yolox_s.onnx`` and verifies its sha256 against the pinned constant on
EVERY run (mantis manifest discipline: artifacts are pinned, not trusted).
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Official YOLOX release asset — repo and weights are Apache-2.0 (license-clean).
# Verified resolving via GitHub's release-asset CDN on 2026-07-03.
MODEL_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
)
MODEL_LICENSE = "Apache-2.0 (Megvii-BaseDetection/YOLOX, COCO-pretrained YOLOX-S)"
# sha256 of the 35 858 002-byte asset, computed at pin time (2026-07-03).
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
# models/ is gitignored; path anchored to the repo root, not the cwd.
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolox_s.onnx"

_CHUNK = 1 << 20  # 1 MiB read chunks — hash while streaming, no full-file buffer


def sha256_of(path: Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> str:
    """Stream ``url`` to ``dest`` atomically (via .part), returning its sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, part.open("wb") as out:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)  # atomic: never leave a half-written model at MODEL_PATH
    return digest.hexdigest()


def _provenance(digest: str) -> None:
    print(f"model : {MODEL_PATH}")
    print(f"url   : {MODEL_URL}")
    print(f"license: {MODEL_LICENSE}")
    print(f"sha256: {digest} (pinned: OK)")


def main() -> int:
    if MODEL_PATH.exists():
        digest = sha256_of(MODEL_PATH)
        if digest != MODEL_SHA256:
            print(
                f"ERROR: {MODEL_PATH} sha256 mismatch\n"
                f"  expected {MODEL_SHA256}\n"
                f"  got      {digest}\n"
                f"Delete the file and re-run `python -m mtmc.get_model`.",
                file=sys.stderr,
            )
            return 1
        print("model already present, hash verified")
        _provenance(digest)
        return 0

    print(f"downloading {MODEL_URL}")
    digest = _download(MODEL_URL, MODEL_PATH)
    if digest != MODEL_SHA256:
        MODEL_PATH.unlink(missing_ok=True)  # never keep an unverified artifact
        print(
            f"ERROR: downloaded file sha256 mismatch\n"
            f"  expected {MODEL_SHA256}\n"
            f"  got      {digest}\n"
            f"Upstream asset changed — re-verify the release before re-pinning.",
            file=sys.stderr,
        )
        return 1
    print("download complete, hash verified")
    _provenance(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
