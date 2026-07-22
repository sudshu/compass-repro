"""Reviewer-side: download the COMPASS data bundles from Zenodo and unpack.

Usage:
  python scripts/fetch_data.py --record <zenodo-record-id> [--only geoscf|tempo]

Downloads into data/zenodo_staging/, verifies SHA256SUMS, and unpacks the
tars into the repository layout (data/geoscf/eval_full, data/tempo/eval_test,
weights/, data/*/stats/).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
API = "https://zenodo.org/api/records"


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--only", default=None, choices=[None, "geoscf", "tempo"])
    ap.add_argument("--full", action="store_true",
                    help="also download the 14 GB full GEOS-CF archive "
                         "(default: the 3.5 GB stride-4 standard bundle)")
    args = ap.parse_args()

    import json
    with urllib.request.urlopen(f"{API}/{args.record}") as r:
        rec = json.load(r)
    staging = PKG / "data/zenodo_staging"
    staging.mkdir(parents=True, exist_ok=True)

    files = rec["files"]
    if not args.full:
        files = [f for f in files if f["key"] != "compass_geoscf_eval_full.tar"]
    if args.only:
        files = [f for f in files
                 if args.only in f["key"] or f["key"] in ("SHA256SUMS", "MANIFEST.json")]
    for f in files:
        dst = staging / f["key"]
        if dst.exists() and dst.stat().st_size == f["size"]:
            print(f"skip (exists): {f['key']}")
            continue
        print(f"downloading {f['key']} ({f['size'] / 1e9:.2f} GB)...", flush=True)
        urllib.request.urlretrieve(f["links"]["self"], dst)

    sums = (staging / "SHA256SUMS").read_text().strip().splitlines()
    for line in sums:
        digest, name = line.split()
        p = staging / name
        if not p.exists():
            continue
        ok = sha256(p) == digest
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            sys.exit(f"checksum mismatch: {name}")

    for tar in staging.glob("compass_*.tar"):
        print(f"unpacking {tar.name}...", flush=True)
        subprocess.run(["tar", "-xf", str(tar), "-C", str(PKG)], check=True)
    lk = staging / "compass_tempo_hour_lookup.npz"
    if lk.exists():
        (PKG / "data/tempo/stats").mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", str(lk), str(PKG / "data/tempo/stats/hour_lookup.npz")],
                       check=True)
    print("done — data unpacked into the repository layout.")


if __name__ == "__main__":
    main()
