"""Bundle the large distribution files for the Zenodo record.

Zenodo caps a record at ~100 files, so the per-snapshot extracts are tarred
(uncompressed — the .npz members are already deflate-compressed). Produces
in --out:

  compass_geoscf_eval_full.tar   (~14 GB, 738 extracted GEOS-CF snapshots)
  compass_tempo_eval_test.tar    (~8.3 GB, 1,227 extracted TEMPO test scans)
  compass_tempo_hour_lookup.npz  (104 MB, train-split scan-hour lookup)
  compass_weights_stats.tar      (checkpoints + stats/climatology files)
  SHA256SUMS                     (checksums of the above)
  MANIFEST.json                  (contents, sizes, provenance)

Usage:
  python package_zenodo.py --out /path/to/zenodo_staging
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]

BUNDLES = {
    "compass_geoscf_eval_full.tar": ["data/geoscf/eval_full"],
    "compass_tempo_eval_test.tar": ["data/tempo/eval_test"],
    "compass_weights_stats.tar": ["weights", "data/geoscf/stats", "data/tempo/stats"],
}
SINGLE = {"compass_tempo_hour_lookup.npz": "data/tempo/stats/hour_lookup.npz"}


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = []
    for tar_name, srcs in BUNDLES.items():
        dst = out / tar_name
        if dst.exists():
            print(f"skip (exists): {tar_name}", flush=True)
        else:
            print(f"tar -> {tar_name}", flush=True)
            cmd = ["tar", "-cf", str(dst), "-C", str(PKG)] + srcs
            subprocess.run(cmd, check=True)
        made.append(dst)
    for name, rel in SINGLE.items():
        dst = out / name
        if not dst.exists():
            subprocess.run(["cp", str(PKG / rel), str(dst)], check=True)
        made.append(dst)

    manifest = {}
    sums = []
    for p in made:
        print(f"sha256 {p.name} ...", flush=True)
        digest = sha256(p)
        sums.append(f"{digest}  {p.name}")
        manifest[p.name] = {"bytes": p.stat().st_size, "sha256": digest}
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "MANIFEST.json").write_text(json.dumps({
        "package": "compass-repro Zenodo data record",
        "provenance": "extracted evaluation variables + model weights; see the "
                      "GitHub repository README for formats and protocols",
        "files": manifest}, indent=2))
    print("done:", *(p.name for p in made), sep="\n  ")


if __name__ == "__main__":
    sys.exit(main())
