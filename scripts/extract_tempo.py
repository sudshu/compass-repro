"""Extract compact TEMPO test snapshots for the interspersed evaluation.

Maintainer-side script (reviewers receive its OUTPUT). For each test snapshot
listed in the run's ``files.json`` this stores, as float16:

- ``no2_n``  (H,W): the model's z-scored log10-NO2 channel, NaN where the
  TEMPO retrieval is invalid (raw columns ~1e15 molec/cm2 overflow float16;
  the z-scored channel is O(1)). ``isfinite(no2_n)`` reproduces the original
  validity mask ``isfinite(no2) & (no2 > 0)`` exactly.
- ``<extra>_n`` (H,W) for the six TEMPO extras (o3_col, uv_aerosol_index,
  so2_index, hcho_col, cloud_fraction, cloud_pressure): sanitised
  (fill sentinels and out-of-range cloud values -> NaN, as dataset_v9 does)
  then z-scored with the run's stats.json; NaN preserved.
- ``hrrr_u10``, ``hrrr_v10``, ``hrrr_pblh`` (H,W): physical units
  (float16 is exact enough: max |error| ~0.03 m/s for winds, ~4 m for PBLH,
  i.e. <1% of the anomaly scale; NaN coverage mask preserved).
- ``ts``: the snapshot timestamp string.

The run's ``stats.json`` ships alongside, so the normalisation is auditable.

Usage:
  python extract_tempo.py --grid-dir <grids_compact> --run-dir weights/tempo \
      --dst data/tempo/eval_test [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXTRA_KEYS = ("o3_col", "uv_aerosol_index", "so2_index",
              "hcho_col", "cloud_fraction", "cloud_pressure")


def sanitize_extra(k: str, a: np.ndarray) -> np.ndarray:
    """Identical to dataset_v9.load_snapshot_v9 sentinel handling."""
    a = np.where(np.abs(a) > 1e20, np.nan, a)
    if k == "cloud_fraction":
        a = np.where((a >= -0.2) & (a <= 1.5), a, np.nan)
    elif k == "cloud_pressure":
        a = np.where((a > 0.0) & (a <= 1100.0), a, np.nan)
    return a


def extract_one(fp: Path, dst: Path, stats: dict) -> None:
    z = np.load(fp, allow_pickle=True)
    no2 = np.array(z["tempo_no2"]).astype(np.float32)
    out = {"ts": str(z["ts"])}

    valid = np.isfinite(no2) & (no2 > 0)
    mu, sd = stats["no2_log"]["mean"], stats["no2_log"]["std"]
    with np.errstate(invalid="ignore", divide="ignore"):
        no2_log = np.log10(no2 + 1e-3)
    no2_n = np.where(valid, (no2_log - mu) / sd, np.nan).astype(np.float16)
    out["no2_n"] = no2_n

    for k in EXTRA_KEYS:
        if k in z.files:
            a = sanitize_extra(k, np.array(z[k]).astype(np.float32))
            a_n = (a - stats[k]["mean"]) / stats[k]["std"]
            out[f"{k}_n"] = a_n.astype(np.float16)
    for k in ("hrrr_u10", "hrrr_v10", "hrrr_pblh"):
        out[k] = np.array(z[k]).astype(np.float16)
    z.close()
    np.savez_compressed(dst / fp.name, **out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--run-dir", required=True,
                    help="dir with files.json + stats.json (e.g. weights/tempo)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rd, gd, dst = Path(args.run_dir), Path(args.grid_dir), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    stats = json.loads((rd / "stats.json").read_text())
    files = [gd / Path(p).name
             for p in json.loads((rd / "files.json").read_text())[args.split]]
    if args.limit:
        files = files[:args.limit]
    for i, fp in enumerate(files):
        extract_one(fp, dst, stats)
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            print(f"[{i + 1}/{len(files)}] {fp.name}", flush=True)
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
