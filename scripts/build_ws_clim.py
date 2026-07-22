"""Precompute the training-split wind-speed climatology (maintainer-side).

Verbatim convention from blocked_results_kit.py::build_ws_clim /
eval_ws_fig1.py: time-mean of hypot(u, v) over the 70% random-split
(rng seed 0) training files of the GEOS-CF training archive — NOT the
hypot of the component climatologies. Needed for the Fig. 3a wind-SPEED
pooled ACC (manuscript: 500-hPa wind speed 0.71 vs 0.59 persistence).

Output ws_clim.npz: ws_sfc (H,W), ws_500 (H,W) float32, full globe.

Usage:
  python build_ws_clim.py --train-root <geos_cf_smoke> \
      --out data/geoscf/stats/ws_clim.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import netCDF4 as nc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_files = sorted(Path(args.train_root).rglob("geos_cf_*.nc"))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(all_files))
    n_train = int(round(0.70 * len(all_files)))
    train_files = [all_files[i] for i in idx[:n_train]]
    print(f"{len(train_files)} training files (70% of {len(all_files)}, rng seed 0)")

    s = {"ws_sfc": 0.0, "ws_500": 0.0}
    for i, f in enumerate(train_files):
        with nc.Dataset(f) as ds:
            for key, (uv, vv) in {"ws_sfc": ("u10m", "v10m"),
                                  "ws_500": ("u500", "v500")}.items():
                u = np.array(ds[uv][:], np.float32)
                v = np.array(ds[vv][:], np.float32)
                u = np.nan_to_num(np.where(np.abs(u) > 1e10, np.nan, u), nan=0.0)
                v = np.nan_to_num(np.where(np.abs(v) > 1e10, np.nan, v), nan=0.0)
                s[key] = s[key] + np.hypot(u, v)
        if (i + 1) % 50 == 0 or i + 1 == len(train_files):
            print(f"  [{i + 1}/{len(train_files)}]", flush=True)

    np.savez_compressed(args.out,
                        ws_sfc=(s["ws_sfc"] / len(train_files)).astype(np.float32),
                        ws_500=(s["ws_500"] / len(train_files)).astype(np.float32))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
