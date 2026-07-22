"""Extract the CAMS O3-pw winds500 spot-check data (maintainer-side).

The winner-map single-species evaluation (eval_pixel_temporal_multitarget.py)
subsamples each month to <= 80 snapshots (sub = max(1, T//80)). This script
saves exactly those snapshots for the O3 pressure-weighted-column model and
the 500-hPa zonal-wind truth, for the three 2020 dynamics months (Jan/Apr/Jul),
plus the static orography and the normalisation stats — everything needed to
re-derive the O3 row of the released rmaps_winds500.npz stack end-to-end.

Output per month: cams_spot_o3pw_2020_MM.npz with
  gases_col (S,1,H,W) f32 raw, u500 (S,H,W) f32 raw, sel (S,) indices
Plus once: cams_spot_static.npz (lat, lon, hgt) and the stats JSONs.

Usage:
  python extract_cams_spot.py --preproc <PREPROC_DIR> --dst data/cams_spot
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import netCDF4 as nc

MONTHS = (1, 4, 7)
YEAR = 2020


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    pp, dst = Path(args.preproc), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for m in MONTHS:
        with nc.Dataset(pp / f"cams_preprocessed_{YEAR}_{m:02d}_o3pwonly.nc") as ds:
            cv = "gases_col" if "gases_col" in ds.variables else "gases"
            g = np.array(ds[cv][:], dtype=np.float32)
            lat = np.array(ds["latitude"][:], dtype=np.float32)
            lon = np.array(ds["longitude"][:], dtype=np.float32)
        T = g.shape[0]
        sub = max(1, T // 80)
        sel = list(range(0, T, sub))[:80]
        with nc.Dataset(pp / f"cams_dynamics_targets_{YEAR}_{m:02d}.nc") as ds:
            u500 = np.array(ds["winds_500"][:], dtype=np.float32)[sel, 0]
        np.savez_compressed(dst / f"cams_spot_o3pw_{YEAR}_{m:02d}.npz",
                            gases_col=g[sel], u500=u500,
                            sel=np.array(sel), T_month=T)
        print(f"month {m:02d}: T={T} -> {len(sel)} snapshots", flush=True)

    with nc.Dataset(pp / "static_hgt.nc") as ds:
        hgt = np.array(ds["z"][:], dtype=np.float32).squeeze(0)
    np.savez_compressed(dst / "cams_spot_static.npz", lat=lat, lon=lon, hgt=hgt)
    shutil.copy(pp / "stats_o3pwonly.json", dst / "stats_o3pwonly.json")
    shutil.copy(pp / "stats_dynamics.json", dst / "stats_dynamics.json")
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
