"""Extract compact evaluation snapshots from the GEOS-CF NetCDF archive.

Maintainer-side script (reviewers receive its OUTPUT). For each
``geos_cf_YYYYMMDD_HH.nc`` this stores exactly what the blocked evaluation
needs, as float16:

- ``col_n`` (4,H,W): CH4/CO/O3/NO2 column fields after per-file zonal-mean
  removal, NaN->0, and z-scoring with the TRAINING stats_geos_cf.json —
  i.e. the model's input channels exactly as built by
  ml_geos_cf/data.py::GeosCfDataset (values O(1), safe in float16; the raw
  physical fields span ~1e-19..1e16 and are NOT float16-representable);
- ``sfc_n`` (4,H,W): same for the near-surface fields;
- ``met`` (5,H,W): u10m, v10m, u500, v500, zpbl with |x|>1e10 fills -> 0
  (physical units; float16 error <0.05% of the anomaly scale).

The training stats file ships alongside, so the normalisation is auditable.
Grid (lat/lon) is not duplicated per file; it ships once in clim_train.npz.

Usage:
  python extract_geoscf.py --src <eval7 dir> --dst <out dir> --stats <stats_geos_cf.json>
                           [--stride N] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import netCDF4 as nc

GASES = ["ch4", "co", "o3", "no2"]


def extract_one(fp: Path, dst: Path, stats: dict) -> None:
    with nc.Dataset(fp) as ds:
        col = np.stack([np.array(ds[f"{g}_col"][:], dtype=np.float32) for g in GASES], 0)
        sfc = np.stack([np.array(ds[f"{g}_sfc"][:], dtype=np.float32) for g in GASES], 0)

        def _clean_met(a):
            a = np.array(a, dtype=np.float32)
            a = np.where(np.abs(a) > 1e10, np.nan, a)
            return np.nan_to_num(a, nan=0.0)

        met = np.stack([_clean_met(ds["u10m"][:]), _clean_met(ds["v10m"][:]),
                        _clean_met(ds["u500"][:]), _clean_met(ds["v500"][:]),
                        _clean_met(ds["zpbl"][:])], axis=0)

    col_anom = np.nan_to_num(col - np.nanmean(col, axis=-1, keepdims=True), nan=0.0)
    sfc_anom = np.nan_to_num(sfc - np.nanmean(sfc, axis=-1, keepdims=True), nan=0.0)

    x_mean_col = np.array(stats["x_mean_col"], dtype=np.float32)
    x_std_col = np.array(stats["x_std_col"], dtype=np.float32)
    x_mean_sfc = np.array(stats["x_mean_sfc"], dtype=np.float32)
    x_std_sfc = np.array(stats["x_std_sfc"], dtype=np.float32)
    col_n = (col_anom - x_mean_col[:, None, None]) / (x_std_col[:, None, None] + 1e-8)
    sfc_n = (sfc_anom - x_mean_sfc[:, None, None]) / (x_std_sfc[:, None, None] + 1e-8)

    np.savez_compressed(dst / (fp.stem + ".npz"),
                        col_n=col_n.astype(np.float16),
                        sfc_n=sfc_n.astype(np.float16),
                        met=met.astype(np.float16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--stats", required=True, help="training stats_geos_cf.json")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth file")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(args.stats).read_text())
    files = sorted(src.glob("geos_cf_*.nc"))[::args.stride]
    if args.limit:
        files = files[:args.limit]
    for i, fp in enumerate(files):
        extract_one(fp, dst, stats)
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"[{i + 1}/{len(files)}] {fp.name}", flush=True)
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
