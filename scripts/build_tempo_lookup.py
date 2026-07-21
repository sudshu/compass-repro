"""Precompute the train-split scan-hour lookup for the TEMPO evaluation.

Maintainer-side script. The clock baseline and the scan-hour-conditioned
score need the per-pixel x per-scan-hour mean of the normalised anomaly
target over the 11,798 TRAIN snapshots. Shipping those snapshots is
impractical, so this script reproduces ``build_lookup`` from the original
``diurnal_control_intersp.py`` (float64 accumulation, identical hour
assignment) and stores the result as one npz that the packaged evaluation
loads directly.

Output ``hour_lookup.npz``: for each usable hour h, ``lookup_<h>`` (3,H,W)
float32 and ``count_<h>`` (int); plus ``hours`` (the sorted hour list).

Usage:
  python build_tempo_lookup.py --grid-dir <grids_compact> --run-dir weights/tempo \
      --clim data/tempo/stats/static_clim_interspersed.npz \
      --out data/tempo/stats/hour_lookup.npz --workers 24
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SNAP_RE = re.compile(r"snapshot_(\d{8})T(\d{6})Z\.npz$")


def snap_start(name: str) -> datetime:
    m = SNAP_RE.search(name)
    d, t = m.group(1), m.group(2)
    return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]),
                    int(t[:2]), int(t[2:4]), int(t[4:6]))


def target_hour(name: str) -> int:
    ts = snap_start(name)
    h = ts.replace(minute=0, second=0, microsecond=0)
    if ts.minute >= 30:
        h += timedelta(hours=1)
    return h.hour


_CL = {}


def _accum_one(args):
    fp, hour = args
    z = np.load(fp, allow_pickle=True)
    out = []
    for k in ("hrrr_u10", "hrrr_v10", "hrrr_pblh"):
        out.append(np.array(z[k]).astype(np.float64))
    z.close()
    y = np.stack(out, 0)
    a = (y - _CL["clim"]) / _CL["anom_std"][:, None, None]
    v = np.isfinite(a)
    return hour, np.where(v, a, 0.0), v.astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--clim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    global _CL
    cl = np.load(args.clim)
    clim = cl["clim"].astype(np.float64)
    anom_std = cl["anom_std"].astype(np.float64)
    _CL = {"clim": clim, "anom_std": anom_std}

    gd = Path(args.grid_dir)
    train_files = [gd / Path(p).name
                   for p in json.loads((Path(args.run_dir) / "files.json").read_text())["train"]]
    jobs = [(fp, target_hour(fp.name)) for fp in train_files]
    hours = sorted({h for _, h in jobs})
    C, H, W = clim.shape
    s = {h: np.zeros((3, H, W)) for h in hours}
    n = {h: np.zeros((3, H, W), dtype=np.int64) for h in hours}
    done = 0
    with Pool(args.workers) as pool:
        for hour, a, v in pool.imap_unordered(_accum_one, jobs, chunksize=16):
            s[hour] += a
            n[hour] += v
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}]", flush=True)

    out = {"hours": np.array(hours, dtype=np.int64)}
    for h in hours:
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s[h] / n[h]
        out[f"lookup_{h}"] = np.where(n[h] > 0, m, 0.0).astype(np.float32)
        out[f"count_{h}"] = np.array(int(n[h][0].max()), dtype=np.int64)
    np.savez_compressed(args.out, **out)
    print("counts:", {h: int(out[f'count_{h}']) for h in hours})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
