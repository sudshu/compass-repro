"""Diurnal-shortcut control for the TEMPO->HRRR result (runbook §2; eval-only).

Question: how much of the corrected TEMPO skill (winds 0.59/0.58, PBLH 0.80
pooled ACC) is solar-time predictability? The anomaly targets subtract a STATIC
per-pixel climatology, so the diurnal cycle remains in the target and
photochemistry encodes the clock in the inputs.

Two designs, both leakage-free (lookup built from the TRAIN split only):

A) Clock baseline: per-pixel x per-scan-hour mean of the normalised anomaly
   target over the train split; the baseline "prediction" for a test patch is
   the lookup at that snapshot's analysis hour. Scored with the identical
   patch protocol and fp32 masked_anom_corr as the network. If this rivals the
   network, the skill is diurnal bookkeeping.

B) Hour-conditioned skill: subtract the same lookup from BOTH the network
   prediction and the truth, then score. This is the network's skill beyond
   the hour-resolved climatology — the defensible headline qualifier.

Also reports var(lookup)/var(y) per target (how large the diurnal component is
in the target itself) and the plain fp32 network score (sanity: must reproduce
corrected_test_metrics).

Runs against v9 runs (--dataset v9) and the mask-only control (--dataset
maskonly). Paths in files.json are remapped onto --grid-dir by basename, so
the same run dirs work on any machine holding the snapshots.

Usage (az-ms):
  python3 diurnal_control.py --run-dir runs/<v17_s0> --dataset v9 \
      --grid-dir /data/pandeysu/stage2b_train/grids_fast --workers 32
"""
from __future__ import annotations

# --- blocked-clim injection (auto; matches train_v9_blocked) ---
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import dataset_v5 as _dv5
_dv5.STATIC_CLIM = _pl.Path(__file__).resolve().parent / 'static_clim_interspersed.npz'
assert _dv5.STATIC_CLIM.exists(), 'static_clim_interspersed.npz missing'
print('[blocked-eval] clim ->', _dv5.STATIC_CLIM.name, flush=True)
# --- end injection ---

import argparse, json, re
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import UNet2to3

HERE = Path(__file__).resolve().parent
SNAP_RE = re.compile(r"snapshot_(\d{8})T(\d{6})Z\.npz$")
MIN_HOUR_COUNT = 20   # train snapshots required for an hour bin to be usable


def snap_start(name: str) -> datetime:
    m = SNAP_RE.search(name)
    d, t = m.group(1), m.group(2)
    return datetime(int(d[:4]), int(d[4:6]), int(d[6:8]),
                    int(t[:2]), int(t[2:4]), int(t[4:6]))


def target_hour(name: str) -> int:
    """UTC hour of the HRRR analysis paired at build time (nearest to start)."""
    ts = snap_start(name)
    h = ts.replace(minute=0, second=0, microsecond=0)
    if ts.minute >= 30:
        h += timedelta(hours=1)
    return h.hour


def remap(files, grid_dir: Path):
    return [grid_dir / Path(p).name for p in files]


# ---------------------------------------------------------------- lookup ----
_CL = {}


def _accum_one(args):
    fp, hour = args
    z = np.load(fp, allow_pickle=True)
    out = []
    for k in ("hrrr_u10", "hrrr_v10", "hrrr_pblh"):
        out.append(np.array(z[k]).astype(np.float64))
    z.close()
    y = np.stack(out, 0)                       # physical units (3,H,W)
    a = (y - _CL["clim"]) / _CL["anom_std"][:, None, None]
    v = np.isfinite(a)
    return hour, np.where(v, a, 0.0), v.astype(np.int32)


def build_lookup(train_files, grid_dir, workers, clim, anom_std):
    """Per-hour per-pixel mean normalised anomaly from the TRAIN split."""
    global _CL
    _CL = {"clim": clim.astype(np.float64), "anom_std": anom_std.astype(np.float64)}
    jobs = [(fp, target_hour(fp.name)) for fp in train_files]
    hours = sorted({h for _, h in jobs})
    C, H, W = clim.shape
    s = {h: np.zeros((3, H, W)) for h in hours}
    n = {h: np.zeros((3, H, W), dtype=np.int64) for h in hours}
    with Pool(workers) as pool:
        for hour, a, v in pool.imap_unordered(_accum_one, jobs, chunksize=16):
            s[hour] += a
            n[hour] += v
    lookup, counts = {}, {}
    for h in hours:
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s[h] / n[h]
        lookup[h] = np.where(n[h] > 0, m, 0.0).astype(np.float32)
        counts[h] = int(n[h][0].max())         # snapshots contributing
    return lookup, counts


# --------------------------------------------------------------- dataset ----
def make_dataset(kind, files, stats, n_eval, seed):
    if kind == "v9":
        from dataset_v9 import AnomalyPatchDatasetV9 as Base
    elif kind == "maskonly":
        from train_v8maskonly import AnomalyPatchDatasetV8MaskOnly as Base
    else:
        raise ValueError(kind)

    class WithMeta(Base):
        def __getitem__(self, idx):
            i_file = idx // self.n
            out = super().__getitem__(idx)
            x, y, m = out[0], out[1], out[2]
            # recover the patch corner chosen by the parent: the parent already
            # consumed the rng; re-deriving would desync. Instead the parent is
            # patched below to record it.
            i, j = self._last_ij
            return x, y, m, i_file, i, j

        def _patch_idx(self, no2_finite, hrrr_finite):
            ij = super()._patch_idx(no2_finite, hrrr_finite)
            self._last_ij = ij
            return ij

    return WithMeta(files, stats, n_eval, train=False, seed=seed)


def corr_fp32(pred, target, mask):
    rs = []
    for c in range(pred.shape[1]):
        p = pred[:, c].float(); t = target[:, c].float()
        v = mask > 0.5
        if v.sum() == 0:
            rs.append(float("nan")); continue
        pv = p[v] - p[v].mean(); tv = t[v] - t[v].mean()
        denom = pv.pow(2).sum().sqrt() * tv.pow(2).sum().sqrt()
        rs.append(float((pv * tv).sum() / (denom + 1e-8)))
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True, choices=["v9", "maskonly"])
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.run_dir); grid_dir = Path(args.grid_dir)
    cfg = json.load(open(rd / "config.json"))
    stats = json.load(open(rd / "stats.json"))
    fj = json.load(open(rd / "files.json"))
    train_f = remap(fj["train"], grid_dir)
    test_f = remap(fj["test"], grid_dir)
    seed = int(cfg.get("seed", 0))

    from dataset_v5 import get_clim
    cl = get_clim()
    clim = cl["clim"].astype(np.float32); anom_std = cl["anom_std"].astype(np.float32)

    print(f"building hour lookup from {len(train_f)} train snapshots...", flush=True)
    lookup, counts = build_lookup(train_f, grid_dir, args.workers, clim, anom_std)
    ok_hours = {h for h, c in counts.items() if c >= MIN_HOUR_COUNT}
    print("hour bins (snapshots):",
          {h: counts[h] for h in sorted(counts)}, "| usable:", sorted(ok_hours),
          flush=True)

    # diurnal variance share of the target itself
    stack = np.stack([lookup[h] for h in sorted(ok_hours)], 0)   # (Nh,3,H,W)
    var_clock = np.nanvar(stack, axis=0).mean(axis=(1, 2))       # (3,)

    ds = make_dataset(args.dataset, test_f, stats, cfg.get("patches_per_snap_eval", 16),
                      seed=seed + 2)
    dl = DataLoader(ds, batch_size=int(cfg.get("batch", 128)), shuffle=False,
                    num_workers=0)   # workers=0: _last_ij metadata must stay in-process
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2to3(c_in=ds.n_in, c_out=3, base=int(cfg.get("base", 48)),
                     depth=int(cfg.get("depth", 3)))
    ck = torch.load(rd / "best.pt", map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.to(dev).eval()

    file_hours = [target_hour(fp.name) for fp in test_f]
    P = ds.patch
    names = ("r_anom_u10", "r_anom_v10", "r_anom_pblh")
    rows = {k: [[], [], []] for k in
            ("network_fp32", "clock_baseline", "network_hourcond")}
    n_skip = 0

    with torch.no_grad():
        for x, y, m, i_file, ii, jj in dl:
            B = x.shape[0]
            clock = torch.zeros_like(y)
            keep = torch.ones(B)
            for b in range(B):
                h = file_hours[int(i_file[b])]
                if h not in ok_hours:
                    keep[b] = 0.0; continue
                i0, j0 = int(ii[b]), int(jj[b])
                clock[b] = torch.from_numpy(
                    lookup[h][:, i0:i0 + P, j0:j0 + P])
            n_skip += int((keep == 0).sum())
            m_eff = m * keep.view(-1, 1, 1)

            x = x.to(dev); y = y.to(dev)
            m_eff = m_eff.to(dev); clock = clock.to(dev)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                p = model(x)
            p = p.float()

            for ci, r in enumerate(corr_fp32(p, y, m_eff)):
                rows["network_fp32"][ci].append(r)
            for ci, r in enumerate(corr_fp32(clock, y, m_eff)):
                rows["clock_baseline"][ci].append(r)
            for ci, r in enumerate(corr_fp32(p - clock, y - clock, m_eff)):
                rows["network_hourcond"][ci].append(r)

    out = {"run": rd.name, "dataset": args.dataset,
           "hour_counts": {str(h): counts[h] for h in sorted(counts)},
           "skipped_patches_offhour": n_skip,
           "clock_variance_share_note":
               "variance of the hour lookup across usable hours, per target, "
               "in normalised-anomaly units (target variance ~1 by construction)",
           "clock_variance": {k: float(var_clock[ci]) for ci, k in enumerate(names)}}
    for key, rr in rows.items():
        out[key] = {k: float(np.nanmean(rr[ci])) for ci, k in enumerate(names)}
    dst = Path(args.out) if args.out else rd / "diurnal_control.json"
    json.dump(out, open(dst, "w"), indent=2)

    print(f"\n=== {rd.name} ({args.dataset}) ===")
    for key in ("network_fp32", "clock_baseline", "network_hourcond"):
        mvals = out[key]
        print(f"  {key:18s} u10={mvals['r_anom_u10']:.3f} "
              f"v10={mvals['r_anom_v10']:.3f} pblh={mvals['r_anom_pblh']:.3f}")
    print(f"  clock variance share: " +
          " ".join(f"{k.split('_')[-1]}={out['clock_variance'][k]:.3f}"
                   for k in names))
    print(f"  off-hour skipped patches: {n_skip}")
    print(f"saved {dst}", flush=True)


if __name__ == "__main__":
    main()
