"""Anomaly-target patch dataset.

Target = (HRRR truth − per-cell climatology) / anom_std.

Inputs (variable):
  always: log10(NO2), valid_mask, hgt_norm, landmask, sin(lat), cos(lon)
  optional (if include_o3): o3_trop, o3_strat, o3_layer_15, o3_layer_22, o3_layer_23

The model predicts the normalised anomaly; inference adds back climatology
to recover physical units. With this target, MSE training only rewards
*time-varying* skill — climatology is no longer the free lunch.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
# Allow pointing at a compact/compressed snapshot dir via env (cache-friendly).
GRID_DIR = Path(os.environ.get("ML_GRID_DIR", str(HERE / "grids")))
STATIC_AUX = HERE / "static_aux.npz"
STATIC_CLIM = HERE / "static_clim.npz"

PATCH = 16
MIN_TEMPO_COV = 0.7
MIN_HRRR_COV = 0.95
N_TRIES = 30

O3_KEYS = ("o3_trop", "o3_strat", "o3_layer_15", "o3_layer_22", "o3_layer_23")


def load_aux():
    z = np.load(STATIC_AUX, allow_pickle=False)
    return {k: np.array(z[k]) for k in ("lat", "lon", "hgt", "lsm")}


def load_clim():
    z = np.load(STATIC_CLIM, allow_pickle=False)
    return {k: np.array(z[k]) for k in ("lat", "lon", "clim", "anom_std")}


_aux = None; _clim = None
def get_aux():
    global _aux
    if _aux is None: _aux = load_aux()
    return _aux
def get_clim():
    global _clim
    if _clim is None: _clim = load_clim()
    return _clim


def load_snapshot(p: Path):
    z = np.load(p, allow_pickle=True)
    out = {
        "no2": np.array(z["tempo_no2"]).astype(np.float32),
        "u10": np.array(z["hrrr_u10"]).astype(np.float32),
        "v10": np.array(z["hrrr_v10"]).astype(np.float32),
        "pblh": np.array(z["hrrr_pblh"]).astype(np.float32),
        "ts": str(z["ts"]),
    }
    keys = set(z.files)
    for k in O3_KEYS:
        if k in keys:
            out[k] = np.array(z[k]).astype(np.float32)
    z.close()
    return out


def discover(require_o3: bool = False):
    files = sorted(GRID_DIR.glob("snapshot_*.npz"))
    if not require_o3:
        return files
    keep = []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        if "o3_trop" in z.files:
            keep.append(fp)
        z.close()
    return keep


def split_files(files, val_frac=0.15, test_frac=0.15, seed=0):
    files = sorted(files)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_test = int(len(files) * test_frac); n_val = int(len(files) * val_frac)
    return ([files[i] for i in sorted(idx[n_test+n_val:])],
            [files[i] for i in sorted(idx[n_test:n_test+n_val])],
            [files[i] for i in sorted(idx[:n_test])])


def compute_input_stats(snapshots, include_o3, max_files=200):
    """Means/std for input channels only."""
    files = snapshots[:min(max_files, len(snapshots))]
    no2_vals = []
    o3_vals = {k: [] for k in O3_KEYS}
    for fp in files:
        s = load_snapshot(fp)
        no2 = s["no2"]
        v = np.isfinite(no2) & (no2 > 0)
        no2_vals.append(np.log10(no2[v] + 1e-3))
        if include_o3:
            for k in O3_KEYS:
                if k in s:
                    arr = s[k]; vv = np.isfinite(arr); o3_vals[k].append(arr[vv].ravel())
    aux = get_aux()
    hgt = aux["hgt"][np.isfinite(aux["hgt"])].ravel()
    stats = {
        "no2_log": {"mean": float(np.concatenate(no2_vals).mean()),
                    "std":  float(np.concatenate(no2_vals).std() + 1e-8)},
        "hgt":     {"mean": float(hgt.mean()), "std": float(hgt.std() + 1e-8)},
    }
    if include_o3:
        for k in O3_KEYS:
            vals = np.concatenate(o3_vals[k]) if o3_vals[k] else np.array([0.0, 1.0])
            stats[k] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-8)}
    return stats


class AnomalyPatchDataset(Dataset):
    def __init__(self, files, stats, n_patches_per_snapshot: int, train: bool,
                 include_o3: bool = False, patch: int = PATCH, seed: int = 0):
        self.files = list(files)
        self.stats = stats
        self.train = train
        self.include_o3 = include_o3
        self.n = int(n_patches_per_snapshot)
        self.patch = patch
        # Bounded LRU cache: at 14k+ snapshots the unbounded dict ate ~100 GB
        # per DataLoader worker × N seeds → OOM. 256 snapshots ≈ 3.3 GB cache,
        # safe per-worker even with 8 workers × 3 seeds.
        from collections import OrderedDict
        self._cache: OrderedDict[Path, dict] = OrderedDict()
        self._cache_max = 256
        self._rng = np.random.default_rng(seed)

        aux = get_aux(); cl = get_clim()
        self.lat = aux["lat"]; self.lon = aux["lon"]
        self.H = self.lat.size; self.W = self.lon.size
        LON2D, LAT2D = np.meshgrid(self.lon, self.lat)
        self.sin_lat = np.sin(np.deg2rad(LAT2D)).astype(np.float32)
        self.cos_lon = np.cos(np.deg2rad(LON2D)).astype(np.float32)
        h_mu = stats["hgt"]["mean"]; h_sd = stats["hgt"]["std"]
        self.hgt_norm = np.where(np.isfinite(aux["hgt"]),
                                 (aux["hgt"] - h_mu) / h_sd, 0.0).astype(np.float32)
        self.lsm_norm = np.where(np.isfinite(aux["lsm"]), aux["lsm"], 0.0).astype(np.float32)
        self.clim = cl["clim"].astype(np.float32)            # (3, H, W)
        self.anom_std = cl["anom_std"].astype(np.float32)    # (3,)

    @property
    def n_in(self):
        return 6 + (len(O3_KEYS) if self.include_o3 else 0)

    @property
    def n_out(self):
        return 3

    def __len__(self):
        return len(self.files) * self.n

    def _get(self, i):
        fp = self.files[i]
        if fp in self._cache:
            self._cache.move_to_end(fp)
            return self._cache[fp]
        snap = load_snapshot(fp)
        self._cache[fp] = snap
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)   # evict oldest
        return snap

    def _patch_idx(self, no2_finite, hrrr_finite):
        H, W = no2_finite.shape; P = self.patch
        for _ in range(N_TRIES):
            i = int(self._rng.integers(0, H - P + 1))
            j = int(self._rng.integers(0, W - P + 1))
            if (no2_finite[i:i+P, j:j+P].mean() >= MIN_TEMPO_COV and
                hrrr_finite[i:i+P, j:j+P].mean() >= MIN_HRRR_COV):
                return i, j
        for _ in range(N_TRIES):
            i = int(self._rng.integers(0, H - P + 1))
            j = int(self._rng.integers(0, W - P + 1))
            if hrrr_finite[i:i+P, j:j+P].mean() >= MIN_HRRR_COV:
                return i, j
        return 0, 0

    def __getitem__(self, idx):
        i_file = idx // self.n
        s = self._get(i_file)
        no2 = s["no2"]
        u10 = s["u10"]; v10 = s["v10"]; pblh = s["pblh"]
        no2_finite = np.isfinite(no2) & (no2 > 0)
        hrrr_finite = np.isfinite(u10) & np.isfinite(v10) & np.isfinite(pblh)
        i, j = self._patch_idx(no2_finite, hrrr_finite)
        P = self.patch

        # NO2 normalised log
        no2_p = no2[i:i+P, j:j+P]
        valid_no2 = (np.isfinite(no2_p) & (no2_p > 0)).astype(np.float32)
        mu = self.stats["no2_log"]["mean"]; sd = self.stats["no2_log"]["std"]
        no2_log = np.where(valid_no2 > 0, np.log10(no2_p + 1e-3), mu)
        no2_n = ((no2_log - mu) / sd).astype(np.float32)

        chans = [no2_n, valid_no2,
                 self.hgt_norm[i:i+P, j:j+P], self.lsm_norm[i:i+P, j:j+P],
                 self.sin_lat[i:i+P, j:j+P], self.cos_lon[i:i+P, j:j+P]]

        # O3 channels if requested
        if self.include_o3:
            for k in O3_KEYS:
                arr = s.get(k)
                if arr is None:
                    chans.append(np.zeros((P, P), dtype=np.float32))
                else:
                    mu_, sd_ = self.stats[k]["mean"], self.stats[k]["std"]
                    a = arr[i:i+P, j:j+P]
                    a_n = (a - mu_) / sd_
                    chans.append(np.where(np.isfinite(a_n), a_n, 0.0).astype(np.float32))

        x = np.stack(chans, axis=0).astype(np.float32)

        # Anomaly target: (truth - clim) / anom_std
        u_p = u10[i:i+P, j:j+P]; v_p = v10[i:i+P, j:j+P]; b_p = pblh[i:i+P, j:j+P]
        clim_u = self.clim[0, i:i+P, j:j+P]
        clim_v = self.clim[1, i:i+P, j:j+P]
        clim_b = self.clim[2, i:i+P, j:j+P]

        def norm_anom(arr, clim, sd):
            anom = arr - clim
            y = anom / sd
            valid = np.isfinite(y)
            return np.where(valid, y, 0.0).astype(np.float32), valid

        y_u, mu_v = norm_anom(u_p, clim_u, self.anom_std[0])
        y_v, mv_v = norm_anom(v_p, clim_v, self.anom_std[1])
        y_b, mp_v = norm_anom(b_p, clim_b, self.anom_std[2])
        y = np.stack([y_u, y_v, y_b], axis=0).astype(np.float32)
        m = (mu_v & mv_v & mp_v).astype(np.float32)

        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m))
