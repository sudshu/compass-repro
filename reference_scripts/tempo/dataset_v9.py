"""Dataset v9 — ALL TEMPO products, NO GOES.

Inputs (12 channels):
  base (6): log10(NO2), valid_mask, hgt_norm, landmask, sin(lat), cos(lon)
  TEMPO extras (6, z-scored, missing->0):
    o3_col (O3TOT total ozone), uv_aerosol_index, so2_index,
    hcho_col (formaldehyde column), cloud_fraction, cloud_pressure

Target: HRRR U10/V10/PBLH anomaly (same as v5-v8).
discover_v9 requires 'hcho_col' present (the gating new product); any individual
extra that is missing is zero-filled.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

from dataset_v5 import (PATCH, MIN_TEMPO_COV, MIN_HRRR_COV, N_TRIES,
                          GRID_DIR, get_aux, get_clim, split_files)

# z-scored extra channels (plain standardization, NOT log)
TEMPO_EXTRA_KEYS = ("o3_col", "uv_aerosol_index", "so2_index",
                     "hcho_col", "cloud_fraction", "cloud_pressure")


def _np_load_retry(p: Path, tries: int = 5):
    import time as _t
    for k in range(tries):
        try:
            return np.load(p, allow_pickle=True)
        except Exception:
            if k == tries - 1:
                raise
            _t.sleep(0.3 * (k + 1))


def load_snapshot_v9(p: Path):
    z = _np_load_retry(p)
    out = {
        "no2":  np.array(z["tempo_no2"]).astype(np.float32),
        "u10":  np.array(z["hrrr_u10"]).astype(np.float32),
        "v10":  np.array(z["hrrr_v10"]).astype(np.float32),
        "pblh": np.array(z["hrrr_pblh"]).astype(np.float32),
        "ts":   str(z["ts"]),
    }
    for k in TEMPO_EXTRA_KEYS:
        if k in z.files:
            a = np.array(z[k]).astype(np.float32)
            # Sanitize TEMPO L3 fill sentinels (-1e30, 9.97e36) that leaked
            # through bin-averaging for products without a QA flag (cloud).
            a = np.where(np.abs(a) > 1e20, np.nan, a)
            if k == "cloud_fraction":
                a = np.where((a >= -0.2) & (a <= 1.5), a, np.nan)
            elif k == "cloud_pressure":
                a = np.where((a > 0.0) & (a <= 1100.0), a, np.nan)
            out[k] = a
    z.close()
    return out


def discover_v9(require_hcho: bool = True):
    files = sorted(GRID_DIR.glob("snapshot_*.npz"))
    keep = []
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        ok = ("hcho_col" in z.files) if require_hcho else True
        z.close()
        if ok: keep.append(fp)
    return keep


def compute_input_stats_v9(snapshots, max_files=300):
    files = snapshots[:min(max_files, len(snapshots))]
    no2_vals = []
    extra = {k: [] for k in TEMPO_EXTRA_KEYS}
    for fp in files:
        s = load_snapshot_v9(fp)
        no2 = s["no2"]
        v = np.isfinite(no2) & (no2 > 0)
        no2_vals.append(np.log10(no2[v] + 1e-3))
        for k in TEMPO_EXTRA_KEYS:
            if k in s:
                a = s[k]; m = np.isfinite(a)
                arr = a[m].astype(np.float32).ravel()
                if arr.size > 50000:
                    arr = np.random.default_rng(0).choice(arr, 50000, replace=False)
                extra[k].append(arr)
    aux = get_aux()
    hgt = aux["hgt"][np.isfinite(aux["hgt"])].ravel()
    stats = {
        "no2_log": {"mean": float(np.concatenate(no2_vals).mean()),
                    "std":  float(np.concatenate(no2_vals).std() + 1e-8)},
        "hgt":     {"mean": float(hgt.mean()), "std": float(hgt.std() + 1e-8)},
    }
    for k in TEMPO_EXTRA_KEYS:
        # float64 to avoid variance overflow on large-magnitude channels (HCHO ~1e16)
        vals = (np.concatenate(extra[k]) if extra[k] else np.array([0.0, 1.0])).astype(np.float64)
        sd = float(vals.std())
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        stats[k] = {"mean": float(vals.mean()), "std": sd + 1e-8}
    return stats


class AnomalyPatchDatasetV9(Dataset):
    def __init__(self, files, stats, n_patches_per_snapshot, train,
                 patch=PATCH, seed=0):
        self.files = list(files); self.stats = stats; self.train = train
        self.n = int(n_patches_per_snapshot); self.patch = patch
        self._cache: OrderedDict[Path, dict] = OrderedDict(); self._cache_max = 64
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
        self.clim = cl["clim"].astype(np.float32)
        self.anom_std = cl["anom_std"].astype(np.float32)

    @property
    def n_in(self):  return 6 + len(TEMPO_EXTRA_KEYS)
    @property
    def n_out(self): return 3
    def __len__(self): return len(self.files) * self.n

    def _get(self, i):
        fp = self.files[i]
        if fp in self._cache:
            self._cache.move_to_end(fp); return self._cache[fp]
        snap = load_snapshot_v9(fp); self._cache[fp] = snap
        if len(self._cache) > self._cache_max: self._cache.popitem(last=False)
        return snap

    def _patch_idx(self, no2_finite, hrrr_finite):
        H, W = no2_finite.shape; P = self.patch
        for _ in range(N_TRIES):
            i = int(self._rng.integers(0, H - P + 1)); j = int(self._rng.integers(0, W - P + 1))
            if (no2_finite[i:i+P, j:j+P].mean() >= MIN_TEMPO_COV
                and hrrr_finite[i:i+P, j:j+P].mean() >= MIN_HRRR_COV):
                return i, j
        for _ in range(N_TRIES):
            i = int(self._rng.integers(0, H - P + 1)); j = int(self._rng.integers(0, W - P + 1))
            if hrrr_finite[i:i+P, j:j+P].mean() >= MIN_HRRR_COV:
                return i, j
        return 0, 0

    def __getitem__(self, idx):
        s = self._get(idx // self.n)
        no2 = s["no2"]; u10 = s["u10"]; v10 = s["v10"]; pblh = s["pblh"]
        no2_finite = np.isfinite(no2) & (no2 > 0)
        hrrr_finite = np.isfinite(u10) & np.isfinite(v10) & np.isfinite(pblh)
        i, j = self._patch_idx(no2_finite, hrrr_finite); P = self.patch

        no2_p = no2[i:i+P, j:j+P]
        valid_no2 = (np.isfinite(no2_p) & (no2_p > 0)).astype(np.float32)
        mu = self.stats["no2_log"]["mean"]; sd = self.stats["no2_log"]["std"]
        with np.errstate(invalid="ignore", divide="ignore"):
            no2_log = np.where(valid_no2 > 0, np.log10(no2_p + 1e-3), mu)
        no2_n = ((no2_log - mu) / sd).astype(np.float32)

        chans = [no2_n, valid_no2,
                 self.hgt_norm[i:i+P, j:j+P], self.lsm_norm[i:i+P, j:j+P],
                 self.sin_lat[i:i+P, j:j+P], self.cos_lon[i:i+P, j:j+P]]
        for k in TEMPO_EXTRA_KEYS:
            a = s.get(k)
            if a is None:
                chans.append(np.zeros((P, P), dtype=np.float32))
            else:
                ap = a[i:i+P, j:j+P]
                m_ = self.stats[k]["mean"]; sd_ = self.stats[k]["std"]
                ap_n = (ap - m_) / sd_
                chans.append(np.where(np.isfinite(ap_n), ap_n, 0.0).astype(np.float32))
        x = np.stack(chans, axis=0).astype(np.float32)

        u_p = u10[i:i+P, j:j+P]; v_p = v10[i:i+P, j:j+P]; b_p = pblh[i:i+P, j:j+P]
        def norm_anom(arr, clim, sd):
            y = (arr - clim) / sd; valid = np.isfinite(y)
            return np.where(valid, y, 0.0).astype(np.float32), valid
        y_u, mu_v = norm_anom(u_p, self.clim[0, i:i+P, j:j+P], self.anom_std[0])
        y_v, mv_v = norm_anom(v_p, self.clim[1, i:i+P, j:j+P], self.anom_std[1])
        y_b, mp_v = norm_anom(b_p, self.clim[2, i:i+P, j:j+P], self.anom_std[2])
        y = np.stack([y_u, y_v, y_b], axis=0).astype(np.float32)
        m = (mu_v & mv_v & mp_v).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m))
