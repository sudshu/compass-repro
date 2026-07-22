"""TEMPO independence controls (Suppl. Fig. S8): primary / no-cloud / mask-only.

Faithful port of the analysis pipeline's ``eval_item20_tmp.py`` (which produced
the manuscript's S8 numbers), reading the packaged snapshot extracts. Protocol:
full-domain 16-px tiles at stride 8 with (Hann + 0.05) blending including the
domain-edge tiles; anomaly = (HRRR - interspersed static climatology)/anom_std;
metrics on the full finite domain and on the TEMPO-covered subset:

- mean of per-scene pooled anomaly r,
- conventional all-pairs pooled ACC (the manuscript's S8 headline definition:
  primary 0.45/0.43/0.77, no-cloud 0.39/0.39/0.70, ...),
- per-cell temporal r maps -> plain and cos-lat weighted medians.

Modes: primary (12 ch), nocloud (10 ch: cloud fraction/pressure dropped),
maskonly (5 ch: validity mask + statics, no measurement values). The stats of
every channel a mode uses are identical across the three runs (verified), so
the primary-stats extracts serve all modes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .models import UNet2to3, load_checkpoint

VARS = ("u10", "v10", "pblh")
TILE = 16
STRIDE = 8
CHUNK = 4096
EXTRA_ALL = ("o3_col", "uv_aerosol_index", "so2_index",
             "hcho_col", "cloud_fraction", "cloud_pressure")
MODE_KEYS = {
    "primary": EXTRA_ALL,
    "nocloud": EXTRA_ALL[:4],
    "maskonly": (),
}


class CellAccum:
    def __init__(self, H, W):
        self.S_p = np.zeros((H, W)); self.S_t = np.zeros((H, W))
        self.S_pp = np.zeros((H, W)); self.S_tt = np.zeros((H, W))
        self.S_pt = np.zeros((H, W)); self.CNT = np.zeros((H, W), dtype=np.int64)

    def add(self, p, t, m):
        self.S_p += m * p; self.S_t += m * t
        self.S_pp += m * p * p; self.S_tt += m * t * t
        self.S_pt += m * p * t; self.CNT += (m > 0)

    def r(self, min_n=10):
        cnt = np.where(self.CNT > 0, self.CNT, 1)
        mp = self.S_p / cnt; mt = self.S_t / cnt
        vp = self.S_pp / cnt - mp ** 2
        vt = self.S_tt / cnt - mt ** 2
        cov = self.S_pt / cnt - mp * mt
        r = cov / (np.sqrt(np.maximum(vp, 1e-12) * np.maximum(vt, 1e-12)) + 1e-12)
        return np.where(self.CNT >= min_n, r, np.nan).astype(np.float32)


def pooled_r_from_stats(S):
    n, Sp, St, Spp, Stt, Spt = [S[k] for k in range(6)]
    num = Spt - Sp * St / n
    vp = Spp - Sp * Sp / n
    vt = Stt - St * St / n
    den = np.sqrt(max(vp, 0.0) * max(vt, 0.0))
    return float(num / den) if den > 0 else float("nan")


def wmedian(vals, weights):
    ok = np.isfinite(vals)
    v = vals[ok]; w = weights[ok]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v = v[order]; w = w[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _build_inputs_extracted(snap, aux_ch, extra_keys):
    """(1,C,H,W) input from an extracted snapshot (z-scored channels stored)."""
    no2_n_raw = snap["no2_n"].astype(np.float32)
    valid_no2 = np.isfinite(no2_n_raw).astype(np.float32)
    no2_n = np.nan_to_num(no2_n_raw, nan=0.0)   # invalid -> (mu-mu)/sd = 0
    chans = [no2_n, valid_no2, aux_ch["hgt_norm"], aux_ch["lsm_norm"],
             aux_ch["sin_lat"], aux_ch["cos_lon"]]
    for k in extra_keys:
        a = snap.get(f"{k}_n")
        if a is None:
            chans.append(np.zeros_like(no2_n))
        else:
            chans.append(np.nan_to_num(a.astype(np.float32), nan=0.0))
    return np.stack(chans, axis=0)[None], valid_no2


def evaluate_mode(mode: str, eval_dir: Path, run_dir: Path, aux_path: Path,
                  clim_path: Path, device: torch.device, limit: int = 0,
                  progress: bool = True) -> dict:
    extra_keys = MODE_KEYS[mode]
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    order = [Path(p).name for p in json.loads((run_dir / "files.json").read_text())["test"]]
    files = [Path(eval_dir) / n for n in order]
    if limit:
        files = files[:limit]

    aux = np.load(aux_path)
    cl = np.load(clim_path)
    clim = cl["clim"].astype(np.float32)
    anom_std = cl["anom_std"].astype(np.float32)
    lat, lon = aux["lat"], aux["lon"]
    H, W = lat.size, lon.size
    LON2D, LAT2D = np.meshgrid(lon, lat)
    stats = json.loads((run_dir / "stats.json").read_text())
    h_mu = stats["hgt"]["mean"]; h_sd = stats["hgt"]["std"]
    aux_ch = {
        "sin_lat": np.sin(np.deg2rad(LAT2D)).astype(np.float32),
        "cos_lon": np.cos(np.deg2rad(LON2D)).astype(np.float32),
        "hgt_norm": np.where(np.isfinite(aux["hgt"]),
                             (aux["hgt"] - h_mu) / h_sd, 0.0).astype(np.float32),
        "lsm_norm": np.where(np.isfinite(aux["lsm"]), aux["lsm"], 0.0).astype(np.float32),
    }
    coslat = np.cos(np.deg2rad(LAT2D)).astype(np.float64)

    n_full = 6 + len(extra_keys)
    c_in = 5 if mode == "maskonly" else n_full
    model = UNet2to3(c_in=c_in, c_out=3, base=cfg["base"], depth=cfg["depth"])
    load_checkpoint(model, run_dir / "best.pt", device)

    acc = {v: CellAccum(H, W) for v in VARS}
    acc_cov = {v: CellAccum(H, W) for v in VARS}
    snap_r = {v: [] for v in VARS}
    snap_r_cov = {v: [] for v in VARS}
    P = {v: np.zeros(6) for v in VARS}
    Pc = {v: np.zeros(6) for v in VARS}

    r_orig = sorted(set(list(range(0, H - TILE + 1, STRIDE)) + [H - TILE]))
    c_orig = sorted(set(list(range(0, W - TILE + 1, STRIDE)) + [W - TILE]))
    origins = [(r0, c0) for r0 in r_orig for c0 in c_orig]
    hann = (np.outer(np.hanning(TILE), np.hanning(TILE)) + 0.05).astype(np.float32)

    def tiled_predict(x_np):
        x_full = torch.from_numpy(x_np[0])
        accum = np.zeros((3, H, W), dtype=np.float32)
        wsum = np.zeros((H, W), dtype=np.float32)
        for i0 in range(0, len(origins), CHUNK):
            chunk = origins[i0:i0 + CHUNK]
            batch = torch.stack([x_full[:, r0:r0 + TILE, c0:c0 + TILE]
                                 for r0, c0 in chunk]).to(device)
            out = model(batch).float().cpu().numpy()
            for (r0, c0), o in zip(chunk, out):
                accum[:, r0:r0 + TILE, c0:c0 + TILE] += o * hann
                wsum[r0:r0 + TILE, c0:c0 + TILE] += hann
        return accum / np.maximum(wsum, 1e-6)

    with torch.no_grad():
        for k, fp in enumerate(files):
            z = np.load(fp, allow_pickle=True)
            snap = {key: np.array(z[key]) for key in z.files if key != "ts"}
            z.close()
            x_np, valid_no2 = _build_inputs_extracted(snap, aux_ch, EXTRA_ALL
                                                      if mode == "maskonly" else extra_keys)
            if mode == "maskonly":
                x_np = x_np[:, 1:6]
            p_anom = tiled_predict(x_np)
            truth = np.stack([snap["hrrr_u10"], snap["hrrr_v10"],
                              snap["hrrr_pblh"]], axis=0).astype(np.float32)
            finite = np.isfinite(truth).all(axis=0)
            t_anom = np.where(finite[None], (truth - clim) / anom_std[:, None, None], 0.0)
            m = finite.astype(np.float64)
            m_cov = (finite & (valid_no2 > 0)).astype(np.float64)
            for i, v in enumerate(VARS):
                pa = p_anom[i].astype(np.float64); ta = t_anom[i].astype(np.float64)
                acc[v].add(pa, ta, m)
                acc_cov[v].add(pa, ta, m_cov)
                pv = pa[finite]; tv = ta[finite]
                if pv.size > 100:
                    pc = pv - pv.mean(); tc = tv - tv.mean()
                    snap_r[v].append(float((pc * tc).sum() /
                                     (np.linalg.norm(pc) * np.linalg.norm(tc) + 1e-12)))
                    P[v] += [pv.size, pv.sum(), tv.sum(),
                             (pv * pv).sum(), (tv * tv).sum(), (pv * tv).sum()]
                fc = finite & (valid_no2 > 0)
                pv = pa[fc]; tv = ta[fc]
                if pv.size > 100:
                    pc = pv - pv.mean(); tc = tv - tv.mean()
                    snap_r_cov[v].append(float((pc * tc).sum() /
                                         (np.linalg.norm(pc) * np.linalg.norm(tc) + 1e-12)))
                    Pc[v] += [pv.size, pv.sum(), tv.sum(),
                              (pv * pv).sum(), (tv * tv).sum(), (pv * tv).sum()]
            if progress and (k + 1) % 200 == 0:
                print(f"  [{mode}] {k + 1}/{len(files)}", flush=True)

    maps = {v: acc[v].r() for v in VARS}
    maps_cov = {v: acc_cov[v].r() for v in VARS}
    return {
        "run": run_dir.name, "mode": mode, "c_in": c_in, "n_test": len(files),
        "pooled_r_anom_full": {v: float(np.nanmean(snap_r[v])) for v in VARS},
        "pooled_r_anom_covered": {v: float(np.nanmean(snap_r_cov[v])) for v in VARS},
        "conventional_pooled_acc_full": {v: pooled_r_from_stats(P[v]) for v in VARS},
        "conventional_pooled_acc_covered": {v: pooled_r_from_stats(Pc[v]) for v in VARS},
        "per_cell_median_r_full": {v: float(np.nanmedian(maps[v])) for v in VARS},
        "per_cell_median_r_covered": {v: float(np.nanmedian(maps_cov[v])) for v in VARS},
        "per_cell_wmedian_r_full": {v: wmedian(maps[v].ravel(), coslat.ravel()) for v in VARS},
        "per_cell_wmedian_r_covered": {v: wmedian(maps_cov[v].ravel(), coslat.ravel())
                                       for v in VARS},
    }
