"""Item 20 control eval — directly comparable to eval_v17_spatial_skill_intersp.py.

Eval-only (no retrain). Mirrors eval_v17_spatial_skill_intersp.py byte-for-byte
inference: interspersed static climatology, 16x16 Hann-blended tiling (TILE=16,
STRIDE=8), anomaly = (HRRR - static_clim_interspersed)/anom_std, same finite +
TEMPO-covered masks. Parametrised by --mode so it reproduces the primary
(12 ch, c_in=12) and adapts to nocloud (10 ch) and maskonly (5 ch).

Modes:
  primary : extra_keys = 6-tuple  -> 12 channels
  nocloud : extra_keys = 4-tuple  -> 10 channels
  maskonly: extra_keys = ()       -> build 6, keep channels [1:6] = 5 channels
            ([valid_no2, hgt, lsm, sin_lat, cos_lon])

Computes, for full-domain and TEMPO-covered masks:
  - mean-of-per-scene pooled anomaly r        (eval_v17 'pooled_r_anom')
  - conventional all-pairs pooled ACC          (bootstrap-script definition)
  - per-cell temporal r map -> plain nanmedian AND cos(lat) area-weighted median
"""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import dataset_v5 as _dv5
_dv5.STATIC_CLIM = _pl.Path(__file__).resolve().parent / 'static_clim_interspersed.npz'
assert _dv5.STATIC_CLIM.exists(), 'static_clim_interspersed.npz missing'
print('[item20-eval] clim ->', _dv5.STATIC_CLIM.name, flush=True)

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(HERE))
from model import UNet2to3
from dataset_v5 import load_aux, load_clim
from dataset_v9 import load_snapshot_v9

VARS = ("u10", "v10", "pblh")
TILE = 16
STRIDE = 8

MODE_KEYS = {
    "primary":  ("o3_col", "uv_aerosol_index", "so2_index", "hcho_col",
                 "cloud_fraction", "cloud_pressure"),
    "nocloud":  ("o3_col", "uv_aerosol_index", "so2_index", "hcho_col"),
    "maskonly": (),
}


def build_inputs(snap, stats, aux_ch, extra_keys):
    """Full-domain (1,C,H,W) input mirroring eval_v17 build_inputs_v9."""
    no2 = snap["no2"]
    valid_no2 = (np.isfinite(no2) & (no2 > 0)).astype(np.float32)
    mu = stats["no2_log"]["mean"]; sd = stats["no2_log"]["std"]
    with np.errstate(invalid="ignore", divide="ignore"):
        no2_log = np.where(valid_no2 > 0, np.log10(no2 + 1e-3), mu)
    no2_n = ((no2_log - mu) / sd).astype(np.float32)
    chans = [no2_n, valid_no2, aux_ch["hgt_norm"], aux_ch["lsm_norm"],
             aux_ch["sin_lat"], aux_ch["cos_lon"]]
    for k in extra_keys:
        a = snap.get(k)
        if a is None:
            chans.append(np.zeros_like(no2_n))
        else:
            m_ = stats[k]["mean"]; sd_ = stats[k]["std"]
            a_n = (a - m_) / sd_
            chans.append(np.where(np.isfinite(a_n), a_n, 0.0).astype(np.float32))
    return np.stack(chans, axis=0)[None], valid_no2


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
    """Weighted median of finite vals."""
    ok = np.isfinite(vals)
    v = vals[ok]; w = weights[ok]
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v = v[order]; w = w[order]
    cw = np.cumsum(w); cutoff = 0.5 * cw[-1]
    return float(v[np.searchsorted(cw, cutoff)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", required=True, choices=list(MODE_KEYS))
    ap.add_argument("--max-snapshots", type=int, default=0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir
    extra_keys = MODE_KEYS[args.mode]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} mode={args.mode} extra_keys={extra_keys}", flush=True)

    aux = load_aux(); cl = load_clim()
    clim = cl["clim"].astype(np.float32)
    anom_std = cl["anom_std"].astype(np.float32)
    lat, lon = aux["lat"], aux["lon"]; H, W = lat.size, lon.size
    LON2D, LAT2D = np.meshgrid(lon, lat)
    aux_ch = {
        "sin_lat": np.sin(np.deg2rad(LAT2D)).astype(np.float32),
        "cos_lon": np.cos(np.deg2rad(LON2D)).astype(np.float32),
    }
    coslat = np.cos(np.deg2rad(LAT2D)).astype(np.float64)

    cfg = json.loads((run_dir / "config.json").read_text())
    stats = json.loads((run_dir / "stats.json").read_text())
    files = json.loads((run_dir / "files.json").read_text())
    test_files = [Path(f) for f in files["test"]]
    if args.max_snapshots > 0:
        test_files = test_files[: args.max_snapshots]
    n_test = len(test_files)

    h_mu = stats["hgt"]["mean"]; h_sd = stats["hgt"]["std"]
    aux_ch["hgt_norm"] = np.where(np.isfinite(aux["hgt"]),
                                  (aux["hgt"] - h_mu) / h_sd, 0.0).astype(np.float32)
    aux_ch["lsm_norm"] = np.where(np.isfinite(aux["lsm"]), aux["lsm"], 0.0).astype(np.float32)

    n_full = 6 + len(extra_keys)
    c_in = 5 if args.mode == "maskonly" else n_full
    model = UNet2to3(c_in=c_in, c_out=3, base=cfg["base"], depth=cfg["depth"]).to(device)
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"c_in={c_in} best_epoch={ck.get('epoch','?')} n_test={n_test}", flush=True)

    acc = {v: CellAccum(H, W) for v in VARS}
    acc_cov = {v: CellAccum(H, W) for v in VARS}
    snap_r = {v: [] for v in VARS}
    snap_r_cov = {v: [] for v in VARS}
    # conventional all-pairs sufficient stats [n,Sp,St,Spp,Stt,Spt]
    P = {v: np.zeros(6) for v in VARS}
    Pc = {v: np.zeros(6) for v in VARS}

    r_orig = sorted(set(list(range(0, H - TILE + 1, STRIDE)) + [H - TILE]))
    c_orig = sorted(set(list(range(0, W - TILE + 1, STRIDE)) + [W - TILE]))
    origins = [(r0, c0) for r0 in r_orig for c0 in c_orig]
    hann = (np.outer(np.hanning(TILE), np.hanning(TILE)) + 0.05).astype(np.float32)
    CHUNK = 4096

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

    t0 = time.time()
    with torch.no_grad():
        for k, fp in enumerate(test_files):
            snap = load_snapshot_v9(fp)
            x_np, valid_no2 = build_inputs(snap, stats, aux_ch, extra_keys)
            if args.mode == "maskonly":
                x_np = x_np[:, 1:6]
            p_anom = tiled_predict(x_np)
            truth = np.stack([snap["u10"], snap["v10"], snap["pblh"]], axis=0).astype(np.float32)
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
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{n_test} ({(time.time()-t0)/60:.1f} min)", flush=True)

    maps = {v: acc[v].r() for v in VARS}
    maps_cov = {v: acc_cov[v].r() for v in VARS}
    out = {
        "run": run_dir.name, "mode": args.mode, "c_in": c_in, "n_test": n_test,
        "best_epoch": ck.get("epoch"),
        "pooled_r_anom_full": {v: float(np.nanmean(snap_r[v])) for v in VARS},
        "pooled_r_anom_covered": {v: float(np.nanmean(snap_r_cov[v])) for v in VARS},
        "conventional_pooled_acc_full": {v: pooled_r_from_stats(P[v]) for v in VARS},
        "conventional_pooled_acc_covered": {v: pooled_r_from_stats(Pc[v]) for v in VARS},
        "per_cell_median_r_full": {v: float(np.nanmedian(maps[v])) for v in VARS},
        "per_cell_median_r_covered": {v: float(np.nanmedian(maps_cov[v])) for v in VARS},
        "per_cell_wmedian_r_full": {v: wmedian(maps[v].ravel(), coslat.ravel()) for v in VARS},
        "per_cell_wmedian_r_covered": {v: wmedian(maps_cov[v].ravel(), coslat.ravel()) for v in VARS},
    }
    print("RESULT_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
