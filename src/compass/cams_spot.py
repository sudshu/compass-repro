"""CAMS single-species spot check: re-derive the O3 row of the winner maps.

Provenance link for the winner-map verification: the released
``rmaps_winds500.npz`` stack is claimed to be the per-pixel temporal ACC of
the eight single-species CAMS models. This module re-runs ONE of those models
end-to-end — the O3 pressure-weighted-column winds500 model (the species the
paper reports leading at 500 hPa in both systems) — from its checkpoint on
the extracted evaluation snapshots, and compares the resulting per-pixel map
against the released stack's O3 row.

Protocol verbatim from eval_pixel_temporal_multitarget.py: 40-px stride-20
unweighted tiled inference (with the edge row/column), de-normalisation with
the dynamics y-stats, per-month mean removal (deseason), per-pixel temporal
correlation over the concatenated Jan/Apr/Jul 2020 snapshots (<=80/month).
CUDA runs use mixed-precision autocast for the forward pass exactly as the
original; CPU/MPS run float32.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .models import UNetWindPBLH

PATCH = 40
STRIDE = 20
LAT_MAX_DEG = 75.0
MONTHS = (1, 4, 7)
YEAR = 2020


def load_model_auto(run_dir: Path, device) -> UNetWindPBLH:
    """Architecture derived from the checkpoint (eval_la_basin_regional.load_model)."""
    ckpt = torch.load(Path(run_dir) / "checkpoints" / "best.pt",
                      map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    first_w = state[[k for k in state if "encoders.0.block.0.weight" in k][0]]
    final_w = state[[k for k in state if "final_conv.weight" in k][0]]
    model = UNetWindPBLH(
        in_channels=first_w.shape[1], out_channels=final_w.shape[0],
        base_channels=first_w.shape[0],
        depth=len([k for k in state
                   if k.startswith("encoders.") and k.endswith(".block.0.weight")]))
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def pixel_tcorr(p, t):
    pa = p - p.mean(0, keepdims=True)
    ta = t - t.mean(0, keepdims=True)
    num = (pa * ta).sum(0)
    den = np.sqrt((pa ** 2).sum(0) * (ta ** 2).sum(0))
    return num / (den + 1e-12)


def evaluate(spot_dir: Path, run_dir: Path, device: torch.device,
             progress: bool = True) -> dict:
    spot_dir = Path(spot_dir)
    st = np.load(spot_dir / "cams_spot_static.npz")
    lat, lon, hgt_full = st["lat"], st["lon"], st["hgt"]
    valid = np.abs(lat) <= LAT_MAX_DEG
    lat0 = int(np.argmax(valid)); nlat = int(valid.sum()); nlon = lon.size

    cst = json.loads((spot_dir / "stats_o3pwonly.json").read_text())
    dst = json.loads((spot_dir / "stats_dynamics.json").read_text())
    xmg = np.asarray(cst["x_mean_col"], np.float32)[[0]].reshape(1, -1, 1, 1)
    xsg = np.asarray(cst["x_std_col"], np.float32)[[0]].reshape(1, -1, 1, 1)
    ym = np.asarray(dst["y_mean_500"], np.float32).reshape(1, -1, 1, 1)
    ys = np.asarray(dst["y_std_500"], np.float32).reshape(1, -1, 1, 1)

    sin_lat2d = np.broadcast_to(
        np.sin(np.deg2rad(lat[lat0:lat0 + nlat])).astype(np.float32)[:, None],
        (nlat, nlon))
    hgt = hgt_full[lat0:lat0 + nlat, :]
    hgt = (hgt - hgt.mean()) / (hgt.std() + 1e-8)

    rs = list(range(0, nlat - PATCH + 1, STRIDE)) + (
        [nlat - PATCH] if (nlat - PATCH) % STRIDE else [])
    cs = list(range(0, nlon - PATCH + 1, STRIDE)) + (
        [nlon - PATCH] if (nlon - PATCH) % STRIDE else [])
    rs, cs = sorted(set(rs)), sorted(set(cs))

    model = load_model_auto(run_dir, device)
    nch = 2  # winds500 model outputs (u500, v500); component 0 evaluated

    P, Tr = [], []
    with torch.no_grad():
        for m in MONTHS:
            z = np.load(spot_dir / f"cams_spot_o3pw_{YEAR}_{m:02d}.npz")
            g = z["gases_col"]; u500 = z["u500"]
            S = g.shape[0]
            pred = np.zeros((S, nch, nlat, nlon), np.float64)
            cnt = np.zeros((1, 1, nlat, nlon), np.float64)
            for ti in range(S):
                gc = g[ti:ti + 1, [0], lat0:lat0 + nlat, :]
                ch = [(gc - xmg) / (xsg + 1e-8),
                      sin_lat2d[None, None], hgt[None, None]]
                xf = np.concatenate(ch, 1).astype(np.float32)
                for r0 in rs:
                    for c0 in cs:
                        xt = torch.from_numpy(
                            xf[:, :, r0:r0 + PATCH, c0:c0 + PATCH]).to(device)
                        with torch.amp.autocast("cuda",
                                                enabled=device.type == "cuda"):
                            yh = model(xt).cpu().numpy()
                        pred[ti, :, r0:r0 + PATCH, c0:c0 + PATCH] += (yh * ys + ym)[0]
                        if ti == 0:
                            cnt[0, 0, r0:r0 + PATCH, c0:c0 + PATCH] += 1.0
            pred /= np.maximum(cnt, 1e-9)
            pm = pred[:, 0]
            tm = u500[:, lat0:lat0 + nlat, :].astype(np.float64)
            pm = pm - pm.mean(0, keepdims=True)
            tm = tm - tm.mean(0, keepdims=True)
            P.append(pm); Tr.append(tm)
            if progress:
                print(f"  month {m:02d}: {S} snapshots done", flush=True)

    rmap = pixel_tcorr(np.concatenate(P, 0), np.concatenate(Tr, 0)).astype(np.float32)
    return {"rmap": rmap, "median_r": float(np.nanmedian(rmap)),
            "lat": lat[lat0:lat0 + nlat], "lon": lon}


def compare_with_released(rmap: np.ndarray, released_rmaps: Path,
                          species: str = "o3") -> dict:
    d = np.load(released_rmaps, allow_pickle=True)
    sp = [str(s) for s in d["species"]]
    ref = d["stack"][sp.index(species)]
    fin = np.isfinite(rmap) & np.isfinite(ref)
    diff = np.abs(rmap - ref)[fin]
    corr = float(np.corrcoef(rmap[fin], ref[fin])[0, 1])
    return {"median_abs_diff": float(np.median(diff)),
            "p99_abs_diff": float(np.percentile(diff, 99)),
            "map_correlation": corr,
            "median_r_reproduced": float(np.nanmedian(rmap)),
            "median_r_released": float(np.nanmedian(ref))}
