"""GEOS-CF temporally blocked evaluation from extracted snapshot files.

Protocol is IDENTICAL to the analysis repository's
``ml_geos_cf/scripts/eval_blocked_aug_oct_dec.py`` (which produced the paper's
blocked headline numbers), except that inputs are read from the compact
extracted ``.npz`` snapshots shipped with this package instead of the full
NetCDF archive:

- extraction already applied the per-file zonal-mean removal AND the z-score
  with the TRAINING stats to the gas column / surface fields (the raw
  physical fields span ~1e-19..1e16 and are not float16-representable; the
  z-scored channels are O(1), where float16 quantisation is negligible);
- the remaining assembly (sin(lat), zero-HGT placeholder), the 80-px
  half-stride Hann-blended tiling, the training-period climatology anomaly
  definition and the pooled Pearson r over |lat| <= 75 deg are unchanged.

Checkpoint, stats and climatology files are the originals from training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .metrics import RAccum
from .models import UNetWindPBLH, load_checkpoint

PATCH = 80
LAT_MAX_DEG = 75.0
TARGET_NAMES = ["u10", "v10", "u500", "v500", "pblh"]
MONTH_NAMES = {8: "Aug", 10: "Oct", 12: "Dec"}


# ------------------------------------------------------------------ tiling
def hann2d(n: int) -> np.ndarray:
    w1 = np.hanning(n + 2)[1:-1]
    return (w1[:, None] * w1[None, :]).astype(np.float32)


def sliding_predict(model, x_full: np.ndarray, device, patch=PATCH, stride=None,
                    batch=32) -> np.ndarray:
    """Hann-blended global inference. Returns (C_out, H, W) in normalised-anomaly space."""
    if stride is None:
        stride = patch // 2
    C, H, W = x_full.shape
    pad_h = (-(H - patch) % stride)
    pad_w = (-(W - patch) % stride)
    x_pad = np.pad(x_full, ((0, 0), (0, pad_h), (0, pad_w)), mode="wrap")
    Hp, Wp = x_pad.shape[1:]
    win = hann2d(patch)

    with torch.no_grad():
        x0 = torch.from_numpy(x_pad[:, :patch, :patch][None]).to(device)
        n_out = model(x0).shape[1]
    out = np.zeros((n_out, Hp, Wp), dtype=np.float32)
    wgt = np.zeros((Hp, Wp), dtype=np.float32)

    coords = [(i, j)
              for i in range(0, Hp - patch + 1, stride)
              for j in range(0, Wp - patch + 1, stride)]

    with torch.no_grad():
        for s in range(0, len(coords), batch):
            chunk = coords[s:s + batch]
            xs = np.stack([x_pad[:, i:i + patch, j:j + patch] for (i, j) in chunk])
            yt = model(torch.from_numpy(xs).to(device)).cpu().numpy()
            for k, (i, j) in enumerate(chunk):
                out[:, i:i + patch, j:j + patch] += yt[k] * win[None]
                wgt[i:i + patch, j:j + patch] += win
    out /= np.clip(wgt[None], 1e-8, None)
    return out[:, :H, :W]


# ------------------------------------------------------------------ data
def load_extracted(fpath: Path, gas_indices, sin_lat_map, hgt_norm, use_sfc):
    """Build the full-globe input tensor + raw met truth from an extracted snapshot.

    The extracted .npz stores (float16): ``col_n`` (4,H,W) and ``sfc_n``
    (4,H,W) — the model's z-scored gas input channels (zonal-mean removed,
    NaN->0, normalised with the training stats) — and ``met`` (5,H,W):
    [u10m, v10m, u500, v500, zpbl] in physical units with fills cleaned.
    """
    z = np.load(fpath)
    col_n = z["col_n"].astype(np.float32)[gas_indices]
    sfc_n = z["sfc_n"].astype(np.float32)[gas_indices]
    met = z["met"].astype(np.float32)

    chans = [col_n]
    if use_sfc:
        chans.append(sfc_n)
    chans += [sin_lat_map[None], hgt_norm[None]]
    x = np.concatenate(chans, axis=0).astype(np.float32)
    return x, met


def month_of(p: Path) -> int:
    return int(p.stem.split("_")[2][4:6])  # geos_cf_YYYYMMDD_HH


# ------------------------------------------------------------------ evaluation
def evaluate(eval_dir: Path, weights: Path, config: Path,
             clim_path: Path, device: torch.device, limit: int = 0,
             stride: int = 1, progress: bool = True) -> dict:
    """Run the blocked evaluation for one seed; returns pooled + per-month ACC."""
    files = sorted(Path(eval_dir).glob("geos_cf_*.npz"))[::max(1, stride)]
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no extracted snapshots in {eval_dir}")

    cl = np.load(clim_path)
    clim = cl["clim"].astype(np.float32)
    anom_std = cl["anom_std"].astype(np.float32)
    lat0 = cl["lat"].astype(np.float32)

    H, W = clim.shape[1:]
    sin_lat_map = np.broadcast_to(
        np.sin(np.deg2rad(lat0))[:, None], (H, W)).astype(np.float32)
    hgt_norm = np.zeros((H, W), dtype=np.float32)
    lat_mask_rows = np.abs(lat0) <= LAT_MAX_DEG

    cfg = json.loads(Path(config).read_text())
    gas_indices = cfg.get("gas_indices", [0, 1, 2, 3])
    use_sfc = bool(cfg.get("use_surface_channels", True))
    n_gas = len(gas_indices)
    in_ch = n_gas + (n_gas if use_sfc else 0) + 1 + 1
    model = UNetWindPBLH(in_channels=in_ch, out_channels=5,
                         base_channels=cfg.get("base_channels", 48),
                         depth=cfg.get("depth", 4),
                         dropout_rate=cfg.get("dropout_rate", 0.05))
    load_checkpoint(model, weights, device)

    acc_all = {t: RAccum() for t in TARGET_NAMES}
    acc_mon = {m: {t: RAccum() for t in TARGET_NAMES} for m in MONTH_NAMES}

    for fi, fp in enumerate(files):
        x, met = load_extracted(fp, gas_indices, sin_lat_map, hgt_norm, use_sfc)
        pred_anom = sliding_predict(model, x, device)
        true_anom = (met - clim) / (anom_std[:, None, None] + 1e-8)
        m = month_of(fp)
        for c, t in enumerate(TARGET_NAMES):
            acc_all[t].add(pred_anom[c][lat_mask_rows], true_anom[c][lat_mask_rows])
            if m in acc_mon:
                acc_mon[m][t].add(pred_anom[c][lat_mask_rows], true_anom[c][lat_mask_rows])
        if progress and ((fi + 1) % 25 == 0 or fi + 1 == len(files)):
            print(f"  [{fi + 1}/{len(files)}] {fp.name}  "
                  f"running v500 r={acc_all['v500'].r():.4f}", flush=True)

    pooled = {t: acc_all[t].r() for t in TARGET_NAMES}
    per_month = {MONTH_NAMES[m]: {t: acc_mon[m][t].r() for t in TARGET_NAMES}
                 for m in MONTH_NAMES if acc_mon[m][TARGET_NAMES[0]].n > 0}
    return {"pooled_anom_acc": pooled, "per_month": per_month,
            "n_files": len(files)}
