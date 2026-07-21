"""TEMPO -> HRRR interspersed evaluation from extracted snapshots.

Faithful port of the analysis pipeline's ``diurnal_control_intersp.py``
(which produced the paper's Fig. 5a numbers), with two packaging changes:

1. Snapshots are the compact extracts written by ``scripts/extract_tempo.py``
   (z-scored input channels stored float16 with NaN validity coding; the raw
   NO2 ~1e15 and HCHO ~1e16 fields are not float16-representable). The
   validity masks — and therefore the seeded rejection-sampling patch draws —
   are bit-identical to the originals.
2. The train-split scan-hour lookup is precomputed
   (``scripts/build_tempo_lookup.py``) and shipped as ``hour_lookup.npz``
   instead of requiring the 11,798 training snapshots.

Protocol (unchanged): 16-px patches, 16 per snapshot, seeded sampler
(seed = run seed + 2); per-batch masked Pearson r in float32 averaged over
batches; clock baseline = hour lookup as the prediction; scan-hour-conditioned
= lookup subtracted from both prediction and truth; hours with fewer than 20
training snapshots are skipped. CUDA runs use mixed-precision autocast for the
forward pass exactly as the original; CPU/MPS run float32.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .models import UNet2to3, load_checkpoint

PATCH = 16
MIN_TEMPO_COV = 0.7
MIN_HRRR_COV = 0.95
N_TRIES = 30
MIN_HOUR_COUNT = 20
EXTRA_KEYS = ("o3_col", "uv_aerosol_index", "so2_index",
              "hcho_col", "cloud_fraction", "cloud_pressure")
SNAP_RE = re.compile(r"snapshot_(\d{8})T(\d{6})Z\.npz$")
TARGETS = ("r_anom_u10", "r_anom_v10", "r_anom_pblh")


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


# ---------------------------------------------------------------- dataset ----
class ExtractedPatchDataset(Dataset):
    """AnomalyPatchDatasetV9 logic on the extracted snapshot format.

    Returns (x, y, m, i_file, i, j) so the evaluation can slice the hour
    lookup at the sampled patch corner (the original recovered i,j via a
    subclass hook; here the dataset returns them directly).
    """

    def __init__(self, files, stats, aux_path, clim_path,
                 n_patches_per_snapshot=16, patch=PATCH, seed=0):
        self.files = list(files)
        self.stats = stats
        self.n = int(n_patches_per_snapshot)
        self.patch = patch
        self._rng = np.random.default_rng(seed)
        self._cache: dict = {}

        aux = np.load(aux_path)
        cl = np.load(clim_path)
        lat, lon = aux["lat"], aux["lon"]
        LON2D, LAT2D = np.meshgrid(lon, lat)
        self.sin_lat = np.sin(np.deg2rad(LAT2D)).astype(np.float32)
        self.cos_lon = np.cos(np.deg2rad(LON2D)).astype(np.float32)
        h_mu = stats["hgt"]["mean"]; h_sd = stats["hgt"]["std"]
        hgt = aux["hgt"]
        self.hgt_norm = np.where(np.isfinite(hgt), (hgt - h_mu) / h_sd, 0.0).astype(np.float32)
        lsm = aux["lsm"]
        self.lsm_norm = np.where(np.isfinite(lsm), lsm, 0.0).astype(np.float32)
        self.clim = cl["clim"].astype(np.float32)
        self.anom_std = cl["anom_std"].astype(np.float32)

    @property
    def n_in(self):
        return 6 + len(EXTRA_KEYS)

    def __len__(self):
        return len(self.files) * self.n

    def _get(self, i):
        fp = self.files[i]
        if fp in self._cache:
            return self._cache[fp]
        z = np.load(fp, allow_pickle=True)
        snap = {k: np.array(z[k]) for k in z.files if k != "ts"}
        z.close()
        self._cache = {fp: snap}  # sequential access: keep only the current file
        return snap

    def _patch_idx(self, no2_finite, hrrr_finite):
        H, W = no2_finite.shape
        P = self.patch
        for _ in range(N_TRIES):
            i = int(self._rng.integers(0, H - P + 1))
            j = int(self._rng.integers(0, W - P + 1))
            if (no2_finite[i:i + P, j:j + P].mean() >= MIN_TEMPO_COV
                    and hrrr_finite[i:i + P, j:j + P].mean() >= MIN_HRRR_COV):
                return i, j
        for _ in range(N_TRIES):
            i = int(self._rng.integers(0, H - P + 1))
            j = int(self._rng.integers(0, W - P + 1))
            if hrrr_finite[i:i + P, j:j + P].mean() >= MIN_HRRR_COV:
                return i, j
        return 0, 0

    def __getitem__(self, idx):
        i_file = idx // self.n
        s = self._get(i_file)
        no2_n = s["no2_n"].astype(np.float32)
        u10 = s["hrrr_u10"].astype(np.float32)
        v10 = s["hrrr_v10"].astype(np.float32)
        pblh = s["hrrr_pblh"].astype(np.float32)

        no2_finite = np.isfinite(no2_n)          # == isfinite(no2) & (no2 > 0)
        hrrr_finite = np.isfinite(u10) & np.isfinite(v10) & np.isfinite(pblh)
        i, j = self._patch_idx(no2_finite, hrrr_finite)
        P = self.patch

        no2_p = no2_n[i:i + P, j:j + P]
        valid_no2 = np.isfinite(no2_p).astype(np.float32)
        no2_c = np.nan_to_num(no2_p, nan=0.0)    # invalid -> (mu - mu)/sd = 0

        chans = [no2_c, valid_no2,
                 self.hgt_norm[i:i + P, j:j + P], self.lsm_norm[i:i + P, j:j + P],
                 self.sin_lat[i:i + P, j:j + P], self.cos_lon[i:i + P, j:j + P]]
        for k in EXTRA_KEYS:
            a = s.get(f"{k}_n")
            if a is None:
                chans.append(np.zeros((P, P), dtype=np.float32))
            else:
                ap = a[i:i + P, j:j + P].astype(np.float32)
                chans.append(np.nan_to_num(ap, nan=0.0))
        x = np.stack(chans, axis=0).astype(np.float32)

        def norm_anom(arr, clim, sd):
            y = (arr - clim) / sd
            valid = np.isfinite(y)
            return np.where(valid, y, 0.0).astype(np.float32), valid

        y_u, mu_v = norm_anom(u10[i:i + P, j:j + P], self.clim[0, i:i + P, j:j + P], self.anom_std[0])
        y_v, mv_v = norm_anom(v10[i:i + P, j:j + P], self.clim[1, i:i + P, j:j + P], self.anom_std[1])
        y_b, mp_v = norm_anom(pblh[i:i + P, j:j + P], self.clim[2, i:i + P, j:j + P], self.anom_std[2])
        y = np.stack([y_u, y_v, y_b], axis=0).astype(np.float32)
        m = (mu_v & mv_v & mp_v).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m),
                i_file, i, j)


# ------------------------------------------------------------------ metric ----
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


# -------------------------------------------------------------- evaluation ----
def evaluate(eval_dir: Path, run_dir: Path, aux_path: Path, clim_path: Path,
             lookup_path: Path, device: torch.device, limit: int = 0,
             progress: bool = True) -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    stats = json.loads((run_dir / "stats.json").read_text())
    order = [Path(p).name for p in json.loads((run_dir / "files.json").read_text())["test"]]
    files = [Path(eval_dir) / n for n in order]        # files.json test order
    if limit:
        files = files[:limit]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} extracted test snapshots missing, "
                                f"e.g. {missing[0]}")

    lk = np.load(lookup_path)
    lookup = {int(h): lk[f"lookup_{int(h)}"] for h in lk["hours"]}
    counts = {int(h): int(lk[f"count_{int(h)}"]) for h in lk["hours"]}
    ok_hours = {h for h, c in counts.items() if c >= MIN_HOUR_COUNT}

    seed = int(cfg.get("seed", 0))
    ds = ExtractedPatchDataset(files, stats, aux_path, clim_path,
                               n_patches_per_snapshot=cfg.get("patches_per_snap_eval", 16),
                               seed=seed + 2)
    dl = DataLoader(ds, batch_size=int(cfg.get("batch", 128)), shuffle=False,
                    num_workers=0)   # sequential: patch rng must stay in-process

    model = UNet2to3(c_in=ds.n_in, c_out=3, base=int(cfg.get("base", 48)),
                     depth=int(cfg.get("depth", 3)))
    load_checkpoint(model, run_dir / "best.pt", device)

    file_hours = [target_hour(fp.name) for fp in files]
    P = ds.patch
    rows = {k: [[], [], []] for k in
            ("network_fp32", "clock_baseline", "network_hourcond")}
    n_skip = 0
    n_batches = 0

    with torch.no_grad():
        for x, y, m, i_file, ii, jj in dl:
            B = x.shape[0]
            clock = torch.zeros_like(y)
            keep = torch.ones(B)
            for b in range(B):
                h = file_hours[int(i_file[b])]
                if h not in ok_hours:
                    keep[b] = 0.0
                    continue
                i0, j0 = int(ii[b]), int(jj[b])
                clock[b] = torch.from_numpy(lookup[h][:, i0:i0 + P, j0:j0 + P])
            n_skip += int((keep == 0).sum())
            m_eff = m * keep.view(-1, 1, 1)

            x = x.to(device); y = y.to(device)
            m_eff = m_eff.to(device); clock = clock.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p = model(x)
            p = p.float()

            for ci, r in enumerate(corr_fp32(p, y, m_eff)):
                rows["network_fp32"][ci].append(r)
            for ci, r in enumerate(corr_fp32(clock, y, m_eff)):
                rows["clock_baseline"][ci].append(r)
            for ci, r in enumerate(corr_fp32(p - clock, y - clock, m_eff)):
                rows["network_hourcond"][ci].append(r)
            n_batches += 1
            if progress and n_batches % 25 == 0:
                u = float(np.nanmean(rows["network_fp32"][0]))
                print(f"  [{n_batches} batches] running u10 r={u:.4f}", flush=True)

    out = {"run": run_dir.name, "n_test_files": len(files),
           "skipped_patches_offhour": n_skip}
    for key, rr in rows.items():
        out[key] = {k: float(np.nanmean(rr[ci])) for ci, k in enumerate(TARGETS)}
    return out
