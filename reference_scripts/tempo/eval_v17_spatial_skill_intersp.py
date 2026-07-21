"""Fig 5 CONUS spatial-skill data — per-pixel anomaly correlation for v17.

For each v17_all_tempo_no_goes_compact seed (s0/s1/s2), run full-domain
inference on that seed's own held-out TEST snapshots (no retrain, no leakage)
and accumulate, per grid cell:

  r_anom(u10), r_anom(v10), r_anom(pblh)   -- Pearson r between predicted and
      true NORMALISED anomalies over test snapshots (same anomaly definition
      as training: (x - static_clim) / anom_std, static_clim.npz)
  TEMPO NO2 valid-coverage fraction        -- for optional masking downstream

Also accumulates pooled error-magnitude stats in PHYSICAL units (RMSE, mean
bias, variance ratio pred/truth) per variable per seed -> feeds the SI
error-magnitude table.

Outputs (out-dir, default runs/eval_v17_spatial/):
  v17_s{K}_spatial_raw.npz   per-seed maps + counts
  v17_seedmean_spatial.npz   seed-mean maps (+ lat/lon + coverage)
  error_magnitude_v17.json   pooled RMSE/bias/var-ratio per seed + seed-mean
  fig_v17_conus_skill_draft.png   draft 3-panel map for [T] integration
  summary.json
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

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model import UNet2to3
from dataset_v5 import load_aux, load_clim
from dataset_v9 import load_snapshot_v9, TEMPO_EXTRA_KEYS

VARS = ("u10", "v10", "pblh")
TILE = 16      # MUST match the training patch size: the v17 U-Net collapses to
STRIDE = 8     # near-zero output variance on inputs larger than its 16x16
               # training patches (verified: full-domain std_pred 0.19 vs 0.72
               # in the patch regime, pooled r 0.13 vs 0.58). Inference is
               # therefore tiled at 16x16 with Hann blending.


def build_inputs_v9(snap: dict, stats: dict, aux_ch: dict) -> tuple[np.ndarray, np.ndarray]:
    """Full-domain (1,12,H,W) input tensor mirroring AnomalyPatchDatasetV9.
    Returns (x, valid_no2_mask)."""
    no2 = snap["no2"]
    valid_no2 = (np.isfinite(no2) & (no2 > 0)).astype(np.float32)
    mu = stats["no2_log"]["mean"]; sd = stats["no2_log"]["std"]
    with np.errstate(invalid="ignore", divide="ignore"):
        no2_log = np.where(valid_no2 > 0, np.log10(no2 + 1e-3), mu)
    no2_n = ((no2_log - mu) / sd).astype(np.float32)

    chans = [no2_n, valid_no2, aux_ch["hgt_norm"], aux_ch["lsm_norm"],
             aux_ch["sin_lat"], aux_ch["cos_lon"]]
    for k in TEMPO_EXTRA_KEYS:
        a = snap.get(k)
        if a is None:
            chans.append(np.zeros_like(no2_n))
        else:
            m_ = stats[k]["mean"]; sd_ = stats[k]["std"]
            a_n = (a - m_) / sd_
            chans.append(np.where(np.isfinite(a_n), a_n, 0.0).astype(np.float32))
    return np.stack(chans, axis=0)[None], valid_no2


class CellAccum:
    """Per-cell Pearson-r accumulator over time."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-glob", default="runs/20260529_1702*_v17_all_tempo_no_goes_compact_s*")
    ap.add_argument("--out-dir", default=str(HERE / "runs/eval_v17_spatial"))
    ap.add_argument("--max-snapshots", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    aux = load_aux(); cl = load_clim()
    clim = cl["clim"].astype(np.float32)          # (3,H,W) physical
    anom_std = cl["anom_std"].astype(np.float32)  # (3,)
    lat, lon = aux["lat"], aux["lon"]; H, W = lat.size, lon.size
    LON2D, LAT2D = np.meshgrid(lon, lat)
    aux_ch = {
        "sin_lat": np.sin(np.deg2rad(LAT2D)).astype(np.float32),
        "cos_lon": np.cos(np.deg2rad(LON2D)).astype(np.float32),
    }

    run_dirs = sorted((HERE).glob(args.run_glob))
    assert run_dirs, f"no runs match {args.run_glob}"
    print("runs:", [d.name for d in run_dirs], flush=True)

    seed_maps = {v: [] for v in VARS}
    seed_maps_cov = {v: [] for v in VARS}
    seed_cov = []
    err_table = {}
    summary = {}

    for run_dir in run_dirs:
        tag = run_dir.name.split("_")[-1]  # s0/s1/s2
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

        model = UNet2to3(c_in=12, c_out=3, base=cfg["base"], depth=cfg["depth"]).to(device)
        ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()
        print(f"\n=== {run_dir.name}: test={n_test} best_epoch={ck.get('epoch','?')} ===", flush=True)

        acc = {v: CellAccum(H, W) for v in VARS}          # all test times
        acc_cov = {v: CellAccum(H, W) for v in VARS}      # only times with valid TEMPO NO2 at the pixel
        cov_sum = np.zeros((H, W)); cov_n = 0
        # pooled per-snapshot anomaly r (sanity vs test_metrics.json)
        snap_r = {v: [] for v in VARS}
        # same, restricted to TEMPO-covered pixels (closer to the >=70%-coverage
        # patch protocol used by train_v9's test_metrics.json)
        snap_r_cov = {v: [] for v in VARS}
        # error accumulators: n, sum(d), sum(d^2), sum(p), sum(p^2), sum(t), sum(t^2)
        # E      -> physical units (clim added back)
        # E_anom -> normalised-anomaly space (the anti-blur variance-ratio lives here:
        #           physical-space variance is dominated by the shared climatology)
        E = {v: np.zeros(7) for v in VARS}
        E_anom = {v: np.zeros(7) for v in VARS}

        # Tile origins (stride STRIDE, final tile snapped to the border)
        r_orig = sorted(set(list(range(0, H - TILE + 1, STRIDE)) + [H - TILE]))
        c_orig = sorted(set(list(range(0, W - TILE + 1, STRIDE)) + [W - TILE]))
        origins = [(r0, c0) for r0 in r_orig for c0 in c_orig]
        hann = (np.outer(np.hanning(TILE), np.hanning(TILE)) + 0.05).astype(np.float32)
        CHUNK = 4096

        def tiled_predict(x_np):
            """16x16-tile inference with Hann blending -> (3,H,W)."""
            x_full = torch.from_numpy(x_np[0])          # (12,H,W) on CPU
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
                x_np, valid_no2 = build_inputs_v9(snap, stats, aux_ch)
                p_anom = tiled_predict(x_np)             # (3,H,W) normalised anomaly

                truth = np.stack([snap["u10"], snap["v10"], snap["pblh"]], axis=0).astype(np.float32)
                finite = np.isfinite(truth).all(axis=0)
                t_anom = np.where(finite[None], (truth - clim) / anom_std[:, None, None], 0.0)

                m = finite.astype(np.float64)
                m_cov = (finite & (valid_no2 > 0)).astype(np.float64)
                for i, v in enumerate(VARS):
                    acc[v].add(p_anom[i].astype(np.float64), t_anom[i].astype(np.float64), m)
                    acc_cov[v].add(p_anom[i].astype(np.float64), t_anom[i].astype(np.float64), m_cov)
                    # pooled per-snapshot anomaly r
                    pv = p_anom[i][finite]; tv = t_anom[i][finite]
                    if pv.size > 100:
                        pc = pv - pv.mean(); tc = tv - tv.mean()
                        snap_r[v].append(float((pc * tc).sum() /
                                         (np.linalg.norm(pc) * np.linalg.norm(tc) + 1e-12)))
                    fc = finite & (valid_no2 > 0)
                    pv = p_anom[i][fc]; tv = t_anom[i][fc]
                    if pv.size > 100:
                        pc = pv - pv.mean(); tc = tv - tv.mean()
                        snap_r_cov[v].append(float((pc * tc).sum() /
                                             (np.linalg.norm(pc) * np.linalg.norm(tc) + 1e-12)))
                    # physical-unit errors
                    p_phys = clim[i] + anom_std[i] * p_anom[i]
                    pp = p_phys[finite].astype(np.float64); tt = truth[i][finite].astype(np.float64)
                    d = pp - tt
                    E[v] += np.array([d.size, d.sum(), (d ** 2).sum(),
                                      pp.sum(), (pp ** 2).sum(), tt.sum(), (tt ** 2).sum()])
                    # anomaly-space errors
                    pa = p_anom[i][finite].astype(np.float64)
                    ta = t_anom[i][finite].astype(np.float64)
                    da = pa - ta
                    E_anom[v] += np.array([da.size, da.sum(), (da ** 2).sum(),
                                           pa.sum(), (pa ** 2).sum(), ta.sum(), (ta ** 2).sum()])
                cov_sum += valid_no2; cov_n += 1
                if (k + 1) % 200 == 0:
                    print(f"  {k+1}/{n_test} ({(time.time()-t0)/60:.1f} min)", flush=True)

        maps = {v: acc[v].r() for v in VARS}
        maps_cov = {v: acc_cov[v].r() for v in VARS}
        cov = (cov_sum / max(cov_n, 1)).astype(np.float32)
        np.savez_compressed(out_dir / f"v17_{tag}_spatial_raw.npz",
                            lat=lat, lon=lon, tempo_cov=cov,
                            cnt=acc["u10"].CNT.astype(np.int32),
                            cnt_cov=acc_cov["u10"].CNT.astype(np.int32),
                            **{f"r_anom_map_{v}": maps[v] for v in VARS},
                            **{f"r_anom_map_cov_{v}": maps_cov[v] for v in VARS})
        for v in VARS:
            seed_maps[v].append(maps[v])
            seed_maps_cov[v].append(maps_cov[v])
        seed_cov.append(cov)

        def finish(acc):
            n, sd_, sd2, sp, sp2, st, st2 = acc
            var_p = sp2 / n - (sp / n) ** 2
            var_t = st2 / n - (st / n) ** 2
            return {"rmse": float(np.sqrt(sd2 / n)), "bias": float(sd_ / n),
                    "var_ratio": float(var_p / var_t),
                    "std_pred": float(np.sqrt(var_p)), "std_truth": float(np.sqrt(var_t)),
                    "n": int(n)}
        err = {v: {"physical": finish(E[v]), "anomaly": finish(E_anom[v])} for v in VARS}
        err_table[tag] = err
        summary[tag] = {
            "n_test": n_test,
            "pooled_r_anom": {v: float(np.nanmean(snap_r[v])) for v in VARS},
            "pooled_r_anom_tempo_covered": {v: float(np.nanmean(snap_r_cov[v])) for v in VARS},
            "per_cell_median_r_anom": {v: float(np.nanmedian(maps[v])) for v in VARS},
            "per_cell_median_r_anom_covered": {v: float(np.nanmedian(maps_cov[v])) for v in VARS},
            "train_test_metrics_json": json.loads((run_dir / "test_metrics.json").read_text()),
        }
        print(f"  pooled r_anom (full-domain): "
              + "  ".join(f"{v}={summary[tag]['pooled_r_anom'][v]:.3f}" for v in VARS), flush=True)
        print(f"  pooled r_anom (TEMPO-covered pixels): "
              + "  ".join(f"{v}={summary[tag]['pooled_r_anom_tempo_covered'][v]:.3f}" for v in VARS), flush=True)
        print(f"  per-cell median r_anom (all times):     "
              + "  ".join(f"{v}={summary[tag]['per_cell_median_r_anom'][v]:.3f}" for v in VARS), flush=True)
        print(f"  per-cell median r_anom (covered times): "
              + "  ".join(f"{v}={summary[tag]['per_cell_median_r_anom_covered'][v]:.3f}" for v in VARS), flush=True)

    # ----- seed-mean products -----
    mean_maps = {v: np.nanmean(np.stack(seed_maps[v]), axis=0).astype(np.float32) for v in VARS}
    mean_maps_cov = {v: np.nanmean(np.stack(seed_maps_cov[v]), axis=0).astype(np.float32) for v in VARS}
    mean_cov = np.nanmean(np.stack(seed_cov), axis=0).astype(np.float32)
    np.savez_compressed(out_dir / "v17_seedmean_spatial.npz",
                        lat=lat, lon=lon, tempo_cov=mean_cov,
                        **{f"r_anom_map_{v}": mean_maps[v] for v in VARS},
                        **{f"r_anom_map_cov_{v}": mean_maps_cov[v] for v in VARS})
    summary["seedmean_per_cell_median_r_anom"] = {
        v: float(np.nanmedian(mean_maps[v])) for v in VARS}
    summary["seedmean_per_cell_median_r_anom_covered"] = {
        v: float(np.nanmedian(mean_maps_cov[v])) for v in VARS}

    # seed-mean error stats (simple average over seeds)
    seeds = [t for t in err_table]
    err_table["seed_mean"] = {
        v: {sp: {k: float(np.mean([err_table[t][v][sp][k] for t in seeds]))
                 for k in ("rmse", "bias", "var_ratio", "std_pred", "std_truth")}
            for sp in ("physical", "anomaly")}
        for v in VARS}
    (out_dir / "error_magnitude_v17.json").write_text(json.dumps(err_table, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:", json.dumps(summary, indent=2), flush=True)

    # ----- draft 3-panel figure -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.style.use("/home/pandeysu/.claude/assets/sp_figs.mplstyle")
    except Exception:
        pass
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    proj = ccrs.PlateCarree()
    titles = {"u10": "10-m U wind", "v10": "10-m V wind", "pblh": "PBLH"}
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.6),
                             subplot_kw={"projection": proj})
    low_cov = mean_cov < 0.05
    for row, v in enumerate(VARS):
        for col, (mm, tag_) in enumerate([(mean_maps, "all test times"),
                                          (mean_maps_cov, "TEMPO-observed times only")]):
            ax = axes[row, col]
            data = np.where(low_cov, np.nan, mm[v])
            pc = ax.pcolormesh(lon, lat, data, transform=proj, cmap="viridis",
                               vmin=0, vmax=1, shading="auto")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.2")
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.3")
            ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="0.5")
            ax.set_extent([-125, -65, 25, 50], crs=proj)
            med = float(np.nanmedian(data))
            ax.set_title(f"{titles[v]} — {tag_} (median r {med:.2f})", fontsize=10)
            cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.05,
                              pad=0.06, shrink=0.8)
            cb.set_label("anomaly correlation r")
    fig.suptitle("TEMPO v17 (composition-only) — per-pixel skill over CONUS, 3-seed mean\n"
                 "left: r over all held-out snapshots; right: conditional on valid TEMPO NO$_2$ "
                 "at the pixel; grey = TEMPO coverage < 5%",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_v17_conus_skill_draft.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out_dir}/fig_v17_conus_skill_draft.png", flush=True)


if __name__ == "__main__":
    main()
