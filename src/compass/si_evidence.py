"""Evidence-level verification of the SI controls (Suppl. Figs. S6/S7/S10).

These analyses ship as their released result files (the same files the SI
figures are drawn from) plus code that recomputes the manuscript's claims
from them — the released-evidence tier of the package (full re-inference for
these SI controls would add several GB for little review value; the design is
stated in the README).

1. Water-vapour comparison (Suppl. Fig. S6, Suppl. Methods 2): per-pixel
   temporal-ACC maps of the trace-gas and water-vapour models
   (q_vs_rich_pixelmap*.npz) -> cos-lat-weighted land/ocean lead shares with
   the Natural Earth 110-m land mask (shipped): trace gases lead 78% of land
   for PBLH; water vapour leads 84% of the PBLH ocean area, 81% of land for
   near-surface wind speed (75% global, 73% ocean).

2. WRF-Chem single-plume control (Suppl. Fig. S7, Suppl. Methods 3):
   single_plume_eval JSONs (magnitude-augmented + fixed-magnitude
   counterfactual) -> held-out-source skill at training-source level;
   wind-speed skill flat over 0.01x-1000x rescaling for the augmented model;
   the fixed-magnitude model collapses at low source strength; shuffled-plume
   floors far below skill.

3. Virtual in-situ stations (Suppl. Fig. S10, Suppl. Methods 5):
   insitu_station_probe.json -> the Discussion's station-mean single-site
   PBLH temporal ACC (0.54), validation-clean selected models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WV_KEYS = ("pblh_land_tracegas_lead", "pblh_ocean_wv_lead", "ws10_land_wv_lead",
           "ws10_global_wv_lead", "ws10_ocean_wv_lead", "ws500_global_wv_lead")


def wv_shares(wv_dir: Path) -> dict:
    """Cos-lat-weighted lead shares from the released per-pixel ACC maps."""
    lm = np.load(Path(wv_dir) / "land_mask_ne110.npz")
    land = lm["land"]
    w = np.broadcast_to(np.cos(np.deg2rad(lm["lat"]))[:, None], land.shape)
    zp = np.load(Path(wv_dir) / "q_vs_rich_pixelmap.npz")
    zw = np.load(Path(wv_dir) / "q_vs_rich_pixelmap_wind.npz")

    def rich_share(r_rich, r_q, region):
        fin = np.isfinite(r_rich) & np.isfinite(r_q) & region
        lead = (r_rich > r_q) & fin
        return float((w * lead).sum() / (w * fin).sum())

    every = np.ones_like(land, bool)
    return {
        "pblh_land_tracegas_lead": rich_share(zp["r_rich"], zp["r_q"], land),
        "pblh_ocean_wv_lead": 1 - rich_share(zp["r_rich"], zp["r_q"], ~land),
        "ws10_land_wv_lead": 1 - rich_share(zw["r_rich_spd1000"], zw["r_q_spd1000"], land),
        "ws10_global_wv_lead": 1 - rich_share(zw["r_rich_spd1000"], zw["r_q_spd1000"], every),
        "ws10_ocean_wv_lead": 1 - rich_share(zw["r_rich_spd1000"], zw["r_q_spd1000"], ~land),
        "ws500_global_wv_lead": 1 - rich_share(zw["r_rich_spd500"], zw["r_q_spd500"], every),
    }


def wrf_checks(wrf_dir: Path) -> dict:
    """Held-out-source, rescaling-invariance and floor checks from the JSONs."""
    aug = json.loads((Path(wrf_dir) / "single_plume_eval_aug.json").read_text())
    fix = json.loads((Path(wrf_dir) / "single_plume_eval_fix.json").read_text())
    alphas = ("alpha_0.01", "alpha_0.1", "alpha_10.0", "alpha_100.0", "alpha_1000.0")

    def r(d, key, var):
        return d[key][var]["r_mean"]

    out = {
        "train_WSPD10": r(aug, "train_sources", "WSPD10"),
        "held_WSPD10": r(aug, "held_sources", "WSPD10"),
        "train_PBLH": r(aug, "train_sources", "PBLH"),
        "held_PBLH": r(aug, "held_sources", "PBLH"),
        "floor_WSPD10": r(aug, "floor_shuffled", "WSPD10"),
        "floor_PBLH": r(aug, "floor_shuffled", "PBLH"),
        "aug_alpha_WSPD10": {a: r(aug, a, "WSPD10") for a in alphas},
        "fix_alpha_WSPD10": {a: r(fix, a, "WSPD10") for a in alphas},
    }
    base = r(aug, "held_sources", "WSPD10")
    out["claims"] = {
        # held-out sources at training-source skill (both targets, within 0.05)
        "held_at_training_skill": (abs(out["held_WSPD10"] - out["train_WSPD10"]) <= 0.05
                                   and abs(out["held_PBLH"] - out["train_PBLH"]) <= 0.05),
        # augmented model essentially unchanged over 0.01x-1000x (>= 70% of base)
        "aug_invariant": min(out["aug_alpha_WSPD10"].values()) >= 0.7 * base,
        # fixed-magnitude counterfactual degrades at low source strength
        "fix_collapses_low_alpha": out["fix_alpha_WSPD10"]["alpha_0.01"] < 0.3 * base,
        # skill well above the shuffled-plume floor
        "above_floor": (out["held_WSPD10"] > 3 * out["floor_WSPD10"]
                        and out["held_PBLH"] > 3 * out["floor_PBLH"]),
    }
    return out


def insitu_checks(insitu_dir: Path) -> dict:
    """Station-mean single-site PBLH ACC from the released probe JSON."""
    p = json.loads((Path(insitu_dir) / "insitu_station_probe.json").read_text())
    res = p["results"]
    pblh = [res[s]["A_point_instant"]["selected"][2] for s in res]
    u10 = [res[s]["A_point_instant"]["selected"][0] for s in res]
    v10 = [res[s]["A_point_instant"]["selected"][1] for s in res]
    return {"n_stations": len(res),
            "station_mean_pblh_acc": float(np.mean(pblh)),
            "station_mean_u10_acc": float(np.mean(u10)),
            "station_mean_v10_acc": float(np.mean(v10))}


def evaluate(si_dir: Path) -> dict:
    si_dir = Path(si_dir)
    return {"wv": wv_shares(si_dir / "wv"),
            "wrf": wrf_checks(si_dir / "wrf"),
            "insitu": insitu_checks(si_dir / "insitu")}
