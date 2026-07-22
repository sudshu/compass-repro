"""Winner-map verification (paper Fig. 4): tracer partition by lifetime.

The per-gas per-pixel temporal-ACC stacks (``rmaps_<target>.npz``) are the
released output of the single-species evaluations (GEOS-CF: 6 gases on the
blocked archive; CAMS: 8 pressure-weighted-column species on the cross-year
2020 holdout). This module recomputes, from those stacks, exactly what the
paper reports:

- the common-6-gas ({CH4, CO, O3, NO2, HCHO, SO2}) area-weighted winner
  shares per target (the Fig. 4d–f bars and the Results percentages), with
  the identical cos-latitude weighting and argmax convention as the figure
  script (fig2_winner_maps.py::_winner_share, copied verbatim);
- each species' area-weighted median per-pixel ACC.

This verifies the partition claims (e.g. NO2 leads PBLH over 28% of the
area in GEOS-CF and 26% in CAMS; O3 leads at 500 hPa in both systems) from
the released per-pixel evidence. Re-deriving the stacks themselves from the
single-species checkpoints is provided for one representative model as a
spot check (see README).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COMMON6 = ("ch4", "co", "o3", "no2", "hcho", "so2")
TARGETS = ("surface", "winds500", "pblh")


def winner_share(rmaps_npz: Path, species_set=COMMON6) -> dict:
    """Verbatim from fig2_winner_maps.py::_winner_share."""
    d = np.load(rmaps_npz, allow_pickle=True)
    sp = [str(s) for s in d["species"]]
    idx = [sp.index(g) for g in species_set]
    sub = d["stack"][idx]
    wlat = np.broadcast_to(np.cos(np.deg2rad(d["lat"]))[:, None], sub.shape[1:])
    fin = np.isfinite(sub).all(axis=0)
    winner = np.argmax(np.where(np.isfinite(sub), sub, -np.inf), axis=0)
    den = (wlat * fin).sum()
    return {g: float((wlat * ((winner == i) & fin)).sum() / den)
            for i, g in enumerate(species_set)}


def species_median_acc(rmaps_npz: Path) -> dict:
    """Area-weighted median per-pixel ACC per species (cos-lat weights)."""
    d = np.load(rmaps_npz, allow_pickle=True)
    sp = [str(s) for s in d["species"]]
    lat = d["lat"]
    out = {}
    for i, g in enumerate(sp):
        v = d["stack"][i]
        w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], v.shape)
        ok = np.isfinite(v)
        vv, ww = v[ok], w[ok]
        o = np.argsort(vv)
        cw = np.cumsum(ww[o])
        out[g] = float(vv[o][np.searchsorted(cw, 0.5 * cw[-1])])
    return out


def evaluate(winner_dir: Path) -> dict:
    """Winner shares + per-species medians for both systems."""
    res = {}
    for system in ("geoscf", "cams"):
        sd = Path(winner_dir) / system
        res[system] = {
            t: {"winner_share_common6": winner_share(sd / f"rmaps_{t}.npz"),
                "species_median_acc": species_median_acc(sd / f"rmaps_{t}.npz")}
            for t in TARGETS}
    return res
