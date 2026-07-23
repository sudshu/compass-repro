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


def winner_share(rmaps_npz: Path, species_set=COMMON6, acc_floor: float = None) -> dict:
    """Common-set winner shares (cos-lat weighted argmax).

    With ``acc_floor`` (manuscript v1.2+ convention, fig2_winner_maps.py /
    figS_cams_winner.py): pixels where no gas in the set reaches the floor
    are excluded from the analysed area ("grey-masked").
    Without it: the pre-v1.2 unfloored convention.
    """
    d = np.load(rmaps_npz, allow_pickle=True)
    sp = [str(s) for s in d["species"]]
    idx = [sp.index(g) for g in species_set]
    sub = d["stack"][idx]
    wlat = np.broadcast_to(np.cos(np.deg2rad(d["lat"]))[:, None], sub.shape[1:])
    fin = np.isfinite(sub).all(axis=0)
    if acc_floor is not None:
        fin = fin & (np.nanmax(sub, axis=0) >= acc_floor)
    winner = np.argmax(np.where(np.isfinite(sub), sub, -np.inf), axis=0)
    den = (wlat * fin).sum()
    return {g: float((wlat * ((winner == i) & fin)).sum() / den)
            for i, g in enumerate(species_set)}


def winner_share_floored(rmaps_npz: Path, acc_floor: float = 0.2) -> dict:
    """Manuscript v1.2+ convention (fig2_winner_maps.py, 2026-07-22): shares of
    the ANALYSED area — the stack's own winner argmax over its full species
    set, cos-lat weighted, restricted to pixels where the best gas reaches
    ``acc_floor`` (grey-masked otherwise)."""
    d = np.load(rmaps_npz, allow_pickle=True)
    sp = [str(s) for s in d["species"]]
    winner = d["winner"].astype(float)
    stack = d["stack"]
    best = np.nanmax(stack, axis=0)
    low = ~(best >= acc_floor)
    winner = np.where(low, np.nan, winner)
    wlat = np.broadcast_to(np.cos(np.deg2rad(d["lat"]))[:, None], winner.shape)
    fin = np.isfinite(winner)
    den = (wlat * fin).sum()
    return {g: float((wlat * ((winner == k) & fin)).sum() / den)
            for k, g in enumerate(sp)}


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


def evaluate(winner_dir: Path, acc_floor: float = None) -> dict:
    """Winner shares + per-species medians for both systems.

    Pass ``acc_floor=0.2`` for the manuscript v1.2+ wind-speed stacks."""
    res = {}
    for system in ("geoscf", "cams"):
        sd = Path(winner_dir) / system
        res[system] = {
            t: {"winner_share_common6": winner_share(sd / f"rmaps_{t}.npz",
                                                     acc_floor=acc_floor),
                "species_median_acc": species_median_acc(sd / f"rmaps_{t}.npz")}
            for t in TARGETS}
    return res
