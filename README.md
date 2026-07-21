# COMPASS — reproducibility package

Code, model weights and extracted evaluation data for:

> **Neural networks infer winds and boundary-layer height from single
> atmospheric composition scenes.** Sudhanshu Pandey, NASA Jet Propulsion
> Laboratory / Caltech. (Manuscript under review.)

This package lets a reviewer or reader **re-run the paper's evaluations from
the released model weights on the released data and reproduce the reported
numbers** — not merely re-plot them. All models are small 2D U-Nets, so
everything runs on a **Linux CUDA GPU**, an **Apple-Silicon Mac** (PyTorch
MPS) or plain **CPU** with the same commands.

**Scope note.** Within-simulation skill (GEOS-CF, CAMS) is an
information-content statement about the simulated atmosphere, not a
real-world retrieval; the TEMPO→HRRR experiment is the real-observation test.
The paper's claims are calibrated accordingly and this package preserves that
distinction.

## Quick start (smoke test, ~1 minute on GPU / a few minutes on a Mac)

```bash
conda env create -f environment.yml
conda activate compass-repro
python scripts/evaluate_geoscf.py            # 16-snapshot subset shipped in-repo
```

Expected output: pooled anomaly-correlation (ACC) for the five targets with
PASS against `expected/geoscf_subset.json` (reference values produced by the
original analysis pipeline; the packaged pipeline reproduces them to 4
decimal places).

## Full evaluation (reproduces the paper's headline table)

```bash
python scripts/fetch_data.py geoscf          # downloads ~13 GB from Zenodo
python scripts/evaluate_geoscf.py --eval-dir data/geoscf/eval_full \
    --expected expected/geoscf_full.json
```

Headline: 500-hPa meridional wind (v500) pooled ACC **0.81** vs **0.45** for
24-h persistence (3-seed mean; per-seed values in `expected/`).

## What is in the data

The full raw archives (GEOS-CF NetCDF, TEMPO L3, CAMS, HRRR) are tens of TB
and are not redistributed. Instead, `data/` contains the **extracted
variables the evaluation actually consumes**, produced by
`scripts/extract_geoscf.py` (shipped for transparency):

- `col_n`, `sfc_n` — the model's z-scored trace-gas input channels
  (CH4, CO, O3, NO2 columns + near-surface), after per-snapshot zonal-mean
  removal and normalisation with the **training-period** statistics
  (`data/geoscf/stats/stats_geos_cf.json`, shipped);
- `met` — the target fields (u10, v10, u500, v500, PBLH) in physical units;
- `data/geoscf/stats/clim_train.npz` — the training-period climatology that
  defines the anomaly (never recomputed from the evaluation period).

float16 storage was validated against the original float32 NetCDF pipeline:
pooled ACC agrees to 4 decimal places.

## Evaluation protocol (identical to the paper)

Full-globe 80-px half-stride Hann-blended tiling; anomaly =
(truth − training climatology)/anomaly-std; pooled (cell, time) Pearson r
over |lat| ≤ 75°; months of the evaluation archive (Aug/Oct/Dec 2020) are
absent from training. See `src/compass/geoscf.py` for the exact code and
provenance notes.

## Device support

`src/compass/device.py` selects CUDA → MPS → CPU automatically; force one
with `--device cuda|mps|cpu`. The networks use only Conv2d / GroupNorm /
GELU / MaxPool2d / ConvTranspose2d — no CUDA-only ops.

## Layout

```
src/compass/         package: models, metrics, device, per-system evaluation
scripts/             CLI entry points + maintainer-side extraction
weights/             released checkpoints (per system / seed)
data/                extracted evaluation data + training stats/climatology
expected/            reference numbers the run scripts check against
```

## Status

| System | Result verified | Status |
|---|---|---|
| GEOS-CF blocked (headline) | v500 ACC 0.81 vs persistence 0.45 | **working** (seed 42 shipped; s43/s44 via Zenodo) |
| TEMPO → HRRR | 0.45/0.48/0.72 | in preparation |
| CAMS winner maps / tracer partition | lifetime partition | in preparation |
| Water-vapour control | — | in preparation |
| WRF-Chem plume control | held-out-source generalisation | in preparation |

## Citation / licence

Code: MIT (see LICENSE — pending author confirmation). Data extracts derive
from NASA GMAO GEOS-CF (public), NASA TEMPO L3 (public), NOAA HRRR (public)
and the Copernicus/CAMS reanalysis (attribution required); the extracts are
redistributed for scientific verification with attribution to the original
providers.
