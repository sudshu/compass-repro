# COMPASS — reproducibility package

Code, model weights and extracted evaluation data for:

> **Recovering atmospheric dynamics from atmospheric composition snapshots
> using machine learning.** Sudhanshu Pandey, NASA Jet Propulsion
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

## Standard evaluation (Zenodo data, reviewer-sized)

```bash
python scripts/fetch_data.py --record <zenodo-id>       # ~12 GB (stride-4 GEOS-CF + TEMPO)
python scripts/evaluate_geoscf.py --eval-dir data/geoscf/eval_stride4 \
    --expected expected/geoscf_stride4.json             # ~2 min on GPU
python scripts/evaluate_tempo.py                        # ~4 min on GPU
python scripts/evaluate_winner.py                       # seconds (Fig. 4 partition)
```

- GEOS-CF standard bundle = every 4th blocked snapshot (185 of 738, 3.5 GB);
  each target agrees with the full-set audited value to within 0.005
  (v500 0.8065 vs 0.8067). The full 738-snapshot archive (14 GB) is optional
  (`--full`) and reproduces the audited table exactly
  (`expected/geoscf_full.json`).
- Headline: 500-hPa meridional wind (v500) pooled ACC **0.81** vs **0.45**
  for 24-h persistence (3-seed mean; per-seed values in `expected/`).
- TEMPO: all nine Fig. 5a numbers (primary / clock / scan-hour-conditioned)
  reproduce to 4 decimals (`expected/tempo_intersp.json`).

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
| GEOS-CF blocked (headline) | all 5 targets to 4 decimals (v500 0.8067; paper 0.81 vs persistence 0.45) | **validated** (738 snapshots, seed 42; s43/s44 via Zenodo) |
| TEMPO → HRRR headline (conventional pooled ACC, manuscript v1.2+) | 0.4456/0.4323/0.7667 → paper 0.43–0.45 / 0.77 | **validated** (`scripts/evaluate_tempo_controls.py --mode primary`) |
| TEMPO diurnal metrics (batch-mean; clock + scan-hour-conditioned) | all 9 numbers to 4 decimals: network 0.4485/0.4775/0.7150, clock 0.2192/0.1647/0.6310, conditioned 0.4041/0.4549/0.5623 | **validated** (1,227 test scans; `scripts/evaluate_tempo.py`) |
| Winner maps / tracer partition (Fig. 4, manuscript v1.2+ wind-speed stacks) | all five share claims exact (GEOS-CF HCHO 46% surface / 45% PBLH, O3 48% 500-hPa; NO2 PBLH 30% GEOS-CF vs 26% CAMS; common-6, ACC>=0.2 analysed-area floor) | **validated** (`scripts/evaluate_winner.py`; legacy zonal stacks + expected retained) |
| GEOS-CF wind-speed ACC + persistence (Fig. 3a) | ws500 0.7140 vs persistence 0.5931, + all components incl. v500 persistence 0.4473 — 14/14 to 4 decimals | **validated** (seed 42; `--baselines`) |
| Water-vapour control (Suppl. Fig. S6) | all lead shares exact (trace gases 78% of PBLH land; WV 84% ocean, 81% ws10 land) | **validated** (evidence tier: `scripts/evaluate_si_evidence.py`) |
| WRF-Chem plume control (Suppl. Fig. S7) | held-out at training skill; 0.01x–1000x invariance; fixed-magnitude collapse; floors | **validated** (evidence tier) |
| Virtual-station analysis (Suppl. Fig. S10) | station-mean PBLH ACC 0.5366 → 0.54 | **validated** (evidence tier) |
| TEMPO interspersed mask-only / no-cloud controls (Suppl. Fig. S8) | all 36 metrics to 4 decimals (primary 0.4456/0.4323/0.7667; no-cloud 0.3934/0.3893/0.6947; mask-only 0.1369/0.1652/0.3836) | **validated** (`scripts/evaluate_tempo_controls.py`) |
| CAMS single-species spot check (winner-map provenance) | released rmaps O3 row reproduced to numerical noise (median diff 6e-8, map corr 1.0000) | **validated** (`scripts/evaluate_cams_spot.py`) |
| Trace-gas-value ablation (manuscript v1.3, Suppl. Fig. S8) | 0.3677/0.3633/0.6551 → paper 0.37/0.36/0.66 | validation running (`--mode tracegas`) |

## Citation / licence

Code: MIT (see LICENSE — pending author confirmation). Data extracts derive
from NASA GMAO GEOS-CF (public), NASA TEMPO L3 (public), NOAA HRRR (public)
and the Copernicus/CAMS reanalysis (attribution required); the extracts are
redistributed for scientific verification with attribution to the original
providers.
