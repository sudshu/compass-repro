"""Reproduce the paper's temporally blocked GEOS-CF evaluation.

Runs inference with the released weights on the extracted evaluation
snapshots and prints reproduced vs reported pooled anomaly-correlation (ACC),
with PASS/FAIL per target.

Examples
--------
Smoke test (subset shipped in the repository), any machine:
  python scripts/evaluate_geoscf.py --eval-dir data/geoscf/eval_subset --expected expected/geoscf_subset.json

Full evaluation (738 snapshots fetched from Zenodo):
  python scripts/evaluate_geoscf.py --eval-dir data/geoscf/eval_full --expected expected/geoscf_full.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from compass.device import get_device            # noqa: E402
from compass.geoscf import evaluate, TARGET_NAMES  # noqa: E402

TOL = 0.01  # |reproduced - reported| tolerance on pooled ACC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(PKG / "data/geoscf/eval_subset"))
    ap.add_argument("--weights", default=str(PKG / "weights/geoscf/s42_best.pt"))
    ap.add_argument("--config", default=str(PKG / "weights/geoscf/s42_config.json"))
    ap.add_argument("--clim", default=str(PKG / "data/geoscf/stats/clim_train.npz"))
    ap.add_argument("--expected", default=str(PKG / "expected/geoscf_subset.json"))
    ap.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="take every Nth snapshot")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")

    t0 = time.time()
    res = evaluate(Path(args.eval_dir), Path(args.weights), Path(args.config),
                   Path(args.clim), device, limit=args.limit, stride=args.stride)
    dt = time.time() - t0
    pooled = res["pooled_anom_acc"]
    print(f"\n{res['n_files']} snapshots evaluated in {dt:.0f} s on {device}")

    exp_path = Path(args.expected)
    try:
        expected = json.loads(exp_path.read_text())
    except (OSError, ValueError):
        expected = None
    if expected and expected.get("n_files") not in (None, res["n_files"]):
        print(f"\nNOTE: evaluated {res['n_files']} snapshots but the reference "
              f"was computed on {expected['n_files']} — PASS/FAIL not applicable.")
        expected = None

    print(f"\n{'target':<7s} {'reproduced':>11s} {'reported':>9s} {'status':>7s}")
    print("-" * 40)
    ok = True
    for t in TARGET_NAMES:
        rep = pooled[t]
        if expected and t in expected.get("pooled_anom_acc", {}):
            ref = expected["pooled_anom_acc"][t]
            good = abs(rep - ref) <= TOL
            ok &= good
            print(f"{t:<7s} {rep:>11.4f} {ref:>9.4f} {'PASS' if good else 'FAIL':>7s}")
        else:
            print(f"{t:<7s} {rep:>11.4f} {'--':>9s} {'--':>7s}")
    if expected:
        print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
