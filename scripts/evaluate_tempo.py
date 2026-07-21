"""Reproduce the paper's TEMPO -> HRRR interspersed evaluation (Fig. 5a).

Runs the released v20-interspersed checkpoint on the extracted test snapshots
and prints the three configurations reported in the paper — the primary
network, the clock (scan-hour climatology) baseline, and the scan-hour-
conditioned score — against the reference values, with PASS/FAIL.

Example:
  python scripts/evaluate_tempo.py            # full 1,227-scan test set
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from compass.device import get_device        # noqa: E402
from compass.tempo import evaluate, TARGETS  # noqa: E402

TOL = 0.01
CONFIGS = ("network_fp32", "clock_baseline", "network_hourcond")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(PKG / "data/tempo/eval_test"))
    ap.add_argument("--run-dir", default=str(PKG / "weights/tempo"))
    ap.add_argument("--aux", default=str(PKG / "data/tempo/stats/static_aux.npz"))
    ap.add_argument("--clim", default=str(PKG / "data/tempo/stats/static_clim_interspersed.npz"))
    ap.add_argument("--lookup", default=str(PKG / "data/tempo/stats/hour_lookup.npz"))
    ap.add_argument("--expected", default=str(PKG / "expected/tempo_intersp.json"))
    ap.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="limit #test snapshots")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")

    t0 = time.time()
    res = evaluate(Path(args.eval_dir), Path(args.run_dir), Path(args.aux),
                   Path(args.clim), Path(args.lookup), device, limit=args.limit)
    print(f"\n{res['n_test_files']} test snapshots evaluated in "
          f"{time.time() - t0:.0f} s on {device} "
          f"(off-hour skipped patches: {res['skipped_patches_offhour']})")

    exp_path = Path(args.expected)
    expected = json.loads(exp_path.read_text()) if exp_path.exists() else None
    if expected and expected.get("n_test_files") not in (None, res["n_test_files"]):
        print(f"\nNOTE: evaluated {res['n_test_files']} snapshots but the reference "
              f"used {expected['n_test_files']} — PASS/FAIL not applicable.")
        expected = None

    ok = True
    for key in CONFIGS:
        print(f"\n{key}")
        print(f"  {'target':<12s} {'reproduced':>11s} {'reported':>9s} {'status':>7s}")
        for t in TARGETS:
            rep = res[key][t]
            if expected and key in expected:
                ref = expected[key][t]
                good = abs(rep - ref) <= TOL
                ok &= good
                print(f"  {t:<12s} {rep:>11.4f} {ref:>9.4f} {'PASS' if good else 'FAIL':>7s}")
            else:
                print(f"  {t:<12s} {rep:>11.4f} {'--':>9s} {'--':>7s}")
    if expected:
        print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
