"""Reproduce the TEMPO independence controls (Suppl. Fig. S8).

Runs the primary, no-cloud and mask-only interspersed checkpoints with
full-domain tiled inference over the extracted test scans, and checks the
S8 metrics (conventional pooled ACC, per-scene pooled r, per-cell medians;
full-domain and TEMPO-covered) against the reference evaluation.

Example:
  python scripts/evaluate_tempo_controls.py                # all three modes
  python scripts/evaluate_tempo_controls.py --mode nocloud
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from compass.device import get_device                    # noqa: E402
from compass.tempo_controls import evaluate_mode, VARS   # noqa: E402

TOL = 0.01
RUN_DIRS = {"primary": "weights/tempo", "nocloud": "weights/tempo_nocloud",
            "maskonly": "weights/tempo_maskonly"}
CHECK_KEYS = ("conventional_pooled_acc_full", "pooled_r_anom_full",
              "per_cell_wmedian_r_full", "conventional_pooled_acc_covered")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(PKG / "data/tempo/eval_test"))
    ap.add_argument("--aux", default=str(PKG / "data/tempo/stats/static_aux.npz"))
    ap.add_argument("--clim", default=str(PKG / "data/tempo/stats/static_clim_interspersed.npz"))
    ap.add_argument("--expected", default=str(PKG / "expected/tempo_item20_reference.json"))
    ap.add_argument("--mode", default=None, choices=[None, "primary", "nocloud", "maskonly"])
    ap.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    try:
        expected = json.loads(Path(args.expected).read_text())
    except (OSError, ValueError):
        expected = None

    modes = [args.mode] if args.mode else list(RUN_DIRS)
    ok = True
    for mode in modes:
        t0 = time.time()
        res = evaluate_mode(mode, Path(args.eval_dir), PKG / RUN_DIRS[mode],
                            Path(args.aux), Path(args.clim), device, limit=args.limit)
        print(f"\n=== {mode} ({res['n_test']} scans, {time.time() - t0:.0f} s) ===")
        ref = expected.get(mode) if expected else None
        if ref and args.limit and ref.get("n_test") != res["n_test"]:
            print("  (limit active - PASS/FAIL not applicable)")
            ref = None
        for key in CHECK_KEYS:
            for v in VARS:
                rep = res[key][v]
                if ref and key in ref:
                    e = ref[key][v]
                    good = abs(rep - e) <= TOL
                    ok &= good
                    print(f"  {key:33s} {v:<5s} {rep:8.4f} vs {e:8.4f} "
                          f"{'PASS' if good else 'FAIL'}")
                else:
                    print(f"  {key:33s} {v:<5s} {rep:8.4f}")
    if expected:
        print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
