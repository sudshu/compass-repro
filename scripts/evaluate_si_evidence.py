"""Verify the SI controls (Suppl. Figs. S6/S7/S10) from the released evidence.

Recomputes the water-vapour lead shares, the WRF-Chem plume-control claims
and the virtual-station summary from the shipped result files and checks
them against the reference. Seconds, no GPU, numpy only.

Example:
  python scripts/evaluate_si_evidence.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from compass.si_evidence import evaluate, WV_KEYS  # noqa: E402

TOL = 0.005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--si-dir", default=str(PKG / "data/si_evidence"))
    ap.add_argument("--expected", default=str(PKG / "expected/si_evidence.json"))
    args = ap.parse_args()

    res = evaluate(Path(args.si_dir))
    expected = json.loads(Path(args.expected).read_text())
    ok = True

    print("=== Water-vapour comparison (Suppl. Fig. S6) — lead shares ===")
    for k in WV_KEYS:
        rep, ref = res["wv"][k], expected["wv"][k]
        good = abs(rep - ref) <= TOL
        ok &= good
        print(f"  {k:28s} {100 * rep:5.1f}% vs {100 * ref:5.1f}%  "
              f"{'PASS' if good else 'FAIL'}")

    print("=== WRF-Chem single-plume control (Suppl. Fig. S7) — claims ===")
    for k, v in res["wrf"]["claims"].items():
        ok &= v
        print(f"  {k:28s} {'PASS' if v else 'FAIL'}")
    a = res["wrf"]
    print(f"  (held/train WSPD10 {a['held_WSPD10']:.3f}/{a['train_WSPD10']:.3f}; "
          f"floors {a['floor_WSPD10']:.3f}/{a['floor_PBLH']:.3f})")

    print("=== Virtual stations (Suppl. Fig. S10) ===")
    rep = res["insitu"]["station_mean_pblh_acc"]
    ref = expected["insitu"]["station_mean_pblh_acc"]
    good = abs(rep - ref) <= TOL
    ok &= good
    print(f"  station-mean PBLH ACC        {rep:.4f} vs {ref:.4f}  "
          f"{'PASS' if good else 'FAIL'}  (manuscript: 0.54)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
