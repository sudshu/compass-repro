"""Reproduce the paper's winner-map partition numbers (Fig. 4 / Results).

Recomputes the common-6-gas area-weighted winner shares and per-species
median per-pixel ACC from the released rmaps stacks, for both systems, and
checks them against the reference values.

Example:
  python scripts/evaluate_winner.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from compass.winner import evaluate, COMMON6, TARGETS  # noqa: E402

TOL_SHARE = 0.001   # absolute tolerance on shares (fraction of area)
TOL_ACC = 0.005


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner-dir", default=str(PKG / "data/winner_wspd"),
                    help="manuscript v1.2+ wind-speed stacks (default); pass "
                         "data/winner + expected/winner_shares.json for the "
                         "pre-v1.2 zonal-wind legacy stacks")
    ap.add_argument("--expected", default=str(PKG / "expected/winner_shares_wspd.json"))
    args = ap.parse_args()

    expected = json.loads(Path(args.expected).read_text())
    res = evaluate(Path(args.winner_dir), acc_floor=expected.get("acc_floor"))

    ok = True
    for system in ("geoscf", "cams"):
        print(f"\n=== {system.upper()} — common-6 winner shares (% of area) ===")
        print(f"{'target':<10s}" + "".join(f"{g:>8s}" for g in COMMON6) + f" {'status':>7s}")
        for t in TARGETS:
            rep = res[system][t]["winner_share_common6"]
            ref = expected[system][t]["winner_share_common6"]
            good = all(abs(rep[g] - ref[g]) <= TOL_SHARE for g in COMMON6)
            ok &= good
            print(f"{t:<10s}" + "".join(f"{100 * rep[g]:>8.1f}" for g in COMMON6)
                  + f" {'PASS' if good else 'FAIL':>7s}")
        med_ok = all(
            abs(res[system][t]["species_median_acc"][g]
                - expected[system][t]["species_median_acc"][g]) <= TOL_ACC
            for t in TARGETS for g in expected[system][t]["species_median_acc"])
        ok &= med_ok
        print(f"per-species median ACC: {'PASS' if med_ok else 'FAIL'}")

    print("\nkey paper numbers:")
    g, c = res["geoscf"], res["cams"]
    print(f"  GEOS-CF NO2 PBLH share {100 * g['pblh']['winner_share_common6']['no2']:.1f}% (paper 28%)"
          f" | CAMS {100 * c['pblh']['winner_share_common6']['no2']:.1f}% (paper 26%)")
    print(f"  O3 500-hPa share: GEOS-CF {100 * g['winds500']['winner_share_common6']['o3']:.1f}%"
          f" (paper 56%) | CAMS {100 * c['winds500']['winner_share_common6']['o3']:.1f}% (paper 66%)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
