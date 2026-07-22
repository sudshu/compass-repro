"""CAMS single-species spot check: winner-map provenance.

Re-runs the O3 pressure-weighted-column winds500 model end-to-end on the
extracted Jan/Apr/Jul-2020 snapshots and compares the resulting per-pixel
temporal-ACC map against the O3 row of the released rmaps_winds500.npz stack
(the winner-map evidence). PASS = the released stack is reproduced by the
released checkpoint.

Example:
  python scripts/evaluate_cams_spot.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "src"))

from compass.device import get_device                       # noqa: E402
from compass.cams_spot import evaluate, compare_with_released  # noqa: E402

TOL_MEDIAN_DIFF = 0.01   # median |per-pixel diff| vs released stack
TOL_MAP_CORR = 0.995     # correlation between reproduced and released maps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-dir", default=str(PKG / "data/cams_spot"))
    ap.add_argument("--run-dir", default=str(PKG / "weights/cams_o3pw_winds500"))
    ap.add_argument("--released", default=str(PKG / "data/winner/cams/rmaps_winds500.npz"))
    ap.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    t0 = time.time()
    res = evaluate(Path(args.spot_dir), Path(args.run_dir), device)
    cmp = compare_with_released(res["rmap"], Path(args.released), species="o3")
    print(f"\nspot check completed in {time.time() - t0:.0f} s")
    print(json.dumps(cmp, indent=2))

    ok = (cmp["median_abs_diff"] <= TOL_MEDIAN_DIFF
          and cmp["map_correlation"] >= TOL_MAP_CORR)
    print("\n" + ("SPOT CHECK PASSED — released winner-map stack reproduced "
                  "by the released checkpoint" if ok else "SPOT CHECK FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
