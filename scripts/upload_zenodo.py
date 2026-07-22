"""Create (or update) the DRAFT Zenodo deposition and upload the data bundles.

- NEVER publishes: the record stays a private draft until the author clicks
  Publish (at acceptance). Publishing is irreversible, so it is not automated.
- The DOI is pre-reserved at draft creation and printed — it can be cited in
  the manuscript's Data Availability statement before publication.
- Idempotent: re-running skips files already present with the same size, so
  an interrupted upload resumes where it left off.

Token: read from ~/.zenodo_token (never from the repository or argv).

Usage:
  python upload_zenodo.py --staging /path/to/zenodo_staging          # create + upload
  python upload_zenodo.py --staging ... --deposition-id 12345       # resume/update
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

API = "https://zenodo.org/api"

METADATA = {
    "metadata": {
        "title": ("COMPASS reproducibility data: extracted evaluation variables and "
                  "model weights for 'Recovering atmospheric dynamics from "
                  "atmospheric composition snapshots using machine learning'"),
        "upload_type": "dataset",
        "description": (
            "<p>Extracted evaluation data and trained model weights accompanying the "
            "manuscript <i>Recovering atmospheric dynamics from atmospheric composition "
            "snapshots using machine learning</i> (S. Pandey). Together with the code repository, "
            "this record lets a reader re-run the paper's evaluations from the released "
            "weights on the released data and reproduce the reported anomaly-correlation "
            "metrics.</p>"
            "<p>Contents: GEOS-CF temporally blocked evaluation extracts (738 snapshots), "
            "TEMPO&rarr;HRRR interspersed test extracts (1,227 scans), the train-split "
            "scan-hour lookup, model checkpoints, and normalisation/climatology files. "
            "Formats and the evaluation protocol are documented in the code repository "
            "README and MANIFEST.json.</p>"
            "<p>Source data credits: NASA GMAO GEOS-CF; NASA TEMPO L3; NOAA HRRR; "
            "Copernicus Atmosphere Monitoring Service (CAMS). The extracts are derived "
            "variable subsets redistributed for scientific verification with attribution "
            "to the original providers.</p>"),
        "creators": [{
            "name": "Pandey, Sudhanshu",
            "affiliation": ("NASA Jet Propulsion Laboratory, California Institute of "
                            "Technology, Pasadena, CA, USA"),
        }],
        "license": "cc-by-4.0",
        "keywords": ["atmospheric composition", "trace gases", "boundary-layer height",
                     "neural networks", "wind field retrieval", "TEMPO", "GEOS-CF",
                     "reproducibility"],
        "access_right": "open",
    }
}


def token() -> str:
    return Path("~/.zenodo_token").expanduser().read_text().strip()


def upload_file(bucket_url: str, fp: Path, tok: str, tries: int = 3) -> None:
    for k in range(tries):
        try:
            with open(fp, "rb") as fh:
                r = requests.put(f"{bucket_url}/{fp.name}",
                                 data=fh,
                                 params={"access_token": tok},
                                 timeout=7200)
            r.raise_for_status()
            info = r.json()
            print(f"  uploaded {fp.name}: {info.get('size')} bytes, "
                  f"checksum {info.get('checksum')}", flush=True)
            return
        except Exception as e:                     # noqa: BLE001
            print(f"  attempt {k + 1} failed for {fp.name}: {e}", flush=True)
            if k == tries - 1:
                raise
            time.sleep(30 * (k + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--deposition-id", type=int, default=0,
                    help="existing draft to resume/update (default: create new)")
    args = ap.parse_args()
    tok = token()
    staging = Path(args.staging)
    files = sorted(p for p in staging.iterdir() if p.is_file())
    if not files:
        sys.exit(f"nothing to upload in {staging}")

    if args.deposition_id:
        r = requests.get(f"{API}/deposit/depositions/{args.deposition_id}",
                         params={"access_token": tok}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"resuming deposition {dep['id']}")
    else:
        r = requests.post(f"{API}/deposit/depositions",
                          params={"access_token": tok}, json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"created draft deposition {dep['id']}")

    # metadata (safe to re-apply)
    r = requests.put(f"{API}/deposit/depositions/{dep['id']}",
                     params={"access_token": tok}, json=METADATA, timeout=60)
    r.raise_for_status()
    dep = r.json()

    doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "?")
    bucket = dep["links"]["bucket"]
    have = {f["filename"]: f["filesize"] for f in dep.get("files", [])}

    print(f"reserved DOI : {doi}")
    print(f"draft URL    : {dep['links'].get('html', '?')}")
    for fp in files:
        if have.get(fp.name) == fp.stat().st_size:
            print(f"  skip (already uploaded): {fp.name}", flush=True)
            continue
        print(f"uploading {fp.name} ({fp.stat().st_size / 1e9:.2f} GB)...", flush=True)
        upload_file(bucket, fp, tok)

    print("\nALL FILES UPLOADED — record remains an UNPUBLISHED DRAFT.")
    print(f"deposition id: {dep['id']}   reserved DOI: {doi}")
    print("Publish manually at acceptance (irreversible).")
    state = {"deposition_id": dep["id"], "reserved_doi": doi,
             "html": dep["links"].get("html", "")}
    (staging / "zenodo_deposition.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
