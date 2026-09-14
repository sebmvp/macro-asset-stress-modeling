#!/usr/bin/env python3
"""Download the public Kaggle panel used by the original project.

The CSV is licensed CC BY-NC-SA 4.0 on Kaggle and is not committed here.

Requires a Kaggle account that has accepted the dataset terms, plus an API
token at ~/.kaggle/kaggle.json (chmod 600).

Fallback: download the dataset in a browser and place
  data/Global_Market_Stress_and_Liquidity_Regimes.csv
in this repository.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CANONICAL_NAME = "Global_Market_Stress_and_Liquidity_Regimes.csv"
KAGGLE_DATASET = "kanchana1990/algorithmic-trading-macro-stress-and-asset-regimes"
KAGGLE_URL = (
    "https://www.kaggle.com/datasets/kanchana1990/"
    "algorithmic-trading-macro-stress-and-asset-regimes"
)


def _print_manual() -> None:
    print(
        "\nManual download\n"
        f"  1. Open {KAGGLE_URL}\n"
        "  2. Accept the dataset terms.\n"
        f"  3. Download the CSV and save it as {DATA_DIR / CANONICAL_NAME}\n"
        "  4. Run: make data-check\n",
        file=sys.stderr,
    )


def _kaggle_json() -> Path:
    return Path.home() / ".kaggle" / "kaggle.json"


def download(dest: Path = DATA_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / CANONICAL_NAME
    if target.is_file():
        print(f"already present: {target}")
        return target

    token = _kaggle_json()
    if not token.is_file():
        print(
            "No ~/.kaggle/kaggle.json found.\n"
            "Create a Kaggle API token (Account → API → Create New Token)\n"
            "and save it as ~/.kaggle/kaggle.json with chmod 600.",
            file=sys.stderr,
        )
        _print_manual()
        raise SystemExit(2)

    kaggle = shutil.which("kaggle")
    cmd = [
        kaggle or sys.executable,
        *(["-m", "kaggle"] if kaggle is None else []),
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(dest),
        "--unzip",
    ]
    print("running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Kaggle CLI failed: {exc}", file=sys.stderr)
        _print_manual()
        raise SystemExit(2) from exc

    # Some unzip layouts leave a nested zip or a differently named CSV.
    if not target.is_file():
        for zip_path in dest.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        matches = sorted(dest.rglob("*.csv"))
        for path in matches:
            if path.name == CANONICAL_NAME:
                if path != target:
                    path.replace(target)
                break
        else:
            if len(matches) == 1 and matches[0] != target:
                matches[0].replace(target)

    if not target.is_file():
        print("Download finished but the canonical CSV name was not found.", file=sys.stderr)
        _print_manual()
        raise SystemExit(2)

    print(f"wrote {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DATA_DIR,
        help="Directory to write the CSV into (default: data/)",
    )
    args = parser.parse_args(argv)
    download(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
