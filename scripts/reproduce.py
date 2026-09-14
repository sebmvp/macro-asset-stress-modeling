#!/usr/bin/env python3
"""Entry point for `make reproduce` / `python scripts/reproduce.py`."""

from macro_stress.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["reproduce"]))
