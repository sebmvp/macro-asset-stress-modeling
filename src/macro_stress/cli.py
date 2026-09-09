"""Developer CLI: data-check | reproduce."""

from __future__ import annotations

import argparse
import json
import sys

from .data import CANONICAL_CSV_NAME, DATA_DIR, data_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="macro-stress")
    parser.add_argument("command", choices=["data-check", "reproduce"])
    parser.add_argument("--data", default=None, help="Path to the Kaggle CSV")
    args = parser.parse_args(argv)

    if args.command == "data-check":
        info = data_check(args.data)
        print(json.dumps(info, indent=2))
        if not info["found"]:
            print(
                f"\nPlace {CANONICAL_CSV_NAME} in {DATA_DIR} — see data/README.md",
                file=sys.stderr,
            )
            return 2
        return 0 if not info["issues"] else 1

    from .reproduce import reproduce

    try:
        payload = reproduce(args.data)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    e = next(r for r in payload["model_e"] if r["model"] == "XGBoost")
    print(
        "Reconstructed run written to results/latest/run.json\n"
        f"Model E XGBoost ROC–AUC={e.get('roc_auc'):.4f}  "
        f"(published checksum 0.6835)\n"
        f"Walk-forward mean ROC–AUC={payload.get('walk_forward_mean_roc_auc')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
