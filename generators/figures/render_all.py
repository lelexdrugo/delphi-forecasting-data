"""Render every generated figure from one analysis directory.

    python scripts/figures/render_all.py --analysis-dir <dir> --output-dir figures/generated

Each figure script is a standalone CLI with the same `--analysis-dir /
--output-dir / --dataset-dir / --greyscale` contract; this driver just runs
them in order and reports which succeeded. A script that refuses to plot
(missing artifact, unsupported analysis status, failed cross-check) is
reported as a refusal and does not stop the others, so one gap in the analysis
outputs does not hide the rest of the review.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRIPTS = [
    "fig_interaction_rank",
    "fig_information_ladder",
    "fig_scarcity_curve",
    "fig_conformal_coverage",
    "fig_decision_regret",
    "fig_ranking_quality",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/generated"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--greyscale", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="render only these module names (default: all)",
    )
    args = parser.parse_args(argv)

    base = ["--analysis-dir", str(args.analysis_dir), "--output-dir", str(args.output_dir)]
    if args.dataset_dir is not None:
        base += ["--dataset-dir", str(args.dataset_dir)]
    if args.greyscale:
        base += ["--greyscale"]

    selected = args.only or SCRIPTS
    failures: list[tuple[str, str]] = []
    for name in selected:
        if name not in SCRIPTS:
            failures.append((name, "not a known figure module"))
            continue
        module = importlib.import_module(name)
        try:
            module.main(list(base))
        except SystemExit as exc:
            message = str(exc)
            if message and message != "0":
                failures.append((name, message))
        except Exception as exc:  # pragma: no cover - surfaced to the operator
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    print()
    print(f"[figures] rendered {len(selected) - len(failures)} of {len(selected)} figures")
    for name, message in failures:
        print(f"[figures] REFUSED {name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
