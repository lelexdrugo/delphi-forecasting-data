"""Figure: is one speed factor per target enough?

Panel A shows the measured runtime of every evaluated workload point on every
target, with the rank-1 speed-factor baseline's prediction drawn on the same
mark, so the residual is the visible gap between the filled and the open
marker.

Panel B is the falsification panel. The rank-1 baseline fits
`log runtime = workload terms + one-hot target`, so the runtime *ratio* it
predicts between any two targets is the same for every workload: a horizontal
line. Measured ratios that drift away from that line are exactly the part of
the workload-by-target interaction a single speed factor per target cannot
carry.

Data source
    fold-predictions.csv  (column `rank1-target-speed-factor`, the measured
                           runtime, the target and the origin)
    dataset.csv           (point_id and family, which fold-predictions.csv
                           does not carry)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig-interaction-rank"
RANK1 = "rank1-target-speed-factor"
# Short family names for the rotated tick labels, in the manuscript's words (Table 4).
FAMILY_SHORT = {
    "stress-ng-cpu": "CPU stressor",
    "video-transcode": "video",
    "compress-encrypt": "compress-encrypt",
    "onnx-inference-fp32": "FP32 inference",
    "onnx-inference-int8": "INT8 inference",
    "graph-kernel": "graph",
    "duckdb-tpch": "SQL",
}


def main(argv: list[str] | None = None) -> int:
    parser = C.base_parser(__doc__)
    parser.add_argument("--split", default="test", choices=["test", "calibration", "all"])
    parser.add_argument(
        "--reference-target",
        default=None,
        help="denominator of the ratio panel (default: the first canonical target present)",
    )
    args = parser.parse_args(argv)
    analysis = C.prepare(args)

    if RANK1 not in analysis.predictions.columns:
        raise C.MissingData(
            f"fold-predictions.csv has no '{RANK1}' column, so the rank-1 reference "
            "cannot be drawn. The analysis did not run the registered baseline."
        )

    frame = analysis.with_dataset(["point_id", "family", "primary_size"])
    frame = frame[
        (frame["outcome_status"] == C.COMPLETE_STATUS)
        & frame["execution_runtime_seconds"].notna()
    ]
    if args.split != "all":
        frame = frame[frame["split"] == args.split]
    if frame.empty:
        raise C.MissingData(f"no completed rows in split '{args.split}'.")

    frame["execution_runtime_seconds"] = pd.to_numeric(frame["execution_runtime_seconds"])
    frame[RANK1] = pd.to_numeric(frame[RANK1], errors="coerce")

    targets = C.sort_targets(frame["candidate_cluster"].unique())
    reference = args.reference_target or targets[0]
    if reference not in targets:
        raise C.MissingData(f"reference target '{reference}' is absent from split '{args.split}'.")

    cell = (
        frame.groupby(["point_id", "family", "candidate_cluster"])
        .agg(
            measured=("execution_runtime_seconds", "median"),
            rank1=(RANK1, "median"),
            rows=("execution_runtime_seconds", "size"),
        )
        .reset_index()
    )
    measured = cell.pivot(index=["point_id", "family"], columns="candidate_cluster", values="measured")
    predicted = cell.pivot(index=["point_id", "family"], columns="candidate_cluster", values="rank1")
    complete = measured[targets].notna().all(axis=1) & predicted[targets].notna().all(axis=1)
    if not complete.any():
        raise C.MissingData(
            "no workload point has a measured value on every target in this split; "
            "the ratio panel would compare different point sets across targets."
        )
    dropped = int((~complete).sum())
    measured, predicted = measured[complete], predicted[complete]
    order = measured[reference].sort_values().index
    measured, predicted = measured.loc[order], predicted.loc[order]
    positions = np.arange(len(order))

    fig, (top, bottom) = C.plt.subplots(
        2,
        1,
        figsize=C.figsize(1.0, 0.76),  # shorter family tick labels need less height
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.36},
    )
    C.tidy(top)
    C.tidy(bottom)

    # -- Panel A: measured runtime with the rank-1 prediction on the same mark
    for target in targets:
        style = C.target_style(target)
        colour = C.maybe_grey(style["color"])
        y_measured = measured[target].to_numpy(dtype=float)
        y_predicted = predicted[target].to_numpy(dtype=float)
        top.vlines(positions, y_measured, y_predicted, color=colour, linewidth=0.7, alpha=0.55, zorder=2)
        top.plot(
            positions,
            y_predicted,
            linestyle="none",
            marker="_",
            markersize=6.0,
            markeredgewidth=1.0,
            color=colour,
            zorder=3,
        )
        top.plot(
            positions,
            y_measured,
            linestyle="none",
            marker=style["marker"],
            markersize=3.4,
            markerfacecolor=colour,
            markeredgecolor=C.SURFACE,
            markeredgewidth=0.6,
            color=colour,
            zorder=4,
        )
    top.set_yscale("log")
    top.set_ylabel("runtime (s, log)")
    C.panel_title(
        top, "(a) Measured runtime per workload point and target (marker), with the "
        "rank-one speed-factor prediction on the same mark (tick)"
    )
    fig.legend(
        handles=[C.line_handle(C.tier(t), C.target_style(t)) for t in targets]
        + [
            C.Line2D([], [], color=C.INK_MUTED, marker="_", linestyle="none",
                     markersize=6, label="rank-one prediction"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 1.005),
        ncol=5,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # -- Panel B: ratio to the reference target; rank-1 predicts a flat line
    spreads = {}
    for target in [t for t in targets if t != reference]:
        style = C.target_style(target)
        colour = C.maybe_grey(style["color"])
        ratio_measured = (measured[target] / measured[reference]).to_numpy(dtype=float)
        ratio_predicted = (predicted[target] / predicted[reference]).to_numpy(dtype=float)
        level = float(np.median(ratio_predicted))
        spreads[target] = float(np.max(ratio_predicted) - np.min(ratio_predicted))
        line = bottom.plot(
            [positions[0] - 0.5, positions[-1] + 0.5],
            [level, level],
            color=colour,
            linewidth=0.9,
            alpha=0.85,
            zorder=2,
        )[0]
        line.set_dashes([2.6, 1.6])
        bottom.plot(
            positions,
            ratio_measured,
            linestyle="none",
            marker=style["marker"],
            markersize=3.4,
            markerfacecolor=colour,
            markeredgecolor=C.SURFACE,
            markeredgewidth=0.6,
            color=colour,
            zorder=4,
        )
        bottom.annotate(
            C.tier(target),
            xy=(positions[-1] + 0.6, level),
            fontsize=C.FONT_SMALL_PT,
            color=colour,
            va="center",
            ha="left",
            annotation_clip=False,
        )
    bottom.axhline(1.0, color=C.AXIS, linewidth=0.7, zorder=1)
    bottom.set_yscale("log")
    bottom.set_ylabel(f"ratio to {C.tier(reference)} (log)")
    C.panel_title(
        bottom,
        "(b) Rank one predicts one ratio per target for every workload (dashed line); "
        "the measured ratios (markers) do not follow it",
    )

    # Tick labels name the family in the manuscript's words, not the pipeline's point
    # identifier, which no reader can decode (round-2 review, figure labels).
    labels = [FAMILY_SHORT.get(family, C.display(family)) for _, family in measured.index]
    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels, rotation=90, fontsize=C.FONT_SMALL_PT - 1.0)
    bottom.set_xlim(positions[0] - 0.8, positions[-1] + 3.0)
    bottom.set_xlabel(f"workload point (family), ordered by measured runtime on the {C.tier(reference)} tier")

    worst_spread = max(spreads.values()) if spreads else 0.0
    note = (
        "The rank-one ratio is constant by construction"
        if worst_spread < 0.01
        else f"The rank-one ratio varies across points by at most {worst_spread:.3f}; the dashed "
        "line is its median"
    )
    support = C.support_text(
        rows=int(len(frame)),
        origins=int(frame["origin_id"].nunique()),
        points=int(len(measured)),
        targets=len(targets),
    )
    if dropped:
        support += f"; {dropped} point(s) without all-target coverage excluded"
    C.footnote(fig, f"{support}. {note}.")

    fig.subplots_adjust(left=0.105, right=0.865, top=0.875, bottom=0.305)
    C.stamp(fig, analysis, note=f"split={args.split}; reference={C.tier(reference)}")
    C.save(fig, args.output_dir, NAME, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
