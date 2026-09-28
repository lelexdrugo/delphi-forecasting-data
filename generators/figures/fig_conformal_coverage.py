"""Figure: conformal coverage and interval width, by regime and by family.

Each row is one evaluated subgroup, labelled with its support. Panel (a) is
what the interval bought -- empirical coverage against the nominal level.
Panel (b) is what it cost -- the interval width in seconds. Reading a row
across the two panels gives the coverage/width trade-off for that subgroup;
reading a column down gives the comparison across subgroups.

A dot plot is used rather than a coverage-versus-width scatter because the
subgroup names are the primary key a reader needs, and in a scatter of ten
named subgroups the labels collide and stop being readable in a single print
column.

Support is never implicit: every row carries its row count and its number of
independent forecast origins, and the calibration-origin count -- the quantity
the finite-sample conformal guarantee actually depends on -- is printed under
the panels. A subgroup the analysis marked unsupported is drawn as an explicit
`unsupported` row, never omitted.

Data source
    uncertainty-metrics.json -> methods[m].test,
                                methods[m].subgroups[declared_state_regime|family][value],
                                calibration_origin_count,
                                minimum_independent_calibration_origins
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig-conformal-coverage"
METHOD_LABEL = {
    "split-conformal": "split conformal",
    "cqr": "conformalized quantile regression (CQR)",
}
SUBGROUP_LABEL = {
    "declared_state_regime": "state regime",
    "family": "workload family",
    "candidate_cluster": "candidate target",
    "history_count_bin": "history budget",
}


def main(argv: list[str] | None = None) -> int:
    parser = C.base_parser(__doc__)
    parser.add_argument(
        "--width-metric",
        default="mean_width_seconds",
        choices=["mean_width_seconds", "median_width_seconds"],
    )
    args = parser.parse_args(argv)
    analysis = C.prepare(args)

    block = analysis.uncertainty
    methods = C.require_key(block, "methods", "uncertainty-metrics.json")
    if not methods:
        raise C.MissingData("uncertainty-metrics.json reports no methods at all.")
    nominal = float(block.get("coverage", 0.9))
    calibration_origins = int(block.get("calibration_origin_count", 0))
    minimum_origins = int(block.get("minimum_independent_calibration_origins", 0))

    ordered = [m for m in ("split-conformal", "cqr") if m in methods]
    ordered += [m for m in sorted(methods) if m not in ordered]
    evaluated = {m: methods[m] for m in ordered if methods[m].get("status") == "evaluated"}

    if not evaluated:
        reasons = "; ".join(
            f"{m}: {methods[m].get('reason', methods[m].get('status'))}" for m in ordered
        )
        fig, ax = C.plt.subplots(figsize=C.figsize(1.0, 0.36))
        C.unsupported_panel(
            ax,
            "Conformal coverage and interval width",
            reasons,
            support=f"{calibration_origins} calibration origins "
            f"(registered minimum {minimum_origins})",
        )
        fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.10)
        C.stamp(fig, analysis)
        C.save(fig, args.output_dir, NAME, analysis)
        return 0

    # -- assemble the rows ---------------------------------------------------
    columns: list[str] = []
    for method in evaluated.values():
        for column in method.get("subgroups", {}):
            if column not in columns:
                columns.append(column)

    rows: list[dict] = [{"kind": "row", "label": "pooled test", "key": None, "pooled": True}]
    for column in columns:
        labels: list[str] = []
        for method in evaluated.values():
            for label in sorted(method.get("subgroups", {}).get(column, {})):
                if label not in labels:
                    labels.append(label)
        if not labels:
            continue
        rows.append({"kind": "header", "label": SUBGROUP_LABEL.get(column, column)})
        for label in labels:
            # Targets are printed by tier; C.tier passes every other subgroup value through.
            rows.append({"kind": "row", "label": C.display(label) if label else "(blank)",
                         "key": (column, label)})

    positions: dict[int, float] = {}
    y = 0.0
    for index, row in enumerate(rows):
        positions[index] = -y
        y += 0.75 if row["kind"] == "header" else 1.0

    fig, (coverage_ax, width_ax) = C.plt.subplots(
        1,
        2,
        figsize=C.figsize(1.0, 0.049 * y + 0.30),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 0.85], "wspace": 0.06},
    )
    C.tidy(coverage_ax, grid_axis="x")
    C.tidy(width_ax, grid_axis="x")

    def cell_of(method: dict, row: dict) -> dict | None:
        if row.get("pooled"):
            return method.get("test")
        column, label = row["key"]
        return method.get("subgroups", {}).get(column, {}).get(label)

    widths: list[float] = []
    annotations: list[tuple[float, str]] = []
    for row_index, row in enumerate(rows):
        if row["kind"] == "header":
            continue
        supports: list[str] = []
        drawn = False
        # A cell can carry support and still carry no metrics: the analysis withdraws a regime whose
        # manipulation check did not pass, and writes its `n` with a status and nothing else. Guarding
        # on `n == 0` alone reads such a cell as plottable, so the absence of the metric itself is the
        # condition, and the status is kept for the row's label.
        undrawable: list[str] = []
        for method_index, method in enumerate(evaluated.values()):
            cell = cell_of(method, row)
            if not cell or int(cell.get("n", 0)) == 0:
                continue
            # A cell below the twenty-observation display rule is marked unsupported by the analysis
            # and carries metrics anyway; the table withholds them, and so must the figure.
            if (
                "empirical_coverage" not in cell
                or args.width_metric not in cell
                or cell.get("status") == "unsupported"
            ):
                undrawable.append(
                    f"{cell.get('status', 'no metrics')} (n = {int(cell.get('n', 0))})"
                )
                continue
            style = C.entity_style(method_index)
            colour = C.maybe_grey(style["color"])
            size = 5.6 if row.get("pooled") else 4.0
            coverage_ax.plot(
                [float(cell["empirical_coverage"])],
                [positions[row_index]],
                linestyle="none",
                marker=style["marker"],
                markersize=size,
                color=colour,
                markeredgecolor=C.SURFACE,
                markeredgewidth=0.7,
                zorder=4,
            )
            width = float(cell[args.width_metric])
            widths.append(width)
            width_ax.plot(
                [width],
                [positions[row_index]],
                linestyle="none",
                marker=style["marker"],
                markersize=size,
                color=colour,
                markeredgecolor=C.SURFACE,
                markeredgewidth=0.7,
                zorder=4,
            )
            supports.append(f"{int(cell['n'])}/{int(cell['origin_count'])}")
            drawn = True
        if not drawn:
            coverage_ax.annotate(
                next(iter(dict.fromkeys(undrawable)), "unsupported (n = 0)"),
                xy=(0.03, positions[row_index]),
                xycoords=("axes fraction", "data"),
                fontsize=C.FONT_SMALL_PT - 0.8,
                color=C.maybe_grey(C.STATUS_CRITICAL),
                va="center",
            )
        annotations.append((positions[row_index], " / ".join(dict.fromkeys(supports))))

    # Row labels and the support column live on the shared y axis.
    tick_positions = [positions[i] for i, r in enumerate(rows)]
    tick_labels = [
        (r["label"] if r["kind"] == "row" else r["label"].upper()) for r in rows
    ]
    coverage_ax.set_yticks(tick_positions)
    coverage_ax.set_yticklabels(tick_labels, fontsize=C.FONT_SMALL_PT)
    for tick, row in zip(coverage_ax.get_yticklabels(), rows):
        if row["kind"] == "header":
            tick.set_color(C.INK_MUTED)
            tick.set_fontsize(C.FONT_SMALL_PT - 1.0)
        elif row.get("pooled"):
            tick.set_color(C.INK)
    coverage_ax.set_ylim(min(tick_positions) - 0.9, 0.9)

    coverage_ax.axvline(nominal, color=C.maybe_grey(C.STATUS_WARNING), linewidth=1.0, zorder=2)
    coverage_ax.annotate(
        f"nominal {nominal:.2f}",
        xy=(nominal, 0.99),
        xycoords=("data", "axes fraction"),
        xytext=(3, -2),
        textcoords="offset points",
        fontsize=C.FONT_SMALL_PT - 0.8,
        color=C.INK_SECONDARY,
        ha="left",
        va="top",
    )
    coverage_ax.set_xlim(-0.03, 1.06)
    coverage_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    coverage_ax.set_xlabel("empirical coverage on test")
    C.panel_title(coverage_ax, "(a) Coverage", width=30)

    span = max(widths) - min(widths) if widths else 1.0
    width_ax.set_xlim(0, (max(widths) if widths else 1.0) + 0.12 * max(span, 1.0))
    width_ax.set_xlabel(f"{args.width_metric.replace('_', ' ').replace('seconds', '(s)')}")
    C.panel_title(width_ax, "(b) Interval width", width=30)

    # The support column, right of panel (b).
    width_ax.annotate(
        "rows / origins",
        xy=(1.02, 0.99),
        xycoords="axes fraction",
        fontsize=C.FONT_SMALL_PT - 1.2,
        color=C.INK_MUTED,
        ha="left",
        va="bottom",
        annotation_clip=False,
    )
    for position, text in annotations:
        if not text:
            continue
        width_ax.annotate(
            text,
            xy=(1.02, position),
            xycoords=("axes fraction", "data"),
            fontsize=C.FONT_SMALL_PT - 1.2,
            color=C.INK_SECONDARY,
            va="center",
            ha="left",
            annotation_clip=False,
        )

    fig.legend(
        handles=[
            C.line_handle(METHOD_LABEL.get(m, m), C.entity_style(i))
            for i, m in enumerate(evaluated)
        ],
        loc="upper left",
        bbox_to_anchor=(0.012, 1.005),
        ncol=2,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    # In words, not artifact keys: the unit and the aggregation are what a reader needs.
    unit = block.get("calibration_unit", "origin")
    aggregation = block.get("within_origin_aggregation", "")
    worst = "max" in str(aggregation)
    gate = (
        f"{calibration_origins} calibration origins (registered minimum {minimum_origins}); "
        f"one score per {unit}"
        + (", the worst across its targets and repetitions." if worst else f" ({aggregation}).")
    )
    unsupported = [m for m in ordered if methods[m].get("status") != "evaluated"]
    if unsupported:
        gate += " Not evaluated: " + "; ".join(
            f"{m} ({methods[m].get('reason', methods[m].get('status'))})" for m in unsupported
        )
    C.footnote(fig, gate)
    fig.subplots_adjust(left=0.215, right=0.845, top=0.885, bottom=0.175)
    C.stamp(fig, analysis)
    C.save(fig, args.output_dir, NAME, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
