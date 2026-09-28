"""Figure: the information ladder W -> W+H -> W+H+C -> W+H+C+S.

Panel (a) is the level: the test MAE the analysis reports for each rung.
Panel (b) is the comparison that actually answers the question: the *paired*
change in absolute error from the previous rung, with a 95 % interval.

Where the interval comes from
    `point-metrics.json` emits one MAE per rung and no interval for it, and
    `statistics.json` bootstraps only the selected model against each baseline,
    not the rungs against each other. The paired intervals here are therefore
    recomputed from `fold-predictions.csv`, whose `ladder:*` columns hold the
    per-row predictions of every rung, using the analysis's own definition:
    a one-stage cluster bootstrap over forecast origins with the registered
    seed (20260917), 10 000 draws and a 0.95 confidence level. Panel (a) is
    cross-checked against the emitted `feature_ladder` MAE, so the recomputation
    is provably operating on the same rows.

Data source
    point-metrics.json  -> feature_ladder[rung]
    fold-predictions.csv -> ladder:W, ladder:W+H, ladder:W+H+C, ladder:W+H+C+S
    dataset.csv (only with --facet regime) -> declared_state_regime
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig-information-ladder"
LADDER = ["W", "W+H", "W+H+C", "W+H+C+S"]
RUNG_LABEL = {
    "W": "W\nworkload",
    "W+H": "W+H\n+ history",
    "W+H+C": "W+H+C\n+ target",
    "W+H+C+S": "W+H+C+S\n+ live state",
}


def paired_deltas(frame: pd.DataFrame, current: str, previous: str) -> dict[str, list[float]]:
    """Per-origin lists of |err(current)| - |err(previous)|."""
    actual = frame["execution_runtime_seconds"].to_numpy(dtype=float)
    a = np.abs(actual - pd.to_numeric(frame[current], errors="coerce").to_numpy(dtype=float))
    b = np.abs(actual - pd.to_numeric(frame[previous], errors="coerce").to_numpy(dtype=float))
    out: dict[str, list[float]] = {}
    for origin, delta in zip(frame["origin_id"].astype(str), a - b):
        out.setdefault(origin, []).append(float(delta))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = C.base_parser(__doc__)
    parser.add_argument(
        "--facet",
        default="none",
        choices=["none", "regime"],
        help="'regime' adds one delta row per declared state regime (needs dataset.csv)",
    )
    args = parser.parse_args(argv)
    analysis = C.prepare(args)

    emitted = C.require_key(analysis.point_metrics, "feature_ladder", "point-metrics.json")
    missing_rungs = [r for r in LADDER if r not in emitted]
    if missing_rungs:
        raise C.MissingData(
            f"point-metrics.json reports no feature_ladder entry for {missing_rungs}."
        )
    columns = {rung: f"ladder:{rung}" for rung in LADDER}
    absent = [c for c in columns.values() if c not in analysis.predictions.columns]
    if absent:
        raise C.MissingData(
            f"fold-predictions.csv has no {absent} column(s); the paired ladder "
            "intervals cannot be recomputed and the analysis does not emit them."
        )

    test = analysis.split("test")
    if test.empty:
        raise C.MissingData("the test split has no completed rows.")

    # Cross-check: the recomputed level must equal the emitted level.
    levels: dict[str, dict] = {}
    for rung in LADDER:
        block = emitted[rung]
        if block.get("status") != "evaluated":
            raise C.MissingData(
                f"feature_ladder['{rung}'] is '{block.get('status')}' with n="
                f"{block.get('n', 0)}; the ladder cannot be drawn from this run."
            )
        recomputed = float(
            np.mean(
                np.abs(
                    test["execution_runtime_seconds"].to_numpy(dtype=float)
                    - pd.to_numeric(test[columns[rung]], errors="coerce").to_numpy(dtype=float)
                )
            )
        )
        expected = float(block["mae_seconds"])
        if abs(recomputed - expected) / max(abs(expected), 1e-9) > 1e-6:
            raise C.MissingData(
                f"rung {rung}: recomputed MAE {recomputed:.9g} != emitted {expected:.9g}"
            )
        levels[rung] = block

    deltas = []
    for index in range(1, len(LADDER)):
        current, previous = LADDER[index], LADDER[index - 1]
        interval = C.paired_origin_bootstrap(
            paired_deltas(test, columns[current], columns[previous])
        )
        interval["label"] = f"{previous}  ->  {current}"
        deltas.append(interval)

    facet_rows: list[tuple[str, list[dict]]] = []
    if args.facet == "regime":
        joined = analysis.with_dataset(["declared_state_regime"])
        joined = joined[
            (joined["split"] == "test")
            & (joined["outcome_status"] == C.COMPLETE_STATUS)
            & joined["execution_runtime_seconds"].notna()
        ]
        for regime in sorted(joined["declared_state_regime"].astype(str).unique()):
            block = joined[joined["declared_state_regime"].astype(str) == regime]
            rows = []
            for index in range(1, len(LADDER)):
                current, previous = LADDER[index], LADDER[index - 1]
                interval = C.paired_origin_bootstrap(
                    paired_deltas(block, columns[current], columns[previous])
                )
                interval["label"] = f"{previous}  ->  {current}"
                rows.append(interval)
            facet_rows.append((regime, rows))

    total_rows = 1 + len(facet_rows)
    fig, axes = C.plt.subplots(
        1,
        2,
        figsize=C.figsize(1.0, 0.44 + 0.16 * len(facet_rows)),
        gridspec_kw={"width_ratios": [1.0, 1.35], "wspace": 0.30},
    )
    level_ax, delta_ax = axes
    C.tidy(level_ax, grid_axis="x")
    C.tidy(delta_ax, grid_axis="x")

    # -- Panel (a): the level -------------------------------------------------
    positions = np.arange(len(LADDER))[::-1]
    style = C.entity_style(0)
    colour = C.maybe_grey(style["color"])
    values = [levels[r]["mae_seconds"] for r in LADDER]
    level_ax.hlines(positions, 0, values, color=colour, linewidth=1.6, alpha=0.30, zorder=2)
    level_ax.plot(
        values,
        positions,
        linestyle="none",
        marker=style["marker"],
        markersize=4.2,
        color=colour,
        markeredgecolor=C.SURFACE,
        markeredgewidth=0.7,
        zorder=4,
    )
    for position, rung, value in zip(positions, LADDER, values):
        level_ax.annotate(
            C.fmt_seconds(value),
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=C.FONT_SMALL_PT,
            color=C.INK,
            va="center",
        )
    level_ax.set_yticks(positions)
    level_ax.set_yticklabels([RUNG_LABEL[r] for r in LADDER], fontsize=C.FONT_SMALL_PT)
    level_ax.set_xlim(0, max(values) * 1.22)
    level_ax.set_xlabel("test MAE (s)")
    C.panel_title(level_ax, "(a) Level: test MAE per rung", width=40)

    # -- Panel (b): the paired change ----------------------------------------
    groups: list[tuple[str, list[dict]]] = [("all test origins", deltas)] + facet_rows
    labels: list[str] = []
    y = 0.0
    tick_positions: list[float] = []
    supports: list[str] = []
    for group_index, (group_name, rows) in enumerate(groups):
        group_style = C.entity_style(group_index)
        group_colour = C.maybe_grey(group_style["color"])
        for row in rows:
            if row["status"] != "evaluated":
                delta_ax.annotate(
                    "unsupported (0 origins)",
                    xy=(0, -y),
                    fontsize=C.FONT_SMALL_PT,
                    color=C.maybe_grey(C.STATUS_CRITICAL),
                    va="center",
                    ha="left",
                )
            else:
                delta_ax.hlines(
                    -y,
                    row["lower"],
                    row["upper"],
                    color=group_colour,
                    linewidth=1.4,
                    zorder=3,
                )
                delta_ax.plot(
                    [row["estimate"]],
                    [-y],
                    linestyle="none",
                    marker=group_style["marker"],
                    markersize=4.0,
                    color=group_colour,
                    markeredgecolor=C.SURFACE,
                    markeredgewidth=0.7,
                    zorder=4,
                )
            tick_positions.append(-y)
            prefix = "" if len(groups) == 1 else f"{group_name}: "
            labels.append(f"{prefix}{row['label']}")
            supports.append(str(row.get("origin_count", 0)))
            y += 1.0
        y += 0.6

    delta_ax.axvline(0.0, color=C.AXIS, linewidth=0.8, zorder=1)
    delta_ax.set_yticks(tick_positions)
    delta_ax.set_yticklabels(labels, fontsize=C.FONT_SMALL_PT)
    delta_ax.set_xlabel("paired change in absolute error (s)")
    C.panel_title(
        delta_ax,
        "(b) Paired change vs. the previous rung, 95 % origin bootstrap",
        width=44,
    )
    limit = max(
        abs(v)
        for row in ([r for _, rows in groups for r in rows])
        if row["status"] == "evaluated"
        for v in (row["lower"], row["upper"])
    )
    delta_ax.set_xlim(-limit * 1.18, limit * 1.18)
    delta_ax.set_ylim(min(tick_positions) - 0.8, 0.8)

    delta_ax.annotate(
        "negative = the added rung helps",
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, -2),
        textcoords="offset points",
        fontsize=C.FONT_SMALL_PT - 0.6,
        color=C.INK_MUTED,
        ha="center",
        va="top",
    )

    support = C.support_text(
        rows=int(len(test)),
        origins=int(test["origin_id"].nunique()),
        draws=C.BOOTSTRAP_DRAWS,
    )
    C.footnote(
        fig,
        f"{support}; resampling unit = forecast origin; seed {C.BOOTSTRAP_SEED}; "
        + (f"{supports[0]} origins per interval."
           if len(set(supports)) == 1 else f"origins per interval: {', '.join(supports)}."),
    )
    fig.subplots_adjust(left=0.135, right=0.985, top=0.80, bottom=0.26)
    C.stamp(fig, analysis, note=f"facet={args.facet}")
    C.save(fig, args.output_dir, NAME, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
