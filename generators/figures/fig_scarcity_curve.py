"""Figure: the scarcity curve over causal history budgets k = 0, 1, 2, 4.

Panel (a) is the curve: test MAE of the selected learned model and of the
registered baselines, as the number of prior completions a forecaster may use
is capped at k. Panel (b) is the denominator: how many test rows actually have
each amount of prior family-target history at that budget, because a flat curve
means something different when most cells have no history at all.

Scope note
    `scarcity-generalization.json` emits one MAE per budget per model and **no
    interval**, and the analysis recomputes nothing per-row at other budgets --
    `fold-predictions.csv` holds only the full-history predictions. This figure
    therefore plots point estimates and states that it does; it does not invent
    a band the analysis cannot support.

Data source
    scarcity-generalization.json -> scarcity[k].learned_test,
                                    scarcity[k].baseline_test[name],
                                    scarcity[k].test_pair_history_support
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig-scarcity-curve"

# Named series, in assignment order. Everything else folds into the envelope.
FEATURED_BASELINES = [
    "rank1-target-speed-factor",
    "family-target-median",
    "recency-last",
]
SHORT = {
    "rank1-target-speed-factor": "rank-1 speed factor",
    "family-target-median": "family-target median",
    "recency-last": "recency (last)",
    "family-target-log-log-size": "family-target log-log size",
    "recency-last-two-mean": "recency (last two)",
    "family-median": "family median",
    "global-median": "global median",
}


def main(argv: list[str] | None = None) -> int:
    parser = C.base_parser(__doc__)
    parser.add_argument(
        "--metric",
        default="mae_seconds",
        choices=["mae_seconds", "median_absolute_error_seconds", "rmse_seconds"],
    )
    args = parser.parse_args(argv)
    analysis = C.prepare(args)

    scarcity = C.require_key(
        analysis.scarcity_generalization, "scarcity", "scarcity-generalization.json"
    )
    if not scarcity:
        raise C.MissingData("scarcity-generalization.json carries an empty 'scarcity' block.")
    budgets = sorted(scarcity, key=lambda k: int(k))
    positions = np.arange(len(budgets), dtype=float)

    learned: list[float | None] = []
    learned_support: list[tuple[int, int]] = []
    for budget in budgets:
        block = C.require_key(scarcity[budget], "learned_test", f"scarcity['{budget}']")
        if block.get("status") != "evaluated":
            raise C.MissingData(
                f"scarcity['{budget}'].learned_test is '{block.get('status')}' "
                f"with n={block.get('n', 0)}; the curve would have a silent hole at k={budget}."
            )
        learned.append(float(block[args.metric]))
        learned_support.append((int(block.get("n", 0)), int(block.get("origin_count", 0))))

    baseline_names = sorted(scarcity[budgets[0]].get("baseline_test", {}))
    if not baseline_names:
        raise C.MissingData("scarcity-generalization.json carries no baseline_test block.")
    baseline_curves: dict[str, list[float]] = {}
    for name in baseline_names:
        values = []
        for budget in budgets:
            block = scarcity[budget].get("baseline_test", {}).get(name, {})
            if block.get("status") != "evaluated":
                values = []
                break
            values.append(float(block[args.metric]))
        if values:
            baseline_curves[name] = values

    featured = [n for n in FEATURED_BASELINES if n in baseline_curves]
    folded = [n for n in baseline_curves if n not in featured]

    fig, (curve_ax, support_ax) = C.plt.subplots(
        2,
        1,
        figsize=C.figsize(1.0, 0.88),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.42},
    )
    C.tidy(curve_ax)
    C.tidy(support_ax)

    # -- the folded "other baselines" envelope, drawn first so it recedes
    if folded:
        stack = np.asarray([baseline_curves[n] for n in folded], dtype=float)
        curve_ax.fill_between(
            positions,
            stack.min(axis=0),
            stack.max(axis=0),
            color=C.maybe_grey(C.OTHER_IN),
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )

    handles = []
    selected_label = analysis.selected_model.replace("learned:", "selected learned: ")
    for index, (label, values) in enumerate(
        [(selected_label, learned)] + [(SHORT.get(n, n), baseline_curves[n]) for n in featured]
    ):
        style = C.entity_style(index)
        colour = C.maybe_grey(style["color"])
        line = curve_ax.plot(
            positions,
            values,
            color=colour,
            marker=style["marker"],
            markersize=3.6,
            markeredgecolor=C.SURFACE,
            markeredgewidth=0.6,
            linewidth=1.2,
            zorder=3 + index,
        )[0]
        if style["dashes"] != (None, None):
            line.set_dashes(list(style["dashes"]))
        handles.append(C.line_handle(label, style))
        curve_ax.annotate(
            C.fmt_seconds(values[-1]),
            xy=(positions[-1], values[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=C.FONT_SMALL_PT - 0.6,
            color=colour,
            va="center",
            annotation_clip=False,
        )
    if folded:
        handles.append(
            C.patch_handle(f"other baselines, min-max ({len(folded)})", C.OTHER_IN)
        )

    metric_label = {
        "mae_seconds": "test MAE (s)",
        "median_absolute_error_seconds": "test median |error| (s)",
        "rmse_seconds": "test RMSE (s)",
    }[args.metric]
    curve_ax.set_ylabel(metric_label)
    curve_ax.set_xlim(-0.25, len(budgets) - 1 + 0.55)
    C.panel_title(
        curve_ax,
        "(a) Accuracy against the causal history budget k (point estimates; the "
        "analysis emits no interval per budget)",
        width=74,
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 1.005),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # -- Panel (b): the support behind each budget
    support_blocks = {
        budget: scarcity[budget].get("test_pair_history_support", {}) for budget in budgets
    }
    prior_counts = sorted(
        {int(k) for block in support_blocks.values() for k in block}, key=int
    )
    if not prior_counts:
        C.unsupported_panel(
            support_ax,
            "(b) Test rows by available prior family-target history",
            "scarcity['k'].test_pair_history_support is empty",
        )
    else:
        colours = C.ordinal_colors(len(prior_counts))
        bottoms = np.zeros(len(budgets), dtype=float)
        for slot, prior in enumerate(prior_counts):
            heights = np.asarray(
                [int(support_blocks[b].get(str(prior), 0)) for b in budgets], dtype=float
            )
            support_ax.bar(
                positions,
                heights,
                bottom=bottoms,
                width=0.52,
                color=C.maybe_grey(colours[slot]),
                edgecolor=C.SURFACE,
                linewidth=0.9,
                zorder=3,
            )
            for x, height, base in zip(positions, heights, bottoms):
                if height >= 0.12 * float(bottoms.max() + heights.max() or 1):
                    support_ax.annotate(
                        f"{int(height)}",
                        xy=(x, base + height / 2),
                        fontsize=C.FONT_SMALL_PT - 1.0,
                        color=C.SURFACE if slot >= len(colours) - 2 else C.INK,
                        ha="center",
                        va="center",
                        zorder=5,
                    )
            bottoms = bottoms + heights
        support_ax.set_ylabel("test rows")
        C.panel_title(
            support_ax,
            "(b) Test rows by available prior family-target history at each budget",
            width=74,
            pad=20.0,
        )
        support_ax.legend(
            handles=[
                C.patch_handle(f"{prior} prior", colours[slot])
                for slot, prior in enumerate(prior_counts)
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=min(5, len(prior_counts)),
            handletextpad=0.5,
            columnspacing=1.0,
            fontsize=C.FONT_SMALL_PT - 0.6,
        )

    support_ax.set_xticks(positions)
    support_ax.set_xticklabels([f"k = {b}" for b in budgets])
    support_ax.set_xlabel("causal history budget (most recent completions usable per group)")

    rows, origins = learned_support[0]
    C.footnote(
        fig,
        C.support_text(rows=rows, origins=origins)
        + f"; {len(baseline_curves)} registered baselines, {len(featured)} shown by name. "
        "No interval is drawn: the analysis emits none per budget.",
    )
    fig.subplots_adjust(left=0.115, right=0.945, top=0.865, bottom=0.215)
    C.stamp(fig, analysis, note=f"metric={args.metric}")
    C.save(fig, args.output_dir, NAME, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
