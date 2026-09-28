"""Figure: placement regret per decision rule, against the measured oracle.

One row per registered rule. Every mark is one test forecast origin: its
regret is the measured runtime of the target the rule chose minus the measured
runtime of the best target actually observed at that origin, both truncated at the
origin's latency budget. Zero means the rule picked an optimal target. Because the replay
measured *every* target at every origin, the oracle is observed rather than
assumed, and the whole distribution is shown rather than a single bar.

Origins whose selected target failed or was censored are drawn in the reserved
`critical` status colour with a distinct marker and are counted in the
deadline-violation rate printed beside the row, so a rule cannot look good by
choosing a target that never finished.

Data source
    decision-replay.csv          -> one row per (origin, rule): regret_seconds,
                                    deadline_violation, selected/oracle target
    decision-replay-metrics.json -> rules[rule].mean/median regret,
                                    deadline_violation_rate, origin_count,
                                    reported_failed_origins,
                                    risk_rule_interval_source,
                                    risk_rule_width_varies_across_targets
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig-decision-regret"
# The reduction's rule identifiers, in the manuscript's words. The keys are checked against the
# emitted rule block below, so a renamed rule fails the render instead of printing its identifier.
RULE_LABEL = {
    "deadline-feasible-point-forecast": "point-forecast rule",
    "deadline-feasible-conformal-upper-bound": "risk-aware rule (fallback)",
    "constant-large-tier": "always the large tier",
    "constant-fixed-tier": "always the medium tier",
}


def main(argv: list[str] | None = None) -> int:
    parser = C.base_parser(__doc__)
    parser.add_argument(
        "--linthresh",
        type=float,
        default=None,
        help="width of the symlog linear zone around zero (default: a data-driven decade)",
    )
    parser.add_argument(
        "--scale",
        default="symlog",
        choices=["linear", "symlog"],
        help="symlog keeps the many zero-regret origins readable beside the tail",
    )
    args = parser.parse_args(argv)
    analysis = C.prepare(args)

    metrics = analysis.decision_metrics
    status = metrics.get("status")
    rules_block = metrics.get("rules", {})
    if status != "evaluated" or not rules_block:
        fig, ax = C.plt.subplots(figsize=C.figsize(1.0, 0.34))
        failed = metrics.get("reported_failed_origins", [])
        C.unsupported_panel(
            ax,
            "Decision replay: regret per rule",
            f"decision-replay-metrics.json reports status '{status}'",
            support=f"{len(failed)} origin(s) rejected by the all-targets-measured gate",
        )
        fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.10)
        C.stamp(fig, analysis)
        C.save(fig, args.output_dir, NAME, analysis)
        return 0

    replay = analysis.decisions.copy()
    for column in ("origin_id", "rule", "regret_seconds", "deadline_violation"):
        if column not in replay.columns:
            raise C.MissingData(f"decision-replay.csv has no '{column}' column.")
    replay["regret_seconds"] = pd.to_numeric(replay["regret_seconds"], errors="coerce")
    replay["deadline_violation"] = (
        replay["deadline_violation"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )
    if replay["regret_seconds"].isna().any():
        raise C.MissingData("decision-replay.csv carries non-numeric regret values.")

    rules = [r for r in RULE_LABEL if r in rules_block]
    unknown = [r for r in sorted(rules_block) if r not in RULE_LABEL]
    if unknown:
        raise C.MissingData(
            f"decision-replay-metrics.json reports rules this figure has no name for: {unknown}. "
            "Add them to RULE_LABEL rather than printing their identifiers."
        )
    rules += [r for r in sorted(rules_block) if r not in rules]
    # Order rows worst-first so the reader's eye lands on the reference rule last.
    rules.sort(key=lambda r: -float(rules_block[r].get("mean_regret_seconds", 0.0)))

    # Cross-check: the emitted per-rule summary must match the replay rows.
    for rule in rules:
        rows = replay[replay["rule"] == rule]
        emitted = rules_block[rule]
        if int(emitted.get("origin_count", len(rows))) != len(rows):
            raise C.MissingData(
                f"rule '{rule}': decision-replay.csv has {len(rows)} origins but "
                f"decision-replay-metrics.json reports {emitted.get('origin_count')}."
            )
        recomputed = float(rows["regret_seconds"].mean())
        expected = float(emitted["mean_regret_seconds"])
        if abs(recomputed - expected) > 1e-6 * max(abs(expected), 1.0):
            raise C.MissingData(
                f"rule '{rule}': recomputed mean regret {recomputed:.9g} != "
                f"emitted {expected:.9g}"
            )

    fig, ax = C.plt.subplots(figsize=C.figsize(1.0, 0.09 * len(rules) + 0.24))
    C.tidy(ax, grid_axis="x")

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    positions = {rule: -index for index, rule in enumerate(rules)}
    summaries: list[tuple[float, str]] = []
    for index, rule in enumerate(rules):
        rows = replay[replay["rule"] == rule]
        style = C.entity_style(index)
        colour = C.maybe_grey(style["color"])
        jitter = rng.uniform(-0.17, 0.17, size=len(rows))
        ok = ~rows["deadline_violation"].to_numpy()
        values = rows["regret_seconds"].to_numpy(dtype=float)
        ax.plot(
            values[ok],
            positions[rule] + jitter[ok],
            linestyle="none",
            marker=style["marker"],
            markersize=3.0,
            color=colour,
            markerfacecolor=colour,
            markeredgecolor=C.SURFACE,
            markeredgewidth=0.45,
            alpha=0.85,
            zorder=3,
        )
        if (~ok).any():
            ax.plot(
                values[~ok],
                positions[rule] + jitter[~ok],
                linestyle="none",
                marker="X",
                markersize=4.6,
                color=C.maybe_grey(C.STATUS_CRITICAL),
                markeredgecolor=C.SURFACE,
                markeredgewidth=0.5,
                zorder=5,
            )
        emitted = rules_block[rule]
        median = float(emitted["median_regret_seconds"])
        mean = float(emitted["mean_regret_seconds"])
        ax.plot(
            [median],
            [positions[rule]],
            marker="|",
            markersize=11,
            markeredgewidth=1.5,
            color=C.INK,
            linestyle="none",
            zorder=6,
        )
        ax.plot(
            [mean],
            [positions[rule]],
            marker="d",
            markersize=4.4,
            markerfacecolor=C.SURFACE,
            markeredgecolor=C.INK,
            markeredgewidth=1.0,
            linestyle="none",
            zorder=6,
        )
        violations = float(emitted.get("deadline_violation_rate", 0.0))
        # The zero-regret origins overplot at x = 0 however they are jittered, so their count is
        # printed: without it the row reads as a handful of marks.
        at_zero = int((values == 0.0).sum())
        summaries.append(
            (
                positions[rule],
                f"mean {C.fmt_seconds(mean)} s | median {C.fmt_seconds(median)} s | "
                f"{at_zero} of {len(rows)} origins at zero | "
                f"{violations * 100:.1f}% deadline violation",
            )
        )

    ax.axvline(0.0, color=C.AXIS, linewidth=0.9, zorder=2)
    ax.annotate(
        "0 = the rule chose a measured-optimal target",
        xy=(0.0, 0.995),
        xycoords=("data", "axes fraction"),
        xytext=(3, -1),
        textcoords="offset points",
        fontsize=C.FONT_SMALL_PT - 0.8,
        color=C.INK_SECONDARY,
        ha="left",
        va="top",
    )
    highest = float(replay["regret_seconds"].max())
    if args.scale == "symlog":
        positive = replay.loc[replay["regret_seconds"] > 0, "regret_seconds"]
        if args.linthresh is not None:
            linthresh = float(args.linthresh)
        elif positive.empty:
            linthresh = 1.0
        else:
            # A round decade at the lower quartile of the positive regrets: the
            # linear zone absorbs the near-zero mass without hiding the tail.
            quartile = float(np.quantile(positive, 0.25))
            linthresh = 10.0 ** np.floor(np.log10(max(quartile, 1e-6)))
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.45)
        ax.set_xlim(-linthresh, highest * 1.6 if highest > 0 else 1.0)
    else:
        ax.set_xlim(-0.02 * max(highest, 1.0), highest * 1.10 if highest > 0 else 1.0)
    ax.set_yticks([positions[r] for r in rules])
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(RULE_LABEL.get(r, r), width=20)) for r in rules],
        fontsize=C.FONT_SMALL_PT,
    )
    ax.set_ylim(min(positions.values()) - 0.65, 0.95)
    # Regret is truncated at the origin's latency budget (Equation 6), not at the 900 s job
    # timeout; the manuscript keeps the two apart and so must the axis.
    ax.set_xlabel("placement regret against the measured best choice (s, truncated at the latency budget)")
    C.panel_title(
        ax,
        "Regret per test forecast origin, by decision rule",
        width=62,
    )
    for position, text in summaries:
        ax.annotate(
            text,
            xy=(0.0, position - 0.40),
            xycoords=("axes fraction", "data"),
            fontsize=C.FONT_SMALL_PT - 1.2,
            color=C.INK_SECONDARY,
            va="center",
            ha="left",
        )

    handles = [
        C.Line2D([], [], color=C.INK, marker="|", linestyle="none", markersize=9,
                 markeredgewidth=1.5, label="median"),
        C.Line2D([], [], color=C.INK, marker="d", linestyle="none", markersize=4.4,
                 markerfacecolor=C.SURFACE, markeredgewidth=1.0, label="mean"),
        C.Line2D([], [], color=C.maybe_grey(C.STATUS_CRITICAL), marker="X", linestyle="none",
                 markersize=4.6, label="deadline violation"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.005),
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    failed = metrics.get("reported_failed_origins", [])
    interval_source = metrics.get("risk_rule_interval_source")
    varies = metrics.get("risk_rule_width_varies_across_targets")
    # Said in words: the gate status and the interval source are artifact keys, which a reader
    # of the figure cannot decode (round-2 review, figure labels).
    gate = metrics.get("all_targets_measured_gate", "n/a")
    gate_words = (
        "every origin stays in the denominator"
        if gate == "evaluated-origins-retained-in-denominator"
        else f"all-targets gate: {gate}"
    )
    source_words = {"cqr": "conformalized quantile regression", "split-conformal": "split conformal"}
    note = (
        f"{gate_words}; "
        + (f"{len(failed)} failed the all-targets gate" if failed else "none failed the all-targets gate")
        + ". The risk-aware rule "
        f"uses the {source_words.get(str(interval_source), str(interval_source))} band, whose width "
        + ("varies" if varies else "does not vary")
        + " across targets."
    )
    note = note[0].upper() + note[1:]
    if varies is False:
        note += (
            " A constant-radius band cannot reorder targets, so that rule is a "
            "relabelled copy of the point rule in this run."
        )
    C.footnote(fig, note)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.865, bottom=0.215)
    C.stamp(fig, analysis, note=f"scale={args.scale}")
    C.save(fig, args.output_dir, NAME, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
