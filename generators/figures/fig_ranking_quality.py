"""Figure: per-origin ranking quality -- Kendall tau and top-1 hit rate.

Placement consumes the *ordering* of the candidate targets, not the magnitude
of the forecast, so a model can be poor on MAE and still be useful, or accurate
and still order the targets wrongly. Panel (a) shows the whole per-origin
Kendall tau distribution, one mark per test forecast origin, with the median.
Panel (b) shows the top-1 hit rate -- how often the model's best-ranked target
really was the fastest measured one -- with its support as a count.

Where the per-origin values come from
    `analysis_v3.ranking_metrics` emits only the aggregates
    (`mean_kendall_tau`, `median_kendall_tau`, `top1_target_hit_rate`,
    `origin_count`); the per-origin values a distribution needs are not
    written anywhere. They are therefore recomputed from
    `fold-predictions.csv` with the analysis's own definition -- repetitions
    collapsed to the per-target median, pairwise concordance over the shared
    targets, ties broken by target name -- and the aggregates are then
    cross-checked against the emitted ones. A mismatch is a hard failure.

Data source
    fold-predictions.csv -> per-row predictions per model
    point-metrics.json   -> metrics[model].ranking (the cross-check)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NAME = "fig-ranking-quality"
SHORT = {
    "rank1-target-speed-factor": "rank-one speed factor",
    "family-target-median": "family–target median",
    "family-target-log-log-size": "family–target size rate",
    "recency-last": "recency (last)",
    "recency-last-two-mean": "recency (last two)",
    "family-median": "family median",
    "global-median": "global median",
}


LEARNED = {
    "learned:ridge": "ridge",
    "learned:random_forest": "random forest",
    "learned:xgboost": "gradient boosting (XGBoost)",
    "learned:bilinear": "bilinear interaction",
}


def label_for(name: str, analysis: C.Analysis) -> str:
    if name == analysis.selected_model:
        return f"{LEARNED.get(name, name)} (selected learned)"
    if name.startswith("learned:"):
        return LEARNED.get(name, name)
    if name.startswith("ladder:"):
        return name.replace("ladder:", "rungs ")
    return SHORT.get(name, name)


def main(argv: list[str] | None = None) -> int:
    parser = C.base_parser(__doc__)
    parser.add_argument(
        "--include",
        default="models-and-baselines",
        choices=["models-and-baselines", "all"],
        help="'all' adds the feature-ladder rungs as extra rows",
    )
    args = parser.parse_args(argv)
    analysis = C.prepare(args)

    test = analysis.split("test")
    if test.empty:
        raise C.MissingData("the test split has no completed rows.")

    emitted = C.require_key(analysis.point_metrics, "metrics", "point-metrics.json")
    candidates = [
        name
        for name in analysis.prediction_columns
        if args.include == "all" or not name.startswith("ladder:")
    ]
    if not candidates:
        raise C.MissingData("fold-predictions.csv carries no prediction columns.")

    rows: list[dict] = []
    problems: list[str] = []
    for name in candidates:
        taus, hits = C.per_origin_ranking(test, name)
        if not hits:
            problems.append(f"{name}: no test origin yielded a ranking")
            continue
        reference = emitted.get(name, {}).get("ranking", {})
        mean_tau = float(np.mean(list(taus.values()))) if taus else None
        hit_rate = float(np.mean(list(hits.values())))
        for key, value in (
            ("mean_kendall_tau", mean_tau),
            ("top1_target_hit_rate", hit_rate),
            ("origin_count", float(len(hits))),
        ):
            expected = reference.get(key)
            if expected is None or value is None:
                continue
            if abs(float(expected) - value) > 1e-9:
                problems.append(
                    f"{name}.{key}: recomputed {value:.9g} != emitted {float(expected):.9g}"
                )
        rows.append(
            {
                "name": name,
                "taus": taus,
                "hits": hits,
                "mean_tau": mean_tau,
                "median_tau": float(np.median(list(taus.values()))) if taus else None,
                "hit_rate": hit_rate,
            }
        )
    if problems:
        raise C.MissingData(
            "the recomputed per-origin ranking does not reproduce the analysis's "
            "own aggregates:\n  - " + "\n  - ".join(problems)
        )
    if not rows:
        raise C.MissingData("no model produced a per-origin ranking on the test split.")

    rows.sort(key=lambda r: (-(r["mean_tau"] if r["mean_tau"] is not None else -2), r["name"]))
    positions = {row["name"]: -index for index, row in enumerate(rows)}
    highlighted = {analysis.selected_model, analysis.strongest_baseline, "rank1-target-speed-factor"}

    fig, (tau_ax, hit_ax) = C.plt.subplots(
        1,
        2,
        figsize=C.figsize(1.0, 0.058 * len(rows) + 0.30),
        sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.07},
    )
    C.tidy(tau_ax, grid_axis="x")
    C.tidy(hit_ax, grid_axis="x")

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    for index, row in enumerate(rows):
        emphasis = row["name"] in highlighted
        style = C.entity_style(0 if row["name"] == analysis.selected_model else (1 if emphasis else 3))
        colour = C.maybe_grey(style["color"] if emphasis else C.OTHER_IN)
        values = np.asarray(list(row["taus"].values()), dtype=float)
        jitter = rng.uniform(-0.16, 0.16, size=values.size)
        tau_ax.plot(
            values,
            positions[row["name"]] + jitter,
            linestyle="none",
            marker=style["marker"] if emphasis else "o",
            markersize=2.8,
            color=colour,
            markeredgecolor=C.SURFACE,
            markeredgewidth=0.35,
            alpha=0.9,
            zorder=3,
        )
        if row["median_tau"] is not None:
            tau_ax.plot(
                [row["median_tau"]],
                [positions[row["name"]]],
                marker="|",
                markersize=11,
                markeredgewidth=1.5,
                color=C.INK if emphasis else C.INK_SECONDARY,
                linestyle="none",
                zorder=5,
            )
        hit_ax.plot(
            [row["hit_rate"]],
            [positions[row["name"]]],
            linestyle="none",
            marker=style["marker"] if emphasis else "o",
            markersize=4.6 if emphasis else 3.6,
            color=colour,
            markeredgecolor=C.SURFACE,
            markeredgewidth=0.7,
            zorder=4,
        )
        hits = int(sum(row["hits"].values()))
        total = len(row["hits"])
        hit_ax.annotate(
            f"{row['hit_rate']:.2f}  ({hits}/{total})",
            xy=(row["hit_rate"], positions[row["name"]]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=C.FONT_SMALL_PT - 1.0,
            color=C.INK if emphasis else C.INK_SECONDARY,
            va="center",
        )

    tau_ax.axvline(0.0, color=C.AXIS, linewidth=0.8, zorder=2)
    tau_ax.set_xlim(-1.08, 1.08)
    tau_ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    tau_ax.set_xlabel("per-origin Kendall $\\tau$ over the four targets")
    tau_ax.set_yticks([positions[r["name"]] for r in rows])
    tau_ax.set_yticklabels(
        ["\n".join(textwrap.wrap(label_for(r["name"], analysis), width=22)) for r in rows],
        fontsize=C.FONT_SMALL_PT,
    )
    for tick, row in zip(tau_ax.get_yticklabels(), rows):
        tick.set_color(C.INK if row["name"] in highlighted else C.INK_SECONDARY)
    tau_ax.set_ylim(min(positions.values()) - 0.8, 0.8)
    C.panel_title(tau_ax, "(a) Ordering quality per origin", width=40)

    hit_ax.set_xlim(-0.03, 1.28)
    hit_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    hit_ax.set_xlabel("top-1 target hit rate")
    C.panel_title(hit_ax, "(b) Top-1 hit rate", width=40)

    origins = int(test["origin_id"].nunique())
    C.footnote(
        fig,
        C.support_text(rows=int(len(test)), origins=origins)
        + "; one mark per held-out origin; vertical bar = median. Models other than the "
        "selected learned model, the strongest baseline and the rank-one speed factor are "
        "drawn in grey.",
    )
    fig.subplots_adjust(left=0.225, right=0.985, top=0.905, bottom=0.215)
    C.stamp(fig, analysis, note=f"include={args.include}")
    C.save(fig, args.output_dir, NAME, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
