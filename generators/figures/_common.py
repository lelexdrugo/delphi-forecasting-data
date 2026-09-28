"""Shared plotting contract for the DELPHI Forecasting figures.

Every figure script in this directory imports this module. It fixes four
things so that no individual script re-decides them:

1. the page geometry and typography of the MDPI `forecasting` class;
2. the validated categorical palette and the redundant (shape / dash / hatch)
   encodings that keep the figures legible in greyscale and under CVD;
3. the loader for the frozen analysis outputs, including the completeness rule
   and the consistency assertions that prove the loader reproduces the numbers
   the analysis itself reported;
4. the refusal rules: a missing artifact is an error, an `unsupported` analysis
   status is drawn as an explicit annotation, and a support count is printed
   beside every estimate.

The module never contacts a cluster, a registry or a network. It only reads
files under the analysis directory it is given.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the Agg selection)
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


# ---------------------------------------------------------------------------
# Page geometry and typography
# ---------------------------------------------------------------------------
# Measured, not assumed: compiled a scratch document with the manuscript's own
# preamble, `\documentclass[forecasting,article,submit,moreauthors]{Definitions/mdpi}`,
# and read `\the\linewidth` from the log.
TEXT_WIDTH_PT = 394.35522
PT_PER_INCH = 72.27
TEXT_WIDTH_IN = TEXT_WIDTH_PT / PT_PER_INCH  # 5.4567 in

# The class loads `mathpazo`, i.e. Palatino. Palatino Linotype is metrically
# compatible and ships with Windows; the remaining entries are fallbacks so the
# scripts still render (with a printed warning) on a machine without it.
SERIF_STACK = [
    "Palatino Linotype",
    "TeX Gyre Pagella",
    "Palatino",
    "URW Palladio L",
    "Book Antiqua",
    "DejaVu Serif",
]

# Manuscript sizes, measured from the same scratch compile:
# normalsize 10pt, small 9pt, footnotesize 8pt, scriptsize 7pt.
FONT_BASE_PT = 8.0      # \footnotesize: the MDPI figure-text size
FONT_SMALL_PT = 7.0     # \scriptsize: annotations and support counts
FONT_TITLE_PT = 9.0     # \small: panel titles


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Slots are the documented data-viz reference palette, unchanged hexes. Four
# targets must be separable in scatter/small-multiple forms, where *any* two
# marks can sit side by side, so the palette was validated with `--pairs all`
# against the print surface (white), not the screen surface:
#
#   validate_palette.py "#2a78d6,#eb6834,#1baf7a,#eda100" --pairs all --surface #ffffff
#       FAIL normal-vision floor: yellow<->orange dE 13.7 (< 15)
#   validate_palette.py "#2a78d6,#eb6834,#1baf7a,#4a3aa7" --pairs all --surface #ffffff
#       PASS lightness band, PASS chroma floor,
#       PASS CVD separation (worst all-pairs aqua<->orange dE 9.2 deutan),
#       PASS normal-vision floor (worst all-pairs violet<->blue dE 16.3),
#       WARN contrast: aqua 2.82:1 -> relief rule (direct labels are mandatory).
#
# So the four-target set is slots 1, 2, 3, 7 of the documented order. Slots 4-6
# are skipped rather than re-stepped: the hexes stay documented values and the
# skipped slots are the pair that fails all-pairs separation.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

# Reserved status colours. Never used for a series.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

# Chart chrome and ink (light / print).
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Colour follows the entity, never its rank: the mapping is keyed by the
# canonical target identifier from the acquisition spec, so a panel that shows
# a subset of targets does not repaint the survivors.
TARGET_ORDER = ["public-cloud", "on-prem", "edge-1", "edge-2"]

# The manuscript names targets by resource tier, and states that the operational cluster identifiers
# appear only in the fleet table and the testbed figure. The artifacts carry the identifiers, so every
# generated figure must translate them before printing one. Anything not in this map is passed through
# unchanged, so a value that is not a target (a family, a regime) is unaffected.
TIER_LABEL = {
    "public-cloud": "large",
    "on-prem": "medium",
    "edge-1": "small-1",
    "edge-2": "small-2",
}


# Display names for the reduction's slugs. The manuscript names families and regimes in words
# (Table 4 and Section 4.3); a figure that prints a slug forces the reader back to the pipeline.
FAMILY_LABEL = {
    "stress-ng-cpu": "CPU stressor",
    "video-transcode": "Video transcode",
    "compress-encrypt": "Compress–encrypt–hash",  # en dash; "--" is LaTeX, not matplotlib
    "onnx-inference-fp32": "Inference, FP32",
    "onnx-inference-int8": "Inference, INT8",
    "graph-kernel": "Graph kernels",
    "duckdb-tpch": "Analytical SQL",
}
REGIME_LABEL = {
    "none": "no background",
    "r1-moderate": "moderate",
    "r2-heavy": "heavy",
    "r3-cpu-pressure": "CPU pressure",
}
RULE_LABEL = {
    "deadline-feasible-point-forecast": "point forecast",
    "deadline-feasible-conformal-upper-bound": "interval upper bound",
    "constant-large-tier": "always large",
    "constant-fixed-tier": "always medium",
}


def display(name: Any) -> str:
    """The name a figure may print: tier, family, regime or rule, else the value unchanged."""
    key = str(name)
    for table in (TIER_LABEL, FAMILY_LABEL, REGIME_LABEL, RULE_LABEL):
        if key in table:
            return table[key]
    return key


def tier(name: Any) -> str:
    """The tier name a figure may print for a target; other values are returned unchanged."""
    return TIER_LABEL.get(str(name), str(name))
TARGET_COLOR = {
    "public-cloud": BLUE,
    "on-prem": ORANGE,
    "edge-1": AQUA,
    "edge-2": VIOLET,
}
# Redundant channels. Colour alone never carries target identity: every target
# also owns a marker shape and a dash pattern, which survive greyscale printing
# and photocopying.
TARGET_MARKER = {"public-cloud": "o", "on-prem": "s", "edge-1": "^", "edge-2": "D"}
TARGET_DASH = {
    "public-cloud": (None, None),
    "on-prem": (3.2, 1.4),
    "edge-1": (1.2, 1.2),
    "edge-2": (5.0, 1.4, 1.2, 1.4),
}

# A second entity space: models and decision rules. Same four validated hues,
# same redundant channels, so a reader who has learned the target encoding is
# not asked to learn a second colour vocabulary in a different order.
MODEL_COLOR_SLOTS = [BLUE, ORANGE, AQUA, VIOLET]
MODEL_MARKER_SLOTS = ["o", "s", "^", "D"]
MODEL_DASH_SLOTS = [(None, None), (3.2, 1.4), (1.2, 1.2), (5.0, 1.4, 1.2, 1.4)]

# The "Other" fold: anything past four entities collapses into this envelope
# rather than being given a generated hue.
OTHER_IN = "#bfbeb6"

# Ordinal ramp (one hue, light -> dark) for ordered categories such as "0, 1, 2,
# 4 prior executions". Steps 250/350/450/550 of the documented blue ramp: the
# lightest is the ordinal floor that still clears 2:1 against a light surface.
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def ordinal_colors(count: int) -> list[str]:
    """`count` evenly spaced steps of the ordinal ramp, light -> dark."""
    if count <= 1:
        return [ORDINAL_BLUE[2]]
    if count <= len(ORDINAL_BLUE):
        indices = np.linspace(0, len(ORDINAL_BLUE) - 1, count).round().astype(int)
        return [ORDINAL_BLUE[i] for i in indices]
    # More ordered levels than documented steps: fold the tail rather than
    # generating new hues, and let the caller label the folded bucket.
    return [ORDINAL_BLUE[i % len(ORDINAL_BLUE)] for i in range(count)]


def target_style(target: str) -> dict[str, Any]:
    """Colour, marker and dash for one target; deterministic for unknown ids."""
    if target in TARGET_COLOR:
        return {
            "color": TARGET_COLOR[target],
            "marker": TARGET_MARKER[target],
            "dashes": TARGET_DASH[target],
        }
    index = (abs(hash(target)) % 4)
    return {
        "color": MODEL_COLOR_SLOTS[index],
        "marker": MODEL_MARKER_SLOTS[index],
        "dashes": MODEL_DASH_SLOTS[index],
    }


def entity_style(index: int) -> dict[str, Any]:
    """Colour, marker and dash for the index-th entity of a non-target series."""
    slot = index % 4
    return {
        "color": MODEL_COLOR_SLOTS[slot],
        "marker": MODEL_MARKER_SLOTS[slot],
        "dashes": MODEL_DASH_SLOTS[slot],
    }


def sort_targets(targets: Iterable[str]) -> list[str]:
    """Canonical target order: spec order first, then anything unknown, sorted."""
    known = [t for t in TARGET_ORDER if t in set(targets)]
    extra = sorted(set(targets) - set(TARGET_ORDER))
    return known + extra


def to_greyscale(hex_color: str) -> str:
    """Relative-luminance grey for the greyscale proof render."""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
    step = int(round((luminance ** (1 / 2.2)) * 255))
    return "#{0:02x}{0:02x}{0:02x}".format(max(0, min(255, step)))


GREYSCALE = False


def maybe_grey(hex_color: str) -> str:
    return to_greyscale(hex_color) if GREYSCALE else hex_color


# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
def apply_style(greyscale: bool = False) -> None:
    """Install the manuscript's typography and the recessive chart chrome."""
    global GREYSCALE
    GREYSCALE = greyscale
    resolved = _resolve_serif()
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "font.size": FONT_BASE_PT,
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": FONT_TITLE_PT,
            "axes.labelsize": FONT_BASE_PT,
            "xtick.labelsize": FONT_SMALL_PT,
            "ytick.labelsize": FONT_SMALL_PT,
            "legend.fontsize": FONT_SMALL_PT,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.9,
            "legend.columnspacing": 1.1,
            "legend.labelspacing": 0.3,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
            "svg.fonttype": "none",
            "figure.constrained_layout.use": False,
        }
    )
    if resolved not in SERIF_STACK[:1]:
        print(
            f"[figures] note: Palatino Linotype was not found; using '{resolved}'. "
            "Figure text will not match the manuscript face exactly.",
        )


def _resolve_serif() -> str:
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in SERIF_STACK:
        if name in available:
            return name
    return "DejaVu Serif"


def figsize(width_fraction: float = 1.0, aspect: float = 0.62) -> tuple[float, float]:
    """Width in text-width fractions; height = width * aspect."""
    width = TEXT_WIDTH_IN * width_fraction
    return (width, width * aspect)


def panel_title(ax: plt.Axes, text: str, width: int = 74, pad: float = 4.0) -> None:
    """Left-aligned panel title, wrapped so it cannot run past the text block."""
    import textwrap

    ax.set_title(
        "\n".join(textwrap.wrap(text, width=width)),
        loc="left",
        color=INK,
        fontsize=FONT_TITLE_PT,
        pad=pad,
    )


def footnote(fig: plt.Figure, text: str, y: float = 0.034, width: int = 118) -> None:
    """Support counts and caveats, under the plot block, in muted ink."""
    import textwrap

    fig.text(
        0.012,
        y,
        "\n".join(textwrap.wrap(text, width=width)),
        fontsize=FONT_SMALL_PT - 0.6,
        color=INK_SECONDARY,
        va="bottom",
    )


def tidy(ax: plt.Axes, *, grid_axis: str = "y") -> plt.Axes:
    """Recessive chrome: no box, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return ax


# ---------------------------------------------------------------------------
# Provenance stamp
# ---------------------------------------------------------------------------
FIXTURE_BANNER = "SYNTHETIC FIXTURE - LAYOUT ONLY, NOT A RESULT"


def stamp(fig: plt.Figure, analysis: "Analysis", note: str = "") -> None:
    """Every render states where its numbers came from. Fixtures say so loudly.

    A render of real data carries its provenance in the PDF's metadata, as a generated table
    carries it in its TeX header, so the printed figure shows only what the reader needs. A
    fixture also prints it, beside the watermark, so a layout proof can never pass for a result.
    """
    parts = [p for p in (analysis.provenance(), note) if p]
    fig._provenance = "  |  ".join(parts)  # read by save()
    if analysis.is_fixture:
        fig.text(
            0.005,
            0.004,
            fig._provenance,
            fontsize=FONT_SMALL_PT - 1.2,
            color=INK_MUTED,
            ha="left",
            va="bottom",
        )
        fig.text(
            0.5,
            0.5,
            FIXTURE_BANNER,
            fontsize=13,
            color=maybe_grey(STATUS_CRITICAL),
            alpha=0.13,
            ha="center",
            va="center",
            rotation=24,
            zorder=50,
            transform=fig.transFigure,
        )


def save(fig: plt.Figure, output_dir: Path, name: str, analysis: "Analysis") -> Path:
    """Write a deterministic vector PDF plus a raster preview for review.

    A `--greyscale` run is a *proof*, not a deliverable: it writes only the
    raster preview, so the repository never carries two versions of the same
    figure that a reader could confuse.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "FIXTURE-" if analysis.is_fixture else ""
    if GREYSCALE:
        proof_dir = output_dir / "greyscale-proof"
        proof_dir.mkdir(parents=True, exist_ok=True)
        path = proof_dir / f"{prefix}{name}.png"
        fig.savefig(path, format="png", dpi=220)
        plt.close(fig)
        print(f"[figures] wrote {path}")
        return path
    pdf_path = output_dir / f"{prefix}{name}.pdf"
    # metadata CreationDate=None keeps the bytes stable across runs, so a
    # committed figure does not re-dirty the tree on every render.
    provenance = getattr(fig, "_provenance", "")
    fig.savefig(
        pdf_path,
        format="pdf",
        metadata={"CreationDate": None, "Subject": provenance, "Creator": f"scripts/figures ({name})"},
    )
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        preview_dir / f"{prefix}{name}.png", format="png", dpi=220,
        metadata={"Description": provenance},
    )
    plt.close(fig)
    print(f"[figures] wrote {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Refusal helpers
# ---------------------------------------------------------------------------
class MissingData(SystemExit):
    """Raised instead of drawing an empty or silently-imputed panel."""

    def __init__(self, message: str) -> None:
        super().__init__(f"[figures] refusing to plot: {message}")


def require_file(path: Path, why: str) -> Path:
    if not path.is_file():
        raise MissingData(f"{path} is missing. It is required for {why}.")
    return path


def require_key(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise MissingData(f"'{key}' is absent from {where}. The analysis did not emit it.")
    return mapping[key]


def unsupported_panel(ax: plt.Axes, title: str, reason: str, support: str = "") -> None:
    """Draw the explicit 'the analysis reported this as unsupported' panel."""
    import textwrap

    tidy(ax, grid_axis="none")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    lines = ["unsupported"] + textwrap.wrap(reason, width=76)
    if support:
        lines += textwrap.wrap(support, width=76)
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=FONT_SMALL_PT,
        color=maybe_grey(STATUS_CRITICAL),
        linespacing=1.6,
    )
    panel_title(ax, title, width=76)


def support_text(**counts: Any) -> str:
    """Uniform support-count annotation: `n = 96 rows, 24 origins`."""
    order = ["rows", "origins", "targets", "points", "families", "draws"]
    parts = []
    for key in order + [k for k in counts if k not in order]:
        if key in counts and counts[key] is not None:
            parts.append(f"{counts[key]:,} {key}" if isinstance(counts[key], int) else f"{counts[key]} {key}")
    return "n = " + ", ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Analysis loading
# ---------------------------------------------------------------------------
# Artifacts are loaded *lazily*, by the attribute name a figure asks for, so
# one absent file only stops the figures that actually read it. The refusal
# still names the file and the figure input it belongs to.
ARTIFACTS = {
    "model_selection": ("model-selection.json", "the selected model and baseline"),
    "uncertainty": ("uncertainty-metrics.json", "the conformal coverage figure"),
    "scarcity_generalization": ("scarcity-generalization.json", "the scarcity figure"),
    "decision_metrics": ("decision-replay-metrics.json", "the decision-replay figure"),
    "statistics": ("statistics.json", "the model-versus-baseline comparisons"),
}
# `point-metrics.json` and `fold-predictions.csv` are the two files every
# figure depends on, directly or through `check_consistency`, so they are
# loaded eagerly and their absence is an immediate refusal.
POINT_METRICS = "point-metrics.json"
MANIFEST = "analysis-manifest.json"
FOLD_PREDICTIONS = "fold-predictions.csv"
DECISION_REPLAY = "decision-replay.csv"
FIXTURE_MARKER = "FIXTURE.json"

# Columns written by `_write_predictions`; everything else in that file is a
# model or ladder prediction column.
PREDICTION_INDEX_COLUMNS = [
    "origin_id",
    "candidate_cluster",
    "repetition",
    "split",
    "outcome_status",
    "execution_runtime_seconds",
]

# Grouping metadata the prediction file MAY carry. The v4 reduction emits these beside the
# predictions; earlier ones left them to the dataset. They are deliberately kept out of
# PREDICTION_INDEX_COLUMNS, which doubles as the required-columns check, so that an analysis
# directory without them still loads. Every column here is metadata: a figure that treated one as a
# model prediction would try to average workload names.
OPTIONAL_METADATA_COLUMNS = [
    "point_id",
    "family",
    "declared_state_regime",
    "primary_size",
]

# `analysis_v3._complete` is `_truth(row["completed"]) and runtime is not None`,
# and `dataset_v3` sets `completed = (outcome_status == "completed")`. The
# prediction file carries `outcome_status` but not `completed`, so the rule is
# reconstructed here -- and then proved correct by `check_consistency`, which
# re-derives the analysis's own reported MAE and origin counts through it.
COMPLETE_STATUS = "completed"


@dataclass
class Analysis:
    directory: Path
    point_metrics: dict
    manifest: dict
    predictions: pd.DataFrame
    dataset: pd.DataFrame | None
    is_fixture: bool
    fixture_note: str = ""
    _cache: dict = field(default_factory=dict)

    # -- lazily loaded artifacts -------------------------------------------
    def artifact(self, name: str) -> dict:
        """Load one analysis JSON on first use; refuse loudly if it is absent."""
        if name in self._cache:
            return self._cache[name]
        filename, why = ARTIFACTS[name]
        path = require_file(self.directory / filename, why)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._cache[name] = payload
        return payload

    @property
    def model_selection(self) -> dict:
        return self.artifact("model_selection")

    @property
    def uncertainty(self) -> dict:
        return self.artifact("uncertainty")

    @property
    def scarcity_generalization(self) -> dict:
        return self.artifact("scarcity_generalization")

    @property
    def decision_metrics(self) -> dict:
        return self.artifact("decision_metrics")

    @property
    def statistics(self) -> dict:
        return self.artifact("statistics")

    @property
    def decisions(self) -> pd.DataFrame:
        if "decisions" not in self._cache:
            path = require_file(self.directory / DECISION_REPLAY, "the decision-replay figure")
            self._cache["decisions"] = pd.read_csv(path, encoding="utf-8-sig")
        return self._cache["decisions"]

    # -- derived views ------------------------------------------------------
    @property
    def prediction_columns(self) -> list[str]:
        return [
            c
            for c in self.predictions.columns
            if c not in PREDICTION_INDEX_COLUMNS
            and c not in OPTIONAL_METADATA_COLUMNS
            and is_numeric_dtype(self.predictions[c])
            # A column with no values at all is not a forecast. The reduction emits provenance
            # columns that are numeric and entirely absent; treating one as a model puts a
            # method in the figures that appears in no table.
            and bool(self.predictions[c].notna().any())
        ]

    @property
    def selected_model(self) -> str:
        return str(require_key(self.point_metrics, "selected_learned_model", "point-metrics.json"))

    @property
    def strongest_baseline(self) -> str:
        return str(
            require_key(self.point_metrics, "selected_strongest_baseline", "point-metrics.json")
        )

    def split(self, name: str, complete_only: bool = True) -> pd.DataFrame:
        frame = self.predictions[self.predictions["split"] == name]
        if complete_only:
            frame = frame[
                (frame["outcome_status"] == COMPLETE_STATUS)
                & frame["execution_runtime_seconds"].notna()
            ]
        return frame.copy()

    def with_dataset(self, columns: Sequence[str]) -> pd.DataFrame:
        """Join dataset columns the prediction file does not carry (point_id,
        family, declared_state_regime, ...) on the measurement key."""
        if self.dataset is None:
            raise MissingData(
                "dataset.csv was not found beside the analysis outputs, and "
                f"fold-predictions.csv does not carry {list(columns)}. "
                "Pass --dataset-dir pointing at the built dataset directory."
            )
        key = ["origin_id", "candidate_cluster", "repetition"]
        missing = [c for c in columns if c not in self.dataset.columns]
        if missing:
            raise MissingData(f"dataset.csv does not carry {missing}.")
        left = self.predictions.copy()
        right = self.dataset[key + list(columns)].copy()
        for frame in (left, right):
            frame["repetition"] = pd.to_numeric(frame["repetition"], errors="coerce")
        if right.duplicated(subset=key).any():
            raise MissingData(
                "dataset.csv is not unique on (origin_id, candidate_cluster, repetition); "
                "the join used by this figure would fan out."
            )
        # The v4 prediction file already carries some of these columns, so a plain merge would
        # suffix both copies and leave the requested name absent. Keep both sides, then require that
        # they agree: two independent files describing the same measurement must say the same thing,
        # and a disagreement means the analysis and dataset directories are from different runs.
        overlap = [c for c in columns if c in left.columns]
        merged = left.merge(right, on=key, how="left", validate="one_to_one", suffixes=("", "_ds"))
        for column in overlap:
            mine, theirs = merged[column].astype(str), merged[f"{column}_ds"].astype(str)
            if not mine.equals(theirs):
                differing = int((mine != theirs).sum())
                raise MissingData(
                    f"fold-predictions.csv and dataset.csv disagree on {column} for {differing} "
                    "rows; the analysis directory and dataset directory do not describe the same run."
                )
            merged = merged.drop(columns=[f"{column}_ds"])
        if merged[list(columns)].isna().any().any():
            raise MissingData(
                "the dataset join left unmatched prediction rows; "
                "the analysis directory and dataset directory do not describe the same run."
            )
        return merged

    def provenance(self) -> str:
        if self.is_fixture:
            return f"source: SYNTHETIC FIXTURE ({self.fixture_note or self.directory.name})"
        revision = str(self.manifest.get("analysis_revision", "unknown"))[:12]
        dataset = str(self.manifest.get("input_dataset_sha256", "unknown"))[:12]
        return f"source: {self.directory.name} | analysis {revision} | dataset {dataset}"


def load_analysis(analysis_dir: Path, dataset_dir: Path | None = None) -> Analysis:
    analysis_dir = Path(analysis_dir)
    if not analysis_dir.is_dir():
        raise MissingData(f"{analysis_dir} is not a directory.")
    point_metrics = json.loads(
        require_file(
            analysis_dir / POINT_METRICS, "every figure, and for the consistency assertion"
        ).read_text(encoding="utf-8")
    )
    manifest_path = analysis_dir / MANIFEST
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    predictions = pd.read_csv(
        require_file(analysis_dir / FOLD_PREDICTIONS, "every prediction-derived figure"),
        encoding="utf-8-sig",
    )
    missing = [c for c in PREDICTION_INDEX_COLUMNS if c not in predictions.columns]
    if missing:
        raise MissingData(f"{FOLD_PREDICTIONS} is missing required columns {missing}.")
    predictions["repetition"] = pd.to_numeric(predictions["repetition"], errors="coerce")
    predictions["execution_runtime_seconds"] = pd.to_numeric(
        predictions["execution_runtime_seconds"], errors="coerce"
    )

    dataset = None
    for candidate in [
        Path(dataset_dir) / "dataset.csv" if dataset_dir else None,
        analysis_dir / "dataset.csv",
        analysis_dir.parent / "dataset" / "dataset.csv",
    ]:
        if candidate is not None and candidate.is_file():
            dataset = pd.read_csv(candidate, encoding="utf-8-sig", low_memory=False)
            break
    if dataset_dir is not None and dataset is None:
        raise MissingData(f"no dataset.csv under {dataset_dir}.")

    marker = analysis_dir / FIXTURE_MARKER
    is_fixture = marker.is_file()
    note = ""
    if is_fixture:
        note = str(json.loads(marker.read_text(encoding="utf-8")).get("note", ""))

    analysis = Analysis(
        directory=analysis_dir,
        point_metrics=point_metrics,
        manifest=manifest,
        predictions=predictions,
        dataset=dataset,
        is_fixture=is_fixture,
        fixture_note=note,
    )
    check_consistency(analysis)
    return analysis


def check_consistency(analysis: Analysis, tolerance: float = 1e-6) -> None:
    """Prove the loader reproduces the analysis's own reported numbers.

    If the completeness rule, the split filter or the column mapping were
    wrong, the recomputed test MAE would not match `point-metrics.json`. This
    runs on every load; a mismatch is a hard failure, never a warning.
    """
    metrics = analysis.point_metrics.get("metrics", {})
    if not metrics:
        raise MissingData("point-metrics.json carries no 'metrics' block.")
    test = analysis.split("test")
    problems: list[str] = []
    checked = 0
    for name, block in metrics.items():
        reported = block.get("test", {})
        if reported.get("status") != "evaluated" or name not in test.columns:
            continue
        actual = test["execution_runtime_seconds"].to_numpy(dtype=float)
        predicted = pd.to_numeric(test[name], errors="coerce").to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - predicted)))
        expected = float(reported["mae_seconds"])
        scale = max(abs(expected), 1e-9)
        if abs(mae - expected) / scale > tolerance:
            problems.append(f"{name}: recomputed test MAE {mae:.9g} != reported {expected:.9g}")
        if int(reported.get("n", len(test))) != len(test):
            problems.append(f"{name}: recomputed test n {len(test)} != reported {reported.get('n')}")
        origins = test["origin_id"].nunique()
        if int(reported.get("origin_count", origins)) != origins:
            problems.append(
                f"{name}: recomputed test origins {origins} != reported {reported.get('origin_count')}"
            )
        checked += 1
    if checked == 0:
        problems.append(
            "no evaluated model in point-metrics.json has a matching column in fold-predictions.csv"
        )
    if problems:
        raise MissingData(
            "the loader does not reproduce the analysis's own numbers, so the "
            "figures would not describe this run:\n  - " + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Statistics reproduced from the registered contract
# ---------------------------------------------------------------------------
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260917  # specs/controlled-analysis-v3.yaml: random_seed


def paired_origin_bootstrap(
    values_by_origin: Mapping[str, Sequence[float]],
    draws: int = BOOTSTRAP_DRAWS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """One-stage cluster bootstrap over forecast origins.

    Identical in definition to `analysis_v3.hierarchical_bootstrap` with the
    registered `hierarchy: [origin]`: an origin is drawn with replacement and
    contributes the mean of its paired differences. Used only for quantities
    the analysis does not itself emit an interval for (the feature ladder);
    the seed, draw count and confidence level are the registered ones.
    """
    origins = sorted(values_by_origin)
    if not origins:
        return {"status": "unsupported", "origin_count": 0}
    per_origin = {o: float(np.mean(np.asarray(values_by_origin[o], dtype=float))) for o in origins}
    rng = np.random.default_rng(seed)
    means = np.asarray([per_origin[o] for o in origins], dtype=float)
    index = rng.integers(0, len(origins), size=(draws, len(origins)))
    draw_values = means[index].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "status": "evaluated",
        "origin_count": len(origins),
        "draws": draws,
        "resampling_unit": "origin",
        "estimate": float(np.mean(means)),
        "lower": float(np.quantile(draw_values, alpha / 2)),
        "upper": float(np.quantile(draw_values, 1 - alpha / 2)),
    }


def per_origin_ranking(
    frame: pd.DataFrame, column: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-origin Kendall tau and top-1 hit, by the analysis's own definition.

    `analysis_v3.ranking_metrics` collapses repetitions to the per-target median
    and compares orderings pairwise, but reports only the mean/median tau and
    the pooled hit rate. The per-origin values a distribution figure needs are
    therefore recomputed here with exactly that definition, and the aggregates
    are cross-checked against the emitted ones by the caller.
    """
    taus: dict[str, float] = {}
    hits: dict[str, float] = {}
    for origin, block in frame.groupby("origin_id", sort=True):
        actual = block.groupby("candidate_cluster")["execution_runtime_seconds"].median()
        predicted = block.groupby("candidate_cluster")[column].median()
        shared = sorted(set(actual.index) & set(predicted.index))
        if len(shared) >= 2:
            concordant = discordant = 0
            for i, left in enumerate(shared):
                for right in shared[i + 1 :]:
                    product = (actual[left] - actual[right]) * (predicted[left] - predicted[right])
                    if product > 0:
                        concordant += 1
                    elif product < 0:
                        discordant += 1
            pairs = concordant + discordant
            taus[str(origin)] = (concordant - discordant) / pairs if pairs else 0.0
        if len(shared) >= 1:
            best_actual = min(shared, key=lambda t: (actual[t], t))
            best_predicted = min(shared, key=lambda t: (predicted[t], t))
            hits[str(origin)] = float(best_actual == best_predicted)
    return taus, hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="directory written by experiments/forecasting/analysis_v3.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/generated"),
        help="where the vector PDF is written (default: figures/generated)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="directory holding dataset.csv, for the columns fold-predictions.csv omits",
    )
    parser.add_argument(
        "--greyscale",
        action="store_true",
        help="proof render: map every series colour to its luminance grey",
    )
    return parser


def prepare(args: argparse.Namespace) -> Analysis:
    apply_style(greyscale=bool(getattr(args, "greyscale", False)))
    return load_analysis(args.analysis_dir, args.dataset_dir)


def legend(ax: plt.Axes, handles: Sequence[Any], **kwargs: Any) -> None:
    """A legend is always present for two or more series."""
    defaults = dict(loc="upper left", ncol=min(4, len(handles)), handletextpad=0.5)
    defaults.update(kwargs)
    ax.legend(handles=handles, **defaults)


def line_handle(label: str, style: Mapping[str, Any], filled: bool = True) -> Line2D:
    dashes = style.get("dashes", (None, None))
    handle = Line2D(
        [],
        [],
        color=maybe_grey(style["color"]),
        marker=style.get("marker", "o"),
        markersize=3.6,
        linewidth=1.1,
        label=label,
        markerfacecolor=maybe_grey(style["color"]) if filled else SURFACE,
        markeredgecolor=maybe_grey(style["color"]),
        markeredgewidth=0.8,
    )
    if dashes != (None, None):
        handle.set_dashes(list(dashes))
    return handle


def patch_handle(label: str, color: str, hatch: str | None = None) -> Patch:
    return Patch(
        facecolor=maybe_grey(color),
        edgecolor=SURFACE,
        linewidth=0.8,
        hatch=hatch,
        label=label,
    )


def fmt_seconds(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"
