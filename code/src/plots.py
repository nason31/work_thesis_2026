"""Figures for the analysis of a scoring run.

Every time-series panel follows two rules from CLAUDE.md: House and Senate are
drawn side by side, never pooled (O8), and the 2016-09-10 source break is
marked as a dashed line at the 114th Congress -- the Congress that holds both
sources. Presidential terms are shaded behind the data.

Colours follow the dataviz method (validated for colour-blind readers, all
pairs): Democrats blue and Republicans red; the two models orange and violet,
so a model line is never mistaken for a party. Text stays in ink colours;
series colours mark identity only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from src.config import PLOTS_DIR, PRESIDENTS_BY_CONGRESS, SOURCE_BREAK_CONGRESS

PARTY_COLORS = {"D": "#2a78d6", "R": "#e34948"}
PARTY_NAMES = {"D": "Democrats", "R": "Republicans"}
#: Assigned to a run's models in order; never cycled.
MODEL_PALETTE = ("#eb6834", "#4a3aa7")
#: Speech types (SPEECH_TYPES order): categorical slots 2-7, skipping the party
#: blue and red. Validated adjacent-pair safe for colour-blind readers in this
#: order; three sit under 3:1 contrast, so every line is also labelled directly.
SPEECH_TYPE_PALETTE = ("#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
#: Topic blocks (TOPIC_BLOCKS order): the first five of the same slots, so the
#: adjacent pairs stay the validated ones.
TOPIC_BLOCK_PALETTE = SPEECH_TYPE_PALETTE[:5]
CHAMBERS = (("H", "House"), ("S", "Senate"))

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
TERM_SHADE = "#f0efec"

LINE_WIDTH = 1.75
MARKER_SIZE = 5
BAND_ALPHA = 0.15


def congress_year(congress: int) -> int:
    """The year a Congress begins (the 107th began in 2001)."""
    return 1789 + 2 * (congress - 1)


# --- shared panel furniture -------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)


def _congress_axis(ax, congresses: list[int]) -> None:
    ax.set_xticks(congresses)
    ax.set_xticklabels(
        [f"{c}\n{congress_year(c)}" for c in congresses], color=INK_SECONDARY
    )
    ax.set_xlim(min(congresses) - 0.6, max(congresses) + 0.6)
    ax.set_xlabel("Congress (start year)", color=INK_SECONDARY, fontsize=8)


def shade_presidencies(ax, congresses: list[int], names: bool = True) -> None:
    """Alternate light bands per presidential term, named along the top.

    ``names=False`` leaves out the names, for small panels.
    """
    shown = [c for c in sorted(PRESIDENTS_BY_CONGRESS) if c in congresses]
    terms = [
        (name, [c for c, _ in group])
        for name, group in groupby(
            ((c, PRESIDENTS_BY_CONGRESS[c][0]) for c in shown), key=lambda t: t[1]
        )
    ]
    for index, (name, members) in enumerate(terms):
        low, high = min(members) - 0.5, max(members) + 0.5
        if index % 2 == 0:
            ax.axvspan(low, high, color=TERM_SHADE, zorder=0, linewidth=0)
        if not names:
            continue
        ax.text(
            (low + high) / 2,
            1.0,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK_SECONDARY,
        )


def mark_source_break(ax, label: bool = True, x: float = SOURCE_BREAK_CONGRESS) -> None:
    """Dashed line at the 114th Congress, where Stanford hands over to govinfo.

    ``label=False`` leaves out the text, for small panels. ``x`` places the
    line on an axis that is not in Congresses (e.g. presidential terms).
    """
    ax.plot(
        [x, x],
        [0, 1],
        transform=ax.get_xaxis_transform(),
        color=INK_MUTED,
        linestyle=(0, (4, 3)),
        linewidth=1,
        zorder=1,
    )
    if not label:
        return
    ax.text(
        x + 0.12,
        0.02,
        "source change\n2016-09-10",
        transform=ax.get_xaxis_transform(),
        fontsize=7,
        color=INK_MUTED,
        va="bottom",
    )


def _panels(title: str, subtitle: str | None, ylabel: str) -> tuple[Figure, list]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.02)
    if subtitle:
        fig.text(0.01, 0.955, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY)
    for ax, (_, name) in zip(axes, CHAMBERS):
        _style(ax)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=14)
    axes[0].set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
    return fig, list(axes)


def _series(ax, data: pd.DataFrame, x: str, y: str, color: str, label: str) -> None:
    """A line with markers and its 95% band."""
    data = data.sort_values(x)
    ax.fill_between(
        data[x], data["ci_low"], data["ci_high"], color=color, alpha=BAND_ALPHA,
        linewidth=0, zorder=2,
    )  # fmt: skip
    ax.plot(
        data[x], data[y], color=color, linewidth=LINE_WIDTH, marker="o",
        markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=0.8,
        label=label, zorder=3,
    )  # fmt: skip


def _line_end(data: pd.DataFrame, x: str, y: str, text: str) -> tuple:
    """Where a series ends, and the label it gets there."""
    last = data.sort_values(x).iloc[-1]
    return last[x], last[y], text


#: Smallest vertical distance between two direct labels, as a share of the axis.
LABEL_GAP = 0.045


def _direct_labels(ax, ends: list[tuple]) -> None:
    """Label each series at its end, nudging labels apart where lines meet.

    Call after the y-limits are final: the gap is a share of the axis height.
    """
    low, high = ax.get_ylim()
    gap = LABEL_GAP * (high - low)
    placed: list[tuple] = []
    for x, y, text in sorted(ends, key=lambda end: end[1]):
        if placed and y - placed[-1][1] < gap:
            y = placed[-1][1] + gap
        placed.append((x, y, text))
    for x, y, text in placed:
        ax.annotate(
            text, (x, y), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=7.5, color=INK_SECONDARY, annotation_clip=False,
        )  # fmt: skip


# --- the figures -------------------------------------------------------------


def plot_party_positions(
    cells: pd.DataFrame,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Each party's mean per Congress with 95% intervals; House | Senate."""
    fig, axes = _panels(title, subtitle, ylabel)
    congresses = sorted(cells["congress_number"].unique())
    for ax, (chamber, _) in zip(axes, CHAMBERS):
        shade_presidencies(ax, congresses)
        ends = []
        for party in ("D", "R"):
            data = cells[(cells["chamber"] == chamber) & (cells["party"] == party)]
            _series(ax, data, "congress_number", "mean", PARTY_COLORS[party],
                    PARTY_NAMES[party])  # fmt: skip
            ends.append(_line_end(data, "congress_number", "mean", PARTY_NAMES[party]))
        mark_source_break(ax)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        _congress_axis(ax, congresses)
        if ylim:
            ax.set_ylim(*ylim)
        _direct_labels(ax, ends)
    axes[1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    return fig


def plot_party_positions_by_group(
    cells: pd.DataFrame,
    group: str,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    order: Sequence[str] = (),
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """Each party's mean per Congress, one row per value of ``group``; House | Senate.

    ``cells`` is `analysis.cell_means` by Congress, chamber, ``group`` and
    party. All panels share one y-axis, so the rows compare directly; each marks
    the source break and shades the presidencies (named on the top row only).
    """
    groups = list(order) or sorted(cells[group].unique())
    congresses = sorted(cells["congress_number"].unique())
    height = 3.2 * len(groups) + 1.0
    fig, axes = plt.subplots(
        len(groups), 2, figsize=(12, height), sharex=True, sharey=True,
        facecolor=SURFACE, squeeze=False,
    )  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", va="top", fontsize=12, color=INK, y=0.998)
    if subtitle:
        fig.text(
            0.01, 1 - 0.42 / height, subtitle, ha="left", va="top", fontsize=8.5,
            color=INK_SECONDARY,
        )  # fmt: skip
    for row, value in enumerate(groups):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            shade_presidencies(ax, congresses, names=row == 0)
            for party in ("D", "R"):
                data = cells[
                    (cells["chamber"] == chamber)
                    & (cells["party"] == party)
                    & (cells[group] == value)
                ]
                if not data.empty:
                    _series(ax, data, "congress_number", "mean",
                            PARTY_COLORS[party], PARTY_NAMES[party])  # fmt: skip
            mark_source_break(ax, label=row == len(groups) - 1)
            ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
            _congress_axis(ax, congresses)
            if row < len(groups) - 1:
                ax.set_xlabel("")
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"{name} · {value}", loc="left", fontsize=10, color=INK,
                         pad=14 if row == 0 else 6)  # fmt: skip
        axes[row][0].set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / height))
    return fig


def plot_gap(gaps: pd.DataFrame, title: str, subtitle: str | None = None) -> Figure:
    """R - D per Congress with 95% intervals; House | Senate, one series each."""
    fig, axes = _panels(title, subtitle, "Party gap (R − D)")
    congresses = sorted(gaps["congress_number"].unique())
    for ax, (chamber, _) in zip(axes, CHAMBERS):
        shade_presidencies(ax, congresses)
        data = gaps[gaps["chamber"] == chamber]
        _series(ax, data, "congress_number", "gap", INK_SECONDARY, "R − D")
        mark_source_break(ax)
        ax.set_ylim(0, max(1.0, gaps["ci_high"].max() + 0.05))
        _congress_axis(ax, congresses)
    return fig


def plot_models(
    cells_by_model: Mapping[str, pd.DataFrame],
    title: str,
    subtitle: str | None = None,
) -> Figure:
    """Each model's party means per Congress; House | Senate, colour = model."""
    fig, axes = _panels(title, subtitle, "Ideology (−1 liberal … +1 conservative)")
    first = next(iter(cells_by_model.values()))
    congresses = sorted(first["congress_number"].unique())
    for ax, (chamber, _) in zip(axes, CHAMBERS):
        shade_presidencies(ax, congresses)
        ends = []
        for color, (model, cells) in zip(MODEL_PALETTE, cells_by_model.items()):
            for party in ("D", "R"):
                data = cells[(cells["chamber"] == chamber) & (cells["party"] == party)]
                _series(ax, data, "congress_number", "mean", color,
                        model if party == "D" else None)  # fmt: skip
                ends.append(_line_end(data, "congress_number", "mean",
                                      f"{model.split('-')[0]} · {party}"))  # fmt: skip
        mark_source_break(ax)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        _congress_axis(ax, congresses)
        _direct_labels(ax, ends)
    axes[1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    return fig


def plot_member_validation(
    members: pd.DataFrame,
    value: str = "ideology",
    benchmark: str = "nominate_dim1",
    title: str = "",
    subtitle: str | None = None,
) -> Figure:
    """Each member's average score against their DW-NOMINATE score."""
    fig, ax = plt.subplots(figsize=(6.5, 5.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for party in ("D", "R"):
        data = members[members["party"] == party]
        r = data[value].corr(data[benchmark])
        ax.scatter(
            data[benchmark], data[value], s=22, color=PARTY_COLORS[party],
            edgecolor=SURFACE, linewidth=0.6, alpha=0.85,
            label=f"{PARTY_NAMES[party]}: r = {r:+.2f} within party (n = {len(data)})",
        )  # fmt: skip
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.set_xlabel(f"DW-NOMINATE ({benchmark})", color=INK_SECONDARY, fontsize=8.5)
    ax.set_ylabel("Member's mean LLM ideology score", color=INK_SECONDARY, fontsize=8.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK, y=1.0)
    if subtitle:
        fig.text(0.02, 0.935, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


#: The two rows of `plot_validation`: which columns are drawn, how the r on
#: the panel is labelled, and which `design_row` statistic it shows.
VALIDATION_ROWS = (
    (
        "score", "bench", "Across parties", "overall_pearson",
        "Member's mean ideology score", "Nokken-Poole (DW-NOMINATE dim. 1, per Congress)",
    ),
    (
        "score_within", "bench_within", "Within party × chamber × Congress", "within_cell_pearson",
        "Score relative to party, chamber, Congress", "Nokken-Poole relative to party, chamber, Congress",
    ),
)  # fmt: skip


def plot_validation(
    units_by_set: Mapping[str, pd.DataFrame],
    results_by_set: Mapping[str, Mapping[str, float]],
    title: str,
    subtitle: str | None = None,
) -> Figure:
    """Member x Congress units against DW-NOMINATE, one column per speech set.

    Top row: each unit's mean score against its benchmark, both parties -- the
    party gap dominates it. Bottom row: both taken relative to the unit's
    party x chamber x Congress cell, the validation the thesis reports (A5).
    ``units_by_set`` holds `validation_design.within_cell` tables (``party``,
    ``score``, ``bench``, ``score_within``, ``bench_within``); ``results_by_set``
    the matching `validation_design.design_row` results. The r written on each
    panel is that result's, with its member-bootstrap interval -- not one
    recomputed here, so the figure and the reported number cannot drift apart.
    """
    n = len(units_by_set)
    fig, axes = plt.subplots(
        2, n, figsize=(5.8 * n, 10), sharex="row", sharey="row", facecolor=SURFACE,
        squeeze=False,
    )  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", va="top", fontsize=12, color=INK, y=0.998)
    if subtitle:
        fig.text(
            0.01, 0.972, subtitle, ha="left", va="top", fontsize=8.5,
            color=INK_SECONDARY,
        )  # fmt: skip
    for col, (name, units) in enumerate(units_by_set.items()):
        result = results_by_set[name]
        for row, (y, x, label, stat, ylabel, xlabel) in enumerate(VALIDATION_ROWS):
            ax = axes[row][col]
            _style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            for party in ("D", "R"):
                data = units[units["party"] == party]
                ax.scatter(
                    data[x], data[y], s=9, color=PARTY_COLORS[party], alpha=0.45,
                    linewidth=0, label=f"{PARTY_NAMES[party]} ({len(data):,} units)",
                    zorder=2,
                )  # fmt: skip
            ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
            ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
            ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
            ax.text(
                0.02, 0.98,
                f"{label}\nr = {result[stat]:+.2f} "
                f"[{result[f'{stat}_ci_low']:+.2f}, {result[f'{stat}_ci_high']:+.2f}]",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK,
            )  # fmt: skip
            ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
            if row == 0:
                ax.legend(frameon=False, fontsize=8, loc="lower right", markerscale=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    return fig


def plot_label_shares(
    shares: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    order: tuple[str, ...] = (),
    column: str = "speech_type",
    palette: Sequence[str] = SPEECH_TYPE_PALETTE,
    ylabel: str = "Share of labelled speeches",
) -> Figure:
    """Each label's share per Congress; House | Senate, colour = label.

    ``shares`` is `label_eval.label_shares` by Congress and chamber, for the
    label ``column`` (speech type by default, or topic block). Labels are
    coloured in ``order`` (the codebook's), so a label keeps its colour
    whichever labels a run happens to contain.
    """
    fig, axes = _panels(title, subtitle, ylabel)
    congresses = sorted(shares["congress_number"].unique())
    types = list(order) or sorted(shares[column].unique())
    colors = dict(zip(types, palette))
    for ax, (chamber, _) in zip(axes, CHAMBERS):
        shade_presidencies(ax, congresses)
        ends = []
        for speech_type in types:
            data = shares[
                (shares["chamber"] == chamber) & (shares[column] == speech_type)
            ]
            if data.empty:
                continue
            # A Congress without this type has a share of 0, not a gap to bridge.
            data = (
                data.set_index("congress_number")["share"]
                .reindex(congresses, fill_value=0.0)
                .rename_axis("congress_number")
                .reset_index()
            )
            ax.plot(
                data["congress_number"], data["share"], color=colors[speech_type],
                linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=speech_type,
                zorder=3,
            )  # fmt: skip
            ends.append(_line_end(data, "congress_number", "share", speech_type))
        mark_source_break(ax)
        _congress_axis(ax, congresses)
        ax.set_ylim(0, 1)
        _direct_labels(ax, ends)
    # From both panels: a type may appear in one chamber only.
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(text, handle)
    axes[1].legend(
        list(handles.values()), list(handles), frameon=False, fontsize=8,
        loc="upper left", bbox_to_anchor=(1.18, 1.0),
    )  # fmt: skip
    return fig


def plot_gap_by_group(
    gaps: pd.DataFrame,
    group: str,
    title: str,
    subtitle: str | None = None,
    order: Sequence[str] = (),
    palette: Sequence[str] = TOPIC_BLOCK_PALETTE,
) -> Figure:
    """The R - D gap per Congress, one line per value of ``group``; House | Senate.

    ``gaps`` is `analysis.party_gap` by Congress, chamber and ``group``. Colours
    follow ``order``, so a group keeps its colour across figures. Bands are 95%
    intervals, kept faint so that crossing lines stay readable.
    """
    fig, axes = _panels(title, subtitle, "Party gap (R − D)")
    congresses = sorted(gaps["congress_number"].unique())
    groups = list(order) or sorted(gaps[group].unique())
    colors = dict(zip(groups, palette))
    for ax, (chamber, _) in zip(axes, CHAMBERS):
        shade_presidencies(ax, congresses)
        ends = []
        for value in groups:
            data = gaps[(gaps["chamber"] == chamber) & (gaps[group] == value)]
            if data.empty:
                continue
            data = data.sort_values("congress_number")
            ax.fill_between(
                data["congress_number"], data["ci_low"], data["ci_high"],
                color=colors[value], alpha=0.08, linewidth=0, zorder=2,
            )  # fmt: skip
            ax.plot(
                data["congress_number"], data["gap"], color=colors[value],
                linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE - 1,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=value, zorder=3,
            )  # fmt: skip
            ends.append(_line_end(data, "congress_number", "gap", value))
        mark_source_break(ax)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        _congress_axis(ax, congresses)
        _direct_labels(ax, ends)
    handles: dict[str, object] = {}
    for ax in axes:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(text, handle)
    axes[1].legend(
        list(handles.values()), list(handles), frameon=False, fontsize=8,
        loc="upper left", bbox_to_anchor=(1.18, 1.0),
    )  # fmt: skip
    return fig


def plot_gap_trends(
    trends: pd.DataFrame,
    group: str,
    title: str,
    subtitle: str | None = None,
    order: Sequence[str] = (),
    colors: Mapping[str, str] | None = None,
) -> Figure:
    """How much the party gap changed over the window, per group; House | Senate.

    ``trends`` is `analysis.gap_trend` by chamber and ``group``: the fitted
    change from the first Congress to the last, with its 95% interval (the
    slope's interval times the Congresses spanned). Groups run top to bottom
    in ``order``; ``colors`` maps a group to its colour (e.g. by topic block).
    """
    groups = list(order) or sorted(trends[group].unique())
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 0.42 * len(groups) + 1.6), sharey=True, sharex=True,
        facecolor=SURFACE,
    )  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.02)
    if subtitle:
        fig.text(0.01, 0.97, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY,
                 va="top")  # fmt: skip
    y = {value: len(groups) - 1 - i for i, value in enumerate(groups)}
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        data = trends[(trends["chamber"] == chamber) & trends[group].isin(groups)]
        for _, row in data.iterrows():
            first, last = (int(c) for c in str(row["span"]).split("-"))
            low, high = row["ci_low"] * (last - first), row["ci_high"] * (last - first)
            color = (colors or {}).get(row[group], INK_SECONDARY)
            ax.plot([low, high], [y[row[group]]] * 2, color=color, linewidth=2,
                    alpha=0.45, solid_capstyle="round", zorder=2)  # fmt: skip
            ax.plot(row["change_over_span"], y[row[group]], "o", color=color,
                    markersize=MARKER_SIZE + 1, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, zorder=3)  # fmt: skip
        ax.set_xlabel("Change in the party gap (R − D), first to last Congress",
                      color=INK_SECONDARY, fontsize=8)  # fmt: skip
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(list(y), fontsize=8, color=INK_SECONDARY)
    axes[0].set_ylim(-0.7, len(groups) - 0.3)
    return fig


def plot_member_trajectories(
    trajectories: pd.DataFrame,
    members: Mapping[int, str],
    title: str,
    subtitle: str | None = None,
    ncols: int = 4,
) -> Figure:
    """One small panel per member: their mean per Congress beside their party's.

    ``trajectories`` is `analysis.member_trajectories`; ``members`` maps ICPSR to
    the panel title, in panel order. The member's line carries a 95% band; the
    dashed line is their party's mean in their chamber that Congress. A
    member recorded under two parties gets both, each in its colour. The
    source break is marked; presidential terms are shaded, unnamed.
    """
    rows = math.ceil(len(members) / ncols)
    fig, axes = plt.subplots(
        rows, ncols, figsize=(3.3 * ncols, 2.5 * rows), sharex=True, sharey=True,
        facecolor=SURFACE, squeeze=False,
    )  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.0)
    if subtitle:
        fig.text(0.01, 0.975, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY,
                 va="top")  # fmt: skip
    congresses = sorted(trajectories["congress_number"].unique())
    flat = axes.ravel()
    for ax, (icpsr, name) in zip(flat, members.items()):
        _style(ax)
        shade_presidencies(ax, congresses, names=False)
        mark_source_break(ax, label=False)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        member = trajectories[trajectories["icpsr"] == icpsr]
        for party, data in member.groupby("party"):
            # A Congress without enough speeches is a gap, not a point to draw
            # a line through.
            first, last = data["congress_number"].min(), data["congress_number"].max()
            data = (
                data.set_index("congress_number")
                .reindex(range(first, last + 1))
                .rename_axis("congress_number")
                .reset_index()
            )
            color = PARTY_COLORS.get(party, INK_SECONDARY)
            ax.plot(data["congress_number"], data["party_mean"], color=color,
                    linewidth=1, linestyle=(0, (3, 2)), alpha=0.7, zorder=2,
                    label="party mean (chamber)")  # fmt: skip
            ax.fill_between(data["congress_number"], data["ci_low"], data["ci_high"],
                            color=color, alpha=BAND_ALPHA, linewidth=0,
                            zorder=2)  # fmt: skip
            ax.plot(data["congress_number"], data["mean"], color=color,
                    linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE - 1,
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3,
                    label="member")  # fmt: skip
        ax.set_title(name, loc="left", fontsize=9, color=INK)
    for ax in flat[len(members) :]:
        ax.set_visible(False)
    ticks = [c for c in congresses if (c - congresses[0]) % 2 == 0]
    for ax in axes[-1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{c}\n{congress_year(c)}" for c in ticks], fontsize=7,
                           color=INK_SECONDARY)  # fmt: skip
        ax.set_xlim(min(congresses) - 0.6, max(congresses) + 0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Ideology", color=INK_SECONDARY, fontsize=8)
    handles, labels = flat[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    fig.legend(list(keep.values()), list(keep), frameon=False, fontsize=8,
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncols=2)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def plot_member_slopes(
    trends: pd.DataFrame, title: str, subtitle: str | None = None, label_top: int = 4
) -> Figure:
    """Each member's speech trend against their voting trend, one dot per member.

    x: slope of their per-Congress Nokken-Poole score; y: slope of their speech
    score (both per Congress). One axis each, no shared scale implied. The
    ``label_top`` members furthest from zero on the speech trend, per party,
    are named.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    data = trends.dropna(subset=["slope", "slope_nokken_poole"])
    for party in ("D", "R"):
        d = data[data["party"] == party]
        r = d["slope"].corr(d["slope_nokken_poole"])
        ax.scatter(
            d["slope_nokken_poole"], d["slope"], s=22, color=PARTY_COLORS[party],
            edgecolor=SURFACE, linewidth=0.6, alpha=0.85,
            label=f"{PARTY_NAMES[party]}: r = {r:+.2f} (n = {len(d)})",
        )  # fmt: skip
        for _, row in (
            d.reindex(d["slope"].abs().sort_values().index[::-1])
            .head(label_top)
            .iterrows()
        ):
            ax.annotate(
                row["bioname"].split(",")[0].title(), (row["slope_nokken_poole"], row["slope"]),
                xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK_SECONDARY,
            )  # fmt: skip
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.set_xlabel("Change per Congress in DW-NOMINATE (Nokken-Poole)",
                  color=INK_SECONDARY, fontsize=8.5)  # fmt: skip
    ax.set_ylabel("Change per Congress in mean speech score", color=INK_SECONDARY,
                  fontsize=8.5)  # fmt: skip
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK, y=1.0)
    if subtitle:
        fig.text(0.02, 0.935, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def plot_group_means(
    table: pd.DataFrame,
    group: str,
    title: str,
    ylabel: str,
    subtitle: str | None = None,
    order: Sequence[str] = (),
    ticklabels: Mapping[str, str] | None = None,
    source_break: float | None = None,
) -> Figure:
    """Means per group with 95% intervals, one series per party; House | Senate.

    ``table`` is `analysis.cell_means` by chamber, party and ``group``; groups
    run along x in ``order``. The parties are set slightly apart so their
    intervals do not overlap on the page. ``ticklabels`` maps a group to a
    shorter tick label (a group it leaves out keeps its name).

    Groups that pool every Congress (age, tenure) are not a time series and
    get no source break. Groups in time order (presidential terms) are, and
    ``source_break`` gives the x position of the 2016-09-10 break, the first
    group at 0.
    """
    groups = list(order) or sorted(table[group].unique())
    x = {value: i for i, value in enumerate(groups)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.03)
    if subtitle:
        fig.text(0.01, 0.965, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY)
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        for offset, party in ((-0.08, "D"), (0.08, "R")):
            data = table[(table["chamber"] == chamber) & (table["party"] == party)]
            data = data[data[group].isin(groups)]
            data = data.assign(x=data[group].map(x) + offset).sort_values("x")
            ax.vlines(data["x"], data["ci_low"], data["ci_high"],
                      color=PARTY_COLORS[party], linewidth=2, alpha=0.45,
                      zorder=2)  # fmt: skip
            ax.plot(data["x"], data["mean"], color=PARTY_COLORS[party],
                    linewidth=1, marker="o", markersize=MARKER_SIZE + 1,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=PARTY_NAMES[party], zorder=3)  # fmt: skip
        ax.set_xticks(list(x.values()))
        ax.set_xticklabels(
            [(ticklabels or {}).get(g, g) for g in groups],
            fontsize=8,
            color=INK_SECONDARY,
        )
        ax.set_xlim(-0.5, len(groups) - 0.5)
        if source_break is not None:
            mark_source_break(ax, x=source_break)
    axes[0].set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left",
                   bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    return fig


def save_figure(fig: Figure, name: str, directory: Path = PLOTS_DIR) -> Path:
    """Save as PNG under results/plots (committed) and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return path
