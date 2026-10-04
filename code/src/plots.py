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

from collections.abc import Mapping
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


def shade_presidencies(ax, congresses: list[int]) -> None:
    """Alternate light bands per presidential term, named along the top."""
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


def mark_source_break(ax) -> None:
    """Dashed line at the 114th Congress, where Stanford hands over to govinfo."""
    ax.plot(
        [SOURCE_BREAK_CONGRESS, SOURCE_BREAK_CONGRESS],
        [0, 1],
        transform=ax.get_xaxis_transform(),
        color=INK_MUTED,
        linestyle=(0, (4, 3)),
        linewidth=1,
        zorder=1,
    )
    ax.text(
        SOURCE_BREAK_CONGRESS + 0.12,
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


def save_figure(fig: Figure, name: str, directory: Path = PLOTS_DIR) -> Path:
    """Save as PNG under results/plots (committed) and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return path
