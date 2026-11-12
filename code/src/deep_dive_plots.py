"""Figures for the deep dive (notebook 08), beside those in `plots.py`.

The same rules and furniture as `plots.py`, imported rather than copied:
House and Senate side by side, never pooled (O8), presidential terms shaded;
the 2016-09-10 source break is not drawn (docs/decisions.md A10). Chamber
control is drawn as a thin strip along the top of each panel, coloured by the
majority party, so presidency and chamber majority stay readable in one figure.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.config import PRESIDENTS_BY_CONGRESS, TOPIC_BLOCKS
from src.deep_dive import (
    HOUSE_MAJORITY,
    MARGIN_TERCILES,
    OWN_MARGIN_TERCILES,
    POWER_LEVELS,
    SENATE_MAJORITY,
    SENATE_MAJORITY_CHANGES,
    congress_of,
    election_day,
)
from src.plots import (
    BASELINE,
    CHAMBERS,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MARKER_SIZE,
    PARTY_COLORS,
    PARTY_NAMES,
    SPEECH_TYPE_PALETTE,
    SURFACE,
    TERM_SHADE,
    TOPIC_BLOCK_PALETTE,
    _style,
    congress_year,
    plot_party_positions,
    shade_presidencies,
)

#: The strip's bottom and height, as shares of the axis height: just inside
#: the top, clear of the presidents' names above the axis. Leave headroom in
#: ``ylim`` so the data stay below it.
CONTROL_STRIP = (0.93, 0.035)
CONTROL_ALPHA = 0.55

#: Models A-C in the coefficient plot: ink shades and marker shapes, so the
#: party colours stay reserved for parties.
MODEL_STYLES = {
    "A": (INK_MUTED, "o"),
    "B": (INK_SECONDARY, "s"),
    "C": (INK, "D"),
}

#: Components of the two decompositions. Categorical slots from the dataviz
#: palette that are not the party blue and red.
COMPONENT_COLORS = {
    "positions": "#6250d6",
    "agenda": "#eda100",
    "adaptation": "#6250d6",
    "replacement": "#eda100",
    "unmatched": INK_MUTED,
}


def _congress_days(congress: int) -> tuple[dt.date, dt.date]:
    """First and last day of a Congress (3 January to 3 January, two years on)."""
    start = 1789 + 2 * (congress - 1)
    return dt.date(start, 1, 3), dt.date(start + 2, 1, 3)


def _majority_spans(chamber: str, congress: int) -> list[tuple[float, float, str]]:
    """Where each majority sits along the x-axis within one Congress's slot.

    A Congress occupies [congress - 0.5, congress + 0.5]; a Senate that changed
    hands is split by the share of days each majority held.
    """
    if chamber == "H" or congress not in SENATE_MAJORITY_CHANGES:
        party = (
            HOUSE_MAJORITY[congress] if chamber == "H" else SENATE_MAJORITY[congress]
        )
        return [(congress - 0.5, congress + 0.5, party)]
    first, last = _congress_days(congress)
    length = (last - first).days
    periods = SENATE_MAJORITY_CHANGES[congress]
    spans = []
    for (start, party), following in zip(periods, [*periods[1:], (last, None)]):
        low = congress - 0.5 + (start - first).days / length
        high = congress - 0.5 + (following[0] - first).days / length
        spans.append((low, high, party))
    return spans


def shade_chamber_control(ax, chamber: str, congresses: Sequence[int]) -> None:
    """A thin strip along the top of the panel, coloured by the majority party."""
    bottom, height = CONTROL_STRIP
    for congress in congresses:
        for low, high, party in _majority_spans(chamber, congress):
            ax.axvspan(
                low, high, ymin=bottom, ymax=bottom + height, color=PARTY_COLORS[party],
                alpha=CONTROL_ALPHA, linewidth=0, zorder=4,
            )  # fmt: skip


def plot_positions_with_control(
    cells: pd.DataFrame,
    ylabel: str,
    title: str,
    subtitle: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """`plots.plot_party_positions` with the chamber majority along the top.

    Presidential terms stay shaded behind the data; the strip shows which
    party held the panel's chamber, split by date where the Senate changed
    hands within a Congress.
    """
    fig = plot_party_positions(cells, ylabel, title, subtitle, ylim)
    congresses = sorted(cells["congress_number"].unique())
    axes = fig.axes[:2]
    for ax, (chamber, _) in zip(axes, CHAMBERS):
        shade_chamber_control(ax, chamber, congresses)
    # One legend below both panels, clear of the direct labels at the lines' ends.
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].get_legend().remove()
    handles += [
        Patch(color=PARTY_COLORS[p], alpha=CONTROL_ALPHA, linewidth=0)
        for p in ("D", "R")
    ]
    labels += [f"{PARTY_NAMES[p]} hold the chamber (top strip)" for p in ("D", "R")]
    fig.legend(
        handles, labels, frameon=False, fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.02), ncols=len(handles),
    )  # fmt: skip
    return fig


def plot_opposition_models(
    models: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    terms: Sequence[tuple[str, str]] = (
        ("opposition", "Out of the White House"),
        ("chamber_minority", "Minority in own chamber"),
    ),
) -> Figure:
    """Coefficients of models A-C with 95% intervals, one panel per term.

    ``models`` is `deep_dive.opposition_models` for one value. Rows are party x
    chamber; the models sit side by side within a row, in ink shades, so a
    model is never mistaken for a party. A term a model leaves out is absent.
    """
    groups = [(c, p) for c, _ in CHAMBERS for p in ("D", "R")]
    names = dict(CHAMBERS)
    y = {group: len(groups) - 1 - i for i, group in enumerate(groups)}
    offsets = {"A": 0.22, "B": 0.0, "C": -0.22}
    fig, axes = plt.subplots(
        1, len(terms), figsize=(5.2 * len(terms), 0.75 * len(groups) + 1.6),
        sharey=True, facecolor=SURFACE,
    )  # fmt: skip
    axes = list(axes) if len(terms) > 1 else [axes]
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.03)
    if subtitle:
        fig.text(0.01, 0.975, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY,
                 va="top")  # fmt: skip
    for ax, (term, label) in zip(axes, terms):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(label, loc="left", fontsize=10, color=INK, pad=8)
        for model, (color, marker) in MODEL_STYLES.items():
            data = models[(models["term"] == term) & (models["model"] == model)]
            for _, row in data.iterrows():
                at = y[(row["chamber"], row["party"])] + offsets[model]
                ax.plot([row["ci_low"], row["ci_high"]], [at, at], color=color,
                        linewidth=2, alpha=0.45, solid_capstyle="round",
                        zorder=2)  # fmt: skip
                ax.plot(row["coef"], at, marker=marker, color=color, linestyle="none",
                        markersize=MARKER_SIZE + 1, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, zorder=3)  # fmt: skip
        ax.set_xlabel("Coefficient (95% interval, clustered by member)",
                      color=INK_SECONDARY, fontsize=8)  # fmt: skip
    axes[0].set_yticks(list(y.values()))
    axes[0].set_yticklabels(
        [f"{names[c]} {PARTY_NAMES[p]}" for c, p in y], fontsize=8, color=INK_SECONDARY
    )
    axes[0].set_ylim(-0.6, len(groups) - 0.4)
    legend = [
        plt.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="none",
            markersize=MARKER_SIZE + 1,
            label=f"Model {model}",
        )
        for model, (color, marker) in MODEL_STYLES.items()
    ]
    axes[-1].legend(handles=legend, frameon=False, fontsize=8, loc="upper left",
                    bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    return fig


def _value(table: pd.DataFrame, **match: str) -> pd.DataFrame:
    """The rows of ``table`` whose columns equal ``match``."""
    keep = pd.Series(True, index=table.index)
    for column, value in match.items():
        keep &= table[column] == value
    return table[keep]


def _stacked_bars(
    ax,
    positions: Sequence[float],
    parts: Sequence[tuple[str, Sequence[float]]],
    totals: pd.DataFrame,
    width: float = 0.55,
) -> None:
    """Bars stacked from zero, positive parts upward and negative downward,
    with the total as a dot and its 95% interval (``value``, ``ci_low``,
    ``ci_high`` in ``totals``, one row per position)."""
    up = [0.0] * len(positions)
    down = [0.0] * len(positions)
    for name, values in parts:
        for i, (x, value) in enumerate(zip(positions, values)):
            base = up[i] if value >= 0 else down[i]
            ax.bar(x, value, width=width, bottom=base, color=COMPONENT_COLORS[name],
                   linewidth=0, zorder=2, label=name if i == 0 else None)  # fmt: skip
            if value >= 0:
                up[i] += value
            else:
                down[i] += value
    ax.vlines(positions, totals["ci_low"], totals["ci_high"], color=INK, linewidth=1.2,
              zorder=3)  # fmt: skip
    ax.plot(positions, totals["value"], "o", color=INK, markersize=MARKER_SIZE,
            markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4, label="total")  # fmt: skip
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)


def plot_gap_decomposition(
    decomposition: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    comparisons: Sequence[str] = (),
    blocks: Sequence[str] = (),
) -> Figure:
    """The change in the policy-debate gap, split into positions and agenda.

    ``decomposition`` is `deep_dive.bootstrap_gap_decomposition`. Top row,
    one panel per comparison: per chamber, the two parts stacked and the
    total change as a dot with its interval. Bottom row: each block's
    contribution to the two parts, with intervals.
    """
    comparisons = list(comparisons) or list(decomposition["comparison"].unique())
    blocks = list(blocks) or [
        b for b in decomposition["topic_block"].unique() if b != "all"
    ]
    fig, axes = plt.subplots(
        2, len(comparisons), figsize=(5.6 * len(comparisons), 8.4),
        facecolor=SURFACE, squeeze=False, gridspec_kw={"height_ratios": (1, 1.3)},
    )  # fmt: skip
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.02)
    if subtitle:
        fig.text(0.01, 0.985, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY,
                 va="top")  # fmt: skip
    chambers = [c for c, _ in CHAMBERS]
    names = dict(CHAMBERS)
    for column, comparison in enumerate(comparisons):
        data = decomposition[decomposition["comparison"] == comparison]
        top = axes[0][column]
        _style(top)
        overall = data[data["topic_block"] == "all"]
        parts = [
            (
                name,
                [
                    float(_value(overall, component=name, chamber=c)["value"].iloc[0])
                    for c in chambers
                ],
            )
            for name in ("positions", "agenda")
        ]
        totals = pd.concat(
            [_value(overall, component="total", chamber=c) for c in chambers]
        )
        _stacked_bars(top, range(len(chambers)), parts, totals)
        top.set_xticks(range(len(chambers)))
        top.set_xticklabels(
            [names[c] for c in chambers], fontsize=9, color=INK_SECONDARY
        )
        top.set_title(comparison, loc="left", fontsize=10, color=INK, pad=8)
        bottom = axes[1][column]
        _style(bottom)
        bottom.grid(axis="x", color=GRID, linewidth=0.6)
        bottom.grid(axis="y", visible=False)
        bottom.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        rows = [(b, c) for b in blocks for c in chambers]
        y = {row: len(rows) - 1 - i for i, row in enumerate(rows)}
        for offset, component in ((0.17, "positions"), (-0.17, "agenda")):
            part = data[data["component"] == component].set_index(
                ["topic_block", "chamber"]
            )
            for row, at in y.items():
                if row not in part.index:
                    continue
                r = part.loc[row]
                bottom.barh(at + offset, r["value"], height=0.32,
                            color=COMPONENT_COLORS[component], linewidth=0,
                            zorder=2)  # fmt: skip
                bottom.hlines(at + offset, r["ci_low"], r["ci_high"], color=INK,
                              linewidth=1, zorder=3)  # fmt: skip
        bottom.set_yticks(list(y.values()))
        bottom.set_yticklabels([f"{b} · {names[c]}" for b, c in y], fontsize=7.5,
                               color=INK_SECONDARY)  # fmt: skip
        bottom.set_xlabel("Contribution to the change in the gap (R − D)",
                          color=INK_SECONDARY, fontsize=8)  # fmt: skip
    axes[0][0].set_ylabel(
        "Change in the gap (R − D)", color=INK_SECONDARY, fontsize=8.5
    )
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left",
                       bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_turnover(
    cumulative: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    windows: Sequence[str] = (),
) -> Figure:
    """Each party's cumulative move, split into adaptation and replacement.

    ``cumulative`` is `deep_dive.bootstrap_turnover`. One panel per chamber;
    per party and window, adaptation, replacement and the small unmatched
    part are stacked, and the total move is a dot with its interval.
    """
    windows = list(windows) or list(cumulative["window"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=1.03)
    if subtitle:
        fig.text(0.01, 0.965, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY)
    slots = [(p, w) for p in ("D", "R") for w in windows]
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        data = cumulative[cumulative["chamber"] == chamber]
        parts = [
            (
                component,
                [
                    float(
                        _value(data, component=component, party=p, window=w)[
                            "value"
                        ].iloc[0]
                    )
                    for p, w in slots
                ],
            )
            for component in ("adaptation", "replacement", "unmatched")
        ]
        totals = pd.concat(
            [_value(data, component="delta", party=p, window=w) for p, w in slots]
        )
        _stacked_bars(ax, range(len(slots)), parts, totals)
        ax.set_xticks(range(len(slots)))
        ax.set_xticklabels(
            [f"{PARTY_NAMES[p]}\n{w.split(' ')[0]}" for p, w in slots],
            fontsize=8,
            color=INK_SECONDARY,
        )
        for tick, (party, _) in zip(ax.get_xticklabels(), slots):
            tick.set_color(PARTY_COLORS[party])
    axes[0].set_ylabel("Change in mean ideology", color=INK_SECONDARY, fontsize=8.5)
    handles, labels = axes[1].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    axes[1].legend(list(keep.values()), list(keep), frameon=False, fontsize=8,
                   loc="upper left", bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    return fig


# =================================================================================
# Notebook 09: extensions of the deep dive
# =================================================================================

#: Series colours for the keyword lists and the common-ground measures:
#: categorical slots of `plots.SPEECH_TYPE_PALETTE`, never the party colours.
LIST_COLORS = {
    "cooperation": SPEECH_TYPE_PALETTE[1],
    "hostility": SPEECH_TYPE_PALETTE[0],
    "deficit": SPEECH_TYPE_PALETTE[5],
    "undo": SPEECH_TYPE_PALETTE[3],
    "other_party": SPEECH_TYPE_PALETTE[4],
}
MEASURE_STYLES = {
    "d_right_of_r_median": (PARTY_COLORS["D"], "D speeches right of the R median"),
    "r_left_of_d_median": (PARTY_COLORS["R"], "R speeches left of the D median"),
    "overlap": (INK_SECONDARY, "overlap of the two distributions"),
}
ROWS = [(c, p) for c, _ in CHAMBERS for p in ("D", "R")]


def _header(fig: Figure, title: str, subtitle: str | None, y: float = 1.02) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=y)
    if subtitle:
        fig.text(0.01, y - 0.045, subtitle, ha="left", fontsize=8.5, color=INK_SECONDARY,
                 va="top")  # fmt: skip


def plot_event_effects(
    effects: pd.DataFrame, measure: str, title: str, subtitle: str | None = None
) -> Figure:
    """Change from before to after each event, in its topics and in the rest.

    ``effects`` is `deep_dive.event_effects`. One panel per chamber, one row
    per event; filled markers are the event's topics, open markers every other
    topic, each with its 95% interval. ``measure`` "gap" draws R - D in ink,
    "tone" each party in its colour. With ``small_cell`` from
    `deep_dive.flag_small_event_cells`, an estimate resting on a small cell
    gets a † beside its interval and a footnote.
    """
    data = effects[
        (effects["measure"] == measure) & (effects["group"] != "event − other")
    ]
    events = list(dict.fromkeys(effects["event"]))
    parties = ["R−D"] if measure == "gap" else ["D", "R"]
    slots = [(p, g) for p in parties for g in ("event", "other")]
    step = 0.6 / max(len(slots) - 1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.7 * len(events) + 1.8), sharey=True,
                             facecolor=SURFACE)  # fmt: skip
    _header(fig, title, subtitle, y=1.06)
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        for i, (party, group) in enumerate(slots):
            part = _value(data, chamber=chamber, party=party, group=group)
            y = [
                len(events) - 1 - events.index(e) + 0.3 - i * step
                for e in part["event"]
            ]
            color = INK if party == "R−D" else PARTY_COLORS[party]
            label = f"{'' if party == 'R−D' else PARTY_NAMES[party] + ', '}" + (
                "event's topics" if group == "event" else "other topics"
            )
            ax.hlines(y, part["ci_low"], part["ci_high"], color=color, linewidth=1.2,
                      zorder=2)  # fmt: skip
            ax.plot(part["estimate"], y, "o", color=color, markersize=MARKER_SIZE + 1,
                    markerfacecolor=color if group == "event" else SURFACE,
                    markeredgewidth=1.2, linestyle="none", zorder=3, label=label)  # fmt: skip
            if "small_cell" in part:
                for x, at, small in zip(part["ci_high"], y, part["small_cell"]):
                    if small:
                        ax.annotate("†", (x, at), xytext=(3, 0), textcoords="offset points",
                                    va="center", fontsize=9, color=color)  # fmt: skip
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels(events[::-1], fontsize=8.5, color=INK_SECONDARY)
        what = "R − D" if measure == "gap" else "tone"
        ax.set_xlabel(f"Change in {what}, after minus before", color=INK_SECONDARY,
                      fontsize=8)  # fmt: skip
    axes[1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    if "small_cell" in data and data["small_cell"].any():
        fig.text(0.01, -0.02, "† rests on a cell with fewer than 30 speeches (see the table).",
                 fontsize=8, color=INK_SECONDARY, va="top")  # fmt: skip
    return fig


def plot_event_quarterly(
    quarterly: pd.DataFrame,
    value: str,
    events: Sequence[str],
    title: str,
    subtitle: str | None = None,
) -> Figure:
    """Each party's mean in the event's topics, bin by bin around the event.

    ``quarterly`` is `deep_dive.event_quarterly`; one row per event in
    ``events``, House | Senate. The event day is a solid line, the nearest
    Congress boundary a dashed one.
    """
    data = quarterly[quarterly["value"] == value]
    fig, axes = plt.subplots(len(events), 2, figsize=(12, 2.6 * len(events) + 1), sharex=True,
                             sharey=True, facecolor=SURFACE, squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.0 + 0.02 * 6 / max(len(events), 1))
    for row, event in enumerate(events):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            part = _value(data, event=event, chamber=chamber)
            for party in ("D", "R"):
                series = part[part["party"] == party].sort_values("quarter")
                ax.fill_between(series["quarter"] + 0.5, series["ci_low"], series["ci_high"],
                                color=PARTY_COLORS[party], alpha=0.15, linewidth=0)  # fmt: skip
                ax.plot(series["quarter"] + 0.5, series["mean"], color=PARTY_COLORS[party],
                        marker="o", markersize=MARKER_SIZE - 1, linewidth=1.5,
                        label=PARTY_NAMES[party])  # fmt: skip
            ax.axvline(0, color=INK, linewidth=1, zorder=1)
            if len(part):
                boundary = float(part["congress_boundary"].iloc[0])
                ax.axvline(
                    boundary, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1
                )
            ax.set_title(f"{name} · {event}", loc="left", fontsize=9.5, color=INK)
        axes[row][0].set_ylabel(value, color=INK_SECONDARY, fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("Quarters (91 days) from the event; dashed: nearest Congress boundary",
                      color=INK_SECONDARY, fontsize=8)  # fmt: skip
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_term_coefficients(
    fits: pd.DataFrame, term: str, title: str, subtitle: str | None = None
) -> Figure:
    """One coefficient per chamber and party, with its 95% interval; one panel
    per ``value`` in ``fits`` (a `within_cell_ols` table with ``value``)."""
    values = list(dict.fromkeys(fits["value"]))
    fig, axes = plt.subplots(1, len(values), figsize=(5.6 * len(values), 3.4), facecolor=SURFACE,
                             squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.08)
    names = dict(CHAMBERS)
    for ax, value in zip(axes[0], values):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
        part = fits[(fits["value"] == value) & (fits["term"] == term)]
        for i, (chamber, party) in enumerate(ROWS):
            r = _value(part, chamber=chamber, party=party)
            if r.empty:
                continue
            y = len(ROWS) - 1 - i
            ax.hlines(
                y, r["ci_low"], r["ci_high"], color=PARTY_COLORS[party], linewidth=1.4
            )
            ax.plot(r["coef"], [y], "o", color=PARTY_COLORS[party], markersize=MARKER_SIZE + 1,
                    markeredgecolor=SURFACE)  # fmt: skip
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([f"{names[c]} · {PARTY_NAMES[p]}" for c, p in ROWS[::-1]],
                           fontsize=8.5, color=INK_SECONDARY)  # fmt: skip
        ax.set_title(value, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Coefficient (95% interval)", color=INK_SECONDARY, fontsize=8)
    return fig


def shade_presidencies_years(ax, years: Sequence[int]) -> None:
    """`plots.shade_presidencies` on an axis with one point per year: each
    president's Congresses (PRESIDENTS_BY_CONGRESS) as years, alternately
    shaded and named along the top."""
    shown = sorted(set(years))
    terms: list[tuple[str, list[int]]] = []
    for year in shown:
        name = PRESIDENTS_BY_CONGRESS.get(congress_of(dt.date(year, 6, 1)), ("", ""))[0]
        if terms and terms[-1][0] == name:
            terms[-1][1].append(year)
        else:
            terms.append((name, [year]))
    for index, (name, members) in enumerate(terms):
        low, high = min(members) - 0.5, max(members) + 0.5
        if index % 2 == 0:
            ax.axvspan(low, high, color=TERM_SHADE, zorder=0, linewidth=0)
        ax.text((low + high) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.5, color=INK_SECONDARY)  # fmt: skip


def _mark_incomplete_year(ax, year: int | None) -> None:
    if year is not None:
        ax.axvspan(year - 0.5, year + 0.5, facecolor="none", edgecolor=INK_MUTED, hatch="///",
                   linewidth=0, zorder=1)  # fmt: skip


def plot_yearly_keyword_lines(
    shares: pd.DataFrame,
    lists: Sequence[str],
    title: str,
    subtitle: str | None = None,
    incomplete: int | None = None,
) -> Figure:
    """Share of speeches with a hit per year, one line per list.

    ``shares`` is `deep_dive.keyword_shares` by year, chamber and party. Rows
    are the parties, columns the chambers; presidencies shaded, the source
    break marked, an incomplete last year hatched.
    """
    years = sorted(shares["year"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True,
                             facecolor=SURFACE)  # fmt: skip
    _header(fig, title, subtitle, y=1.03)
    for row, party in enumerate(("D", "R")):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            shade_presidencies_years(ax, years)
            for name_ in lists:
                part = _value(shares, chamber=chamber, party=party, list=name_)
                part = part.sort_values("year")
                color = LIST_COLORS[name_]
                ax.fill_between(part["year"], 100 * part["ci_low"], 100 * part["ci_high"],
                                color=color, alpha=0.15, linewidth=0)  # fmt: skip
                ax.plot(part["year"], 100 * part["mean"], color=color, linewidth=1.6,
                        marker="o", markersize=MARKER_SIZE - 2, label=name_)  # fmt: skip
            _mark_incomplete_year(ax, incomplete)
            ax.set_title(f"{name} · {PARTY_NAMES[party]}", loc="left", fontsize=10, color=INK,
                         pad=14)  # fmt: skip
        axes[row][0].set_ylabel(
            "Speeches with a hit (%)", color=INK_SECONDARY, fontsize=8.5
        )
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_keyword_by_opposition(
    shares: pd.DataFrame, lists: Sequence[str], title: str, subtitle: str | None = None
) -> Figure:
    """Share of speeches with a hit, with the White House against in
    opposition, per list and party; House | Senate. ``shares`` is
    `deep_dive.keyword_shares` by chamber, party and ``in_opposition``.
    Solid bars: party holds the White House; hatched: in opposition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.05)
    slots = [(p, o) for p in ("D", "R") for o in (False, True)]
    width = 0.8 / len(slots)
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        for i, name_ in enumerate(lists):
            for j, (party, opposition) in enumerate(slots):
                r = _value(shares, chamber=chamber, party=party, list=name_)
                r = r[r["in_opposition"].astype(bool) == opposition]
                if r.empty:
                    continue
                x = i - 0.4 + width * (j + 0.5)
                ax.bar(x, 100 * r["mean"], width=width * 0.92, color=PARTY_COLORS[party],
                       alpha=0.85 if not opposition else 0.35,
                       hatch="///" if opposition else None, edgecolor=SURFACE, linewidth=0,
                       label=(f"{PARTY_NAMES[party]}, "
                              f"{'in opposition' if opposition else 'White House'}")
                       if i == 0 else None)  # fmt: skip
                ax.vlines(
                    x, 100 * r["ci_low"], 100 * r["ci_high"], color=INK, linewidth=1
                )
        ax.set_xticks(range(len(lists)))
        ax.set_xticklabels(list(lists), fontsize=8.5, color=INK_SECONDARY)
    axes[0].set_ylabel("Speeches with a hit (%)", color=INK_SECONDARY, fontsize=8.5)
    axes[1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    return fig


def plot_swing_states(
    means: pd.DataFrame, title: str, subtitle: str | None = None
) -> Figure:
    """Senators' extremity and tone relative to their party, chamber and
    Congress, by their state's presidential margin (tercile). ``means`` is the
    first table of `deep_dive.swing_state_effects`."""
    values = list(dict.fromkeys(means["value"]))
    fig, axes = plt.subplots(1, len(values), figsize=(11, 4.2), facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.05)
    x = {t: i for i, t in enumerate(MARGIN_TERCILES)}
    for ax, value in zip(axes, values):
        _style(ax)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(f"Senate · {value}", loc="left", fontsize=10, color=INK, pad=8)
        for offset, party in ((-0.08, "D"), (0.08, "R")):
            part = means[(means["value"] == value) & (means["party"] == party)]
            part = part.assign(x=part["tercile"].map(x) + offset).sort_values("x")
            ax.vlines(part["x"], part["ci_low"], part["ci_high"], color=PARTY_COLORS[party],
                      linewidth=2, alpha=0.45)  # fmt: skip
            ax.plot(part["x"], part["mean"], color=PARTY_COLORS[party], marker="o",
                    markersize=MARKER_SIZE + 1, linewidth=1, label=PARTY_NAMES[party])  # fmt: skip
        ax.set_xticks(list(x.values()))
        ax.set_xticklabels([f"{t} states" for t in MARGIN_TERCILES], fontsize=8.5,
                           color=INK_SECONDARY)  # fmt: skip
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylabel(f"{value} vs. own party, chamber, Congress", color=INK_SECONDARY,
                      fontsize=8.5)  # fmt: skip
    axes[-1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    return fig


def plot_common_ground(
    table: pd.DataFrame, groups: Sequence[str], title: str, subtitle: str | None = None
) -> Figure:
    """Crossing shares and overlap per Congress, one row per ``group`` value
    of ``topic_block``; House | Senate. ``table`` is
    `deep_dive.bootstrap_common_ground` output with a ``topic_block`` column
    ("all" for every substantive speech)."""
    congresses = sorted(table["congress_number"].unique())
    height = 2.8 * len(groups) + 1.2
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, height), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.0 + 0.3 / height)
    for row, group in enumerate(groups):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            shade_presidencies(ax, congresses, names=row == 0)
            for measure, (color, label) in MEASURE_STYLES.items():
                part = _value(
                    table, topic_block=group, chamber=chamber, measure=measure
                )
                part = part.sort_values("congress_number")
                ax.fill_between(part["congress_number"], part["ci_low"], part["ci_high"],
                                color=color, alpha=0.15, linewidth=0)  # fmt: skip
                ax.plot(part["congress_number"], part["value"], color=color, linewidth=1.6,
                        marker="o", markersize=MARKER_SIZE - 1, label=label)  # fmt: skip
            ax.set_xticks(congresses)
            ax.set_xticklabels([f"{c}\n{congress_year(c)}" for c in congresses], fontsize=7.5,
                               color=INK_SECONDARY)  # fmt: skip
            ax.set_title(f"{name} · {group}", loc="left", fontsize=10, color=INK,
                         pad=14 if row == 0 else 6)  # fmt: skip
        axes[row][0].set_ylabel("Share", color=INK_SECONDARY, fontsize=8.5)
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / height))
    return fig


def plot_laws_and_gap(
    table: pd.DataFrame, title: str, subtitle: str | None = None
) -> Figure:
    """The party gap (line, left axis) and public laws enacted (bars, right
    axis) per Congress; House | Senate. An incomplete Congress's bar is
    hatched. ``table`` holds ``congress_number``, ``chamber``, ``gap`` with
    ``ci_low``/``ci_high``, ``public_laws`` and ``complete``."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.05)
    congresses = sorted(table["congress_number"].unique())
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        part = table[table["chamber"] == chamber].sort_values("congress_number")
        # Presidencies are shaded on the bars' axis, which sits behind the line's,
        # so the shading never covers the bars.
        laws = ax.twinx()
        shade_presidencies(laws, congresses)
        laws.bar(part["congress_number"], part["public_laws"], width=0.6, color=INK_MUTED, alpha=0.3, zorder=1,
                 hatch=["///" if not c else None for c in part["complete"]],
                 edgecolor=INK_MUTED, linewidth=0)  # fmt: skip
        laws.set_ylabel("Public laws enacted (bars)", color=INK_MUTED, fontsize=8)
        laws.tick_params(colors=INK_MUTED, labelsize=8)
        laws.spines["top"].set_visible(False)
        ax.set_zorder(laws.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.fill_between(part["congress_number"], part["ci_low"], part["ci_high"], color=INK,
                        alpha=0.12, linewidth=0)  # fmt: skip
        ax.plot(part["congress_number"], part["gap"], color=INK, marker="o",
                markersize=MARKER_SIZE, linewidth=1.75)  # fmt: skip
        ax.set_xticks(congresses)
        ax.set_xticklabels([f"{c}\n{congress_year(c)}" for c in congresses], fontsize=7.5,
                           color=INK_SECONDARY)  # fmt: skip
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=14)
        ax.set_ylabel("Party gap, R − D (line)", color=INK_SECONDARY, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_keyword_lines_by_congress(
    shares: pd.DataFrame, lists: Sequence[str], title: str, subtitle: str | None = None
) -> Figure:
    """Share of speeches with a hit per Congress, one line per list.

    ``shares`` is `deep_dive.keyword_shares` by Congress, chamber and party.
    Rows are the parties, columns the chambers; presidencies shaded, as in
    `plots.plot_party_positions`.
    """
    congresses = sorted(shares["congress_number"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True,
                             facecolor=SURFACE)  # fmt: skip
    _header(fig, title, subtitle, y=1.03)
    for row, party in enumerate(("D", "R")):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            shade_presidencies(ax, congresses, names=row == 0)
            for name_ in lists:
                part = _value(shares, chamber=chamber, party=party, list=name_)
                part = part.sort_values("congress_number")
                color = LIST_COLORS[name_]
                ax.fill_between(part["congress_number"], 100 * part["ci_low"],
                                100 * part["ci_high"], color=color, alpha=0.15,
                                linewidth=0)  # fmt: skip
                ax.plot(part["congress_number"], 100 * part["mean"], color=color,
                        linewidth=1.6, marker="o", markersize=MARKER_SIZE - 1,
                        label=name_)  # fmt: skip
            ax.set_xticks(congresses)
            ax.set_xticklabels([f"{c}\n{congress_year(c)}" for c in congresses], fontsize=7.5,
                               color=INK_SECONDARY)  # fmt: skip
            ax.set_title(f"{name} · {PARTY_NAMES[party]}", loc="left", fontsize=10, color=INK,
                         pad=14)  # fmt: skip
        axes[row][0].set_ylabel(
            "Speeches with a hit (%)", color=INK_SECONDARY, fontsize=8.5
        )
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# =================================================================================
# Notebook 10: designs, topics over time, Voteview
# =================================================================================

#: Topic blocks in `plots.TOPIC_BLOCK_PALETTE`'s colours, as everywhere else.
BLOCK_COLORS = dict(zip(TOPIC_BLOCKS, TOPIC_BLOCK_PALETTE))
#: Variants in the coefficient plots: ink shades and marker shapes, so the
#: party colours stay reserved for parties.
VARIANT_STYLES = (
    (INK, "o"),
    (INK_SECONDARY, "s"),
    (INK_MUTED, "D"),
    (INK_MUTED, "^"),
)


def _forest_axis(ax) -> None:
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)


def _congress_ticks(ax, congresses: Sequence[int]) -> None:
    ax.set_xticks(list(congresses))
    ax.set_xticklabels([f"{c}\n{congress_year(c)}" for c in congresses], fontsize=7.5,
                       color=INK_SECONDARY)  # fmt: skip


def plot_block_differences(
    tests: pd.DataFrame,
    rows: Sequence[tuple[str, str]],
    title: str,
    subtitle: str | None = None,
) -> Figure:
    """Change per policy block with its 95% interval, two spans.

    ``tests`` is `deep_dive.policy_block_tests` output for both spans, with
    ``measure`` ("gap" or "tone") and ``comparison`` ("main", "control").
    One row of panels per (measure, party) in ``rows``, House | Senate.
    Filled markers: the main span; open: the control span. Each panel's title
    carries the main span's joint test that the four blocks changed alike.
    """
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.4 * len(rows) + 1.2), sharey=True,
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.0 + 0.25 / len(rows))
    names = dict(CHAMBERS)
    for row, (measure, party) in enumerate(rows):
        data = tests[(tests["measure"] == measure) & (tests["party"] == party)]
        blocks = list(dict.fromkeys(data.loc[data["kind"] == "block", "topic_block"]))
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _forest_axis(ax)
            color = INK if party == "R−D" else PARTY_COLORS[party]
            part = data[data["chamber"] == chamber]
            for offset, comparison in ((0.12, "main"), (-0.12, "control")):
                blocks_part = part[
                    (part["kind"] == "block") & (part["comparison"] == comparison)
                ]
                y = [
                    len(blocks) - 1 - blocks.index(b) + offset
                    for b in blocks_part["topic_block"]
                ]
                ax.hlines(y, blocks_part["ci_low"], blocks_part["ci_high"], color=color,
                          linewidth=1.3)  # fmt: skip
                span = blocks_part["span"].iloc[0] if len(blocks_part) else ""
                ax.plot(blocks_part["change"], y, "o", color=color, markersize=MARKER_SIZE + 1,
                        markerfacecolor=color if comparison == "main" else SURFACE,
                        markeredgewidth=1.2, linestyle="none",
                        label=f"{span.replace('-', '→')} ({comparison})")  # fmt: skip
            joint = part[(part["kind"] == "joint") & (part["comparison"] == "main")]
            test = ""
            if len(joint):
                j = joint.iloc[0]
                test = (
                    f" · joint χ²({int(j['df'])}) = {j['chi2']:.1f}, p = {j['p']:.2g}"
                )
            who = "R − D" if party == "R−D" else f"tone, {PARTY_NAMES[party]}"
            ax.set_title(f"{names[chamber]} · {who}{test}", loc="left", fontsize=9.5,
                         color=INK)  # fmt: skip
            ax.set_yticks(range(len(blocks)))
            ax.set_yticklabels(blocks[::-1], fontsize=8.5, color=INK_SECONDARY)
            if row == len(rows) - 1:
                ax.set_xlabel("Change (95% interval)", color=INK_SECONDARY, fontsize=8)
        axes[row][1].legend(frameon=False, fontsize=8, loc="upper left",
                            bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_steps_decomposition(
    steps: pd.DataFrame, title: str, subtitle: str | None = None
) -> Figure:
    """The change in the policy-debate gap at every step, split by block.

    ``steps`` is `deep_dive.bootstrap_step_decomposition`'s first table, with
    ``congress_from`` (the step's first Congress). One panel per chamber; per
    step, the positions part of each block and the agenda total stacked (up
    when positive, down when negative), the total change as a dot with its
    95% interval. Each step sits between its two Congresses, so a change of
    the White House falls on the boundary of the shaded presidencies.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.6), sharex=True, facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.03)
    congresses = sorted(
        {c for c in steps["congress_from"]} | {c + 1 for c in steps["congress_from"]}
    )
    parts = [(b, "positions", BLOCK_COLORS[b], None) for b in TOPIC_BLOCKS]
    parts.append(("all", "agenda", INK_MUTED, "///"))
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        shade_presidencies(ax, congresses, names=chamber == "H")
        data = steps[steps["chamber"] == chamber]
        order = data[data["component"] == "total"].sort_values("congress_from")
        x = (order["congress_from"] + 0.5).to_numpy()
        up, down = [0.0] * len(x), [0.0] * len(x)
        for block, component, color, hatch in parts:
            values = [
                float(
                    _value(data, comparison=c, component=component, topic_block=block)[
                        "value"
                    ].iloc[0]
                )
                for c in order["comparison"]
            ]
            for i, value in enumerate(values):
                base = up[i] if value >= 0 else down[i]
                label = (
                    f"positions: {block}"
                    if component == "positions"
                    else "agenda (all blocks)"
                )
                ax.bar(x[i], value, width=0.62, bottom=base, color=color, hatch=hatch,
                       edgecolor=SURFACE if hatch is None else INK_SECONDARY, linewidth=0,
                       zorder=2, label=label if (i == 0 and chamber == "H") else None)  # fmt: skip
                if value >= 0:
                    up[i] += value
                else:
                    down[i] += value
        ax.vlines(
            x, order["ci_low"], order["ci_high"], color=INK, linewidth=1.2, zorder=3
        )
        ax.plot(x, order["value"], "o", color=INK, markersize=MARKER_SIZE, markeredgecolor=SURFACE,
                zorder=4, label="total change" if chamber == "H" else None)  # fmt: skip
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=14)
        ax.set_ylabel(
            "Change in R − D, policy debates", color=INK_SECONDARY, fontsize=8.5
        )
        ax.set_xticks(x)
        ax.set_xticklabels(list(order["comparison"]), fontsize=7.5, color=INK_SECONDARY)
    axes[0].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_did_cases(
    table: pd.DataFrame,
    variants: Sequence[str],
    title: str,
    subtitle: str | None = None,
    xlabel: str = "Difference-in-differences (95% interval)",
) -> Figure:
    """Coefficients per case with 95% intervals, one panel per ``value``.

    ``table`` holds ``case``, ``value``, ``variant``, ``coef``, ``ci_low``,
    ``ci_high``; rows are the cases in order of appearance (a "pooled" row
    last), one marker style per entry of ``variants``.
    """
    values = list(dict.fromkeys(table["value"]))
    cases = list(dict.fromkeys(c for c in table["case"] if c != "pooled"))
    if (table["case"] == "pooled").any():
        cases.append("pooled")
    step = 0.6 / max(len(variants) - 1, 1)
    fig, axes = plt.subplots(1, len(values), figsize=(5.8 * len(values), 0.55 * len(cases) + 2),
                             sharey=True, facecolor=SURFACE, squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.08)
    for ax, value in zip(axes[0], values):
        _forest_axis(ax)
        for i, variant in enumerate(variants):
            color, marker = VARIANT_STYLES[i % len(VARIANT_STYLES)]
            part = table[(table["value"] == value) & (table["variant"] == variant)]
            y = [len(cases) - 1 - cases.index(c) + 0.3 - i * step for c in part["case"]]
            ax.hlines(y, part["ci_low"], part["ci_high"], color=color, linewidth=1.3)
            ax.plot(part["coef"], y, marker, color=color, markersize=MARKER_SIZE + 1,
                    markeredgecolor=SURFACE, linestyle="none", label=variant)  # fmt: skip
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels(cases[::-1], fontsize=8.5, color=INK_SECONDARY)
        ax.set_title(value, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=8)
    axes[0][-1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    return fig


def plot_event_study(
    effects: pd.DataFrame,
    paths: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    reference: int = -1,
) -> Figure:
    """Event study around the changes of the White House.

    ``effects`` is `deep_dive.wh_event_study` (rows with ``k``), ``paths``
    `deep_dive.party_own_paths`; both for several ``value``s. Rows of panels
    are the values, columns the chambers. Black: the losing party's change
    beyond the winning party's, with its 95% interval; thin lines: each
    side's own change. Everything is relative to k = ``reference`` (zero).
    """
    values = list(dict.fromkeys(effects["value"]))
    fig, axes = plt.subplots(len(values), 2, figsize=(12, 3.3 * len(values) + 1), sharex=True,
                             facecolor=SURFACE, squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.0 + 0.2 / len(values))
    sides = (
        ("loses White House", INK_SECONDARY, "-"),
        ("gains White House", INK_MUTED, "--"),
    )
    for row, value in enumerate(values):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
            ax.axvline(-0.5, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=1)
            for role, color, style in sides:
                own = paths[(paths["value"] == value) & (paths["chamber"] == chamber)
                            & (paths["role"] == role)]  # fmt: skip
                own = pd.concat(
                    [
                        own[["k", "coef"]],
                        pd.DataFrame({"k": [reference], "coef": [0.0]}),
                    ]
                )
                own = own.sort_values("k")
                ax.plot(own["k"], own["coef"], color=color, linestyle=style, linewidth=1.1,
                        label=f"own change, party that {role.split()[0]} it")  # fmt: skip
            effect = effects[(effects["value"] == value) & (effects["chamber"] == chamber)
                             & effects["k"].notna()]  # fmt: skip
            ax.vlines(effect["k"], effect["ci_low"], effect["ci_high"], color=INK, linewidth=1.4,
                      zorder=3)  # fmt: skip
            ax.plot(effect["k"], effect["coef"], "o", color=INK, markersize=MARKER_SIZE + 1,
                    markeredgecolor=SURFACE, zorder=4,
                    label="losing − winning party (95% interval)")  # fmt: skip
            ax.plot([reference], [0], "o", color=SURFACE, markeredgecolor=INK, markersize=MARKER_SIZE + 1,
                    zorder=4)  # fmt: skip
            ax.set_title(f"{name} · {value}", loc="left", fontsize=10, color=INK)
            ax.set_xticks(sorted(set(effect["k"].dropna().astype(int)) | {reference}))
        axes[row][0].set_ylabel(f"Change in {value} vs. k = {reference}", color=INK_SECONDARY,
                                fontsize=8.5)  # fmt: skip
    for ax in axes[-1]:
        ax.set_xlabel("k = Congresses from the new president's first (dotted: the change)",
                      color=INK_SECONDARY, fontsize=8)  # fmt: skip
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_own_margin(
    means: pd.DataFrame, title: str, subtitle: str | None = None
) -> Figure:
    """Senators' extremity and tone relative to their party, chamber and
    Congress, by terciles of their own party's margin in their state.
    ``means`` is the first table of `deep_dive.own_margin_effects`."""
    values = list(dict.fromkeys(means["value"]))
    fig, axes = plt.subplots(1, len(values), figsize=(11, 4.2), facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.05)
    x = {t: i for i, t in enumerate(OWN_MARGIN_TERCILES)}
    labels = ("leans to the\nother party", "middle", "stronghold of\nthe own party")
    for ax, value in zip(axes, values):
        _style(ax)
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(f"Senate · {value}", loc="left", fontsize=10, color=INK, pad=8)
        for offset, party in ((-0.08, "D"), (0.08, "R")):
            part = means[(means["value"] == value) & (means["party"] == party)]
            part = part.assign(x=part["own_tercile"].map(x) + offset).sort_values("x")
            ax.vlines(part["x"], part["ci_low"], part["ci_high"], color=PARTY_COLORS[party],
                      linewidth=2, alpha=0.45)  # fmt: skip
            ax.plot(part["x"], part["mean"], color=PARTY_COLORS[party], marker="o",
                    markersize=MARKER_SIZE + 1, linewidth=1, label=PARTY_NAMES[party])  # fmt: skip
        ax.set_xticks(list(x.values()))
        ax.set_xticklabels(labels, fontsize=8.5, color=INK_SECONDARY)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylabel(f"{value} vs. own party, chamber, Congress", color=INK_SECONDARY,
                      fontsize=8.5)  # fmt: skip
    axes[-1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    return fig


def plot_event_block_grid(
    effects: pd.DataFrame, measure: str, title: str, subtitle: str | None = None
) -> Figure:
    """A grid of events x topic blocks with each change written in its cell.

    ``effects`` is `deep_dive.flag_small_block_cells` output. One panel per
    chamber; for ``measure`` "gap" one column per block, for "tone" and
    "agenda" one per block and party. A value whose 95% interval excludes
    zero is bold on a tinted cell (purple up, amber down); † marks a value
    resting on a cell under 30 speeches; "–" an empty cell.
    """
    data = effects[effects["measure"] == measure]
    events = list(dict.fromkeys(data["event"]))
    parties = ["R−D"] if measure == "gap" else ["D", "R"]
    columns = [(b, p) for b in TOPIC_BLOCKS for p in parties]
    scale = 100 if measure == "agenda" else 1
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.15 * len(columns) + 2), 1.05 * len(events) * 2 + 2.4),
                             facecolor=SURFACE)  # fmt: skip
    _header(fig, title, subtitle, y=1.02)
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(len(events) - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        for i, event in enumerate(events):
            for j, (block, party) in enumerate(columns):
                cell = _value(
                    data, event=event, chamber=chamber, topic_block=block, party=party
                )
                if cell.empty or pd.isna(cell["estimate"].iloc[0]):
                    ax.text(
                        j, i, "–", ha="center", va="center", fontsize=9, color=INK_MUTED
                    )
                    continue
                r = cell.iloc[0]
                clear = r["ci_low"] > 0 or r["ci_high"] < 0
                if clear:
                    tint = "#6250d6" if r["estimate"] > 0 else "#eda100"
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.45), 0.96, 0.9, color=tint,
                                               alpha=0.18, linewidth=0))  # fmt: skip
                text = f"{scale * r['estimate']:+.{1 if scale == 100 else 2}f}" + (
                    "†" if r["small_cell"] else ""
                )
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5,
                        color=INK if clear else INK_MUTED,
                        fontweight="bold" if clear else "normal")  # fmt: skip
        ax.set_yticks(range(len(events)))
        ax.set_yticklabels(events, fontsize=8.5, color=INK_SECONDARY)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(
            [b if p == "R−D" else f"{b}\n{p}" for b, p in columns],
            fontsize=7.5,
            color=INK_SECONDARY,
        )
        ax.xaxis.tick_top()
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=26)
    fig.text(0.01, -0.01, "Bold on a tint: 95% interval excludes zero (purple up, amber down). "
             "† rests on a cell under 30 speeches. – no speeches in a cell.",
             fontsize=8, color=INK_SECONDARY, va="top")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_rollcall_lines(
    summary: pd.DataFrame,
    series: Sequence[tuple[str, str, str]],
    title: str,
    subtitle: str | None = None,
    incomplete: int | None = None,
    ylabel: str = "Share of roll calls (%)",
) -> Figure:
    """Per-Congress roll-call measures, House | Senate.

    ``summary`` has ``congress_number``, ``chamber``, ``variant`` and the
    measure columns; ``series`` lists (column, variant, label): a column's
    colour is fixed, its variants differ in line style. Presidencies shaded,
    an incomplete Congress hatched.
    """
    columns = list(dict.fromkeys(s[0] for s in series))
    variants = list(dict.fromkeys(s[1] for s in series))
    colors = dict(
        zip(
            columns,
            (SPEECH_TYPE_PALETTE[5], SPEECH_TYPE_PALETTE[1], SPEECH_TYPE_PALETTE[0]),
        )
    )
    styles = dict(zip(variants, ("-", "--", ":", "-.")))
    congresses = sorted(summary["congress_number"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.06)
    for ax, (chamber, name) in zip(axes, CHAMBERS):
        _style(ax)
        shade_presidencies(ax, congresses)
        for column, variant, label in series:
            part = summary[
                (summary["chamber"] == chamber) & (summary["variant"] == variant)
            ]
            part = part.sort_values("congress_number")
            ax.plot(part["congress_number"], 100 * part[column], color=colors[column],
                    linestyle=styles[variant], marker="o" if variant == variants[0] else None,
                    markersize=MARKER_SIZE - 1, linewidth=1.6, label=label)  # fmt: skip
        if incomplete is not None:
            ax.axvspan(incomplete - 0.5, incomplete + 0.5, facecolor="none", edgecolor=INK_MUTED,
                       hatch="///", linewidth=0, zorder=1)  # fmt: skip
        _congress_ticks(ax, congresses)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=14)
    axes[0].set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8.5)
    axes[1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_rollcall_undo(
    undo: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    incomplete: int | None = None,
) -> Figure:
    """Votes to undo per Congress: the share of roll calls (top) and the
    Congressional Review Act votes, all and passed (bottom); House | Senate.
    ``undo`` is `deep_dive.rollcall_undo_summary`; a Congress that is the
    first after a change of the White House is ringed."""
    congresses = sorted(undo["congress_number"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, facecolor=SURFACE)
    _header(fig, title, subtitle, y=1.03)
    for column, (chamber, name) in enumerate(CHAMBERS):
        part = undo[undo["chamber"] == chamber].sort_values("congress_number")
        first = part[part["first_after_wh_change"]]
        top, bottom = axes[0][column], axes[1][column]
        for ax in (top, bottom):
            _style(ax)
            shade_presidencies(ax, congresses, names=ax is top)
            if incomplete is not None:
                ax.axvspan(incomplete - 0.5, incomplete + 0.5, facecolor="none",
                           edgecolor=INK_MUTED, hatch="///", linewidth=0, zorder=1)  # fmt: skip
        top.plot(part["congress_number"], 100 * part["share_undo"], color=LIST_COLORS["undo"],
                 marker="o", markersize=MARKER_SIZE, linewidth=1.6, label="votes to undo")  # fmt: skip
        top.plot(first["congress_number"], 100 * first["share_undo"], "o", markersize=MARKER_SIZE + 6,
                 markerfacecolor="none", markeredgecolor=INK, linestyle="none",
                 label="first Congress after a change of the White House")  # fmt: skip
        bottom.bar(part["congress_number"] - 0.18, part["n_cra"], width=0.36, color=INK_MUTED,
                   alpha=0.6, label="CRA votes")  # fmt: skip
        bottom.bar(part["congress_number"] + 0.18, part["n_cra_passed"], width=0.36,
                   color=LIST_COLORS["undo"], label="CRA votes passed")  # fmt: skip
        top.set_title(name, loc="left", fontsize=10, color=INK, pad=14)
        _congress_ticks(bottom, congresses)
    axes[0][0].set_ylabel("Roll calls to undo (%)", color=INK_SECONDARY, fontsize=8.5)
    axes[1][0].set_ylabel(
        "Congressional Review Act votes", color=INK_SECONDARY, fontsize=8.5
    )
    axes[0][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    axes[1][1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --- notebook 10, 4b and 4c ------------------------------------------------------------------

#: Election types in the post-election plot: ink shade, marker, pooled line.
ELECTION_STYLES = {
    "presidential": (INK, "o", "-"),
    "midterm": (INK_SECONDARY, "s", "--"),
}


def plot_power_ladder(
    ladder: pd.DataFrame, title: str, subtitle: str | None = None
) -> Figure:
    """The power ladder: each level against "full power", within members.

    ``ladder`` is `deep_dive.power_ladder` (the ``ladder`` rows are drawn),
    for one or more ``value``s, one panel each; rows are chamber x party, one
    marker per level (`plot_did_cases`).
    """
    names = dict(CHAMBERS)
    rows = ladder[ladder["model"] == "ladder"].sort_values(
        ["chamber", "party"], kind="stable"
    )
    table = rows.assign(
        case=[
            f"{names[c]} · {PARTY_NAMES[p]}"
            for c, p in zip(rows["chamber"], rows["party"])
        ],
        variant=rows["power_level"],
    )
    return plot_did_cases(
        table, list(POWER_LEVELS[1:]), title, subtitle,
        xlabel="Against full power, within members (95% interval)",
    )  # fmt: skip


def _decimal_year(day: dt.date) -> float:
    start = dt.date(day.year, 1, 1)
    return day.year + (day - start).days / (dt.date(day.year + 1, 1, 1) - start).days


def plot_post_election(
    effects: pd.DataFrame, title: str, subtitle: str | None = None
) -> Figure:
    """After minus before each election, the losing side beyond the winning one.

    ``effects`` is `deep_dive.post_election_effects` with ``small_cell`` from
    `deep_dive.flag_small_election_cells`, for one or more ``value``s. Rows of
    panels are the values, columns the chambers; one marker per election at
    its date (presidential and midterm told apart), open where a cell holds
    fewer than the minimum (†); the pooled values as lines with a 95% band.
    """
    values = list(dict.fromkeys(effects["value"]))
    years = sorted({int(e) for e in effects["election"] if e.isdigit()})
    fig, axes = plt.subplots(len(values), 2, figsize=(12, 3.3 * len(values) + 1), sharex=True,
                             sharey="row", facecolor=SURFACE, squeeze=False)  # fmt: skip
    _header(fig, title, subtitle, y=1.03)
    for row, value in enumerate(values):
        for ax, (chamber, name) in zip(axes[row], CHAMBERS):
            _style(ax)
            ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
            part = effects[
                (effects["value"] == value) & (effects["chamber"] == chamber)
            ]
            for kind, (color, marker, style) in ELECTION_STYLES.items():
                pooled = part[
                    (part["kind"] == "pooled") & (part["election_type"] == kind)
                ]
                for r in pooled.to_dict("records"):
                    ax.axhspan(
                        r["ci_low"], r["ci_high"], color=color, alpha=0.08, linewidth=0
                    )
                    ax.axhline(r["coef"], color=color, linestyle=style, linewidth=1,
                               label=f"pooled {kind} (band: 95%)")  # fmt: skip
                shown = part[
                    (part["kind"] == "election") & (part["election_type"] == kind)
                ]
                shown = shown.dropna(subset=["coef"])
                x = [_decimal_year(election_day(int(e))) for e in shown["election"]]
                ax.vlines(
                    x, shown["ci_low"], shown["ci_high"], color=color, linewidth=1.3
                )
                small = shown["small_cell"].astype(bool).to_numpy()
                for mask, face, label in ((~small, color, f"{kind} election"),
                                          (small, SURFACE, f"{kind}, a cell under 30 speeches (†)")):  # fmt: skip
                    if mask.any():
                        ax.plot([v for v, m in zip(x, mask) if m], shown["coef"][mask], marker,
                                color=color, markerfacecolor=face, markersize=MARKER_SIZE + 1,
                                linestyle="none", label=label, zorder=3)  # fmt: skip
            ax.set_title(f"{name} · {value}", loc="left", fontsize=10, color=INK)
            ax.set_xticks([_decimal_year(election_day(y)) for y in years])
            ax.set_xticklabels(
                [str(y) for y in years], fontsize=7.5, color=INK_SECONDARY
            )
        axes[row][0].set_ylabel(f"Losers' change beyond winners'\n({value})",
                                color=INK_SECONDARY, fontsize=8.5)  # fmt: skip
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    axes[0][1].legend(handles, labels, frameon=False, fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.0, 1.0))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
