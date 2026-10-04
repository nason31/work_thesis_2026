"""Tests for the analysis plots: the contract CLAUDE.md sets for time series.

Every time-series panel must mark the 2016-09-10 source break, and House and
Senate are drawn apart. Rendering details are matplotlib's business; these
tests check only what the thesis depends on.
"""

from __future__ import annotations

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src.config import SOURCE_BREAK_CONGRESS
from src.plots import plot_gap, plot_models, plot_party_positions


def _cells(**overrides) -> pd.DataFrame:
    rows = []
    for congress in (113, 114, 115):
        for chamber in ("H", "S"):
            for party, mean in (("D", -0.3), ("R", 0.3)):
                rows.append(
                    {
                        "congress_number": congress,
                        "chamber": chamber,
                        "party": party,
                        "mean": mean,
                        "se": 0.03,
                        "ci_low": mean - 0.06,
                        "ci_high": mean + 0.06,
                    }
                )
    return pd.DataFrame(rows).assign(**overrides)


def _marks_the_break(ax) -> bool:
    return any(
        list(line.get_xdata()) == [SOURCE_BREAK_CONGRESS, SOURCE_BREAK_CONGRESS]
        for line in ax.get_lines()
    )


def _gaps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "congress_number": [113, 114, 115] * 2,
            "chamber": ["H"] * 3 + ["S"] * 3,
            "gap": [0.6] * 6,
            "ci_low": [0.5] * 6,
            "ci_high": [0.7] * 6,
        }
    )


@pytest.mark.parametrize(
    "figure",
    [
        lambda: plot_party_positions(_cells(), "Ideology", "t"),
        lambda: plot_gap(_gaps(), "t"),
        lambda: plot_models({"m1": _cells(), "m2": _cells()}, "t"),
    ],
    ids=["positions", "gap", "models"],
)
def test_every_time_series_panel_marks_the_source_break(figure) -> None:
    fig = figure()

    panels = [ax for ax in fig.axes if ax.get_lines()]
    assert len(panels) == 2  # House and Senate, never pooled
    assert all(_marks_the_break(ax) for ax in panels)


def test_position_panels_are_titled_by_chamber() -> None:
    fig = plot_party_positions(_cells(), "Ideology", "t")

    assert [ax.get_title(loc="left") for ax in fig.axes[:2]] == ["House", "Senate"]


def test_presidential_terms_are_labelled_on_the_panels() -> None:
    fig = plot_party_positions(_cells(), "Ideology", "t")

    labels = {text.get_text() for text in fig.axes[0].texts}
    # 113-114 Obama, 115 Trump.
    assert {"Obama", "Trump"} <= labels


def test_direct_labels_do_not_overlap_when_lines_end_together() -> None:
    """Two models' Democratic lines often end at nearly the same value."""
    fig = plot_models({"m1": _cells(), "m2": _cells()}, "t")

    for ax in fig.axes[:2]:
        ys = sorted(a.xy[1] for a in ax.texts if hasattr(a, "xy"))
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        assert all(b - a >= 0.04 * span for a, b in pairwise(ys))
