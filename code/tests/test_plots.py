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
from src.plots import (
    plot_gap,
    plot_group_means,
    plot_models,
    plot_party_positions,
    plot_party_positions_by_group,
    plot_validation,
)


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


def _validation_units(shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for party, sign in (("D", -1), ("R", 1)):
        for i in range(5):
            score = sign * 0.4 + 0.05 * i + shift
            rows.append(
                {
                    "party": party,
                    "score": score,
                    "bench": sign * 0.4 + 0.04 * i,
                    "score_within": 0.05 * (i - 2),
                    "bench_within": 0.04 * (i - 2),
                }
            )
    return pd.DataFrame(rows)


def _validation_result(within: float) -> dict[str, float]:
    return {
        "overall_pearson": 0.9,
        "overall_pearson_ci_low": 0.88,
        "overall_pearson_ci_high": 0.92,
        "within_cell_pearson": within,
        "within_cell_pearson_ci_low": within - 0.05,
        "within_cell_pearson_ci_high": within + 0.05,
    }


def test_validation_figure_shows_the_reported_r_per_speech_set() -> None:
    """The r on a panel is the validation's own, not one recomputed from the dots."""
    fig = plot_validation(
        {"All": _validation_units(), "Substantive": _validation_units(0.1)},
        {"All": _validation_result(0.123), "Substantive": _validation_result(0.456)},
        "t",
    )

    assert len(fig.axes) == 4  # two rows (across parties, within cells) x two sets
    texts = [" ".join(t.get_text() for t in ax.texts) for ax in fig.axes]
    top_left, top_right, bottom_left, bottom_right = texts
    assert "+0.90" in top_left and "+0.90" in top_right
    assert "+0.12 [+0.07, +0.17]" in bottom_left
    assert "+0.46 [+0.41, +0.51]" in bottom_right
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == ["All", "Substantive", "All", "Substantive"]


def test_group_means_can_relabel_the_ticks() -> None:
    """Long group names (presidential terms) get shorter tick labels."""
    table = pd.DataFrame(
        {
            "chamber": ["H", "H", "S", "S"],
            "party": ["D", "R", "D", "R"],
            "term": ["Bush 2001-2004"] * 4,
            "mean": [-0.3, 0.3, -0.3, 0.3],
            "ci_low": [-0.4, 0.2, -0.4, 0.2],
            "ci_high": [-0.2, 0.4, -0.2, 0.4],
        }
    )

    fig = plot_group_means(
        table, "term", "t", "y", ticklabels={"Bush 2001-2004": "Bush\n2001-2004"}
    )

    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["Bush\n2001-2004"]


def test_group_means_over_time_can_mark_the_source_break() -> None:
    """Presidential terms run in time order, so the break is drawn between them."""
    table = pd.DataFrame(
        {
            "chamber": ["H", "H", "S", "S"] * 2,
            "party": ["D", "R"] * 4,
            "term": ["Obama 2013-2016"] * 4 + ["Trump 2017-2020"] * 4,
            "mean": [-0.3, 0.3] * 4,
            "ci_low": [-0.4, 0.2] * 4,
            "ci_high": [-0.2, 0.4] * 4,
        }
    )

    fig = plot_group_means(table, "term", "t", "y", source_break=0.42)

    for ax in fig.axes[:2]:
        assert any(list(line.get_xdata()) == [0.42, 0.42] for line in ax.get_lines())


def test_positions_by_group_draw_one_row_per_group_each_marking_the_break() -> None:
    """A row per topic block, House | Senate in each, never pooled."""
    cells = pd.concat(
        [_cells(topic_block="economic"), _cells(topic_block="governance")]
    )

    fig = plot_party_positions_by_group(
        cells, "topic_block", "Ideology", "t", order=["economic", "governance"]
    )

    assert len(fig.axes) == 4
    assert all(_marks_the_break(ax) for ax in fig.axes)
    assert [ax.get_title(loc="left") for ax in fig.axes] == [
        "House · economic",
        "Senate · economic",
        "House · governance",
        "Senate · governance",
    ]
