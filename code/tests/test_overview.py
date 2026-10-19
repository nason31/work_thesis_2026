"""Tests for the overview analysis: the substantive filter, the trend of the
party gap, members over time, and their figures.

Expected values are worked out by hand in the comments.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.analysis import gap_trend, member_trajectories, member_trends, substantive
from src.config import SOURCE_BREAK_CONGRESS
from src.plots import plot_gap_by_group, plot_member_slopes, plot_member_trajectories

# --- the substantive filter ----------------------------------------------------------


def test_procedural_ceremonial_and_unlabelled_speeches_are_left_out() -> None:
    frame = pd.DataFrame(
        {
            "speech_id": list("abcdef"),
            "speech_type": ["procedural", "ceremonial", "policy_debate",
                            "bill_introduction", None, "other"],
        }
    )  # fmt: skip

    assert substantive(frame)["speech_id"].tolist() == ["c", "d", "f"]


# --- the trend of the gap ------------------------------------------------------------


def _gap_frame() -> pd.DataFrame:
    """D at -0.2 throughout; R at 0.2 + 0.05 per Congress; noise that cancels."""
    rows = []
    for congress in range(107, 112):
        for party in ("D", "R"):
            base = -0.2 if party == "D" else 0.2 + 0.05 * (congress - 107)
            for i, noise in enumerate((-0.01, 0.01)):
                rows.append(
                    {
                        "congress_number": congress,
                        "party": party,
                        "chamber": "H",
                        "ideology": base + noise,
                        "member_key": f"{party}{congress}{i}",
                    }
                )
    return pd.DataFrame(rows)


def test_the_gap_trend_recovers_a_known_widening() -> None:
    # Gap 0.4 in the 107th, +0.05 per Congress: 0.6 by the 111th.
    row = gap_trend(_gap_frame()).iloc[0]

    assert row["gap_first"] == pytest.approx(0.4)
    assert row["gap_slope"] == pytest.approx(0.05)
    assert row["change_over_span"] == pytest.approx(0.20)
    assert row["span"] == "107-111"


def test_the_gap_trend_is_kept_apart_by_group() -> None:
    frame = pd.concat(
        [
            _gap_frame().assign(topic_block="economic"),
            _gap_frame()
            .assign(topic_block="governance")
            .assign(ideology=lambda f: np.where(f["party"] == "R", 0.2, -0.2)),
        ]
    )

    trends = gap_trend(frame, by=("chamber", "topic_block")).set_index("topic_block")

    assert trends.loc["economic", "gap_slope"] == pytest.approx(0.05)
    assert trends.loc["governance", "gap_slope"] == pytest.approx(0.0, abs=1e-12)


# --- members over time ----------------------------------------------------------------


def _member_frame() -> pd.DataFrame:
    """Member 1 (D) moves -0.1 per Congress, as their party does; member 2 (D)
    stays at -0.5. Ten speeches each per Congress, 107th-111th; member 3 has
    only two speeches a Congress."""
    rows = []
    for congress in range(107, 112):
        step = congress - 107
        for member, level, n in ((1, -0.1 * step, 10), (2, -0.5, 10), (3, 0.0, 2)):
            for _ in range(n):
                rows.append(
                    {
                        "icpsr": float(member),
                        "party": "D",
                        "chamber": "S",
                        "congress_number": congress,
                        "ideology": level,
                        "bioname": f"MEMBER{member}, A",
                        "state": "VT",
                        "nokken_poole_dim1": (
                            -0.3 - 0.02 * step if member == 1 else -0.4
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_trajectories_keep_member_congresses_with_enough_speeches() -> None:
    trajectories = member_trajectories(_member_frame(), min_speeches=10)

    assert set(trajectories["icpsr"]) == {1, 2}
    row = trajectories[
        (trajectories["icpsr"] == 1) & (trajectories["congress_number"] == 109)
    ].iloc[0]
    # 109th: member 1 at -0.2; the party mean over all 22 speeches is
    # (10 * -0.2 + 10 * -0.5 + 2 * 0.0) / 22 = -7/22.
    assert row["mean"] == pytest.approx(-0.2)
    assert row["party_mean"] == pytest.approx(-7 / 22)
    assert row["relative"] == pytest.approx(-0.2 + 7 / 22)
    assert row["n"] == 10 and row["se"] == pytest.approx(0.0)


def test_member_trends_find_who_moved() -> None:
    trends = member_trends(_member_frame(), min_speeches=10, min_congresses=5)
    trends = trends.set_index("icpsr")

    assert trends.loc[1, "slope"] == pytest.approx(-0.1)
    assert trends.loc[1, "change_over_span"] == pytest.approx(-0.4)
    assert trends.loc[1, "slope_nokken_poole"] == pytest.approx(-0.02)
    assert trends.loc[2, "slope"] == pytest.approx(0.0, abs=1e-12)
    assert trends.loc[1, "congresses"] == 5 and trends.loc[1, "chambers"] == "S"


def test_a_member_with_too_few_congresses_has_no_trend() -> None:
    frame = _member_frame()
    frame = frame[~((frame["icpsr"] == 2) & (frame["congress_number"] > 108))]

    trends = member_trends(frame, min_speeches=10, min_congresses=5)

    assert set(trends["icpsr"]) == {1}


# --- figures -------------------------------------------------------------------------


def _marks_the_break(ax) -> bool:
    return any(
        list(line.get_xdata()) == [SOURCE_BREAK_CONGRESS, SOURCE_BREAK_CONGRESS]
        for line in ax.get_lines()
    )


def test_the_topic_gap_figure_keeps_chambers_apart_and_marks_the_break() -> None:
    gaps = pd.DataFrame(
        [
            {"congress_number": c, "chamber": ch, "topic_block": b, "gap": 0.5,
             "ci_low": 0.4, "ci_high": 0.6}
            for c in (113, 114, 115) for ch in ("H", "S") for b in ("economic", "other")
        ]
    )  # fmt: skip

    fig = plot_gap_by_group(gaps, "topic_block", "t", order=("economic", "other"))

    house, senate = fig.axes[:2]
    assert _marks_the_break(house) and _marks_the_break(senate)
    assert house.get_title(loc="left") == "House"


def test_member_panels_mark_the_break_and_hide_spare_panels() -> None:
    trajectories = member_trajectories(
        _member_frame().assign(congress_number=lambda f: f["congress_number"] + 6),
        min_speeches=10,
    )

    fig = plot_member_trajectories(trajectories, {1: "One", 2: "Two", 3: "Three"},
                                   "t", ncols=2)  # fmt: skip

    visible = [ax for ax in fig.axes if ax.get_visible() and ax.get_title(loc="left")]
    assert [ax.get_title(loc="left") for ax in visible] == ["One", "Two", "Three"]
    assert all(_marks_the_break(ax) for ax in visible)
    assert sum(not ax.get_visible() for ax in fig.axes) == 1


def test_the_slope_figure_draws_one_dot_per_member() -> None:
    trends = pd.DataFrame(
        {
            "party": ["D", "D", "R", "R"],
            "slope": [-0.02, 0.01, 0.03, 0.0],
            "slope_nokken_poole": [-0.01, 0.0, 0.02, 0.01],
            "bioname": ["A, X", "B, X", "C, X", "D, X"],
        }
    )

    fig = plot_member_slopes(trends, "t")

    dots = sum(len(c.get_offsets()) for c in fig.axes[0].collections)
    assert dots == 4


def test_the_topic_trend_figure_scales_the_interval_to_the_span() -> None:
    from src.plots import plot_gap_trends

    trends = pd.DataFrame(
        {
            "chamber": ["H", "S"],
            "topic": ["immigration", "immigration"],
            "span": ["107-119", "107-119"],
            "gap_slope": [0.05, 0.08],
            "ci_low": [0.04, 0.06],
            "ci_high": [0.06, 0.10],
            "change_over_span": [0.6, 0.96],
        }
    )

    fig = plot_gap_trends(trends, "topic", "t", order=("immigration",))

    # House: interval 0.04 * 12 = 0.48 to 0.06 * 12 = 0.72, point at 0.6.
    house = fig.axes[0]
    interval, point = house.get_lines()[1:3]
    assert list(interval.get_xdata()) == pytest.approx([0.48, 0.72])
    assert list(point.get_xdata()) == pytest.approx([0.6])


def test_the_share_figure_takes_topic_blocks() -> None:
    from src.plots import plot_label_shares

    shares = pd.DataFrame(
        {
            "congress_number": [113, 114, 113, 114],
            "chamber": ["H"] * 4,
            "topic_block": ["economic", "economic", "governance", "governance"],
            "share": [0.6, 0.5, 0.4, 0.5],
        }
    )

    fig = plot_label_shares(shares, "t", column="topic_block",
                            order=("economic", "governance"))  # fmt: skip

    labels = [ln.get_label() for ln in fig.axes[0].get_lines()]
    assert "economic" in labels and "governance" in labels


def test_a_member_line_breaks_where_a_congress_is_missing() -> None:
    frame = _member_frame()
    frame = frame[~((frame["icpsr"] == 1) & (frame["congress_number"] == 109))]
    trajectories = member_trajectories(frame, min_speeches=10)

    fig = plot_member_trajectories(trajectories, {1: "One"}, "t", ncols=1)

    member_line = next(
        ln for ln in fig.axes[0].get_lines() if ln.get_label() == "member"
    )
    assert list(member_line.get_xdata()) == [107, 108, 109, 110, 111]
    assert np.isnan(member_line.get_ydata()[2])


# --- age, generation, tenure ------------------------------------------------------


def test_background_counts_every_congress_served_since_the_first(tmp_path) -> None:
    from src.analysis import member_background

    path = tmp_path / "members.csv"
    pd.DataFrame(
        {
            # Member 1: House 100-101, gap, Senate 105-107 (one row per chamber
            # in 105 counts once). Member 2: a President row, ignored.
            "congress": [100, 101, 105, 105, 106, 107, 107],
            "chamber": ["House", "House", "House", "Senate", "Senate", "Senate",
                        "President"],
            "icpsr": [1, 1, 1, 1, 1, 1, 2],
            "born": [1950.0] * 6 + [1940.0],
        }
    ).to_csv(path, index=False)  # fmt: skip

    background = member_background(path).set_index("congress_number")

    assert background.loc[107, "tenure"] == 5  # 100, 101, 105, 106, 107
    assert background.loc[105, "tenure"] == 3
    assert (background["first_congress"] == 100).all()
    assert set(background["icpsr"]) == {1}


def test_age_generation_and_tenure_groups() -> None:
    from src.analysis import add_background

    frame = pd.DataFrame(
        {
            "icpsr": [1.0, 2.0, np.nan],
            "congress_number": [115, 115, 115],
            "date": pd.to_datetime(["2017-03-01", "2018-06-01", "2017-01-05"]),
        }
    )
    background = pd.DataFrame(
        {
            "icpsr": pd.array([1, 2], dtype="Int64"),
            "congress_number": [115, 115],
            "born": [1972.0, 1945.0],
            "tenure": [1, 11],
            "first_congress": [115, 105],
        }
    )

    out = add_background(frame, background)

    # 2017 - 1972 = 45: the first year of "45–54" (bins are left-closed).
    assert out["age"].tolist()[:2] == [45, 73]
    assert out["age_group"].tolist()[:2] == ["45–54", "65–74"]
    assert out["generation"].tolist()[:2] == ["Gen X", "Silent and older"]
    assert out["tenure_group"].tolist()[:2] == ["1st Congress", "11+"]
    assert out.loc[2, ["age", "age_group", "tenure"]].isna().all()


def test_extremity_points_toward_each_partys_side() -> None:
    from src.analysis import toward_party

    frame = pd.DataFrame({"party": ["D", "R"], "ideology": [-0.6, 0.4]})

    assert toward_party(frame).tolist() == [0.6, 0.4]


def test_relative_to_cell_compares_within_party_chamber_and_congress() -> None:
    from src.analysis import relative_to_cell

    frame = pd.DataFrame(
        {
            "congress_number": [110, 110, 111],
            "chamber": ["H", "H", "H"],
            "party": ["D", "D", "D"],
            "x": [1.0, 3.0, 5.0],
        }
    )

    assert relative_to_cell(frame, "x").tolist() == [-1.0, 1.0, 0.0]


def test_within_cell_ols_recovers_a_slope_despite_cell_levels() -> None:
    from src.analysis import within_cell_ols

    # Within every cell y rises 0.1 per year of age; the cells' levels differ
    # by 1.0, which a pooled regression would mistake for an age effect.
    rows = []
    for cell, level, ages in ((110, 0.0, (40, 50, 60)), (111, 1.0, (60, 70, 80))):
        for i, age in enumerate(ages):
            rows.append(
                {
                    "congress_number": cell,
                    "chamber": "H",
                    "party": "D",
                    "age": float(age),
                    "y": level + 0.1 * (age - ages[0]) + (0.01 if i == 1 else 0),
                    "member_key": f"m{cell}{i}",
                }
            )

    row = within_cell_ols(pd.DataFrame(rows), "y", ["age"]).iloc[0]

    assert row["term"] == "age"
    assert row["coef"] == pytest.approx(0.1)


def test_the_group_figure_puts_groups_in_order_for_both_chambers() -> None:
    from src.plots import plot_group_means

    table = pd.DataFrame(
        [
            {"chamber": ch, "party": p, "age_group": g, "mean": 0.1, "ci_low": 0.0,
             "ci_high": 0.2}
            for ch in ("H", "S") for p in ("D", "R") for g in ("75+", "under 45")
        ]
    )  # fmt: skip

    fig = plot_group_means(table, "age_group", "t", "y", order=("under 45", "75+"))

    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == ["under 45", "75+"]
    assert fig.axes[1].get_title(loc="left") == "Senate"
