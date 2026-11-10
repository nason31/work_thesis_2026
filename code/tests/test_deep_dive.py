"""Tests for the deep dive: chamber control, opposition within members, the
topic model, and the two decompositions.

Expected values are worked out by hand in the comments.
"""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.deep_dive import (
    CONSTELLATIONS,
    _bootstrap,
    add_chamber_control,
    block_cells,
    bootstrap_gap_decomposition,
    bootstrap_turnover,
    chamber_majority,
    chamber_membership,
    check_reproduces,
    classify_turnover,
    control_cells,
    cumulative_turnover,
    decompose_gap,
    fit_topic_model,
    gap_decomposition,
    identifying_members,
    majority_table,
    opposition_models,
    robustness_samples,
    specter_miscoded,
    stayers_without_speeches,
    topic_cell_sizes,
    topic_gap_tests,
    topic_gaps,
    turnover_decomposition,
    turnover_robustness,
)
from src.deep_dive_plots import (
    _majority_spans,
    plot_gap_decomposition,
    plot_opposition_models,
    plot_positions_with_control,
    plot_turnover,
)

# --- chamber control ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("chamber", "congress", "date", "expected"),
    [
        ("S", 107, "2001-01-10", "D"),  # Gore breaks the 50-50 tie
        ("S", 107, "2001-01-20", "R"),  # Cheney from inauguration day
        ("S", 107, "2001-06-05", "R"),
        ("S", 107, "2001-06-06", "D"),  # Jeffords: the Democrats organize
        ("S", 107, "2002-12-01", "D"),
        ("S", 117, "2021-01-19", "R"),
        ("S", 117, "2021-01-20", "D"),  # Ossoff, Warnock, Padilla; Harris
        ("S", 108, "2003-05-01", "R"),
        ("H", 110, "2007-05-01", "D"),
        ("H", 112, "2011-05-01", "R"),
    ],
)
def test_the_majority_is_resolved_by_date(chamber, congress, date, expected) -> None:
    assert chamber_majority(chamber, congress, date) == expected


def test_chamber_control_columns() -> None:
    frame = pd.DataFrame(
        {
            "chamber": ["H", "S", "S", "S"],
            "congress_number": [112, 111, 107, 107],
            "party": ["D", "D", "R", "R"],
            # As the corpus delivers them: datetime.date, not strings.
            "date": [
                dt.date(2011, 5, 1),
                dt.date(2010, 5, 1),
                dt.date(2001, 3, 1),
                dt.date(2001, 7, 1),
            ],
        }
    )

    out = add_chamber_control(frame)

    # House Democrats under Obama, House held by R: minority.
    # Senate Democrats in the 111th: majority.
    # Senate Republicans 2001: majority (Cheney) until 6 June, then minority.
    assert out["chamber_majority"].tolist() == ["R", "D", "R", "D"]
    assert out["chamber_minority"].tolist() == [1, 0, 0, 1]


def test_an_unknown_congress_stops_chamber_control() -> None:
    frame = pd.DataFrame(
        {
            "chamber": ["H"],
            "congress_number": [120],
            "party": ["D"],
            "date": [dt.date(2027, 5, 1)],
        }
    )
    with pytest.raises(KeyError, match="120"):
        add_chamber_control(frame)


def test_the_majority_table_lists_each_senate_period() -> None:
    table = majority_table()

    # 13 Congresses, plus two more rows for the 107th and one for the 117th.
    assert len(table) == 16
    assert table[table["congress_number"] == 107]["senate_majority"].tolist() == [
        "D",
        "R",
        "D",
    ]


def test_control_cells_split_by_constellation() -> None:
    frame = pd.DataFrame(
        {
            "chamber": ["H"] * 4,
            "party": ["D"] * 4,
            "in_opposition": [True, True, False, False],
            "chamber_minority": [1, 0, 1, 1],
            "tone": [0.6, 0.4, 0.3, 0.5],
            "member_key": ["a", "b", "c", "d"],
        }
    )

    cells = control_cells(frame, "tone").set_index("constellation")

    assert set(cells.index) <= set(CONSTELLATIONS)
    assert cells.loc["opposition, chamber minority", "mean"] == pytest.approx(0.6)
    assert cells.loc["White House, chamber minority", "mean"] == pytest.approx(0.4)
    assert cells.loc["White House, chamber minority", "n_members"] == 2


# --- opposition within members ------------------------------------------------------


def _opposition_frame() -> pd.DataFrame:
    """Tone = member level + 0.1 in opposition + 0.05 in the chamber minority.

    Members 0-3 sit through all six Congresses (levels 0.0-0.3). Members 4-5
    speak only in Congresses where their party is in opposition, and are
    harsh people (level 1.0): between members, opposition looks far stronger
    than 0.1.
    """
    opposition = {107: 0, 108: 1, 109: 0, 110: 1, 111: 1, 112: 0}
    minority = {107: 1, 108: 1, 109: 0, 110: 0, 111: 1, 112: 0}
    rows = []
    for member in range(6):
        level = 0.1 * member if member < 4 else 1.0
        for congress, out in opposition.items():
            if member >= 4 and not out:
                continue
            rows.append(
                {
                    "chamber": "H",
                    "party": "R",
                    "member_key": f"m{member}",
                    "congress_number": congress,
                    "opposition": float(out),
                    "chamber_minority": minority[congress],
                    "tone": level + 0.1 * out + 0.05 * minority[congress],
                }
            )
    return pd.DataFrame(rows)


def test_member_fixed_effects_remove_who_is_speaking() -> None:
    models = opposition_models(_opposition_frame(), "tone").set_index(["model", "term"])

    # Model C compares each member with themselves: exactly the 0.1 and 0.05
    # built in, and no trend.
    assert models.loc[("C", "opposition"), "coef"] == pytest.approx(0.1)
    assert models.loc[("C", "chamber_minority"), "coef"] == pytest.approx(0.05)
    assert models.loc[("C", "congress_number"), "coef"] == pytest.approx(0.0, abs=1e-12)
    # Models A and B also count members 4-5's harshness as opposition.
    assert models.loc[("A", "opposition"), "coef"] > 0.2
    assert models.loc[("B", "opposition"), "coef"] > 0.2
    assert ("A", "chamber_minority") not in models.index


def test_identifying_members_are_those_whose_status_changes() -> None:
    counts = identifying_members(_opposition_frame()).set_index("term")

    # Members 4-5 are always in opposition, but their minority status moves
    # (108th: 1, 110th: 0, 111th: 1).
    assert counts.loc["opposition", "n_members"] == 6
    assert counts.loc["opposition", "n_identifying"] == 4
    assert counts.loc["chamber_minority", "n_identifying"] == 6


def test_the_reproduction_check() -> None:
    fit = pd.DataFrame(
        {
            "chamber": ["H", "H"],
            "party": ["D", "D"],
            "term": ["congress_number", "opposition"],
            "coef": [0.01, 0.1],
            "se": [0.001, 0.02],
        }
    )

    check_reproduces(fit, fit.copy())
    with pytest.raises(ValueError, match="coef differs"):
        check_reproduces(fit, fit.assign(coef=[0.01, 0.11]))
    with pytest.raises(ValueError, match="one side only"):
        check_reproduces(fit, fit.iloc[:1])


def _robustness_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bioname": ["SPECTER, Arlen"] * 3 + ["KENNEDY, Edward Moore", None],
            "chamber": ["S"] * 5,
            "party": ["D", "D", "R", "D", "D"],
            "date": [
                dt.date(2006, 3, 1),
                dt.date(2009, 5, 1),
                dt.date(2003, 3, 1),
                dt.date(2006, 3, 1),
                dt.date(2006, 3, 1),
            ],
            "party_switch": [False, True, None, False, None],
        }
    )


def test_specters_miscoded_speeches_are_the_democratic_ones_before_his_switch() -> None:
    # As a D before 28 April 2009: miscoded. As a D after it, or as an R: not.
    assert specter_miscoded(_robustness_frame()).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_robustness_samples() -> None:
    samples = robustness_samples(_robustness_frame())

    assert len(samples["all speeches"]) == 5
    assert len(samples["without party switchers"]) == 4
    assert len(samples["without Specter 2005–2009"]) == 4


# --- the topic model -------------------------------------------------------------------


def _topic_frame() -> pd.DataFrame:
    """House policy debates, all five blocks, 107th and 118th; two speeches per
    cell from different members, with noise that cancels.

    economic: D -0.2 throughout, R 0.2 then 0.5 (gap 0.4 -> 0.7).
    every other block: D -0.1, R 0.1 throughout (gap 0.2 -> 0.2).
    """
    means = {
        ("D", "economic"): (-0.2, -0.2),
        ("R", "economic"): (0.2, 0.5),
    }
    for block in ("social_cultural", "security_foreign", "governance", "other"):
        means[("D", block)] = (-0.1, -0.1)
        means[("R", block)] = (0.1, 0.1)
    rows = []
    for (party, block), levels in means.items():
        for congress, level in zip((107, 118), levels):
            for i, noise in enumerate((-0.01, 0.01)):
                rows.append(
                    {
                        "speech_id": f"{party}{block}{congress}{i}",
                        "chamber": "H",
                        "party": party,
                        "topic_block": block,
                        "congress_number": congress,
                        "ideology": level + noise,
                        "member_key": f"{party}{block}{congress}{i}",
                    }
                )
    return pd.DataFrame(rows)


def test_topic_cell_sizes_cover_every_cell_and_flag_small_ones() -> None:
    frame = _topic_frame()
    sizes = topic_cell_sizes(
        frame[frame["topic_block"] != "security_foreign"], minimum=3
    )

    # 2 Congresses x 1 chamber x 5 blocks x 2 parties.
    assert len(sizes) == 20
    empty = sizes[sizes["topic_block"] == "security_foreign"]
    assert (empty["n"] == 0).all() and empty["below_min"].all()
    assert (sizes[sizes["topic_block"] == "economic"]["n"] == 2).all()


def test_the_topic_model_gives_the_cell_gaps() -> None:
    fit = fit_topic_model(_topic_frame(), "H")
    gaps = topic_gaps(fit, "H").set_index(["topic_block", "congress_number"])

    assert gaps.loc[("economic", 107), "gap"] == pytest.approx(0.4)
    assert gaps.loc[("economic", 118), "gap"] == pytest.approx(0.7)
    assert gaps.loc[("governance", 118), "gap"] == pytest.approx(0.2)
    assert (gaps["ci_low"] < gaps["gap"]).all()


def test_the_topic_tests_find_which_block_widened() -> None:
    fit = fit_topic_model(_topic_frame(), "H")
    tests = topic_gap_tests(fit, "H", (107, 118)).set_index("topic_block")

    assert tests.loc["economic", "change"] == pytest.approx(0.3)
    assert tests.loc["governance", "change"] == pytest.approx(0.0, abs=1e-12)
    assert tests.loc["economic", "p"] < 0.01
    # The blocks' changes differ: four restrictions for five blocks.
    assert tests.loc["all (joint)", "df"] == 4
    assert tests.loc["all (joint)", "p"] < 0.01


def test_the_topic_tests_refuse_a_block_with_an_empty_cell() -> None:
    frame = _topic_frame()
    empty = (frame["topic_block"] == "other") & (frame["party"] == "D")
    fit = fit_topic_model(frame[~(empty & (frame["congress_number"] == 118))], "H")

    with pytest.raises(ValueError, match="D:other:118"):
        topic_gap_tests(fit, "H", (107, 118))


# --- agenda against positions -------------------------------------------------------------


def _shares_and_means(r_shares, r_means, d_shares, d_means) -> pd.DataFrame:
    rows = []
    for party, shares, means in (("R", r_shares, r_means), ("D", d_shares, d_means)):
        for block, share, mean in zip(("economic", "governance"), shares, means):
            rows.append(
                {"chamber": "H", "party": party, "topic_block": block,
                 "share": share, "mean": mean}
            )  # fmt: skip
    return pd.DataFrame(rows).set_index(["chamber", "party", "topic_block"])


def test_the_decomposition_splits_the_change_exactly() -> None:
    # R: shares .5/.5, means .2/.4 (G .3) -> shares .25/.75, means .2/.6 (G .5).
    # D: .5/.5 at -.2/-.2 throughout (G -.2). Gap .5 -> .7, change .2.
    base = _shares_and_means((0.5, 0.5), (0.2, 0.4), (0.5, 0.5), (-0.2, -0.2))
    compare = _shares_and_means((0.25, 0.75), (0.2, 0.6), (0.5, 0.5), (-0.2, -0.2))

    out = decompose_gap(base, compare).set_index(["component", "topic_block"])

    assert out.loc[("total", "all"), "value"] == pytest.approx(0.2)
    assert out.loc[("total", "all"), "gap_base"] == pytest.approx(0.5)
    assert out.loc[("total", "all"), "gap_compare"] == pytest.approx(0.7)
    # Positions: governance .625 x .2 = .125; economic did not move.
    assert out.loc[("positions", "all"), "value"] == pytest.approx(0.125)
    assert out.loc[("positions", "governance"), "value"] == pytest.approx(0.125)
    assert out.loc[("positions", "economic"), "value"] == pytest.approx(0.0)
    # Agenda: .2 x -.25 + .5 x .25 = .075. Per block, against R's centre .4:
    # economic (.2 - .4) x -.25 = .05, governance (.5 - .4) x .25 = .025.
    assert out.loc[("agenda", "all"), "value"] == pytest.approx(0.075)
    assert out.loc[("agenda", "economic"), "value"] == pytest.approx(0.05)
    assert out.loc[("agenda", "governance"), "value"] == pytest.approx(0.025)
    # A block neither period has contributes nothing.
    assert out.loc[("agenda", "other"), "value"] == pytest.approx(0.0)


def _decomposition_speeches() -> pd.DataFrame:
    """Speeches with the shares and means of the test above: R 2 economic at .2
    and 2 governance at .4 in the 107th, 1 economic at .2 and 3 governance at
    .6 in the 118th; D one of each at -.2 in both."""
    plan = {
        ("R", 107): [("economic", 0.2)] * 2 + [("governance", 0.4)] * 2,
        ("R", 118): [("economic", 0.2)] + [("governance", 0.6)] * 3,
        ("D", 107): [("economic", -0.2), ("governance", -0.2)],
        ("D", 118): [("economic", -0.2), ("governance", -0.2)],
    }
    rows = []
    for (party, congress), speeches in plan.items():
        for i, (block, value) in enumerate(speeches):
            rows.append(
                {
                    "chamber": "H",
                    "party": party,
                    "congress_number": congress,
                    "topic_block": block,
                    "ideology": value,
                    "member_key": f"{party}{i}",
                }
            )
    return pd.DataFrame(rows)


def test_block_cells_give_shares_within_each_party() -> None:
    cells = block_cells(_decomposition_speeches()).set_index(
        ["party", "congress_number", "topic_block"]
    )

    assert cells.loc[("R", 118, "governance"), "share"] == pytest.approx(0.75)
    assert cells.loc[("R", 118, "governance"), "mean"] == pytest.approx(0.6)


def test_the_decomposition_from_speeches_and_over_periods() -> None:
    speeches = _decomposition_speeches()
    # A period of two identical Congresses averages to the same values.
    repeated = pd.concat([speeches, speeches[speeches["congress_number"] == 107].assign(
        congress_number=108)])  # fmt: skip
    comparisons = {"single": ((107,), (118,)), "period": ((107, 108), (118,))}

    out = gap_decomposition(repeated, comparisons).set_index(
        ["comparison", "component", "topic_block"]
    )

    for name in comparisons:
        assert out.loc[(name, "total", "all"), "value"] == pytest.approx(0.2)
        assert out.loc[(name, "agenda", "all"), "value"] == pytest.approx(0.075)


def test_a_period_weighs_each_congress_equally() -> None:
    # R, 107th: one economic at .2, one governance at .4 (shares .5/.5).
    # R, 108th: two economic at .4 (share 1, no governance).
    # Period means of the per-Congress values: economic share .75 at (.2+.4)/2
    # = .3, governance share .25 at .4 -> G_R = .225 + .1 = .325; D -.2
    # throughout, so the base gap is .525. Pooling the speeches instead would
    # give R (.2+.4+.4+.4)/4 = .35 and a gap of .55.
    plan = [
        ("R", 107, "economic", 0.2), ("R", 107, "governance", 0.4),
        ("R", 108, "economic", 0.4), ("R", 108, "economic", 0.4),
        ("R", 118, "economic", 0.2), ("R", 118, "governance", 0.4),
        ("D", 107, "economic", -0.2), ("D", 108, "economic", -0.2),
        ("D", 118, "economic", -0.2),
    ]  # fmt: skip
    speeches = pd.DataFrame(
        plan, columns=["party", "congress_number", "topic_block", "ideology"]
    ).assign(chamber="H", member_key=lambda f: f["party"] + f.index.astype(str))

    out = gap_decomposition(speeches, {"period": ((107, 108), (118,))})
    total = out[(out["component"] == "total")].iloc[0]

    assert total["gap_base"] == pytest.approx(0.525)
    assert total["gap_compare"] == pytest.approx(0.5)


def test_the_bootstrap_draws_whole_members() -> None:
    # Every member gives exactly three speeches. If members are drawn whole,
    # every member's count in every draw is a multiple of three; drawing
    # single speeches would break that almost surely.
    frame = pd.DataFrame(
        {
            "chamber": "H",
            "party": ["D", "R"] * 30,
            "member_key": [f"m{i // 6}{i % 2}" for i in range(60)],
            "value": 1.0,
        }
    )

    def whole(f: pd.DataFrame) -> pd.DataFrame:
        counts = f.groupby("member_key").size()
        return pd.DataFrame(
            {"chamber": ["H"], "value": [float((counts % 3 == 0).all())]}
        )

    out = _bootstrap(frame, whole, ("chamber",), "value", n_draws=50, seed=7)

    assert out.loc[0, "ci_low"] == 1.0


def test_the_decomposition_bootstrap_is_reproducible() -> None:
    speeches = _decomposition_speeches()
    comparisons = {"single": ((107,), (118,))}

    first = bootstrap_gap_decomposition(speeches, comparisons, n_draws=20, seed=1)
    second = bootstrap_gap_decomposition(speeches, comparisons, n_draws=20, seed=1)

    pd.testing.assert_frame_equal(first, second)
    point = gap_decomposition(speeches, comparisons)
    assert np.allclose(first["value"], point["value"])
    assert first[["ci_low", "ci_high"]].notna().all().all()


# --- replacement against adaptation ---------------------------------------------------------


def _membership() -> pd.DataFrame:
    """A sits in the House in the 107th-109th; B leaves after the 107th; C
    enters in the 108th; E stays but speaks only in the 107th; F moves from the
    House (107th) to the Senate (108th)."""
    rows = [
        (1, 107, "H"), (1, 108, "H"), (1, 109, "H"),
        (2, 107, "H"),
        (3, 108, "H"), (3, 109, "H"),
        (5, 107, "H"), (5, 108, "H"),
        (6, 107, "H"), (6, 108, "S"),
    ]  # fmt: skip
    return pd.DataFrame(
        {
            "icpsr": pd.array([r[0] for r in rows], dtype="Int64"),
            "congress_number": [r[1] for r in rows],
            "chamber": [r[2] for r in rows],
            "party": "R",
        }
    )


def _turnover_speeches() -> pd.DataFrame:
    """House Republicans. 107th: A 2 x .2, B 2 x .4 (mean .3). 108th: A 2 x .3,
    C 2 x .6, one unmatched speech at 0 (mean 1.8/5 = .36). 109th: A and C
    1 x .4 each."""
    plan = [
        (1, 107, 0.2), (1, 107, 0.2), (2, 107, 0.4), (2, 107, 0.4),
        (1, 108, 0.3), (1, 108, 0.3), (3, 108, 0.6), (3, 108, 0.6), (None, 108, 0.0),
        (1, 109, 0.4), (3, 109, 0.4),
    ]  # fmt: skip
    return pd.DataFrame(
        {
            "icpsr": [float(m) if m is not None else np.nan for m, _, _ in plan],
            "congress_number": [c for _, c, _ in plan],
            "ideology": [v for _, _, v in plan],
            "chamber": "H",
            "party": "R",
            "member_key": [
                f"m{m}" if m is not None else "unmatched" for m, _, _ in plan
            ],
        }
    )


def test_the_turnover_identity_holds_exactly() -> None:
    steps = turnover_decomposition(_turnover_speeches(), _membership()).set_index(
        "step"
    )
    step = steps.loc["107-108"]

    # Adaptation .3 - .2 = .1; entry 2/5 x (.6 - .3) = .12;
    # exit -(2/4) x (.4 - .2) = -.1; unmatched 1/5 x (0 - .3) = -.06.
    assert step["delta"] == pytest.approx(0.06)
    assert step["adaptation"] == pytest.approx(0.1)
    assert step["entry"] == pytest.approx(0.12)
    assert step["exit"] == pytest.approx(-0.1)
    assert step["unmatched"] == pytest.approx(-0.06)
    assert step["replacement"] == pytest.approx(0.02)
    parts = steps[["adaptation", "entry", "exit", "unmatched"]].sum(axis=1)
    assert np.allclose(parts, steps["delta"])


def test_a_chamber_move_and_a_party_switch_are_exit_and_entry() -> None:
    speeches = pd.DataFrame(
        {
            # F (6) in the House, then the Senate; a switcher's old (7) and
            # new (8) ICPSR, as Voteview gives them.
            "icpsr": [6.0, 6.0, 7.0, 8.0],
            "congress_number": [107, 108, 107, 108],
            "chamber": ["H", "S", "H", "H"],
        }
    )
    membership = pd.concat(
        [
            _membership(),
            pd.DataFrame(
                {
                    "icpsr": pd.array([7, 8], dtype="Int64"),
                    "congress_number": [107, 108],
                    "chamber": ["H", "H"],
                    "party": ["D", "R"],
                }
            ),
        ]
    )

    roles = classify_turnover(speeches, membership)

    assert roles["as_earlier"].tolist()[::2] == ["exit", "exit"]
    assert roles.loc[[1, 3], "as_later"].tolist() == ["entrant", "entrant"]


def test_stayers_without_speeches_are_counted() -> None:
    counts = stayers_without_speeches(_turnover_speeches(), _membership()).set_index(
        ["chamber", "step"]
    )

    # 107 -> 108 in the House: stayers A and E; E has no speech in either.
    row = counts.loc[("H", "107-108")]
    assert row["n_stayers"] == 2
    assert row["no_speech_before"] == 1
    assert row["no_speech_after"] == 1


def test_cumulative_turnover_adds_up_to_the_whole_move() -> None:
    speeches = _turnover_speeches()
    steps = turnover_decomposition(speeches, _membership())
    windows = {"first": (107, 108), "all": (107, 109)}

    out = cumulative_turnover(steps, windows).set_index(["window", "component"])

    # 107th mean .3, 109th mean .4: the whole move is .1.
    assert out.loc[("first", "delta"), "value"] == pytest.approx(0.06)
    assert out.loc[("all", "delta"), "value"] == pytest.approx(0.1)
    parts = sum(
        out.loc[("all", c), "value"]
        for c in ("adaptation", "entry", "exit", "unmatched")
    )
    assert parts == pytest.approx(0.1)


def _many_members(party: str = "R") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twenty House members in the 107th-109th, two speeches each per Congress,
    each a little further out every Congress: enough stayers in every
    bootstrap draw, as real Congresses have."""
    members = range(1, 21)
    membership = pd.DataFrame(
        [(m, c, "H") for m in members for c in (107, 108, 109)],
        columns=["icpsr", "congress_number", "chamber"],
    ).assign(icpsr=lambda f: f["icpsr"].astype("Int64"), party=party)
    speeches = pd.DataFrame(
        [
            (float(m), c, 0.01 * m + 0.05 * (c - 107))
            for m in members
            for c in (107, 108, 109)
            for _ in range(2)
        ],
        columns=["icpsr", "congress_number", "ideology"],
    ).assign(
        chamber="H", party=party, member_key=lambda f: "m" + f["icpsr"].astype(str)
    )
    return speeches, membership


def test_the_turnover_bootstrap_is_reproducible() -> None:
    windows = {"all": (107, 109)}
    speeches, membership = _many_members()
    args = (speeches, membership, "ideology", windows, 10, 3)

    first = bootstrap_turnover(*args)

    pd.testing.assert_frame_equal(first, bootstrap_turnover(*args))
    assert {"ci_low", "ci_high"} <= set(first.columns)


def test_turnover_robustness_runs_each_speech_set() -> None:
    speeches, membership = _many_members()
    speeches = speeches.assign(
        bioname="MEMBER, A",
        date=dt.date(2003, 3, 1),
        party_switch=speeches["icpsr"] == 1.0,  # member 1's six speeches
    )
    windows = {"all": (107, 109)}

    out = turnover_robustness(speeches, membership, windows=windows)

    counts = out.groupby("sample")["n_speeches"].first()
    assert counts["all speeches"] == len(speeches)
    assert counts["without party switchers"] == len(speeches) - 6
    assert counts["without Specter 2005–2009"] == len(speeches)
    every = out[out["sample"] == "all speeches"].drop(columns=["sample", "n_speeches"])
    expected = cumulative_turnover(
        turnover_decomposition(speeches, membership), windows
    )
    pd.testing.assert_frame_equal(every.reset_index(drop=True), expected)


def test_membership_leaves_out_delegates(tmp_path) -> None:
    rows = [
        # congress, chamber, icpsr, state, party code
        (108, "House", 1, "TX", 200),
        (108, "House", 2, "DC", 100),  # a delegate, not in the corpus (D10)
        (108, "Senate", 3, "VT", 328),  # an independent: no party, still a member
        (108, "President", 4, "USA", 200),
    ]
    frame = pd.DataFrame(
        {
            "congress": [r[0] for r in rows],
            "chamber": [r[1] for r in rows],
            "icpsr": [r[2] for r in rows],
            "bioguide_id": None,
            "bioname": "X, Y",
            "state_abbrev": [r[3] for r in rows],
            "district_code": 1,
            "party_code": [r[4] for r in rows],
            "nominate_dim1": 0.0,
            "nominate_dim2": 0.0,
            "nokken_poole_dim1": 0.0,
            "nominate_number_of_votes": 100,
        }
    )
    path = tmp_path / "HSall_members.csv"
    frame.to_csv(path, index=False)

    out = chamber_membership(path).set_index("icpsr")

    assert sorted(out.index) == [1, 3]
    assert out.loc[1, "chamber"] == "H" and out.loc[1, "party"] == "R"
    assert out.loc[3, "chamber"] == "S" and pd.isna(out.loc[3, "party"])


# --- figures ------------------------------------------------------------------------------


def test_a_senate_that_changed_hands_is_split_by_days() -> None:
    spans = _majority_spans("S", 107)

    assert [party for *_, party in spans] == ["D", "R", "D"]
    assert spans[0][0] == pytest.approx(106.5) and spans[-1][1] == pytest.approx(107.5)
    # 17 of the Congress's 730 days under Gore.
    assert spans[0][1] - spans[0][0] == pytest.approx(17 / 730)
    assert _majority_spans("H", 107) == [(106.5, 107.5, "R")]


def _cells() -> pd.DataFrame:
    rows = []
    for congress in (107, 108, 109):
        for chamber in ("H", "S"):
            for party, mean in (("D", -0.3), ("R", 0.3)):
                rows.append(
                    {"congress_number": congress, "chamber": chamber, "party": party,
                     "mean": mean, "ci_low": mean - 0.05, "ci_high": mean + 0.05}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_the_figures_draw() -> None:
    models = opposition_models(_opposition_frame(), "tone")
    models = pd.concat([models, models.assign(party="D"), models.assign(chamber="S")])
    decomposition = bootstrap_gap_decomposition(
        pd.concat(
            [_decomposition_speeches(), _decomposition_speeches().assign(chamber="S")]
        ),
        {"single": ((107,), (118,))},
        n_draws=5,
    )
    republicans, membership = _many_members("R")
    democrats = republicans.assign(party="D", ideology=lambda f: -f["ideology"])
    turnover = bootstrap_turnover(
        pd.concat([republicans, democrats]),
        membership,
        windows={"2001–2003 (107th–108th)": (107, 108)},
        n_draws=5,
    )
    turnover = pd.concat([turnover, turnover.assign(chamber="S")])

    figures = [
        plot_positions_with_control(_cells(), "Ideology", "Positions", "Preliminary."),
        plot_opposition_models(models, "Opposition", "Preliminary."),
        plot_gap_decomposition(decomposition, "Decomposition", "Preliminary."),
        plot_turnover(turnover, "Turnover", "Preliminary."),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)


# =================================================================================
# Notebook 09: extensions
# =================================================================================

from src.deep_dive import (
    KEYWORD_LISTS,
    add_last_congress,
    bootstrap_common_ground,
    cohort_extremity,
    common_ground,
    congress_of,
    dissent_table,
    event_cell_sizes,
    event_effects,
    event_quarterly,
    event_windows,
    fisher_interval,
    flag_small_event_cells,
    incomplete_year,
    keyword_hits,
    keyword_precision,
    keyword_sample,
    keyword_shares,
    last_complete_congress,
    laws_gap_correlation,
    load_corpus_keyword_hits,
    moderate_exit,
    moderates,
    prior_service_switchers,
    read_keyword_check,
    source_break_year_position,
    state_margins,
    state_two_party_share,
    topic_party_change_tests,
    topic_party_values,
    white_house_changes,
    write_keyword_check,
)
from src.deep_dive_plots import (
    plot_common_ground,
    plot_event_effects,
    plot_event_quarterly,
    plot_keyword_by_opposition,
    plot_keyword_lines_by_congress,
    plot_laws_and_gap,
    plot_swing_states,
    plot_term_coefficients,
    plot_yearly_keyword_lines,
)

# --- shared helpers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        (dt.date(2003, 1, 2), 107),  # the 107th runs to 3 January 2003
        (dt.date(2003, 1, 3), 108),
        (dt.date(2025, 12, 18), 119),
    ],
)
def test_the_congress_of_a_date(date, expected) -> None:
    assert congress_of(date) == expected


def test_complete_congresses_and_years_are_read_from_the_last_date() -> None:
    # Scores end 2025-12-18: the 119th has only its first year, the 118th is complete.
    assert last_complete_congress(dt.date(2025, 12, 18)) == 118
    # Data reaching 1 December of the 119th's second year complete it.
    assert last_complete_congress(dt.date(2026, 12, 5)) == 119
    assert last_complete_congress(dt.date(2026, 9, 30)) == 118
    assert incomplete_year(dt.date(2025, 12, 18)) is None
    assert incomplete_year(dt.date(2026, 9, 30)) == 2026


# --- A. events ------------------------------------------------------------------------


def test_event_windows_stop_at_the_congress_boundary() -> None:
    windows = event_windows().set_index("event")

    # The 108th began 2003-01-03, 76 days before the Iraq War.
    assert windows.loc["Iraq War", "days"] == 76
    assert windows.loc["Iraq War", "truncated"] and windows.loc["Iraq War", "too_short"]
    # The 110th ended 2009-01-03, 110 days after Lehman: cut, but over 91 days.
    assert windows.loc["Lehman collapse", "days"] == 110
    assert windows.loc["Lehman collapse", "truncated"]
    assert not windows.loc["Lehman collapse", "too_short"]
    assert windows.loc["9/11", "days"] == 182 and not windows.loc["9/11", "truncated"]
    # Both sides equally long.
    iraq = windows.loc["Iraq War"]
    assert iraq["date"] - iraq["before_from"] == iraq["after_to"] - iraq["date"]


_EVENT = {"Test": (dt.date(2010, 3, 23), ("health",))}


def _event_frame() -> pd.DataFrame:
    """House policy debates 60 days before and after 2010-03-23; two speeches
    per cell from different members, noise +-0.01 that cancels.

    health (the event's topic): R 0.4 -> 0.6, D -0.4 -> -0.5, gap 0.8 -> 1.1.
    other topics: R 0.3, D -0.3 throughout, gap 0.6 -> 0.6.
    Tone: D health 0.2 -> 0.5, everything else 0.2 throughout.
    """
    means = {
        ("R", "health"): (0.4, 0.6),
        ("D", "health"): (-0.4, -0.5),
        ("R", "economy_tax_budget"): (0.3, 0.3),
        ("D", "economy_tax_budget"): (-0.3, -0.3),
    }
    rows = []
    for (party, topic), levels in means.items():
        for when, level in zip(("before", "after"), levels):
            day = dt.date(2010, 1, 22) if when == "before" else dt.date(2010, 5, 22)
            tone = 0.5 if (party, topic, when) == ("D", "health", "after") else 0.2
            for i, noise in enumerate((-0.01, 0.01)):
                rows.append(
                    {"speech_id": f"{party}{topic}{when}{i}", "chamber": "H",
                     "party": party, "topic": topic, "date": day,
                     "ideology": level + noise, "tone": tone + noise,
                     "member_key": f"{party}{topic}{when}{i}"}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_event_effects_split_the_event_topics_from_the_rest() -> None:
    effects = event_effects(_event_frame(), event_windows(_EVENT))
    gap = effects[effects["measure"] == "gap"].set_index("group")["estimate"]

    assert gap["event"] == pytest.approx(0.3)  # 1.1 - 0.8
    assert gap["other"] == pytest.approx(0.0, abs=1e-12)
    assert gap["event − other"] == pytest.approx(gap["event"] - gap["other"])
    tone = effects[effects["measure"] == "tone"].set_index(["party", "group"])[
        "estimate"
    ]
    assert tone[("D", "event")] == pytest.approx(0.3)
    assert tone[("D", "event − other")] == pytest.approx(0.3)
    assert tone[("R", "event")] == pytest.approx(0.0, abs=1e-12)


def test_event_cell_sizes_list_every_cell() -> None:
    sizes = event_cell_sizes(_event_frame(), event_windows(_EVENT), minimum=3)

    # 2 chambers x 2 parties x 2 periods x 2 groups; the Senate is empty.
    assert len(sizes) == 16
    house = sizes[sizes["chamber"] == "H"]
    assert (house["n"] == 2).all() and sizes["below_min"].all()
    assert (sizes[sizes["chamber"] == "S"]["n"] == 0).all()


def test_small_event_cells_are_flagged_on_the_rows_that_use_them() -> None:
    windows = event_windows(_EVENT)
    effects = event_effects(_event_frame(), windows)
    sizes = event_cell_sizes(_event_frame(), windows, minimum=1)
    # Make one cell small: D after, event topic.
    small = (
        (sizes["party"] == "D")
        & (sizes["period"] == "after")
        & (sizes["group"] == "event")
    )
    sizes.loc[small & (sizes["chamber"] == "H"), "below_min"] = True
    flagged = flag_small_event_cells(effects, sizes).set_index(
        ["measure", "party", "group"]
    )

    # The gap uses both parties; event − other uses the event group too.
    assert flagged.loc[("gap", "R−D", "event"), "small_cell"]
    assert flagged.loc[("gap", "R−D", "event − other"), "small_cell"]
    assert not flagged.loc[("gap", "R−D", "other"), "small_cell"]
    assert not flagged.loc[("tone", "R", "event"), "small_cell"]
    assert (
        flagged.loc[("tone", "D", "event"), "small_cells"]
        == "D after event: n=2, 2 members"
    )


def test_event_quarterly_bins_from_the_event_day() -> None:
    quarterly = event_quarterly(_event_frame(), _EVENT)
    ideology = quarterly[quarterly["value"] == "ideology"].set_index(
        ["party", "quarter"]
    )

    # 2010-01-22 is 60 days before: bin -1; 2010-05-22, 60 days after: bin 0.
    assert ideology.loc[("R", -1), "mean"] == pytest.approx(0.4)
    assert ideology.loc[("R", 0), "mean"] == pytest.approx(0.6)
    # The 111th began 444 days before the event and ended 286 days after it
    # (2011-01-03): the nearer boundary is its end.
    assert quarterly["congress_boundary"].iloc[0] == pytest.approx(286 / 91)


# --- B. cohorts -----------------------------------------------------------------------


def test_prior_service_switchers_are_found_by_bioguide(tmp_path) -> None:
    rows = [
        (110, "House", 1, "A000001", "SWITCHER, A"),
        (111, "House", 2, "A000001", "SWITCHER, A"),  # new ICPSR, same person
        (110, "House", 3, "B000001", "STAYER, B"),
        (111, "House", 3, "B000001", "STAYER, B"),
        (110, "President", 9, None, "PRESIDENT, P"),
    ]
    path = tmp_path / "HSall_members.csv"
    pd.DataFrame(
        rows, columns=["congress", "chamber", "icpsr", "bioguide_id", "bioname"]
    ).to_csv(path, index=False)
    switchers = prior_service_switchers(path)

    assert switchers["icpsr"].tolist() == [2]
    assert switchers.loc[0, "person_first_congress"] == 110


def test_cohort_extremity_keeps_first_congresses_only() -> None:
    members = pd.DataFrame(
        {
            "chamber": ["H"] * 6,
            "party": ["R"] * 6,
            "congress_number": [110, 110, 111, 111, 111, 106],
            "first_congress": [110, 110, 110, 111, 111, 106],
            "icpsr": [1, 2, 1, 3, 4, 5],
            "member_key": ["a", "b", "a", "c", "d", "e"],
            "extremity_vs_cell": [0.1, 0.3, 0.9, 0.2, 0.4, 0.9],
        }
    )
    cohorts = cohort_extremity(members, exclude_icpsr=[4]).set_index("cohort")

    # 110: members 1 and 2 in the 110th (member 1's 111th is not a first Congress).
    assert cohorts.loc[110, "mean"] == pytest.approx(0.2)
    # 111: member 3 only; member 4 is excluded as a switcher. 106 is before 107.
    assert cohorts.loc[111, "mean"] == pytest.approx(0.2) and cohorts.loc[111, "n"] == 1
    assert 106 not in cohorts.index


# --- 9. dissenters --------------------------------------------------------------------


def _units() -> pd.DataFrame:
    """One House Congress: D means -0.5, -0.4, 0.05 (median -0.4), R 0.5, 0.4,
    -0.05 (median 0.4); and the same members a Congress later."""
    rows = []
    for congress in (110, 111):
        for icpsr, party, mean in (
            (1, "D", -0.5), (2, "D", -0.4), (3, "D", 0.05),
            (4, "R", 0.5), (5, "R", 0.4), (6, "R", -0.05),
        ):  # fmt: skip
            rows.append({"icpsr": icpsr, "party": party, "chamber": "H",
                         "congress_number": congress, "mean": mean, "n": 10})  # fmt: skip
    return pd.DataFrame(rows)


def test_a_moderate_is_closer_to_the_other_partys_median() -> None:
    flagged = moderates(_units()).set_index(["congress_number", "icpsr"]).loc[110]

    # D at 0.05: 0.35 from the R median, 0.45 from its own -> moderate.
    assert flagged.loc[3, "moderate"] == 1 and flagged.loc[6, "moderate"] == 1
    assert flagged.loc[2, "moderate"] == 0
    # |mean| < 0.2: the robustness definition.
    assert flagged.loc[3, "centrist"] == 1 and flagged.loc[1, "centrist"] == 0


def test_moderate_exit_uses_actual_service_and_drops_the_last_congress() -> None:
    # Members 3 and 4 do not return to the House in the 111th.
    membership = pd.DataFrame(
        {"icpsr": [1, 2, 5, 6], "congress_number": [111] * 4, "chamber": ["H"] * 4}
    )
    out = moderate_exit(_units(), membership, last_congress=111).set_index("icpsr")

    assert set(out["congress_number"]) == {110}
    assert out.loc[3, "left"] == 1 and out.loc[4, "left"] == 1
    assert out.loc[1, "left"] == 0 and out.loc[6, "left"] == 0
    assert out.loc[6, "member_key"] == "icpsr:6"


def test_the_dissent_table_is_reproducible_and_brackets_its_point() -> None:
    first = dissent_table(_units(), n_draws=30, seed=1)
    second = dissent_table(_units(), n_draws=30, seed=1)

    pd.testing.assert_frame_equal(first, second)
    share = first[
        (first["statistic"] == "share_moderate") & (first["congress_number"] == 110)
    ]
    assert share["value"].tolist() == pytest.approx([1 / 3, 1 / 3])
    assert (first["ci_low"] <= first["value"] + 1e-12).all()
    assert (first["ci_high"] >= first["value"] - 1e-12).all()


# --- 4. the last Congress -------------------------------------------------------------


def test_the_last_congress_is_leaving_both_chambers() -> None:
    service = pd.DataFrame(
        {
            "icpsr": [1, 1, 2, 2, 3, 4, 5],
            "person": ["A", "A", "B", "B", "C", "C", "D"],
            "congress_number": [110, 111, 110, 111, 110, 111, 110],
            "chamber": ["H", "H", "H", "S", "H", "H", "H"],
        }
    )
    frame = pd.DataFrame(
        {
            "icpsr": [1, 1, 2, 3, 5, None],
            "congress_number": [110, 111, 110, 110, 110, 110],
        }
    )
    flags = add_last_congress(frame, service, last_congress=111)[
        "last_congress"
    ].tolist()

    # A stays; B moves to the Senate (not leaving); C switches party under a new
    # ICPSR (not leaving); D leaves; A in the 111th is the last Congress in the
    # data (unknown); no ICPSR is unknown.
    assert flags[0] == 0.0 and flags[2] == 0.0 and flags[3] == 0.0
    assert flags[4] == 1.0
    assert np.isnan(flags[1]) and np.isnan(flags[5])


# --- keyword lists --------------------------------------------------------------------


def _speeches(texts: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "speech_id": [str(i) for i in range(len(texts))],
            "party": [p for p, _ in texts],
            "text": [t for _, t in texts],
            "chamber": "H",
            "date": dt.date(2010, 1, 1),
            "bioname": "SPEAKER, A",
            "member_key": [f"m{i}" for i in range(len(texts))],
            "word_count": [len(t.split()) for _, t in texts],
        }
    )


def test_keywords_match_whole_words_and_phrases_case_ignored() -> None:
    frame = _speeches(
        [
            ("D", "This BIPARTISAN bill shows bipartisanship across  the\naisle."),
            ("D", "They compromised. Do not compromise national security."),
            ("D", "The Union of Soviet Socialist Republics; National Socialists."),
            ("R", "The far-left and the radical left want socialism."),
            ("R", "The trade deficit grew, and so did the deficit."),
        ]
    )
    hits = keyword_hits(frame)

    # bipartisan, bipartisanship, across the aisle; never "bipartisan" twice.
    assert hits.loc[0, "cooperation_hits"] == 3
    # "compromised" is another word; "compromise national security" is excluded.
    assert hits.loc[1, "cooperation_hits"] == 0
    # "Soviet Socialist", "Socialist Republics" and "National Socialists" are
    # excluded; "far-left", "radical left" and "socialism" count.
    assert hits.loc[2, "hostility_hits"] == 0
    assert hits.loc[3, "hostility_hits"] == 3
    assert hits.loc[4, "deficit_hits"] == 1
    assert set(hits.columns) >= {f"{n}_hits" for n in KEYWORD_LISTS} | {"other_party"}


def test_naming_the_other_party_depends_on_the_speaker() -> None:
    frame = _speeches(
        [
            ("D", "The Republicans and the Democrats; the Republican Party."),
            ("R", "The Republicans and the Democrats; the Democrat Party."),
            ("D", "A republican form of government and democratic institutions."),
            ("R", "My friends on the other side of the aisle disagree."),
        ]
    )
    hits = keyword_hits(frame)["other_party_hits"].tolist()

    # Own party never counts; "the other side of the aisle" counts once.
    assert hits == [2, 2, 0, 1]


def test_the_keyword_sample_is_reproducible_and_shows_the_hit(tmp_path) -> None:
    frame = _speeches(
        [("D", f"Speech {i} about a bipartisan deal.") for i in range(10)]
    )
    frame = frame.join(keyword_hits(frame))
    first = keyword_sample(frame, "cooperation", n=4, seed=3, context=10)
    second = keyword_sample(frame, "cooperation", n=4, seed=3, context=10)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4 and first["speech_id"].is_unique
    assert all(m.lower() in s.lower() for m, s in zip(first["match"], first["snippet"]))

    path = write_keyword_check(first, tmp_path / "check.csv", "cooperation")
    assert path.read_text().splitlines()[0].startswith("# correct = cooperation")
    with pytest.raises(FileExistsError):
        write_keyword_check(first, path, "cooperation")
    sheet = read_keyword_check(path).assign(correct=[1, 1, 0, 1])
    precision = keyword_precision({"cooperation": sheet})
    assert precision.loc[0, "precision"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="0 or 1"):
        keyword_precision({"cooperation": sheet.assign(correct=[1, 2, 0, 1])})


def test_corpus_hits_are_read_in_batches_without_the_text(tmp_path) -> None:
    corpus = pd.DataFrame(
        {
            "speech_id": ["1", "2", "3"],
            "date": [dt.date(2010, 1, 1)] * 3,
            "member_id": ["a", "b", "c"],
            "party": ["D", "R", "R"],
            "chamber": ["H", "H", "S"],
            "congress_number": [111, 111, 111],
            "text": ["The Republicans agree.", "A bipartisan deal.", "No hit."],
            "source": ["stanford"] * 3,
            "word_count": [3, 3, 2],
            "party_original": ["D", "R", "R"],
        }
    )
    corpus.to_parquet(tmp_path / "corpus.parquet")
    crosswalk = corpus[
        ["source", "member_id", "congress_number", "chamber", "party_original"]
    ]
    crosswalk = crosswalk.assign(icpsr=[10, None, 30])
    crosswalk.to_parquet(tmp_path / "crosswalk.parquet")

    out = load_corpus_keyword_hits(
        tmp_path / "corpus.parquet", tmp_path / "crosswalk.parquet", batch_size=2
    )

    assert "text" not in out.columns and len(out) == 3
    assert out["other_party"].tolist() == [1, 0, 0]
    assert out["cooperation"].tolist() == [0, 1, 0]
    assert out["member_key"].tolist() == ["icpsr:10", "stanford:b", "icpsr:30"]
    assert out["in_opposition"].tolist() == [False, True, True]  # Obama, 111th


def test_keyword_shares_and_rate_per_ten_thousand_words() -> None:
    frame = _speeches([("D", "a bipartisan deal " * 5), ("D", "no hit here at all")])
    shares = keyword_shares(
        frame, keyword_hits(frame), ["cooperation"], by=("chamber",)
    )

    assert shares.loc[0, "mean"] == pytest.approx(0.5)
    # 5 hits in 15 + 5 words.
    assert shares.loc[0, "per_10k_words"] == pytest.approx(1e4 * 5 / 20)


# --- 13. swing states -----------------------------------------------------------------


def test_a_countys_total_row_wins_over_its_modes(tmp_path) -> None:
    columns = ["state", "county_name", "year", "state_po", "county_fips", "office",
               "candidate", "party", "candidatevotes", "totalvotes", "version", "mode"]  # fmt: skip
    rows = [
        # County A: modes and a TOTAL row -> TOTAL only (D 60, R 40).
        ("X", "A", 2020, "XX", "1", "P", "d", "DEMOCRAT", 60, 100, 1, "TOTAL"),
        ("X", "A", 2020, "XX", "1", "P", "r", "REPUBLICAN", 40, 100, 1, "TOTAL"),
        ("X", "A", 2020, "XX", "1", "P", "d", "DEMOCRAT", 50, 100, 1, "ELECTION DAY"),
        ("X", "A", 2020, "XX", "1", "P", "r", "REPUBLICAN", 30, 100, 1, "ELECTION DAY"),
        # County B: modes only (one without a mode) -> summed (D 20, R 80).
        ("X", "B", 2020, "XX", "2", "P", "d", "DEMOCRAT", 15, 100, 1, "ELECTION DAY"),
        ("X", "B", 2020, "XX", "2", "P", "d", "DEMOCRAT", 5, 100, 1, None),
        ("X", "B", 2020, "XX", "2", "P", "r", "REPUBLICAN", 80, 100, 1, "ELECTION DAY"),
        ("X", "B", 2020, "XX", "2", "P", "o", "OTHER", 9, 100, 1, "ELECTION DAY"),
    ]
    path = tmp_path / "countypres.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    shares = state_two_party_share(path)

    assert shares.loc[0, "dem_votes"] == 80 and shares.loc[0, "rep_votes"] == 120
    assert shares.loc[0, "d_share"] == pytest.approx(0.4)


def test_the_margin_uses_the_two_elections_before_the_congress() -> None:
    shares = pd.DataFrame(
        {
            "state": ["AA", "AA", "AA", "BB", "BB", "BB", "CC", "CC", "CC"],
            "year": [2000, 2004, 2008] * 3,
            "d_share": [0.5, 0.6, 0.7, 0.3, 0.2, 0.2, 0.48, 0.52, 0.5],
        }
    )
    margins = state_margins(shares, [107, 109]).set_index(["congress_number", "state"])

    # The 109th began in 2005: 2000 and 2004.
    assert margins.loc[(109, "AA"), "margin"] == pytest.approx(0.05)
    assert margins.loc[(109, "BB"), "margin"] == pytest.approx(0.25)
    # The 107th has only 2000 in the data.
    assert margins.loc[(107, "AA"), "n_elections"] == 1
    assert margins.loc[(107, "BB"), "margin"] == pytest.approx(0.2)
    # 109th: CC 0.02 < AA 0.05 < BB 0.25.
    assert margins.loc[(109, "CC"), "tercile"] == "close"
    assert margins.loc[(109, "AA"), "tercile"] == "middle"
    assert margins.loc[(109, "BB"), "tercile"] == "safe"


# --- D, E, G, 1, 11 --------------------------------------------------------------------


def test_white_house_changes_are_read_from_the_presidents() -> None:
    assert list(white_house_changes(range(107, 120))) == [
        "110→111", "114→115", "116→117", "118→119",
    ]  # fmt: skip


def test_common_ground_on_a_grid() -> None:
    frame = pd.DataFrame(
        {
            "congress_number": 110,
            "chamber": "H",
            "party": ["D"] * 4 + ["R"] * 4,
            "ideology": [-0.5, -0.5, 0.0, 0.6, 0.0, 0.5, 0.5, 0.6],
            "member_key": [f"m{i}" for i in range(8)],
        }
    )
    ground = common_ground(frame).set_index("measure")["value"]

    # R median 0.5: one D speech (0.6) beyond it. D median -0.25: no R below.
    assert ground["d_right_of_r_median"] == pytest.approx(0.25)
    assert ground["r_left_of_d_median"] == pytest.approx(0.0)
    # Shared values: 0.0 (D 1/4, R 1/4) and 0.6 (D 1/4, R 1/4).
    assert ground["overlap"] == pytest.approx(0.5)
    boot = bootstrap_common_ground(frame, n_draws=10)
    assert {"ci_low", "ci_high"} <= set(boot.columns)


def test_each_partys_change_per_block() -> None:
    fit = fit_topic_model(_topic_frame(), "H")
    tests = topic_party_change_tests(fit, "H", (107, 118)).set_index(
        ["party", "topic_block"]
    )

    assert tests.loc[("R", "economic"), "change"] == pytest.approx(0.3)
    assert tests.loc[("D", "economic"), "change"] == pytest.approx(0.0, abs=1e-12)
    assert tests.loc[("R", "all (joint)"), "p"] < 0.01
    values = topic_party_values(fit, "H").set_index(
        ["party", "topic_block", "congress_number"]
    )
    assert values.loc[("R", "economic", 118), "mean"] == pytest.approx(0.5)


def test_the_fisher_interval() -> None:
    # z = atanh(0.5) = 0.5493, SE = 1 / sqrt(12 - 3) = 1/3.
    low, high = fisher_interval(0.5, 12)
    assert low == pytest.approx(np.tanh(0.549306 - 1.959964 / 3), abs=1e-6)
    assert high == pytest.approx(np.tanh(0.549306 + 1.959964 / 3), abs=1e-6)
    # One control costs one degree of freedom.
    assert fisher_interval(0.5, 13, controls=1) == pytest.approx((low, high))


def test_laws_and_gap_correlation_raw_and_detrended() -> None:
    congresses = np.arange(107, 119)
    wiggle = np.array([1, -1] * 6, dtype=float)
    table = pd.DataFrame(
        {
            "congress_number": congresses,
            "chamber": "H",
            "gap": 0.1 * (congresses - 107) + 0.01 * wiggle,
            "public_laws": 500 - 10 * (congresses - 107) + 5 * wiggle,
            "complete": True,
        }
    )
    table = pd.concat(
        [table, table.iloc[[-1]].assign(congress_number=119, complete=False)]
    )
    result = laws_gap_correlation(table).set_index("kind")

    # The trends run opposite; around them the two wiggle together.
    assert result.loc["raw", "r"] < -0.9
    assert result.loc["detrended", "r"] == pytest.approx(1.0)
    assert result.loc["raw", "n_congresses"] == 12


def test_the_source_break_on_a_year_axis() -> None:
    # 2016-09-10 is day 253 of 366: 2015.5 + 253/366.
    assert source_break_year_position() == pytest.approx(2015.5 + 253 / 366)


def test_the_new_figures_draw() -> None:
    effects = event_effects(
        pd.concat([_event_frame(), _event_frame().assign(chamber="S")]),
        event_windows(_EVENT),
    )
    quarterly = event_quarterly(_event_frame(), _EVENT)
    coefficients = pd.DataFrame(
        [{"chamber": c, "party": p, "term": "last_congress", "coef": 0.1, "ci_low": 0.0,
          "ci_high": 0.2, "value": v} for c in ("H", "S") for p in ("D", "R")
         for v in ("extremity", "tone")]
    )  # fmt: skip
    yearly = pd.DataFrame(
        [{"year": y, "chamber": c, "party": p, "list": n, "mean": 0.1, "ci_low": 0.05,
          "ci_high": 0.15} for y in (2015, 2016, 2017) for c in ("H", "S")
         for p in ("D", "R") for n in ("cooperation", "hostility")]
    )  # fmt: skip
    by_opposition = pd.DataFrame(
        [{"chamber": c, "party": p, "in_opposition": o, "list": "undo", "mean": 0.1,
          "ci_low": 0.05, "ci_high": 0.15} for c in ("H", "S") for p in ("D", "R")
         for o in (False, True)]
    )  # fmt: skip
    swing = pd.DataFrame(
        [{"chamber": "S", "party": p, "tercile": t, "value": v, "mean": 0.0,
          "ci_low": -0.1, "ci_high": 0.1} for p in ("D", "R")
         for t in ("close", "middle", "safe") for v in ("extremity", "tone")]
    )  # fmt: skip
    ground = pd.DataFrame(
        [{"congress_number": n, "chamber": c, "topic_block": "all", "measure": m,
          "value": 0.2, "ci_low": 0.1, "ci_high": 0.3} for n in (113, 114, 115)
         for c in ("H", "S")
         for m in ("d_right_of_r_median", "r_left_of_d_median", "overlap")]
    )  # fmt: skip
    by_congress = yearly.rename(columns={"year": "congress_number"}).assign(
        congress_number=lambda f: f["congress_number"] - 1901
    )
    laws = pd.DataFrame(
        [{"congress_number": n, "chamber": c, "gap": 0.5, "ci_low": 0.4, "ci_high": 0.6,
          "public_laws": 300, "complete": n < 115} for n in (113, 114, 115)
         for c in ("H", "S")]
    )  # fmt: skip

    figures = [
        plot_event_effects(
            effects.assign(small_cell=True), "gap", "Events", "Preliminary."
        ),
        plot_event_effects(effects, "tone", "Events", "Preliminary."),
        plot_event_quarterly(
            quarterly, "ideology", ["Test"], "Quarters", "Preliminary."
        ),
        plot_term_coefficients(coefficients, "last_congress", "Last", "Preliminary."),
        plot_yearly_keyword_lines(
            yearly, ["cooperation", "hostility"], "Words", "P.", 2017
        ),
        plot_keyword_by_opposition(by_opposition, ["undo"], "Undo", "Preliminary."),
        plot_keyword_lines_by_congress(
            by_congress, ["cooperation", "hostility"], "Words", "Preliminary."
        ),
        plot_swing_states(swing, "Swing", "Preliminary."),
        plot_common_ground(ground, ["all"], "Ground", "Preliminary."),
        plot_laws_and_gap(laws, "Laws", "Preliminary."),
    ]

    assert all(isinstance(fig, Figure) for fig in figures)
    for fig in figures:
        matplotlib.pyplot.close(fig)


# =================================================================================
# Notebook 10: designs, topics over time, Voteview
# =================================================================================

from src.deep_dive import (
    POLICY_BLOCKS,
    add_own_margin,
    bootstrap_step_decomposition,
    centrist_table,
    chamber_flip_cases,
    chamber_flip_data,
    chamber_flip_did,
    cohort_vs_continuing,
    consecutive_steps,
    cra_resolutions,
    event_block_cell_sizes,
    event_block_effects,
    flag_rollcall_undo,
    flag_small_block_cells,
    is_final_passage,
    load_rollcall_votes,
    load_rollcalls,
    measure_gap_correlation,
    names_power_effects,
    own_margin_effects,
    own_margin_within,
    party_own_paths,
    period_sums,
    policy_block_tests,
    rollcall_keyword_sample,
    rollcall_party_votes,
    rollcall_summary,
    rollcall_undo_summary,
    state_leans,
    step_decomposition,
    step_periods,
    voteview_caucus_party,
    wh_event_study,
    wh_event_study_data,
)
from src.deep_dive_plots import (
    plot_block_differences,
    plot_did_cases,
    plot_event_block_grid,
    plot_event_study,
    plot_own_margin,
    plot_rollcall_lines,
    plot_rollcall_undo,
    plot_steps_decomposition,
)

# --- 1. do the policy blocks differ? -----------------------------------------------------


def test_policy_block_tests_leave_out_other_and_adjust_the_pairs() -> None:
    fit = fit_topic_model(_topic_frame(), "H")
    tests = policy_block_tests(fit, "H", (107, 118))
    blocks = tests[tests["kind"] == "block"].set_index("topic_block")

    # economic: the gap grows 0.4 -> 0.7; every other block stays at 0.2.
    assert list(blocks.index) == list(POLICY_BLOCKS) and "other" not in blocks.index
    assert blocks.loc["economic", "change"] == pytest.approx(0.3)
    assert blocks.loc["governance", "change"] == pytest.approx(0.0, abs=1e-12)
    joint = tests[tests["kind"] == "joint"].iloc[0]
    assert joint["df"] == 3 and joint["p"] < 0.01
    pairs = tests[tests["kind"] == "pair"].set_index("topic_block")
    assert len(pairs) == 6
    assert pairs.loc["economic − social_cultural", "change"] == pytest.approx(0.3)
    # Holm never lowers a p-value.
    assert (pairs["p_holm"] >= pairs["p"] - 1e-15).all()
    tone = policy_block_tests(fit, "H", (107, 118), party="R")
    assert tone.set_index("topic_block").loc["economic", "change"] == pytest.approx(0.3)
    assert (tone["party"] == "R").all()


# --- 2. step decomposition ---------------------------------------------------------------


def _step_frame() -> pd.DataFrame:
    """House policy debates in the 107th-109th, two blocks, shares and means
    changing; each speech from its own member, several per cell."""
    rng = np.random.default_rng(3)
    rows = []
    plan = {  # (party, block): (speeches per Congress, mean per Congress)
        ("D", "economic"): ((6, 4, 5), (-0.2, -0.3, -0.35)),
        ("D", "governance"): ((4, 6, 5), (-0.1, -0.1, -0.2)),
        ("R", "economic"): ((5, 5, 6), (0.2, 0.35, 0.5)),
        ("R", "governance"): ((5, 5, 4), (0.1, 0.2, 0.2)),
    }
    for (party, block), (counts, means) in plan.items():
        for congress, n, mean in zip((107, 108, 109), counts, means):
            for i in range(n):
                rows.append({"speech_id": f"{party}{block}{congress}{i}", "chamber": "H",
                             "party": party, "topic_block": block, "congress_number": congress,
                             "ideology": mean + rng.normal(0, 0.05),
                             "member_key": f"{party}{block}{congress}{i}"})  # fmt: skip
    return pd.DataFrame(rows)


def test_the_steps_add_up_to_the_whole_change() -> None:
    frame = _step_frame()
    steps = consecutive_steps([107, 108, 109])
    out = step_decomposition(frame, steps)
    whole = gap_decomposition(frame, {"107→109": ((107,), (109,))})

    total = out[(out["component"] == "total")]["value"].sum()
    assert total == pytest.approx(whole[whole["component"] == "total"]["value"].iloc[0])
    # contribution = positions + agenda, and the blocks' contributions add to the total.
    step = out[out["comparison"] == "107→108"]
    contribution = step[step["component"] == "contribution"]["value"].sum()
    assert contribution == pytest.approx(
        step[step["component"] == "total"]["value"].iloc[0]
    )
    # within: the unweighted change of R - D inside the block.
    cells = block_cells(frame).set_index(["party", "congress_number", "topic_block"])[
        "mean"
    ]
    expected = (cells[("R", 108, "economic")] - cells[("D", 108, "economic")]) - (
        cells[("R", 107, "economic")] - cells[("D", 107, "economic")]
    )
    within = step[(step["component"] == "within") & (step["topic_block"] == "economic")]
    assert within["value"].iloc[0] == pytest.approx(expected)
    assert within_missing_is_null(step)


def within_missing_is_null(step: pd.DataFrame) -> bool:
    """A block with no speeches has no within-block change."""
    absent = step[(step["component"] == "within") & (step["topic_block"] == "other")]
    return bool(absent["value"].isna().all())


def test_step_periods_and_consecutive_steps() -> None:
    steps = consecutive_steps([109, 110, 111, 112, 114])
    assert list(steps) == ["109→110", "110→111", "111→112"]
    periods = step_periods(steps)
    assert periods["109→110"].startswith("Bush")
    assert periods["110→111"] == "switch 110→111"
    assert periods["111→112"].startswith("Obama")


def test_the_step_bootstrap_matches_the_existing_one_and_sums_periods() -> None:
    frame = _step_frame()
    steps = consecutive_steps([107, 108, 109])
    point, draws = bootstrap_step_decomposition(frame, steps, n_draws=25, seed=4)
    existing = bootstrap_gap_decomposition(frame, steps, n_draws=25, seed=4)
    key = ["comparison", "chamber", "component", "topic_block"]
    merged = existing.merge(point, on=key, suffixes=("", "_new"), validate="one_to_one")
    for column in ("value", "ci_low", "ci_high"):
        assert np.allclose(merged[column], merged[f"{column}_new"], equal_nan=True)
    assert draws["draw"].nunique() == 25

    periods = {"107→108": "first", "108→109": "second"}
    sums = period_sums(point, draws, periods, blocks=("economic", "governance"))
    totals = sums[(sums["kind"] == "sum") & (sums["component"] == "total")]
    assert totals["share_of_growth"].sum() == pytest.approx(1.0)
    joint = sums[sums["kind"] == "joint"]
    assert set(joint["component"]) == {"contribution", "within"}
    assert (joint["df"] == 1).all() and joint["p"].between(0, 1).all()


# --- 3. chamber flips --------------------------------------------------------------------


def test_chamber_flip_cases_are_read_from_the_majorities() -> None:
    cases = chamber_flip_cases(range(107, 120)).set_index("case")

    assert list(cases.index) == ["107→108", "111→112", "113→114", "115→116", "117→118"]
    assert (
        cases.loc["107→108", "status"] == "robustness"
    )  # the 107th Senate changed hands
    assert (cases.drop(index="107→108")["status"] == "main").all()
    assert (
        cases.loc["113→114", "chamber"] == "S" and cases.loc["113→114", "loses"] == "D"
    )
    assert cases.loc["115→116", "loses"] == "R" and cases.loc["115→116", "gains"] == "D"
    assert "117th" in cases.loc["117→118", "note"]


def _flip_frame() -> pd.DataFrame:
    """Democrats in the 111th and 112th, four per chamber; the House ones
    shift by 0.1 + 0.3, the Senate ones by 0.1. Member 9 moves to the Senate."""
    rng = np.random.default_rng(5)
    rows = []
    for member in range(1, 10):
        for congress in (111, 112):
            chamber = "H" if member <= 4 or (member == 9 and congress == 111) else "S"
            shift = (0.1 + (0.3 if chamber == "H" else 0.0)) * (congress == 112)
            for i in range(6):
                rows.append({"member_key": f"icpsr:{member}", "icpsr": member, "party": "D",
                             "chamber": chamber, "congress_number": congress,
                             "source": "govinfo", "extremity": 0.1 * member + shift
                             + rng.normal(0, 0.01)})  # fmt: skip
    return pd.DataFrame(rows)


def test_the_chamber_flip_did_finds_a_built_in_effect() -> None:
    frame = _flip_frame()
    served = {(m, c, "H" if m <= 4 else "S") for m in range(1, 9) for c in (111, 112)}
    served |= {(9, 111, "H"), (9, 112, "S")}
    cases = chamber_flip_cases(range(110, 114))
    data = chamber_flip_data(frame, cases, served)

    assert "icpsr:9" not in set(data["member_key"])  # changed chamber: left out
    out = chamber_flip_did(data, "extremity")
    case = out[out["kind"] == "case"].iloc[0]
    assert case["case"] == "111→112" and case["coef"] == pytest.approx(0.3, abs=0.01)
    pooled = out[out["kind"] == "pooled"].iloc[0]
    assert pooled["coef"] == pytest.approx(0.3, abs=0.01)
    own = out[out["kind"] == "own change"].set_index("chamber")
    assert own.loc["H", "coef"] == pytest.approx(0.4, abs=0.01)
    assert own.loc["S", "coef"] == pytest.approx(0.1, abs=0.01)
    everyone = chamber_flip_data(frame, cases, served, same_chamber=False)
    assert "icpsr:9" in set(everyone["member_key"])


def test_the_chamber_flip_keeps_stanford_only_in_the_mixed_congress() -> None:
    frame = pd.DataFrame(
        {
            "member_key": ["a", "a", "a", "b", "b"],
            "icpsr": [1, 1, 1, 2, 2],
            "party": "D",
            "chamber": ["S", "S", "S", "H", "H"],
            "congress_number": [113, 114, 114, 113, 114],
            "source": ["stanford", "stanford", "govinfo", "stanford", "stanford"],
            "extremity": 0.0,
        }
    )
    cases = chamber_flip_cases(range(112, 116))
    cases = cases[cases["case"] == "113→114"]
    data = chamber_flip_data(frame, cases, set(), same_chamber=False)
    assert (data["source"] == "stanford").all() and len(data) == 4
    placebo = chamber_flip_data(frame, cases, set(), same_chamber=False, placebo=True)
    assert placebo.empty or set(placebo["congress_number"]) <= {112, 113}


# --- 4. event study ------------------------------------------------------------------------


def _event_study_frame() -> pd.DataFrame:
    """113th-116th (Obama -> Trump at 115): Democrats lose the White House and
    move 0.2 further from k = 0 on; no pre-trend. Both parties share a
    Congress effect."""
    rng = np.random.default_rng(6)
    common = {113: 0.0, 114: 0.05, 115: 0.1, 116: 0.12}
    rows = []
    for chamber in ("H", "S"):
        for member in range(8):
            party = "D" if member < 4 else "R"
            for congress in (113, 114, 115, 116):
                effect = 0.2 if party == "D" and congress >= 115 else 0.0
                for i in range(5):
                    rows.append({"member_key": f"{chamber}{member}", "party": party,
                                 "chamber": chamber, "congress_number": congress,
                                 "chamber_minority": (i + member) % 2,
                                 "extremity": member * 0.05 + common[congress] + effect
                                 + rng.normal(0, 0.01)})  # fmt: skip
    return pd.DataFrame(rows)


def test_the_event_study_finds_the_effect_and_no_pre_trend() -> None:
    data = wh_event_study_data(_event_study_frame())
    assert set(data["switch"]) == {"114→115"} and set(data["k"]) == {-2, -1, 0, 1}
    assert (data.loc[data["party"] == "D", "treated"] == 1).all()

    out = wh_event_study(data, "extremity")
    effects = out[out["term"].str.startswith("treated")].set_index(["chamber", "k"])
    assert effects.loc[("H", 0), "coef"] == pytest.approx(0.2, abs=0.01)
    assert effects.loc[("S", 1), "coef"] == pytest.approx(0.2, abs=0.01)
    assert effects.loc[("H", -2), "coef"] == pytest.approx(0.0, abs=0.01)
    paths = party_own_paths(data, "extremity").set_index(["chamber", "role", "k"])
    assert paths.loc[("H", "loses White House", 0), "coef"] == pytest.approx(
        0.25, abs=0.01
    )
    assert paths.loc[("H", "gains White House", 0), "coef"] == pytest.approx(
        0.05, abs=0.01
    )


# --- 5. naming the other party -----------------------------------------------------------


def test_names_power_effects_is_model_c_on_the_flag() -> None:
    rows = []
    for member in range(6):
        for congress in (110, 111, 112):
            minority = int((member + congress) % 2 == 0)
            for i in range(4):
                rows.append({"member_key": f"m{member}", "chamber": "H", "party": "D",
                             "congress_number": congress, "opposition": float(congress == 112),
                             "chamber_minority": minority, "other_party": minority})  # fmt: skip
    out = names_power_effects(pd.DataFrame(rows)).set_index("term")

    assert set(out.index) == {"congress_number", "opposition", "chamber_minority"}
    assert out.loc["chamber_minority", "coef"] == pytest.approx(1.0)


# --- 6. own margin -----------------------------------------------------------------------


def test_own_margin_is_signed_by_the_senators_party() -> None:
    shares = pd.DataFrame(
        {
            "year": [2004, 2004, 2004, 2008, 2008, 2008],
            "state": ["AA", "BB", "CC", "AA", "BB", "CC"],
            "d_share": [0.6, 0.45, 0.5, 0.6, 0.35, 0.5],
        }
    )
    leans = state_leans(shares, [111]).set_index("state")
    assert leans.loc["AA", "d_lean"] == pytest.approx(0.1)
    assert leans.loc["BB", "d_lean"] == pytest.approx(-0.1)
    senate = pd.DataFrame(
        {
            "member_key": ["d1", "d2", "d3", "r1"],
            "party": ["D", "D", "D", "R"],
            "state": ["AA", "BB", "CC", "AA"],
            "congress_number": 111,
        }
    )
    out = add_own_margin(senate, leans.reset_index()).set_index("member_key")
    # A Democrat from AA has a stronghold; a Republican from AA does not.
    assert out.loc["d1", "own_margin"] == pytest.approx(0.1)
    assert out.loc["r1", "own_margin"] == pytest.approx(-0.1)
    assert out.loc["d2", "own_margin_10"] == pytest.approx(-1.0)
    assert (
        out.loc["d1", "own_tercile"] == "high" and out.loc["d2", "own_tercile"] == "low"
    )


def test_own_margin_within_counts_the_senators_who_vary() -> None:
    rows = []
    for member, margins in (("a", (0.1, 0.2)), ("b", (0.05, 0.05)), ("c", (-0.1, 0.0))):
        for congress, margin in zip((110, 111), margins):
            for i in range(4):
                rows.append({"member_key": member, "party": "D", "chamber": "S",
                             "congress_number": congress, "own_margin": margin,
                             "own_margin_10": 10 * margin,
                             "extremity": margin + 0.01 * i, "tone": 0.3})  # fmt: skip
    slopes, variation = own_margin_within(pd.DataFrame(rows))

    assert variation.iloc[0]["n_senators"] == 3
    assert variation.iloc[0]["n_with_variation"] == 2
    assert variation.iloc[0]["median_range_points"] == pytest.approx(10.0)
    extremity = slopes[slopes["value"] == "extremity"].iloc[0]
    assert extremity["coef"] == pytest.approx(0.1)  # 0.1 per 10 points
    means, linear = own_margin_effects(
        pd.DataFrame(rows).assign(
            own_tercile="high", extremity_vs_cell=0.0, tone_vs_cell=0.0
        )
    )
    assert set(means["value"]) == {"extremity", "tone"} and len(linear) == 2


# --- 7. topic blocks around events ----------------------------------------------------------


def _block_window_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One event on 2010-03-23, 30 days a side; House policy debates in two
    blocks. R economic rises by 0.2; D shifts from half economic to all."""
    rows = []
    day = dt.date(2010, 3, 23)
    for period, offset in (("before", -10), ("after", 10)):
        for party in ("D", "R"):
            for i in range(8):
                if party == "D":
                    block = "economic" if (period == "after" or i < 4) else "governance"
                else:
                    block = "economic" if i < 4 else "governance"
                level = 0.2 if party == "R" else -0.2
                if party == "R" and block == "economic" and period == "after":
                    level += 0.2
                rows.append({"speech_id": f"{period}{party}{i}", "chamber": "H",
                             "party": party, "member_key": f"{party}{i}",
                             "date": day + dt.timedelta(days=offset),
                             "topic": "health", "topic_block": block,
                             "ideology": level + 0.001 * i, "tone": 0.3})  # fmt: skip
    windows = pd.DataFrame(
        {"event": ["test"], "date": [day], "days": [30], "topics": ["health"]}
    )
    return pd.DataFrame(rows), windows


def test_event_block_effects_per_block_with_empty_cells_null() -> None:
    policy, windows = _block_window_frame()
    effects = event_block_effects(policy, windows).set_index(
        ["measure", "party", "topic_block"]
    )

    assert effects.loc[("agenda", "D", "economic"), "estimate"] == pytest.approx(0.5)
    assert effects.loc[("agenda", "R", "economic"), "estimate"] == pytest.approx(
        0.0, abs=1e-12
    )
    # D governance after the event is empty: its gap and tone change are null.
    assert pd.isna(effects.loc[("gap", "R−D", "governance"), "estimate"])
    assert effects.loc[("gap", "R−D", "governance"), "note"] == "no speeches in a cell"
    assert pd.isna(effects.loc[("gap", "R−D", "social_cultural"), "estimate"])
    sizes = event_block_cell_sizes(policy, windows)
    assert len(sizes) == 2 * 2 * 2 * 5
    flagged = flag_small_block_cells(event_block_effects(policy, windows), sizes)
    assert flagged["small_cell"].all()  # every cell holds under 30 speeches
    assert (
        "D before: n=8"
        in flagged[flagged["measure"] == "agenda"]["small_cells"].iloc[0]
    )


# --- 8. new against continuing members -------------------------------------------------------


def test_new_members_are_compared_with_continuing_ones() -> None:
    rows = []
    for icpsr, first, value in ((1, 108, 0.5), (2, 108, 0.5), (3, 105, 0.3), (4, 104, 0.3),
                                (5, 108, 0.9)):  # fmt: skip
        for i in range(3):
            rows.append({"icpsr": icpsr, "member_key": f"icpsr:{icpsr}", "first_congress": first,
                         "congress_number": 108, "chamber": "H", "party": "R",
                         "term": "Bush 2001-2008", "extremity": value})  # fmt: skip
    out = cohort_vs_continuing(pd.DataFrame(rows), exclude_icpsr=[5]).set_index("level")

    # New 0.5 (member 5 left out) against continuing 0.3.
    assert out.loc["cohort", "coef"] == pytest.approx(0.2)
    assert out.loc["cohort", "cohort"] == 108
    assert out.loc["pooled", "n_new_members"] == 2
    assert out.loc["presidency", "presidency"] == "Bush 2001-2008"


# --- 9. centrists ----------------------------------------------------------------------------


def test_the_centrist_table_reproduces_the_dissent_table_at_0_2() -> None:
    dissent = dissent_table(_units(), n_draws=40, seed=2)
    dissent = dissent[dissent["statistic"] == "share_centrist"]
    centrists = centrist_table(_units(), bands=(0.15, 0.2, 0.25), n_draws=40, seed=2)
    key = ["chamber", "party", "congress_number"]
    merged = dissent.merge(
        centrists[centrists["band"] == 0.2], on=key, suffixes=("", "_c")
    )
    for column in ("value", "ci_low", "ci_high"):
        assert np.allclose(merged[column], merged[f"{column}_c"])
    # |0.05| is under 0.15 and 0.25 too; |0.4| under none of them.
    wide = centrists.set_index(["party", "congress_number", "band"])["value"]
    assert wide[("D", 110, 0.15)] == pytest.approx(1 / 3)
    assert wide[("D", 110, 0.25)] == pytest.approx(1 / 3)


# --- 10. Voteview ----------------------------------------------------------------------------


def _toy_votes() -> pd.DataFrame:
    rows = []
    # roll call 1: D 3-0 yea, R 0-3 -> party unity.  2: everyone yea -> bipartisan,
    # near-unanimous.  3: D 1-1 -> no D majority.  4: D 2-1 yea only with the
    # independent (I votes nay with... yea), R nay.
    plan = {
        1: [("D", 1)] * 3 + [("R", 6)] * 3,
        2: [("D", 1)] * 3 + [("R", 1)] * 3,
        3: [("D", 1), ("D", 4), ("D", 9), ("R", 1), ("R", 1), ("R", 4)],
        4: [("D", 1), ("D", 6), ("R", 6), ("R", 6), ("R", 6)],
    }
    for roll, casts in plan.items():
        for i, (party, cast) in enumerate(casts):
            rows.append({"congress_number": 111, "chamber": "H", "rollnumber": roll,
                         "icpsr": 100 + i, "party": party, "cast_code": cast,
                         "independent": False})  # fmt: skip
    # The independent caucusing D breaks roll call 4's tie for the Democrats.
    rows.append({"congress_number": 111, "chamber": "H", "rollnumber": 4, "icpsr": 999,
                 "party": "D", "cast_code": 2, "independent": True})  # fmt: skip
    frame = pd.DataFrame(rows)
    frame["yea"] = frame["cast_code"].isin([1, 2, 3])
    frame["nay"] = frame["cast_code"].isin([4, 5, 6])
    return frame


def test_party_unity_bipartisan_and_ties() -> None:
    out = rollcall_party_votes(_toy_votes()).set_index("rollnumber")

    assert out.loc[1, "party_unity"] and not out.loc[1, "bipartisan"]
    assert out.loc[1, "r_yea"] == 0 and out.loc[1, "d_yes_share"] == 1.0
    assert out.loc[2, "bipartisan"] and out.loc[2, "near_unanimous"]
    assert out.loc[3, "no_majority"] and out.loc[3, "d_majority"] == "none"
    assert out.loc[4, "party_unity"]  # D 2-1 with the independent
    without = rollcall_party_votes(_toy_votes(), include_independents=False).set_index(
        "rollnumber"
    )
    assert without.loc[4, "no_majority"]  # D 1-1 without them
    summary = rollcall_summary(rollcall_party_votes(_toy_votes())).iloc[0]
    assert summary["n_votes"] == 4 and summary["share_party_unity"] == pytest.approx(
        0.5
    )
    assert summary["n_no_majority"] == 1


def _members_csv(tmp_path, rows) -> object:
    frame = pd.DataFrame(
        {
            "congress": [r[0] for r in rows],
            "chamber": [r[1] for r in rows],
            "icpsr": [r[2] for r in rows],
            "bioguide_id": [r[3] for r in rows],
            "bioname": [r[4] for r in rows],
            "state_abbrev": [r[5] for r in rows],
            "district_code": 1,
            "party_code": [r[6] for r in rows],
            "nominate_dim1": 0.0,
            "nominate_dim2": 0.0,
            "nokken_poole_dim1": 0.0,
            "nominate_number_of_votes": 100,
        }
    )
    path = tmp_path / "HSall_members.csv"
    frame.to_csv(path, index=False)
    return path


def test_voteview_independents_follow_the_caucus_rule(tmp_path) -> None:
    crosswalk = tmp_path / "crosswalk.parquet"
    pd.DataFrame(
        {"source": ["stanford"], "member_id": ["107113101"], "bioguide_id": ["J000072"]}
    ).to_parquet(crosswalk)
    rows = [
        (
            107,
            "Senate",
            1,
            "J000072",
            "JEFFORDS, James",
            "VT",
            328,
        ),  # CAUCUS_PARTY via crosswalk
        (
            111,
            "Senate",
            2,
            "S000033",
            "SANDERS, Bernard",
            "VT",
            328,
        ),  # GOVINFO_CAUCUS_PARTY
        (107, "Senate", 3, "B001237", "BARKLEY, Dean", "MN", 328),  # excluded (D8)
        (111, "House", 4, "X000001", "SMITH, Ann", "TX", 200),
        (111, "House", 5, "P000596", "PIERLUISI, Pedro", "PR", 328),  # delegate (D10)
    ]
    out = voteview_caucus_party(_members_csv(tmp_path, rows), crosswalk).set_index(
        "icpsr"
    )

    assert out.loc[1, "party"] == "D" and out.loc[1, "rule"] == "caucus"
    assert out.loc[2, "party"] == "D" and out.loc[2, "independent"]
    assert pd.isna(out.loc[3, "party"]) and out.loc[3, "rule"] == "excluded"
    assert out.loc[4, "party"] == "R" and out.loc[4, "rule"] == "party code"
    assert 5 not in out.index
    rows.append((112, "House", 6, "Z000009", "NOBODY, Al", "OH", 328))
    with pytest.raises(ValueError, match="NOBODY"):
        voteview_caucus_party(_members_csv(tmp_path, rows), crosswalk)


def test_rollcalls_and_votes_are_read_with_counts(tmp_path) -> None:
    calls = pd.DataFrame(
        {
            "congress": [106, 111, 111, 119],
            "chamber": ["House", "House", "Senate", "House"],
            "rollnumber": [1, 1, 1, 9],
            "date": ["1999-05-01", "2010-03-21", "2009-12-24", "2026-02-01"],
            "bill_number": ["HR1", "HR3590", "HR3590", "HR2"],
            "vote_result": "Passed",
            "vote_desc": "x",
            "vote_question": "On Passage",
            "dtl_desc": None,
            "session": 1,
        }
    )
    calls_path = tmp_path / "HSall_rollcalls.csv"
    calls.to_csv(calls_path, index=False)
    rollcalls = load_rollcalls(calls_path, last_date=dt.date(2025, 12, 31))
    assert list(rollcalls["congress_number"]) == [111, 111]
    assert set(rollcalls["chamber"]) == {"H", "S"}

    members = _members_csv(tmp_path, [
        (111, "House", 10, "A1", "A, A", "TX", 100),
        (111, "House", 11, "B1", "B, B", "TX", 200),
        (111, "President", 99, None, "OBAMA, Barack", "USA", 100),
    ])  # fmt: skip
    votes = pd.DataFrame(
        {
            "congress": [106, 111, 111, 111, 111, 111, 119],
            "chamber": ["House"] * 7,
            "rollnumber": [1, 1, 1, 1, 1, 1, 9],
            "icpsr": [10, 10, 11, 99, 12, 11, 10],
            "cast_code": [1, 2, 6, 1, 1, 9, 1],
            "prob": 50.0,
        }
    )
    votes_path = tmp_path / "HSall_votes.csv"
    votes.to_csv(votes_path, index=False)
    parties = pd.DataFrame(
        {"congress_number": [111, 111], "chamber": ["H", "H"], "icpsr": [10, 11],
         "party": ["D", "R"], "rule": ["party code"] * 2, "independent": [False, False]}
    )  # fmt: skip
    out, counts = load_rollcall_votes(
        rollcalls, parties, votes_path, members, chunksize=3
    )

    # 106th dropped on reading; the 119th roll call is after the last date.
    assert counts["rows_read"] == 6 and counts["rows_after_last_date"] == 1
    assert counts["president"] == 1 and counts["not_a_member"] == 1
    assert len(out) == 3
    assert out["yea"].tolist() == [True, False, False]  # cast 2 = paired yea
    assert out["nay"].tolist() == [False, True, False]  # cast 9 = not voting
    # A boolean after the left merge, so ~independent is a logical not.
    assert out["independent"].dtype == bool
    assert rollcall_party_votes(out, include_independents=False)["d_yea"].sum() == 1


def test_undo_votes_cra_and_final_passage() -> None:
    calls = pd.DataFrame(
        {
            "congress_number": [111, 111, 115, 115, 115, 115],
            "chamber": ["H", "S", "H", "S", "H", "S"],
            "rollnumber": [1, 2, 3, 4, 5, 6],
            "date": ["2010-01-01"] * 6,
            "bill_number": ["HR3590", "S1", "HJRES36", "SJRES34", "HRES5", "S10"],
            "vote_result": [
                "Passed",
                "Bill Passed",
                "Passed",
                "Joint Resolution Passed",
                "Agreed to",
                "Cloture Motion Agreed to",
            ],
            "vote_desc": [
                "To repeal the Affordable Care Act",
                "Rescissions of unspent funds",
                "Providing for congressional disapproval under chapter 8 of title 5, United States Code",
                "Providing for congressional disapproval under chapter 8 of title 5 of a rule",
                "Providing for consideration of a bill to overturn the rule",
                None,
            ],
            "vote_question": [
                "On Passage",
                "On Passage of the Bill",
                "On Passage",
                "On the Joint Resolution",
                "On Agreeing to the Resolution",
                "On the Cloture Motion",
            ],
            "dtl_desc": None,
        }
    )
    flagged = flag_rollcall_undo(calls, passed=("Passed", "Joint Resolution Passed"))

    assert flagged["undo"].tolist() == [True, False, True, True, True, False]
    assert flagged["cra"].tolist() == [False, False, True, True, False, False]
    assert flagged["cra_passed"].sum() == 2
    assert is_final_passage(calls).tolist() == [True, True, True, True, False, False]
    sample = rollcall_keyword_sample(flagged, n=3, seed=1)
    assert len(sample) == 3 and (sample["correct"] == "").all()
    assert sample.equals(rollcall_keyword_sample(flagged, n=3, seed=1))
    undo = rollcall_undo_summary(flagged).set_index(["congress_number", "chamber"])
    assert undo.loc[(115, "H"), "n_undo"] == 2 and undo.loc[(115, "H"), "n_cra"] == 1
    assert not undo[
        "first_after_wh_change"
    ].any()  # 111 and 115 alone are not consecutive


def test_undo_exclusions_the_wider_cra_rule_and_cra_resolutions() -> None:
    texts = [
        # congress, bill, result, description
        (116, "SJRES37", "Passed", "Providing for congressional disapproval of the proposed export to the UAE"),
        (117, "SJRES35", "Motion Rejected",
         "A joint resolution providing for congressional disapproval of the proposed foreign military sale to Egypt"),
        (116, "HRES833", "Passed", "Providing for consideration of H.Res. 826, expressing disapproval of the actions"),
        (118, "SJRES117", "Motion Rejected", "relating to the disapproval of the Presidential report on Ukraine"),
        (112, "HJRES118", "Passed",
         "Providing for congressional disapproval of the rule submitted by the Office of Family Assistance"),
        (118, "HJRES26", "Passed",
         "Disapproving the action of the District of Columbia Council in approving the Revised Criminal Code Act"),
        (118, "HJRES27", "Passed", "Disapproving the rule submitted by the Department of the Army"),
        (118, "SJRES11", "Joint Resolution Passed", "Disapproving the rule submitted by the Department of the Army"),
    ]  # fmt: skip
    calls = pd.DataFrame(
        {
            "congress_number": [t[0] for t in texts],
            "chamber": ["S", "S", "H", "S", "H", "H", "H", "S"],
            "rollnumber": range(1, len(texts) + 1),
            "bill_number": [t[1] for t in texts],
            "vote_result": [t[2] for t in texts],
            "vote_desc": [t[3] for t in texts],
            "vote_question": "On Passage",
            "dtl_desc": None,
        }
    )
    flagged = flag_rollcall_undo(calls, passed=("Passed", "Joint Resolution Passed"))

    # Arms sales, exports, mere expressions and a presidential report are no
    # longer hits; without the exclusions they were.
    assert flagged["undo"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert flag_rollcall_undo(calls, exclude=False)["undo"].iloc[:4].all()
    # CRA: joint resolutions disapproving a rule, not the DC Council's acts.
    assert flagged["cra"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        True,
    ]
    assert not flagged[
        "cra_title"
    ].any()  # the brief's title text alone misses them all
    resolutions = cra_resolutions(flagged).set_index("congress_number")
    # 118th: HJRES27 (House) and SJRES11 (Senate), both passed one chamber.
    assert resolutions.loc[118, "n_resolutions_voted"] == 2
    assert resolutions.loc[118, "n_resolutions_passed_a_chamber"] == 2
    summary = rollcall_undo_summary(flagged.assign(final_passage=True)).set_index(
        ["congress_number", "chamber"]
    )
    assert summary.loc[(118, "H"), "share_undo_final"] == pytest.approx(1.0)


def test_procedural_motions_are_not_final_passage() -> None:
    calls = pd.DataFrame(
        {
            "bill_number": ["HR1", "HR1", "HR1", "HR1", "S5", "HR2", "HR3", "HR4"],
            "vote_question": [
                "On Agreeing to the Conference Report",
                "On Motion to Recommit the Conference Report",
                "On Motion to Suspend the Rules and Pass, as Amended",
                "On Motion to Concur in the Senate Amendment with an Amendment",
                "On Overriding the Veto",
                "On Motion to Disagree to the Senate Amendment",
                "On Question of Consideration of Conference Report",
                "On Motion to Supend the Rules and Pass, as Amended",
            ],
        }
    )
    assert is_final_passage(calls).tolist() == [
        True, False, True, False, True, False, False, True
    ]  # fmt: skip


def test_a_measure_correlates_with_the_gap_raw_and_detrended() -> None:
    table = pd.DataFrame(
        {
            "congress_number": list(range(107, 113)) * 2,
            "chamber": ["H"] * 6 + ["S"] * 6,
            "gap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 2,
            "share": [0.2, 0.45, 0.55, 0.8, 1.05, 1.15] * 2,
            "complete": True,
        }
    )
    out = measure_gap_correlation(table, "share").set_index(["chamber", "kind"])
    assert out.loc[("H", "raw"), "r"] > 0.98
    assert out.loc[("H", "raw"), "ci_low"] < out.loc[("H", "raw"), "r"]
    assert out.loc[("S", "raw"), "measure"] == "share"


# --- figures ------------------------------------------------------------------------------


def test_the_notebook_10_figures_draw() -> None:
    fit = fit_topic_model(_topic_frame(), "H")
    tests = pd.concat(
        [policy_block_tests(fit, "H", (107, 118)).assign(measure="gap", comparison=c)
         for c in ("main", "control")]
        + [policy_block_tests(fit, "H", (107, 118), party=p).assign(measure="tone", comparison="main")
           for p in ("D", "R")],
        ignore_index=True,
    )  # fmt: skip
    figures = [plot_block_differences(tests, [("gap", "R−D"), ("tone", "D")], "t", "s")]

    frame = _step_frame()
    point, _ = bootstrap_step_decomposition(
        frame, consecutive_steps([107, 108, 109]), n_draws=5
    )
    point["congress_from"] = point["comparison"].str[:3].astype(int)
    figures.append(plot_steps_decomposition(point, "t", "s"))

    did = pd.DataFrame({"case": ["111→112", "pooled"] * 2, "value": ["extremity"] * 2 + ["tone"] * 2,
                        "variant": "loses", "coef": 0.1, "ci_low": 0.0, "ci_high": 0.2})  # fmt: skip
    figures.append(plot_did_cases(did, ["loses"], "t", "s"))

    data = wh_event_study_data(_event_study_frame())
    figures.append(plot_event_study(wh_event_study(data, "extremity"),
                                    party_own_paths(data, "extremity"), "t", "s"))  # fmt: skip

    means = pd.DataFrame({"value": "extremity", "party": ["D", "R"] * 3,
                          "own_tercile": ["low", "low", "middle", "middle", "high", "high"],
                          "mean": 0.0, "ci_low": -0.1, "ci_high": 0.1})  # fmt: skip
    figures.append(
        plot_own_margin(pd.concat([means, means.assign(value="tone")]), "t", "s")
    )

    policy, windows = _block_window_frame()
    effects = flag_small_block_cells(
        event_block_effects(policy, windows), event_block_cell_sizes(policy, windows)
    )
    for measure in ("gap", "tone", "agenda"):
        figures.append(plot_event_block_grid(effects, measure, "t", "s"))

    summary = pd.DataFrame({"congress_number": [110, 111] * 2, "chamber": ["H", "H", "S", "S"],
                            "variant": "all", "share_party_unity": 0.6,
                            "share_bipartisan": 0.3})  # fmt: skip
    figures.append(plot_rollcall_lines(
        summary, [("share_party_unity", "all", "unity"), ("share_bipartisan", "all", "bi")],
        "t", "s", incomplete=111))  # fmt: skip
    undo = pd.DataFrame({"congress_number": [110, 111] * 2, "chamber": ["H", "H", "S", "S"],
                         "share_undo": 0.05, "n_cra": [0, 3, 0, 4], "n_cra_passed": [0, 2, 0, 3],
                         "first_after_wh_change": [False, True] * 2})  # fmt: skip
    figures.append(plot_rollcall_undo(undo, "t", "s", incomplete=111))

    for figure in figures:
        assert isinstance(figure, Figure)
        matplotlib.pyplot.close(figure)


# =================================================================================
# Notebook 10, 4b and 4c: the power ladder, and after elections
# =================================================================================

from src.analysis import add_presidency
from src.deep_dive import (
    ELECTION_MIN_MEMBERS_BOTH,
    POWER_LEVELS,
    add_power_level,
    election_day,
    election_outcomes,
    flag_small_election_cells,
    post_election_cell_sizes,
    post_election_data,
    post_election_effects,
    post_election_windows,
    power_constellations,
    power_ladder,
    power_ladder_cells,
)
from src.deep_dive_plots import plot_post_election, plot_power_ladder

# --- 4b. the power ladder ------------------------------------------------------------------


def test_the_power_levels_follow_the_white_house_and_the_own_chamber() -> None:
    # The 107th Senate by date: D (Gore) to 20 January, R to 6 June, D after;
    # Bush is the 107th's president throughout (PRESIDENTS_BY_CONGRESS).
    dates = ["2001-01-10", "2001-03-01", "2001-07-01"]
    frame = pd.DataFrame(
        {
            "chamber": ["S"] * 6 + ["H", "H"],
            "congress_number": [107] * 6 + [111, 111],
            "date": pd.to_datetime(
                [d for d in dates for _ in "DR"] + ["2009-05-01"] * 2
            ),
            "party": ["D", "R"] * 4,
        }
    )
    out = add_power_level(add_chamber_control(add_presidency(frame)))
    assert out["power_level"].tolist() == [
        "chamber only", "White House only",  # D Senate, Bush in the White House
        "nothing", "full power",             # R Senate
        "chamber only", "White House only",  # D Senate again after 6 June
        "full power", "nothing",             # 111th House: Obama, D majority
    ]  # fmt: skip
    assert out["power_nothing"].tolist() == [0, 0, 1, 0, 0, 0, 0, 1]


def _ladder_frame() -> pd.DataFrame:
    """House Democrats whose party's power varies by member and Congress; built
    in against full power: White House only +0.1, chamber only +0.2, nothing
    +0.5 (so losing both adds 0.2 beyond the two), plus a trend of 0.01 per
    Congress."""
    rng = np.random.default_rng(11)
    effect = {(0, 0): 0.0, (0, 1): 0.1, (1, 0): 0.2, (1, 1): 0.5}
    rows = []
    for member in range(40):
        for congress in range(107, 113):
            opposition, minority = int(rng.integers(2)), int(rng.integers(2))
            for i in range(4):
                rows.append({"speech_id": f"{member}-{congress}-{i}",
                             "member_key": f"icpsr:{member}", "chamber": "H", "party": "D",
                             "congress_number": congress, "in_opposition": bool(opposition),
                             "opposition": float(opposition), "chamber_minority": minority,
                             "tone": 0.02 * member + 0.01 * congress
                             + effect[(opposition, minority)] + rng.normal(0, 0.02)})  # fmt: skip
    return pd.DataFrame(rows)


def test_the_power_ladder_finds_the_levels_and_the_interaction() -> None:
    frame = add_power_level(_ladder_frame())
    out = power_ladder(frame, "tone")

    ladder = out[out["model"] == "ladder"].set_index("power_level")["coef"]
    assert ladder["White House only"] == pytest.approx(0.1, abs=0.01)
    assert ladder["chamber only"] == pytest.approx(0.2, abs=0.01)
    assert ladder["nothing"] == pytest.approx(0.5, abs=0.01)
    interaction = out[out["model"] == "interaction"].iloc[0]
    # the same model, reparametrized: exactly nothing - White House only - chamber only
    expected = ladder["nothing"] - ladder["White House only"] - ladder["chamber only"]
    assert interaction["coef"] == pytest.approx(expected, abs=1e-10)
    assert interaction["coef"] == pytest.approx(0.2, abs=0.015)
    assert interaction["chi2"] == pytest.approx(
        (interaction["coef"] / interaction["se"]) ** 2
    )
    # the additive model is model C of notebook 08
    additive = out[out["model"] == "additive"][
        ["chamber", "party", "term", "coef", "se"]
    ]
    model_c = opposition_models(frame, "tone")
    check_reproduces(additive, model_c[model_c["model"] == "C"])


def test_the_power_ladder_cells_mark_small_and_empty_levels() -> None:
    frame = add_power_level(_ladder_frame())
    frame = frame[(frame["power_level"] != "nothing") & ~(
        (frame["power_level"] == "chamber only") & (frame["member_key"] != "icpsr:0"))]  # fmt: skip
    cells = power_ladder_cells(frame).set_index(["chamber", "party", "power_level"])

    assert len(cells) == 2 * 2 * len(POWER_LEVELS)  # every combination
    assert cells.loc[("H", "D", "nothing"), "n"] == 0
    assert bool(cells.loc[("H", "D", "nothing"), "small"])
    assert cells.loc[("H", "D", "chamber only"), "n_members"] == 1
    assert bool(cells.loc[("H", "D", "chamber only"), "small"])  # one member: under 10
    assert not bool(cells.loc[("H", "D", "full power"), "small"])
    assert cells.loc[("S", "R", "full power"), "n"] == 0


def test_the_power_constellations_read_the_president_and_both_chambers() -> None:
    table = power_constellations()
    by_congress = table.groupby("congress_number")["constellation"].agg(list)

    assert by_congress[107] == ["one chamber", "unified", "one chamber"]
    assert by_congress[110] == ["none"] and by_congress[114] == ["none"]
    assert by_congress[111] == ["unified"] and by_congress[119] == ["unified"]
    assert by_congress[117] == ["one chamber", "unified"]
    noted = table[table["note"] != ""]
    assert noted["congress_number"].tolist() == [107, 117]


# --- 4c. after elections ---------------------------------------------------------------------


def test_the_election_days_follow_the_rule() -> None:
    expected = ["2002-11-05", "2004-11-02", "2006-11-07", "2008-11-04", "2010-11-02",
                "2012-11-06", "2014-11-04", "2016-11-08", "2018-11-06", "2020-11-03",
                "2022-11-08", "2024-11-05"]  # fmt: skip
    assert [election_day(y).isoformat() for y in range(2002, 2025, 2)] == expected
    # 1 November a Tuesday: the first Monday is the 7th, so the 8th
    assert election_day(2005) == dt.date(2005, 11, 8)


def test_the_election_losers_are_read_from_the_constants() -> None:
    out = election_outcomes([2002, 2004, 2006, 2008, 2010, 2014]).set_index(
        ["election", "chamber"]
    )
    # presidential: the party that does not win the presidency, in both chambers
    assert (
        out.loc[("2004", "H"), "loser"] == "D"
        and out.loc[("2004", "S"), "loser"] == "D"
    )
    assert not out.loc[("2004", "H"), "white_house_changed"]  # a re-election
    assert (
        out.loc[("2008", "S"), "loser"] == "R"
        and out.loc[("2008", "S"), "winner"] == "D"
    )
    assert out.loc[("2008", "H"), "white_house_changed"]
    # midterm: the party that loses the chamber's majority
    assert (
        out.loc[("2006", "H"), "loser"] == "R"
        and out.loc[("2006", "S"), "loser"] == "R"
    )
    assert (
        out.loc[("2002", "S"), "loser"] == "D"
    )  # the 107th Senate ended with the Democrats
    for key in (("2002", "H"), ("2010", "S"), ("2014", "H")):
        assert out.loc[key, "status"] == "no change" and pd.isna(out.loc[key, "loser"])
    assert out.loc[("2010", "H"), "loser"] == "D"


def _days(chamber: str, dates: list[str], source: str = "stanford") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chamber": chamber,
            "day": [dt.date.fromisoformat(d) for d in dates],
            "source": source,
        }
    )


def test_the_windows_take_as_many_sitting_days_before_the_election() -> None:
    # 2010: five sitting days before, three after -> the last three before
    in_2010 = ["2010-09-13", "2010-09-14", "2010-09-15", "2010-09-28", "2010-09-29",
               "2010-11-15", "2010-11-16", "2010-12-21"]  # fmt: skip
    # 2002: a day in the 106th must not count; one day before, two after
    in_2002 = ["2000-12-01", "2002-10-17", "2002-11-12", "2002-11-13"]
    # 2016: one Stanford day before the break, two govinfo days, three after
    in_2016 = ["2016-09-12", "2016-09-13", "2016-11-14", "2016-11-15", "2016-11-16"]
    days = pd.concat(
        [
            _days("H", in_2010),
            _days("H", in_2002),
            _days("H", ["2016-09-08"]),
            _days("H", in_2016, "govinfo"),
        ]
    )
    outcomes = election_outcomes([2002, 2010, 2016])
    outcomes = outcomes[outcomes["chamber"] == "H"]
    main = post_election_windows(days, outcomes).set_index("election")
    assert main.loc["2010", "before_days"] == tuple(
        dt.date.fromisoformat(d) for d in ("2010-09-15", "2010-09-28", "2010-09-29")
    )
    assert (
        main.loc["2010", "n_days_after"] == 3 and not main.loc["2010", "short_before"]
    )
    assert main.loc["2002", "before_days"] == (dt.date(2002, 10, 17),)  # not the 106th
    assert main.loc["2002", "short_before"]
    assert main.loc["2016", "n_days_before"] == 2 and main.loc["2016", "short_before"]
    across = post_election_windows(days, outcomes, cut_at_break=False).set_index(
        "election"
    )
    assert across.loc["2016", "before_from"] == dt.date(2016, 9, 8)
    assert across.loc["2016", "n_days_before"] == 3


def _election_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """House speeches around 2006 (R loses), 2010 (D loses) and 2002 (no
    change): ten members a party, two speeches a sitting day, three days a
    side. Built in: everyone +0.1 after, the losing party another +0.3."""
    rng = np.random.default_rng(12)
    sitting = {
        "2002": (["2002-10-15", "2002-10-16", "2002-10-17"], ["2002-11-12", "2002-11-13", "2002-11-14"]),
        "2006": (["2006-09-27", "2006-09-28", "2006-09-29"], ["2006-11-13", "2006-11-14", "2006-11-15"]),
        "2010": (["2010-09-27", "2010-09-28", "2010-09-29"], ["2010-11-15", "2010-11-16", "2010-11-17"]),
    }  # fmt: skip
    outcomes = election_outcomes([2002, 2006, 2010])
    outcomes = outcomes[outcomes["chamber"] == "H"]
    losers = dict(zip(outcomes["election"], outcomes["loser"]))
    rows, days = [], []
    for election, (before, after) in sitting.items():
        days.append(_days("H", before + after))
        for day in before + after:
            post = day in after
            for member in range(20):
                party = "D" if member < 10 else "R"
                for i in range(2):
                    rows.append({"speech_id": f"{election}-{day}-{member}-{i}", "date": pd.Timestamp(day),
                                 "chamber": "H", "party": party, "member_key": f"icpsr:{member}",
                                 "tone": 0.01 * member + 0.1 * post
                                 + 0.3 * (post and party == losers[election])
                                 + rng.normal(0, 0.01)})  # fmt: skip
    windows = post_election_windows(pd.concat(days), outcomes)
    return pd.DataFrame(rows), windows, outcomes


def test_the_post_election_effects_find_a_built_in_effect() -> None:
    frame, windows, outcomes = _election_frame()
    data = post_election_data(frame, windows, outcomes)
    out = post_election_effects(data, "tone")

    per = out[out["kind"] == "election"].set_index("election")
    assert sorted(per.index) == ["2006", "2010"]  # 2002 had no change in the House
    assert per["coef"].to_numpy() == pytest.approx([0.3, 0.3], abs=0.01)
    pooled = out[out["kind"] == "pooled"].iloc[0]
    assert pooled["election_type"] == "midterm" and pooled["coef"] == pytest.approx(
        0.3, abs=0.01
    )
    trend = out[out["kind"] == "trend"].iloc[0]
    assert trend["coef"] == pytest.approx(0.0, abs=0.02)  # the same reaction both times
    own = out[out["kind"] == "own change"].set_index(["election", "party"])["coef"]
    assert own[("2010", "D")] == pytest.approx(0.4, abs=0.01)
    assert own[("2010", "R")] == pytest.approx(0.1, abs=0.01)
    assert own[("2002", "D")] == pytest.approx(0.1, abs=0.01)  # described, not compared

    sizes = post_election_cell_sizes(data, minimum=50)
    assert (sizes["n"] == 60).all() and (sizes["n_members_both"] == 10).all()
    flagged = flag_small_election_cells(out, sizes, minimum=61)
    assert flagged.loc[flagged["kind"] == "election", "small_cell"].all()
    assert not flagged.loc[flagged["kind"] == "pooled", "small_cell"].any()


def test_an_election_with_too_few_members_in_both_windows_is_left_null() -> None:
    frame, windows, outcomes = _election_frame()
    # in 2010 only one Democrat (the losers) speaks after the election
    after = frame["date"] >= pd.Timestamp("2010-11-15")
    keep = ~after | (frame["party"] == "R") | (frame["member_key"] == "icpsr:0")
    data = post_election_data(frame[keep], windows, outcomes)
    out = post_election_effects(data, "tone")
    per = out[out["kind"] == "election"].set_index("election")

    assert pd.isna(per.loc["2010", "coef"])
    assert per.loc["2010", "n_both_loser"] == 1 < ELECTION_MIN_MEMBERS_BOTH
    assert "both windows" in per.loc["2010", "note"]
    assert per.loc["2006", "coef"] == pytest.approx(0.3, abs=0.01)


def test_the_4b_and_4c_figures_draw() -> None:
    ladder = power_ladder(add_power_level(_ladder_frame()), "tone")
    frame, windows, outcomes = _election_frame()
    data = post_election_data(frame, windows, outcomes)
    effects = pd.concat(
        [post_election_effects(data, "tone"),
         post_election_effects(data.assign(extremity=data["tone"]), "extremity")],
        ignore_index=True,
    )  # fmt: skip
    effects = flag_small_election_cells(
        effects, post_election_cell_sizes(data), minimum=61
    )
    figures = [
        plot_power_ladder(ladder, "t", "s"),
        plot_post_election(effects, "t", "s"),
    ]

    for figure in figures:
        assert isinstance(figure, Figure)
        matplotlib.pyplot.close(figure)
