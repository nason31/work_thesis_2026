"""Tests for the validation design comparison (O9, A5): which speeches, which
units, the within-cell statistic, the member bootstrap, reliability, and the
change diagnostic.

Expected values are worked out by hand in the comments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation_design import (
    cluster_draws,
    congress_changes,
    design_row,
    design_statistics,
    icc_oneway,
    procedural_overlap,
    selection_by_congress,
    selection_profile,
    spearman_brown,
    split_half_reliability,
    unit_table,
    validation_speeches,
)


def _speech(
    speech_id: str,
    member: int,
    party: str,
    ideology: float | None,
    congress: int = 107,
    chamber: str = "H",
    nokken_poole: float = 0.0,
    nominate: float = 0.0,
    **extra: object,
) -> dict[str, object]:
    return {
        "speech_id": speech_id,
        "member_key": f"icpsr:{member}",
        "icpsr": float(member),
        "party": party,
        "chamber": chamber,
        "congress_number": congress,
        "ideology": ideology,
        "nokken_poole_dim1": nokken_poole,
        "nominate_dim1": nominate,
        "in_validation": True,
        "procedural": False,
        "speech_type": "policy_debate",
        **extra,
    }


# --- which speeches ------------------------------------------------------------------


def _rule_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _speech("a", 1, "D", -0.3),
            _speech("b", 1, "D", 0.0, procedural=True, speech_type="procedural"),
            _speech("c", 1, "D", -0.1, speech_type="ceremonial"),
            _speech("d", 1, "D", 0.0, procedural=True),  # a policy debate at 0.0
            _speech("e", 2, "D", -0.5, in_validation=False),
            _speech("f", 1, "D", None),
            _speech("g", 1, "D", -0.2, speech_type=None),
        ]
    )


@pytest.mark.parametrize(
    ("rule", "kept"),
    [
        ("none", ["a", "b", "c", "d", "g"]),
        ("zero_score", ["a", "c", "g"]),
        ("labels", ["a", "d"]),  # A3: procedural, ceremonial and unlabelled out
    ],
)
def test_each_procedural_rule_keeps_its_speeches(rule: str, kept: list[str]) -> None:
    assert validation_speeches(_rule_frame(), rule)["speech_id"].tolist() == kept


def test_an_unknown_rule_or_party_stops_the_validation() -> None:
    with pytest.raises(ValueError, match="procedural rule"):
        validation_speeches(_rule_frame(), "fuzzy")
    with pytest.raises(ValueError, match="parties D and R"):
        validation_speeches(pd.DataFrame([_speech("a", 1, "I", 0.1)]), "none")


def test_the_two_procedural_definitions_are_compared() -> None:
    overlap = procedural_overlap(_rule_frame())

    # zero: b, d. labels drop b (procedural), c (ceremonial), g (unlabelled),
    # and f, which has no score -- its type is policy_debate, so labels keep it.
    assert overlap["dropped_by_zero_score"] == 2
    assert overlap["dropped_by_labels"] == 3
    assert overlap["dropped_by_both"] == 1  # b
    assert overlap["dropped_by_zero_score_only"] == 1  # d
    assert overlap["dropped_by_labels_only"] == 2  # c, g


# --- units ---------------------------------------------------------------------------


def _unit_speeches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # member 1: three speeches in the 107th, two in the 108th
            _speech("1a", 1, "D", -0.2, 107, nokken_poole=-0.3, nominate=-0.4),
            _speech("1b", 1, "D", -0.4, 107, nokken_poole=-0.3, nominate=-0.4),
            _speech("1c", 1, "D", -0.6, 107, nokken_poole=-0.3, nominate=-0.4),
            _speech("1d", 1, "D", -0.1, 108, nokken_poole=-0.5, nominate=-0.4),
            _speech("1e", 1, "D", -0.3, 108, nokken_poole=-0.5, nominate=-0.4),
            # member 2: one speech
            _speech("2a", 2, "D", -0.9, 107, nokken_poole=-0.8, nominate=-0.7),
        ]
    )


def test_member_congress_units_are_means_above_the_threshold() -> None:
    units = unit_table(
        _unit_speeches(), "member_congress", "nokken_poole_dim1", min_speeches=2
    )

    # member 2's single speech is below the threshold of 2
    assert units[["congress_number", "n"]].values.tolist() == [[107, 3], [108, 2]]
    assert units["score"].tolist() == pytest.approx([-0.4, -0.2])
    assert units["bench"].tolist() == pytest.approx([-0.3, -0.5])


def test_a_career_averages_nokken_poole_over_its_speeches() -> None:
    units = unit_table(_unit_speeches(), "career", "nokken_poole_dim1", min_speeches=2)

    # 3 x -0.3 and 2 x -0.5 -> -1.9 / 5 = -0.38; the score: -1.6 / 5 = -0.32
    (row,) = units.itertuples()
    assert (row.n, row.congresses) == (5, 2)
    assert row.bench == pytest.approx(-0.38)
    assert row.score == pytest.approx(-0.32)


def test_a_speech_without_a_benchmark_value_is_left_out() -> None:
    speeches = _unit_speeches()
    speeches.loc[0, "nokken_poole_dim1"] = np.nan

    units = unit_table(speeches, "member_congress", "nokken_poole_dim1", min_speeches=3)

    assert units.empty  # member 1 has two usable speeches in the 107th, not three


# --- the statistics ------------------------------------------------------------------


def _units(rows: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    """(party, cell, score, bench) -> a unit table as unit_table returns it."""
    frame = pd.DataFrame(rows, columns=["party", "group", "score", "bench"])
    return frame.assign(
        score_rel=frame["score"],
        bench_rel=frame["bench"],
        cluster=[f"m{i}" for i in range(len(frame))],
    )


def test_a_shared_party_swing_cannot_make_the_within_cell_r() -> None:
    # Two Congresses. Within each, the score orders members against their votes
    # (score = mean + 0.1 o, bench = mean - 0.1 o), but both means rise by 1
    # together: the within-party r pooled over Congresses comes out high.
    rows = [
        ("D", cell, mean + 0.1 * o, mean - 0.1 * o)
        for cell, mean in ((0, 0.0), (1, 1.0))
        for o in (-1, 0, 1)
    ]

    stats = design_statistics(_units(rows))

    assert stats["within_D_pearson"] > 0.9
    assert stats["within_cell_pearson"] == pytest.approx(-1.0)
    assert stats["within_cell_D"] == pytest.approx(-1.0)


def test_the_within_cell_r_survives_an_opposite_swing() -> None:
    # Ordering agrees within each cell, but the score rises where votes fall.
    rows = [
        ("D", cell, s_mean + 0.1 * o, b_mean + 0.1 * o)
        for cell, s_mean, b_mean in ((0, 0.0, 1.0), (1, 1.0, 0.0))
        for o in (-1, 0, 1)
    ]

    stats = design_statistics(_units(rows))

    assert stats["within_D_pearson"] < 0
    assert stats["within_cell_pearson"] == pytest.approx(1.0)


def test_the_within_cell_r_is_also_reported_per_chamber() -> None:
    # House: ordering agrees (r = 1); Senate: it is reversed (r = -1).
    rows = [("D", 0, o, o) for o in (-1.0, 0.0, 1.0)] + [
        ("D", 1, o, -o) for o in (-1.0, 0.0, 1.0)
    ]
    units = _units(rows).assign(chamber=["H"] * 3 + ["S"] * 3)

    stats = design_statistics(units)

    assert stats["within_cell_H"] == pytest.approx(1.0)
    assert stats["within_cell_S"] == pytest.approx(-1.0)
    assert stats["within_cell_pearson"] == pytest.approx(0.0)


def test_the_regression_separates_the_score_from_party() -> None:
    # bench = -0.5 + 0.5 R + 0.2 score, exactly: party leaves variance that the
    # score explains in full.
    rows = [
        (party, 0 if party == "D" else 1, s, -0.5 + 0.5 * (party == "R") + 0.2 * s)
        for party in ("D", "R")
        for s in (-1.0, 0.0, 1.0)
    ]

    stats = design_statistics(_units(rows))

    assert stats["coef_score"] == pytest.approx(0.2)
    assert stats["partial_r2"] == pytest.approx(1.0)
    # bench deviations -0.45, -0.25, -0.05 (D) and 0.05, 0.25, 0.45 (R) from
    # -0.25; the dummy's +-0.5: r = 0.75 / sqrt(1.5 x 0.535)
    assert stats["party_only_r"] == pytest.approx(0.75 / np.sqrt(1.5 * 0.535))


# --- the bootstrap ---------------------------------------------------------------------


def test_a_bootstrap_draw_keeps_members_whole_and_parties_their_size() -> None:
    cluster = np.array(["a", "a", "a", "b", "c", "c", "d"])
    party = np.array(["D", "D", "D", "D", "R", "R", "R"])

    for idx in cluster_draws(cluster, party, n_draws=50, seed=1):
        drawn = pd.Series(cluster[idx]).value_counts()
        rows_of = pd.Series(cluster).value_counts()
        times = drawn / rows_of[drawn.index]
        assert (times == times.round()).all()  # every member drawn whole
        assert times[times.index.isin(["a", "b"])].sum() == 2  # two D members
        assert times[times.index.isin(["c", "d"])].sum() == 2  # two R members


# --- reliability -----------------------------------------------------------------------


def _reliability_speeches(noise: float, seed: int = 0) -> pd.DataFrame:
    """Four D and four R members in one Congress, six speeches each."""
    rng = np.random.default_rng(seed)
    rows = []
    for member in range(8):
        party = "D" if member < 4 else "R"
        position = -0.5 + 0.1 * member if party == "D" else 0.2 + 0.1 * member
        for i in range(6):
            score = position + noise * rng.standard_normal()
            rows.append(
                _speech(f"{member}-{i}", member, party, score, nokken_poole=position)
            )
    return pd.DataFrame(rows)


def test_without_noise_a_member_mean_is_perfectly_reliable() -> None:
    rel = split_half_reliability(
        _reliability_speeches(noise=0.0), "member_congress", "nokken_poole_dim1"
    )

    assert rel == pytest.approx({"D": 1.0, "R": 1.0, "cell": 1.0})


def test_without_a_member_signal_reliability_is_near_zero() -> None:
    rng = np.random.default_rng(3)
    speeches = pd.DataFrame(
        [
            _speech(f"{m}-{i}", m, "D" if m % 2 else "R", rng.standard_normal())
            for m in range(200)
            for i in range(10)
        ]
    )

    rel = split_half_reliability(speeches, "member_congress", "nokken_poole_dim1")

    assert abs(rel["D"]) < 0.3 and abs(rel["R"]) < 0.3 and abs(rel["cell"]) < 0.25


def test_spearman_brown_steps_a_half_up_to_the_whole() -> None:
    assert spearman_brown(0.5) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("values", "groups", "expected"),
    [
        ([1, 1, 3, 3], list("aabb"), 1.0),  # no variance within
        ([0, 2, 0, 2], list("aabb"), -1.0),  # MSB 0, MSW 2, n0 2 -> -2 / 2
        ([0, 2, 4, 6], list("aabb"), 14 / 18),  # MSB 16, MSW 2 -> 14 / 18
    ],
)
def test_icc_by_hand(values: list[float], groups: list[str], expected: float) -> None:
    assert icc_oneway(np.array(values, dtype=float), groups) == pytest.approx(expected)


def test_a_design_row_reports_counts_intervals_and_disattenuation() -> None:
    row = design_row(
        _reliability_speeches(noise=0.05),
        "member_congress",
        "nokken_poole_dim1",
        min_speeches=3,
        n_draws=20,
        n_splits=5,
    )

    assert (row["n_units_D"], row["n_units_R"], row["n_speeches"]) == (4, 4, 48)
    assert row["within_D_pearson_ci_low"] <= row["within_D_pearson_ci_high"]
    expected = row["within_D_pearson"] / np.sqrt(row["rel_D"])
    assert row["disatt_D"] == pytest.approx(expected, nan_ok=True)


# --- who the frequent speakers are ---------------------------------------------------


def test_the_selection_profile_shows_who_a_threshold_keeps() -> None:
    speeches = pd.DataFrame(
        [
            _speech(f"1-{i}", 1, "D", -0.3, nokken_poole=-0.5, nominate=-0.5)
            for i in range(3)
        ]
        + [_speech("2-0", 2, "D", -0.1, nokken_poole=-0.2, nominate=-0.2)]
        + [
            _speech(f"3-{i}", 3, "R", 0.4, nokken_poole=0.6, nominate=0.6)
            for i in range(3)
        ]
    )
    background = pd.DataFrame(
        {"icpsr": [1, 2, 3], "congress_number": 107, "tenure": [5, 1, 2]}
    )
    population = pd.DataFrame(
        {
            "congress": 107,
            "chamber": "House",
            "icpsr": [1, 2, 3, 4],
            "party_code": [100, 100, 200, 100],
            "nokken_poole_dim1": [-0.5, -0.2, 0.6, -0.3],
            "nominate_dim1": [-0.5, -0.2, 0.6, -0.3],
        }
    )

    profile = selection_profile(speeches, (3,), background, population)
    mc = profile[profile["unit"] == "member_congress"].set_index(["threshold", "party"])

    assert mc.loc[(1, "D"), "units"] == 2
    assert mc.loc[(3, "D"), "units"] == 1  # member 2 has one speech
    assert mc.loc[(3, "D"), "speech_share"] == pytest.approx(3 / 4)
    assert mc.loc[(3, "D"), "tenure_mean"] == 5
    assert mc.loc[(3, "D"), "bench_toward_party"] == pytest.approx(0.5)
    assert mc.loc[("voteview", "D"), "units"] == 3  # members 1, 2 and 4
    assert mc.loc[("voteview", "D"), "bench_toward_party"] == pytest.approx(1.0 / 3)


def test_the_selection_by_congress_shows_which_cells_a_threshold_empties() -> None:
    speeches = pd.DataFrame(
        [_speech(f"1-{i}", 1, "D", -0.3, 107, nokken_poole=-0.5) for i in range(3)]
        + [_speech("2-0", 2, "D", -0.1, 107, nokken_poole=-0.2)]
        + [_speech(f"3-{i}", 3, "D", -0.2, 108, nokken_poole=-0.4) for i in range(3)]
    )
    population = pd.DataFrame(
        {
            "congress": [107, 107, 107, 108, 108, 107],
            "chamber": "House",
            "icpsr": [1, 2, 4, 3, 5, 6],
            "party_code": [100, 100, 100, 100, 100, 200],
            "nokken_poole_dim1": [-0.5, -0.2, -0.3, -0.4, -0.6, 0.5],
        }
    )

    table = selection_by_congress(speeches, (3,), population).set_index(
        ["threshold", "congress_number", "party"]
    )

    # Every sampled member-Congress: members 1 and 2 of Voteview's 1, 2 and 4.
    assert table.loc[(1, 107, "D"), "units"] == 2
    assert table.loc[(1, 107, "D"), "coverage"] == pytest.approx(2 / 3)
    assert table.loc[(1, 107, "D"), "toward_kept"] == pytest.approx(0.35)
    assert table.loc[(1, 107, "D"), "toward_all"] == pytest.approx(1 / 3)
    # At 3 speeches member 2 drops out; the 108th keeps member 3 of 3 and 5.
    assert table.loc[(3, 107, "D"), "units"] == 1
    assert table.loc[(3, 108, "D"), "coverage"] == pytest.approx(0.5)
    # A cell with no speaker stays in the table, empty.
    assert table.loc[(3, 107, "R"), "units"] == 0
    assert np.isnan(table.loc[(3, 107, "R"), "toward_kept"])


# --- changes between Congresses ------------------------------------------------------


def test_changes_pair_consecutive_congresses_in_one_chamber_only() -> None:
    units = pd.DataFrame(
        {
            "cluster": ["m|D"] * 3 + ["n|D"] * 2,
            "party": "D",
            "chamber": ["H", "H", "H", "S", "H"],
            "congress_number": [107, 108, 110, 108, 109],
            "score": [-0.2, -0.5, -0.6, -0.1, -0.3],
            "bench": [-0.3, -0.4, -0.4, -0.2, -0.2],
            "group": [0, 1, 2, 3, 4],
        }
    )
    units = units.assign(score_rel=units["score"], bench_rel=units["bench"])

    changes = congress_changes(units)

    # m: 107 -> 108 only (110 does not follow 108); n changed chamber
    assert len(changes) == 1
    row = changes.iloc[0]
    assert [row["congress_number"], row["d_score"], row["d_bench"]] == pytest.approx(
        [108, -0.3, -0.1]
    )


def test_a_shared_move_drops_out_of_the_within_cell_change() -> None:
    # Both members' scores rise by 0.3 from the 107th to the 108th, as their
    # party's did: no change relative to the cell.
    units = pd.DataFrame(
        {
            "cluster": ["m|D", "m|D", "n|D", "n|D"],
            "party": "D",
            "chamber": "H",
            "congress_number": [107, 108, 107, 108],
            "score": [-0.6, -0.3, -0.4, -0.1],
            "bench": [-0.5, -0.5, -0.3, -0.3],
            "group": [0, 1, 0, 1],
        }
    )
    units = units.assign(score_rel=units["score"], bench_rel=units["bench"])

    changes = congress_changes(units)

    assert changes["d_score"].tolist() == pytest.approx([0.3, 0.3])
    assert changes["d_score_within"].tolist() == pytest.approx([0.0, 0.0])
