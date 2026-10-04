"""Tests for the analysis of a scoring run (code/src/analysis.py).

Expected values are worked out by hand in the comments, not computed with the
code under test. The run, corpus and crosswalk fixtures are tiny synthetic files
in ``tmp_path``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.analysis import (
    add_presidency,
    cell_means,
    disagreements,
    load_scored_run,
    member_means,
    party_gap,
    speech_texts,
    trend,
)
from src.crosswalk import CROSSWALK_SCHEMA
from src.merge import MERGED_SCHEMA

# --- member-clustered means ------------------------------------------------


def _frame(values, members, **cols) -> pd.DataFrame:
    n = len(values)
    base = {
        "ideology": values,
        "member_key": members,
        "congress_number": [107] * n,
        "chamber": ["H"] * n,
        "party": ["D"] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_a_cell_mean_has_a_member_clustered_standard_error() -> None:
    # y = 1, 3, 5, 7; mean 4; residuals -3, -1, 1, 3.
    # Two members (a: 1, 3; b: 5, 7): residual sums -4 and +4, squares sum 32.
    # Cluster SE of a mean = sqrt(G/(G-1) * sum e_g^2) / N = sqrt(2 * 32) / 4 = 2.
    cells = cell_means(_frame([1, 3, 5, 7], ["a", "a", "b", "b"]))

    row = cells.iloc[0]
    assert row["mean"] == pytest.approx(4.0)
    assert row["se"] == pytest.approx(2.0)
    assert (row["n"], row["n_members"]) == (4, 2)
    assert row["ci_low"] == pytest.approx(4.0 - 1.959964 * 2.0, rel=1e-4)


def test_clustering_widens_the_error_when_members_repeat() -> None:
    # Same values, four members: sum e_g^2 = 9 + 1 + 1 + 9 = 20,
    # SE = sqrt(4/3 * 20) / 4 = 1.2910.
    independent = cell_means(_frame([1, 3, 5, 7], ["a", "b", "c", "d"])).iloc[0]

    assert independent["se"] == pytest.approx(1.2910, abs=1e-4)


def test_a_cell_with_one_member_gets_no_standard_error() -> None:
    """One member is one observation, however many speeches."""
    row = cell_means(_frame([1, 3], ["a", "a"])).iloc[0]

    assert row["mean"] == pytest.approx(2.0)
    assert math.isnan(row["se"])


def test_cells_are_kept_apart_by_congress_chamber_and_party() -> None:
    frame = pd.concat(
        [
            _frame([0.1, 0.3], ["a", "b"]),
            _frame([0.5, 0.7], ["c", "d"], chamber=["S", "S"]),
            _frame([-0.5, -0.7], ["e", "f"], party=["R", "R"]),
        ]
    )

    cells = cell_means(frame).set_index(["chamber", "party"])

    assert cells.loc[("H", "D"), "mean"] == pytest.approx(0.2)
    assert cells.loc[("S", "D"), "mean"] == pytest.approx(0.6)
    assert cells.loc[("H", "R"), "mean"] == pytest.approx(-0.6)


def test_the_party_gap_is_r_minus_d_with_independent_errors() -> None:
    # Different members in each party, so the errors add in quadrature:
    # gap 0.3 - (-0.4) = 0.7, se sqrt(0.03^2 + 0.04^2) = 0.05.
    cells = pd.DataFrame(
        {
            "congress_number": [110, 110],
            "chamber": ["S", "S"],
            "party": ["R", "D"],
            "mean": [0.3, -0.4],
            "se": [0.03, 0.04],
        }
    )

    row = party_gap(cells).iloc[0]

    assert (row["congress_number"], row["chamber"]) == (110, "S")
    assert row["gap"] == pytest.approx(0.7)
    assert row["se"] == pytest.approx(0.05)
    assert row["ci_high"] == pytest.approx(0.7 + 1.959964 * 0.05, rel=1e-4)


# --- trends ------------------------------------------------------------------


def test_a_trend_recovers_a_known_slope() -> None:
    # 0.05 per Congress, two members each Congress, plus noise that cancels
    # within each Congress (+0.01 / -0.01), so the OLS slope is exactly 0.05.
    congresses = [c for c in range(107, 112) for _ in range(2)]
    values = [
        0.05 * (c - 107) + (0.01 if i % 2 else -0.01) for i, c in enumerate(congresses)
    ]
    frame = _frame(
        values, [f"m{i}" for i in range(len(values))], congress_number=congresses
    )

    row = trend(frame).iloc[0]

    assert row["slope"] == pytest.approx(0.05)
    # 107 -> 111 is four Congresses.
    assert row["change_over_span"] == pytest.approx(0.20)
    assert row["span"] == "107-111"


def test_a_trend_can_be_limited_to_a_congress_range() -> None:
    congresses = [107, 108, 109, 115, 116]
    frame = _frame([0.0, 0.1, 0.2, 0.9, 1.0], list("abcde"), congress_number=congresses)

    row = trend(frame, congress_range=(107, 114)).iloc[0]

    assert row["n"] == 3
    assert row["slope"] == pytest.approx(0.1)
    assert row["span"] == "107-109"


# --- presidency ----------------------------------------------------------------


def test_the_party_outside_the_white_house_is_in_opposition() -> None:
    # 115th: Trump (R). 111th: Obama (D).
    frame = pd.DataFrame({"congress_number": [115, 115, 111], "party": ["D", "R", "R"]})

    out = add_presidency(frame)

    assert out["president"].tolist() == ["Trump", "Trump", "Obama"]
    assert out["in_opposition"].tolist() == [True, False, True]


def test_each_presidential_term_gets_its_own_label() -> None:
    """Trump's two terms are separate terms, not one."""
    frame = pd.DataFrame(
        {"congress_number": [107, 110, 115, 116, 119], "party": ["D"] * 5}
    )

    out = add_presidency(frame)

    assert out["term"].tolist() == [
        "Bush 2001-2008",
        "Bush 2001-2008",
        "Trump 2017-2020",
        "Trump 2017-2020",
        "Trump 2025-2026",
    ]


def test_a_congress_without_a_known_president_is_an_error() -> None:
    with pytest.raises(KeyError, match="130"):
        add_presidency(pd.DataFrame({"congress_number": [130], "party": ["D"]}))


# --- members -------------------------------------------------------------------


def test_member_means_keep_only_members_with_enough_speeches() -> None:
    frame = pd.DataFrame(
        {
            "icpsr": [1] * 5 + [2] * 4,
            "party": ["D"] * 9,
            "bioname": ["A"] * 5 + ["B"] * 4,
            "ideology": [0.0, 0.1, 0.2, 0.3, 0.4, -1, -1, -1, -1],
            "nominate_dim1": [-0.3] * 5 + [-0.5] * 4,
            "nokken_poole_dim1": [-0.2, -0.2, -0.4, -0.4, -0.4, -0.5, -0.5, -0.5, -0.5],
            "in_validation": [True] * 9,
        }
    )

    members = member_means(frame, min_speeches=5)

    assert members["icpsr"].tolist() == [1]
    row = members.iloc[0]
    assert row["ideology"] == pytest.approx(0.2)
    assert row["n_speeches"] == 5
    # Nokken-Poole varies by Congress, so it is averaged over the speeches:
    # (-0.2 * 2 + -0.4 * 3) / 5 = -0.32.
    assert row["nokken_poole_dim1"] == pytest.approx(-0.32)


def test_member_means_leave_out_speeches_outside_the_validation() -> None:
    frame = pd.DataFrame(
        {
            "icpsr": [1] * 5,
            "party": ["D"] * 5,
            "bioname": ["A"] * 5,
            "ideology": [0.0] * 5,
            "nominate_dim1": [-0.3] * 5,
            "nokken_poole_dim1": [-0.3] * 5,
            "in_validation": [True] * 4 + [False],
        }
    )

    assert member_means(frame, min_speeches=5).empty


# --- disagreements -------------------------------------------------------------


def test_disagreements_are_ranked_by_the_size_of_the_gap() -> None:
    frame = pd.DataFrame(
        {
            "speech_id": ["s1", "s2", "s3"],
            "ideology_m1": [0.5, -0.4, 0.0],
            "ideology_m2": [0.4, 0.5, -0.2],
        }
    )

    top = disagreements(frame, ("m1", "m2"), top=2)

    # |gaps| = 0.1, 0.9, 0.2
    assert top["speech_id"].tolist() == ["s2", "s3"]
    assert top["gap"].tolist() == pytest.approx([-0.9, 0.2])


# --- loading a run -------------------------------------------------------------

RUN = "20261004T000000Z"


def _run_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A two-model run over three speeches by two members.

    s1, s2: member A (Stanford, 107th, ICPSR 10); s3: member B (govinfo, 115th,
    unmatched). m2 failed once on s2 and succeeded on the retry.
    """
    scores = tmp_path / "scores"
    scores.mkdir()
    manifest = {"models": {"m1": {}, "m2": {}}}
    (scores / f"s8_manifest_{RUN}.json").write_text(json.dumps(manifest))

    def row(sid, ideo, tone, error=None):
        return {
            "speech_id": sid,
            "party": "D",
            "chamber": "H",
            "congress_number": 107,
            "ideology_score": ideo,
            "tone_score": tone,
            "reasoning": f"why {sid}",
            "error": error,
        }

    files = {
        "m1": [row("s1", -0.4, 0.2), row("s2", 0.0, 0.0), row("s3", -0.6, 0.6)],
        "m2": [
            row("s1", -0.2, 0.4),
            row("s2", None, None, "CallTimeoutError"),
            row("s2", 0.0, 0.0),
            row("s3", -0.8, 0.4),
        ],
    }
    for model, rows in files.items():
        (scores / f"s8_{model}_{RUN}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    ensemble = [
        {"speech_id": "s1", "party": "D", "chamber": "H", "congress_number": 107,
         "ideology_score_mean": -0.3, "ideology_score_std": 0.14,
         "tone_score_mean": 0.3, "tone_score_std": 0.14, "n_models": 2},
        {"speech_id": "s2", "party": "D", "chamber": "H", "congress_number": 107,
         "ideology_score_mean": 0.0, "ideology_score_std": 0.0,
         "tone_score_mean": 0.0, "tone_score_std": 0.0, "n_models": 2},
        {"speech_id": "s3", "party": "D", "chamber": "H", "congress_number": 115,
         "ideology_score_mean": -0.7, "ideology_score_std": 0.14,
         "tone_score_mean": 0.5, "tone_score_std": 0.14, "n_models": 2},
    ]  # fmt: skip
    (scores / f"s8_ensemble_{RUN}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ensemble)
    )

    def speech(sid, member, source, congress, date):
        return {
            "speech_id": sid, "date": pd.Timestamp(date).date(), "member_id": member,
            "party": "D", "chamber": "H", "congress_number": congress, "text": "x",
            "source": source, "last_name": "X", "state": "AK", "word_count": 60,
            "party_original": "D", "icpsr": None,
        }  # fmt: skip

    corpus = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                speech("s1", "107000001", "stanford", 107, "2001-06-01"),
                speech("s2", "107000001", "stanford", 107, "2001-07-01"),
                speech("s3", "B000001", "govinfo", 115, "2017-06-01"),
            ],
            schema=MERGED_SCHEMA,
        ),
        corpus,
    )

    def unit(member, source, congress, icpsr):
        matched = icpsr is not None
        return {
            "source": source, "member_id": member, "congress_number": congress,
            "chamber": "H", "party_original": "D", "state": "AK", "last_name": "X",
            "n_speeches": 1, "icpsr": icpsr, "bioguide_id": None,
            "bioname": "A" if matched else None, "voteview_party_code": None,
            "nominate_dim1": -0.4 if matched else None,
            "nokken_poole_dim1": -0.35 if matched else None,
            "candidate_key": "name_exact", "resolved_by": "unique" if matched else None,
            "unmatched_reason": None if matched else "ambiguous", "n_candidates": 1,
            "party_switch": False, "has_nominate": matched, "in_validation": matched,
        }  # fmt: skip

    xwalk = tmp_path / "crosswalk.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                unit("107000001", "stanford", 107, 10),
                unit("B000001", "govinfo", 115, None),
            ],
            schema=CROSSWALK_SCHEMA,
        ),
        xwalk,
    )
    return scores, corpus, xwalk


def test_a_loaded_run_has_one_row_per_speech_with_every_models_answer(
    tmp_path: Path,
) -> None:
    scores, corpus, xwalk = _run_files(tmp_path)

    frame = load_scored_run(RUN, scores, corpus, xwalk).set_index("speech_id")

    assert len(frame) == 3
    assert frame.loc["s1", "ideology"] == pytest.approx(-0.3)
    assert frame.loc["s1", "ideology_m2"] == pytest.approx(-0.2)
    assert frame.loc["s1", "reasoning_m1"] == "why s1"
    # The retried answer counts, not the failed attempt.
    assert frame.loc["s2", "ideology_m2"] == pytest.approx(0.0)


def test_a_loaded_run_carries_speech_member_and_benchmark_details(
    tmp_path: Path,
) -> None:
    scores, corpus, xwalk = _run_files(tmp_path)

    frame = load_scored_run(RUN, scores, corpus, xwalk).set_index("speech_id")

    assert frame.loc["s3", "source"] == "govinfo"
    assert str(frame.loc["s1", "date"]) == "2001-06-01"
    assert frame.loc["s1", "nominate_dim1"] == pytest.approx(-0.4)
    assert bool(frame.loc["s1", "in_validation"]) is True
    assert bool(frame.loc["s3", "in_validation"]) is False
    # Procedural: every model at exactly 0.0.
    assert frame["procedural"].tolist() == [False, True, False]


def test_members_are_clustered_by_icpsr_and_unmatched_ones_by_source_id(
    tmp_path: Path,
) -> None:
    """ICPSR is career-long; Stanford's speakerid changes every Congress."""
    scores, corpus, xwalk = _run_files(tmp_path)

    frame = load_scored_run(RUN, scores, corpus, xwalk).set_index("speech_id")

    assert frame.loc["s1", "member_key"] == frame.loc["s2", "member_key"] == "icpsr:10"
    assert frame.loc["s3", "member_key"] == "govinfo:B000001"


def test_speech_texts_returns_only_the_requested_speeches(tmp_path: Path) -> None:
    _, corpus, _ = _run_files(tmp_path)

    assert speech_texts(["s3"], corpus) == {"s3": "x"}
