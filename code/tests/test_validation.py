"""Tests for validating LLM ideology scores against DW-NOMINATE.

Synthetic run files, corpus and crosswalk, built in ``tmp_path``. The fixtures
are shaped so the answer is known by hand: perfectly linear groups (r = +1 or
-1), and rows that would wreck the correlation if the code let them in.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.crosswalk import CROSSWALK_SCHEMA
from src.merge import MERGED_SCHEMA
from src.validation import attach_benchmarks, correlate, load_run_scores

RUN = "20260923T103556Z"
MODELS = ("model-a", "model-b")


# --- fixture helpers ---------------------------------------------------


def _write_run(
    tmp_path: Path, per_model: dict[str, list[dict]], label: str = "pilot"
) -> Path:
    """A manifest and per-model JSONL files as pilot_run.py writes them."""
    (tmp_path / f"{label}_manifest_{RUN}.json").write_text("{}")
    for model, rows in per_model.items():
        path = tmp_path / f"{label}_{model}_{RUN}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def _score(speech_id: str, model: str, ideology: float | None, error=None) -> dict:
    return {
        "speech_id": speech_id,
        "party": "D",
        "congress_number": 107,
        "ideology_score": ideology,
        "tone_score": None if ideology is None else 0.1,
        "reasoning": "",
        "model": model,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "error": error,
    }


def _corpus_and_crosswalk(
    tmp_path: Path, members: list[tuple[str, str, float | None, bool]]
) -> tuple[Path, Path]:
    """One speech per member: (speech_id, party, nominate_dim1, in_validation)."""
    speeches, crosswalk = [], []
    for speech_id, party, dim1, in_validation in members:
        member_id = f"107{speech_id:0>6}"
        speeches.append(
            {
                "speech_id": speech_id,
                "date": pd.Timestamp("2001-06-01").date(),
                "member_id": member_id,
                "party": party,
                "chamber": "H",
                "congress_number": 107,
                "text": "x",
                "source": "stanford",
                "last_name": "X",
                "state": "AK",
                "word_count": 60,
                "party_original": party,
                "icpsr": None,
            }
        )
        crosswalk.append(
            {
                "source": "stanford",
                "member_id": member_id,
                "congress_number": 107,
                "chamber": "H",
                "party_original": party,
                "state": "AK",
                "last_name": "X",
                "n_speeches": 1,
                "icpsr": 1 if in_validation else None,
                "bioguide_id": None,
                "bioname": None,
                "voteview_party_code": None,
                "nominate_dim1": dim1,
                "nokken_poole_dim1": dim1,
                "candidate_key": "name_exact",
                "resolved_by": "unique" if in_validation else None,
                "unmatched_reason": None if in_validation else "ambiguous",
                "n_candidates": 1,
                "party_switch": False,
                "has_nominate": in_validation,
                "in_validation": in_validation,
            }
        )
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(speeches, schema=MERGED_SCHEMA), corpus)
    xwalk = tmp_path / "crosswalk.parquet"
    pq.write_table(pa.Table.from_pylist(crosswalk, schema=CROSSWALK_SCHEMA), xwalk)
    return corpus, xwalk


# --- loading a run -----------------------------------------------------


def test_run_scores_are_one_row_per_speech_with_a_column_per_model(
    tmp_path: Path,
) -> None:
    scores_dir = _write_run(
        tmp_path,
        {
            "model-a": [_score("s1", "model-a", -0.5), _score("s2", "model-a", 0.4)],
            "model-b": [_score("s1", "model-b", -0.3), _score("s2", "model-b", 0.8)],
        },
    )
    scores = load_run_scores(scores_dir, RUN, models=MODELS).set_index("speech_id")
    assert scores.loc["s1", "ideology_model-a"] == -0.5
    assert scores.loc["s2", "ideology_model-b"] == 0.8
    assert scores.loc["s1", "ideology_ensemble"] == pytest.approx(-0.4)
    assert scores.loc["s2", "n_models"] == 2


def test_a_failed_model_is_left_out_of_the_ensemble_mean(tmp_path: Path) -> None:
    scores_dir = _write_run(
        tmp_path,
        {
            "model-a": [_score("s1", "model-a", 0.6)],
            "model-b": [_score("s1", "model-b", None, error="timeout")],
        },
    )
    row = load_run_scores(scores_dir, RUN, models=MODELS).iloc[0]
    assert math.isnan(row["ideology_model-b"])
    assert row["ideology_ensemble"] == pytest.approx(0.6) and row["n_models"] == 1


def test_speech_every_model_scored_zero_is_procedural(tmp_path: Path) -> None:
    """The prompt tells models to score procedural speeches 0.0."""
    scores_dir = _write_run(
        tmp_path,
        {
            "model-a": [_score("s1", "model-a", 0.0), _score("s2", "model-a", 0.3)],
            "model-b": [_score("s1", "model-b", 0.0), _score("s2", "model-b", -0.3)],
        },
    )
    scores = load_run_scores(scores_dir, RUN, models=MODELS).set_index("speech_id")
    assert scores.loc["s1", "procedural"]
    # Mean 0.0 from disagreeing models is not procedural.
    assert not scores.loc["s2", "procedural"]


def test_a_retried_speech_counts_once_with_its_successful_score(
    tmp_path: Path,
) -> None:
    """A resumed run keeps the failed attempt and its retry in one file (S13)."""
    scores_dir = _write_run(
        tmp_path,
        {
            "model-a": [
                _score("s1", "model-a", None, error="RateLimitError"),
                _score("s1", "model-a", 0.6),
            ],
            "model-b": [_score("s1", "model-b", 0.2)],
        },
    )
    scores = load_run_scores(scores_dir, RUN, models=MODELS)
    assert len(scores) == 1
    row = scores.iloc[0]
    assert row["ideology_model-a"] == 0.6
    assert row["ideology_ensemble"] == pytest.approx(0.4) and row["n_models"] == 2


def test_a_run_is_read_under_its_own_label(tmp_path: Path) -> None:
    scores_dir = _write_run(
        tmp_path,
        {
            "model-a": [_score("s1", "model-a", 0.1)],
            "model-b": [_score("s1", "model-b", 0.3)],
        },
        label="s8",
    )
    row = load_run_scores(scores_dir, RUN, models=MODELS).iloc[0]
    assert row["ideology_ensemble"] == pytest.approx(0.2)


def test_a_run_is_read_with_the_models_its_manifest_lists(tmp_path: Path) -> None:
    """Old runs keep their own models after the ensemble changes (S15)."""
    scores_dir = _write_run(
        tmp_path,
        {
            "model-a": [_score("s1", "model-a", 0.1)],
            "model-b": [_score("s1", "model-b", 0.3)],
        },
    )
    manifest = {"models": {"model-a": {}, "model-b": {}}}
    (scores_dir / f"pilot_manifest_{RUN}.json").write_text(json.dumps(manifest))

    row = load_run_scores(scores_dir, RUN).iloc[0]

    assert row["n_models"] == 2
    assert row["ideology_ensemble"] == pytest.approx(0.2)


def test_missing_model_file_raises(tmp_path: Path) -> None:
    scores_dir = _write_run(tmp_path, {"model-a": [_score("s1", "model-a", 0.1)]})
    with pytest.raises(FileNotFoundError, match="model-b"):
        load_run_scores(scores_dir, RUN, models=MODELS)


# --- attaching DW-NOMINATE -----------------------------------------------


def test_benchmarks_attach_through_the_corpus_and_crosswalk(tmp_path: Path) -> None:
    corpus, xwalk = _corpus_and_crosswalk(
        tmp_path, [("s1", "D", -0.4, True), ("s2", "R", None, False)]
    )
    scores = pd.DataFrame({"speech_id": ["s1", "s2"], "ideology_ensemble": [0.1, 0.2]})
    frame = attach_benchmarks(scores, corpus=corpus, crosswalk=xwalk).set_index(
        "speech_id"
    )
    assert frame.loc["s1", "nominate_dim1"] == -0.4
    assert frame.loc["s1", "party"] == "D"
    assert bool(frame.loc["s1", "in_validation"])
    # Unmatched speeches stay in the frame, flagged, never dropped silently.
    assert not frame.loc["s2", "in_validation"]


def test_a_scored_speech_absent_from_the_corpus_raises(tmp_path: Path) -> None:
    corpus, xwalk = _corpus_and_crosswalk(tmp_path, [("s1", "D", -0.4, True)])
    scores = pd.DataFrame(
        {"speech_id": ["s1", "gone"], "ideology_ensemble": [0.1, 0.2]}
    )
    with pytest.raises(ValueError, match="gone"):
        attach_benchmarks(scores, corpus=corpus, crosswalk=xwalk)


# --- correlation ---------------------------------------------------------


def _frame(rows: list[tuple[str, float, float, bool]]) -> pd.DataFrame:
    """(party, llm score, nominate_dim1, in_validation) per speech."""
    return pd.DataFrame(
        rows, columns=["party", "ideology_ensemble", "nominate_dim1", "in_validation"]
    )


def test_rows_outside_the_validation_are_not_used() -> None:
    frame = _frame(
        [
            ("D", -0.6, -0.6, True),
            ("D", -0.4, -0.4, True),
            ("R", 0.4, 0.4, True),
            ("R", 0.6, 0.6, True),
            # Would drive r far below 1 if it were counted.
            ("R", -0.9, 0.9, False),
        ]
    )
    result = correlate(frame, "ideology_ensemble", "nominate_dim1")
    assert result["all"]["n"] == 4
    assert result["all"]["pearson_r"] == pytest.approx(1.0)


def test_within_party_correlation_is_reported_separately() -> None:
    """Parties apart overall, but inversely related within each party."""
    frame = _frame(
        [
            ("D", -0.2, -0.6, True),
            ("D", -0.4, -0.5, True),
            ("D", -0.6, -0.4, True),
            ("R", 0.6, 0.4, True),
            ("R", 0.4, 0.5, True),
            ("R", 0.2, 0.6, True),
        ]
    )
    result = correlate(frame, "ideology_ensemble", "nominate_dim1")
    assert result["all"]["pearson_r"] > 0.8
    assert result["D"]["pearson_r"] == pytest.approx(-1.0)
    assert result["R"]["spearman_rho"] == pytest.approx(-1.0)
    assert result["D"]["n"] == 3


def test_speeches_without_a_score_are_skipped() -> None:
    frame = _frame(
        [
            ("D", -0.6, -0.6, True),
            ("D", float("nan"), -0.1, True),
            ("R", 0.4, 0.4, True),
            ("R", 0.6, 0.6, True),
        ]
    )
    assert correlate(frame, "ideology_ensemble", "nominate_dim1")["all"]["n"] == 3


def test_a_group_too_small_to_correlate_gets_no_coefficient() -> None:
    frame = _frame(
        [("D", -0.6, -0.6, True), ("R", 0.4, 0.4, True), ("R", 0.6, 0.7, True)]
    )
    result = correlate(frame, "ideology_ensemble", "nominate_dim1")
    assert result["D"]["n"] == 1 and result["D"]["pearson_r"] is None
