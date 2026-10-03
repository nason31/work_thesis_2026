"""Tests for a scoring run on disk: which attempt counts, what is left, resuming.

A run is extended in stages (docs/decisions.md S13), so a speech can have more
than one row per model -- a failed attempt and its retry. These tests pin down
which row counts, what still needs scoring, and when a run may not be resumed
because something that defines a score has changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.runs import (
    check_nested,
    check_resumable,
    effective_rows,
    find_label,
    latest_run,
    pending_models,
)

MODELS = ("m1", "m2", "m3")


def _row(speech_id: str, ideology: float | None, error: str | None = None) -> dict:
    return {
        "speech_id": speech_id,
        "ideology_score": ideology,
        "tone_score": None if ideology is None else 0.1,
        "error": error,
    }


# --- which attempt counts ----------------------------------------------


@pytest.mark.parametrize(
    "attempts",
    [
        [_row("s1", None, "RateLimitError"), _row("s1", 0.4)],
        [_row("s1", 0.4), _row("s1", None, "RateLimitError")],
    ],
    ids=["retry-succeeded", "success-then-stray-failure"],
)
def test_the_successful_attempt_counts_whatever_its_position(attempts) -> None:
    rows = effective_rows(attempts)

    assert rows["s1"]["ideology_score"] == 0.4
    assert rows["s1"]["error"] is None


def test_a_speech_that_never_succeeded_keeps_its_last_attempt() -> None:
    rows = effective_rows([_row("s1", None, "first"), _row("s1", None, "second")])

    assert rows["s1"]["error"] == "second"


def test_a_row_with_no_error_but_no_score_is_not_a_success() -> None:
    """A null score is a failure even if no error was recorded."""
    rows = effective_rows([_row("s1", None, None)])

    assert pending_models(["s1"], {"m1": list(rows.values())}, ("m1",)) == {
        "s1": ["m1"]
    }


# --- what is left --------------------------------------------------------


def test_pending_lists_only_the_models_without_a_successful_score() -> None:
    rows_by_model = {
        "m1": [_row("s1", 0.1), _row("s2", 0.2)],
        "m2": [_row("s1", 0.1), _row("s2", None, "TruncatedResponseError")],
        "m3": [_row("s1", 0.1)],
    }

    pending = pending_models(["s1", "s2", "s3"], rows_by_model, MODELS)

    assert pending == {"s2": ["m2", "m3"], "s3": ["m1", "m2", "m3"]}


def test_nothing_is_pending_for_a_fully_scored_run() -> None:
    rows_by_model = {m: [_row("s1", 0.0)] for m in MODELS}

    assert pending_models(["s1"], rows_by_model, MODELS) == {}


# --- when a run may be resumed -------------------------------------------


def _settings(**overrides) -> dict:
    settings = {
        "corpus_sha256": "abc",
        "prompt_template": "Score this: {speech_text}",
        "models": {"m1": {"model": "m1", "max_output_tokens": 8192}},
        "random_seed": 42,
        "min_word_count": 50,
        "stratify_by": ["congress_number", "party", "chamber"],
    }
    settings.update(overrides)
    return settings


def test_identical_settings_resume_even_when_the_sample_size_differs() -> None:
    manifest = _settings(per_cell=4, stages=[{"per_cell": 4}], timestamp="T")

    check_resumable(manifest, _settings())  # does not raise


@pytest.mark.parametrize(
    ("key", "changed"),
    [
        ("corpus_sha256", "def"),
        ("prompt_template", "A new prompt: {speech_text}"),
        ("models", {"m1": {"model": "m1", "max_output_tokens": 32768}}),
        ("random_seed", 7),
        ("min_word_count", 100),
        ("stratify_by", ["congress_number", "party"]),
    ],
)
def test_a_changed_score_defining_setting_refuses_the_resume(key, changed) -> None:
    with pytest.raises(ValueError, match=key):
        check_resumable(_settings(), _settings(**{key: changed}))


def test_a_manifest_missing_a_setting_refuses_the_resume() -> None:
    """A run from before a setting was recorded cannot prove it matches."""
    legacy = _settings()
    del legacy["stratify_by"]

    with pytest.raises(ValueError, match="stratify_by"):
        check_resumable(legacy, _settings())


def test_scored_speeches_outside_the_new_sample_refuse_the_resume() -> None:
    """A smaller or differently drawn sample would orphan rows already paid for."""
    with pytest.raises(ValueError, match="s9"):
        check_nested({"s1", "s9"}, {"s1", "s2"})


def test_a_sample_containing_every_scored_speech_is_accepted() -> None:
    check_nested({"s1"}, {"s1", "s2"})  # does not raise


# --- finding a run on disk -----------------------------------------------


def _manifest(directory: Path, label: str, run: str) -> None:
    (directory / f"{label}_manifest_{run}.json").write_text(json.dumps({}))


def test_a_runs_label_comes_from_its_manifest(tmp_path: Path) -> None:
    _manifest(tmp_path, "s8", "20261003T120000Z")
    _manifest(tmp_path, "pilot", "20260923T103556Z")

    assert find_label(tmp_path, "20261003T120000Z") == "s8"


def test_a_run_without_a_manifest_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="20261003T120000Z"):
        find_label(tmp_path, "20261003T120000Z")


def test_the_latest_run_is_chosen_by_timestamp_not_by_label(tmp_path: Path) -> None:
    """'s8' sorts after 'pilot', so a filename sort would pick the older run."""
    _manifest(tmp_path, "s8", "20261001T000000Z")
    _manifest(tmp_path, "pilot", "20261002T000000Z")

    assert latest_run(tmp_path) == "20261002T000000Z"
