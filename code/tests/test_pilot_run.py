"""Tests for running the scoring in stages: label, resume, cost, summary.

The S8 run is one run extended in stages (docs/decisions.md S13): 4 speeches per
cell first, then the same run resumed at 100 per cell. These tests run
`pilot_run.main` end to end on a tiny synthetic corpus. Only the provider call
(`score_one`), the client construction and the tokenizer are replaced -- the
sampling, file writing, resume checks, ensemble and summary are the real ones.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pilot_run
from pilot_run import preflight, summarize

from src.config import ENSEMBLE_MODELS
from src.scoring import ScoreResult

DEEPSEEK, GPT = ENSEMBLE_MODELS


# --- cost of what is left ----------------------------------------------


def test_preflight_prices_only_the_calls_still_pending() -> None:
    estimate = preflight(
        token_counts={"s1": 900, "s2": 900, "s3": 900},
        pending={"s1": [DEEPSEEK, GPT], "s2": [DEEPSEEK]},
        models=ENSEMBLE_MODELS,
    )

    assert estimate["per_model"][DEEPSEEK]["speeches"] == 2
    assert estimate["per_model"][GPT]["speeches"] == 1
    assert estimate["speeches"] == 2 and estimate["calls"] == 3
    assert estimate["estimated_total_usd"] == pytest.approx(
        sum(m["cost_usd"] for m in estimate["per_model"].values())
    )


def test_a_model_with_nothing_pending_costs_nothing() -> None:
    estimate = preflight({"s1": 900}, {"s1": [DEEPSEEK]}, ENSEMBLE_MODELS)

    assert estimate["per_model"][GPT]["speeches"] == 0
    assert estimate["per_model"][GPT]["cost_usd"] == 0


def test_preflight_scales_with_the_length_of_the_pending_speeches() -> None:
    short = preflight({"s1": 900, "s2": 1800}, {"s1": [DEEPSEEK]}, ENSEMBLE_MODELS)
    long = preflight({"s1": 900, "s2": 1800}, {"s2": [DEEPSEEK]}, ENSEMBLE_MODELS)

    ratio = (
        long["per_model"][DEEPSEEK]["prompt_tokens"]
        / short["per_model"][DEEPSEEK]["prompt_tokens"]
    )
    assert ratio == pytest.approx(2.0, rel=0.01)


# --- summary over every attempt ------------------------------------------


def _file_row(speech_id, ideology, stage, prompt_tokens, completion_tokens, error=None):
    return {
        "speech_id": speech_id,
        "ideology_score": ideology,
        "tone_score": None if ideology is None else 0.1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
        "stage": stage,
    }


def test_summary_bills_every_attempt_but_counts_a_retried_failure_once() -> None:
    rows_by_model = {
        DEEPSEEK: [_file_row("s1", -0.2, 1, 100, 400)],
        GPT: [
            _file_row("s1", None, 1, 100, 0, error="RateLimitError: slow down"),
            _file_row("s1", -0.3, 2, 100, 50),
        ],
    }
    ensemble = [
        {
            "speech_id": "s1",
            "party": "D",
            "ideology_score_mean": -0.25,
            "ideology_score_std": 0.07,
            "n_models": 2,
        }
    ]

    summary = summarize(ensemble, rows_by_model)

    gpt = summary["per_model"][GPT]
    assert gpt["calls"] == 2
    assert gpt["failures"] == 0
    assert (gpt["prompt_tokens"], gpt["completion_tokens"]) == (200, 50)
    assert summary["per_stage"]["1"]["calls"] == 2
    assert summary["per_stage"]["2"]["calls"] == 1
    assert summary["total_cost_usd"] == pytest.approx(
        sum(stage["cost_usd"] for stage in summary["per_stage"].values())
    )


# --- end to end: stages of one run ---------------------------------------


def _corpus(tmp_path: Path) -> Path:
    """Two cells (110th House, D and R) of six speeches each."""
    ids, parties, texts = [], [], []
    for party in ("D", "R"):
        for n in range(6):
            speech_id = f"{party}{n}"
            ids.append(speech_id)
            parties.append(party)
            texts.append(f"speech-{speech_id} " + "word " * 60)
    path = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "speech_id": ids,
                "word_count": [61] * len(ids),
                "congress_number": [110] * len(ids),
                "party": parties,
                "chamber": ["H"] * len(ids),
                "text": texts,
            }
        ),
        path,
    )
    return path


class FakeProviders:
    """Answers every call; optionally fails the first call to some models."""

    def __init__(self, fail_first: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_first = set(fail_first)

    async def score_one(self, spec, client, prompt) -> ScoreResult:
        speech_id = re.search(r"speech-(\w+)", prompt).group(1)
        self.calls.append((speech_id, spec.model))
        if spec.model in self.fail_first:
            self.fail_first.discard(spec.model)
            return ScoreResult(
                model=spec.model, error="RateLimitError: slow down", prompt_tokens=10
            )
        return ScoreResult(
            model=spec.model,
            ideology_score=0.5 if speech_id.startswith("R") else -0.5,
            tone_score=0.1,
            reasoning="r",
            prompt_tokens=100,
            completion_tokens=10,
            served_model=f"{spec.model}-served",
        )


@pytest.fixture
def run_env(tmp_path: Path, monkeypatch):
    """Paths plus a `run(*args)` that calls main with the fakes in place."""
    corpus = _corpus(tmp_path)
    scores, metrics = tmp_path / "scores", tmp_path / "metrics"
    providers = FakeProviders()

    monkeypatch.setattr(pilot_run, "build_clients", lambda specs: dict.fromkeys(specs))
    monkeypatch.setattr(
        pilot_run, "count_prompt_tokens", lambda prompts: [900] * len(prompts)
    )
    monkeypatch.setattr("builtins.input", lambda _: "y")

    def run(*args: str, fake: FakeProviders | None = None) -> int:
        monkeypatch.setattr(pilot_run, "score_one", (fake or providers).score_one)
        argv = [
            "--corpus", str(corpus),
            "--output-dir", str(scores),
            "--metrics-dir", str(metrics),
            *args,
        ]  # fmt: skip
        return asyncio.run(pilot_run.main(argv))

    return {"run": run, "scores": scores, "metrics": metrics, "providers": providers}


def _only_run(scores: Path, label: str = "s8") -> str:
    (manifest,) = scores.glob(f"{label}_manifest_*.json")
    return manifest.stem.removeprefix(f"{label}_manifest_")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_a_labelled_run_writes_every_file_under_its_label(run_env) -> None:
    assert run_env["run"]("--label", "s8", "--per-cell", "2") == 0

    run = _only_run(run_env["scores"])
    for model in ENSEMBLE_MODELS:
        rows = _rows(run_env["scores"] / f"s8_{model}_{run}.jsonl")
        assert len(rows) == 4
        assert {row["stage"] for row in rows} == {1}
        assert rows[0]["served_model"] == f"{model}-served"
    assert len(_rows(run_env["scores"] / f"s8_ensemble_{run}.jsonl")) == 4
    summary = json.loads((run_env["metrics"] / f"s8_summary_{run}.json").read_text())
    assert summary["per_model"][GPT]["served_models"] == [f"{GPT}-served"]


def test_resume_scores_only_the_speeches_the_earlier_stage_did_not(run_env) -> None:
    run_env["run"]("--label", "s8", "--per-cell", "2")
    run = _only_run(run_env["scores"])
    first = {speech for speech, _ in run_env["providers"].calls}

    second = FakeProviders()
    assert run_env["run"]("--resume", run, "--per-cell", "4", fake=second) == 0

    resumed = {speech for speech, _ in second.calls}
    assert len(second.calls) == 4 * len(ENSEMBLE_MODELS)
    assert len(resumed) == 4 and not resumed & first

    for model in ENSEMBLE_MODELS:
        rows = _rows(run_env["scores"] / f"s8_{model}_{run}.jsonl")
        assert [row["stage"] for row in rows] == [1] * 4 + [2] * 4
    ensemble = _rows(run_env["scores"] / f"s8_ensemble_{run}.jsonl")
    assert len(ensemble) == 8 and {row["n_models"] for row in ensemble} == {
        len(ENSEMBLE_MODELS)
    }

    manifest = json.loads((run_env["scores"] / f"s8_manifest_{run}.json").read_text())
    assert [stage["per_cell"] for stage in manifest["stages"]] == [2, 4]
    assert manifest["per_cell"] == 4


def test_resume_retries_a_failed_call_and_keeps_the_failed_row(run_env) -> None:
    flaky = FakeProviders(fail_first=(GPT,))
    run_env["run"]("--label", "s8", "--per-cell", "2", fake=flaky)
    run = _only_run(run_env["scores"])
    failed_speech = next(speech for speech, model in flaky.calls if model == GPT)

    second = FakeProviders()
    # No --per-cell: a bare resume finishes the run at its current size.
    run_env["run"]("--resume", run, fake=second)

    assert second.calls == [(failed_speech, GPT)]
    gpt_rows = _rows(run_env["scores"] / f"s8_{GPT}_{run}.jsonl")
    assert [
        r["error"] is None for r in gpt_rows if r["speech_id"] == failed_speech
    ] == [
        False,
        True,
    ]
    ensemble = {
        r["speech_id"]: r for r in _rows(run_env["scores"] / f"s8_ensemble_{run}.jsonl")
    }
    assert ensemble[failed_speech]["n_models"] == len(ENSEMBLE_MODELS)
    summary = json.loads((run_env["metrics"] / f"s8_summary_{run}.json").read_text())
    assert summary["per_model"][GPT]["calls"] == 5
    assert summary["per_model"][GPT]["failures"] == 0


def test_a_resume_dry_run_calls_nothing_and_changes_nothing(run_env) -> None:
    run_env["run"]("--label", "s8", "--per-cell", "2")
    run = _only_run(run_env["scores"])
    before = {p.name: p.read_bytes() for p in run_env["scores"].iterdir()}

    second = FakeProviders()
    assert (
        run_env["run"]("--resume", run, "--per-cell", "4", "--dry-run", fake=second)
        == 0
    )

    assert second.calls == []
    assert {p.name: p.read_bytes() for p in run_env["scores"].iterdir()} == before


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--seed", "7"), "random_seed"),
        (("--per-cell", "1"), "not in the new sample"),
        (("--label", "other"), "label"),
    ],
    ids=["changed-seed", "smaller-sample", "other-label"],
)
def test_a_resume_that_would_mix_runs_is_refused(
    run_env, capsys, args, message
) -> None:
    run_env["run"]("--label", "s8", "--per-cell", "2")
    run = _only_run(run_env["scores"])
    before = {p.name: p.read_bytes() for p in run_env["scores"].iterdir()}

    second = FakeProviders()
    assert run_env["run"]("--resume", run, *args, fake=second) != 0

    assert message in capsys.readouterr().err
    assert second.calls == []
    assert {p.name: p.read_bytes() for p in run_env["scores"].iterdir()} == before
