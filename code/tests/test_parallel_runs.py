"""Tests for parallel calls (S17), single-model runs and the build guard (S18).

`pilot_run.main` and `label_run.main` run end to end on a tiny synthetic
corpus; only the provider call, the client construction and the tokenizer are
faked. The fakes take a little time to answer, so several speeches really are
in flight at once.
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

import label_run
import pilot_run

from src.config import ENSEMBLE_MODELS, LABEL_MODEL
from src.labeling import LabelResult
from src.runs import FINGERPRINT_ERROR, FingerprintGuard
from src.scoring import ScoreResult

DEEPSEEK, GPT = "deepseek-flash", "gpt-4o-2024-11-20"


# --- the guard on its own ---------------------------------------------------------


def test_the_first_answer_sets_the_build_and_another_build_is_rejected() -> None:
    guard = FingerprintGuard([DEEPSEEK], {})

    assert guard.check(DEEPSEEK, "fp_a") is None
    assert guard.check(DEEPSEEK, "fp_a") is None
    message = guard.check(DEEPSEEK, "fp_b")
    assert message.startswith(FINGERPRINT_ERROR) and "fp_b" in message


def test_the_runs_earlier_rows_set_the_build() -> None:
    rows = {DEEPSEEK: [{"system_fingerprint": "fp_old", "error": None}]}

    guard = FingerprintGuard([DEEPSEEK], rows)

    assert guard.check(DEEPSEEK, "fp_new") is not None


def test_a_model_that_is_not_pinned_may_change_build() -> None:
    """GPT-4o rotated three fingerprints within the S8 run, same snapshot."""
    guard = FingerprintGuard([DEEPSEEK], {GPT: [{"system_fingerprint": "fp_1"}]})

    assert guard.check(GPT, "fp_2") is None


def test_rejected_rows_never_define_the_build() -> None:
    """Otherwise resuming a stopped run would accept the new build."""
    rows = {
        DEEPSEEK: [
            {"system_fingerprint": "fp_old", "error": None},
            {"system_fingerprint": "fp_new", "error": f"{FINGERPRINT_ERROR}: x"},
        ]
    }

    assert FingerprintGuard([DEEPSEEK], rows).check(DEEPSEEK, "fp_new") is not None


def test_an_answer_without_a_fingerprint_passes() -> None:
    guard = FingerprintGuard([DEEPSEEK], {DEEPSEEK: [{"system_fingerprint": "a"}]})

    assert guard.check(DEEPSEEK, None) is None


# --- scoring ----------------------------------------------------------------------


def _corpus(tmp_path: Path, per_party: int = 8) -> Path:
    """One cell per party (110th House), `per_party` speeches each."""
    rows = [
        {
            "speech_id": f"{party}{n:02d}",
            "word_count": 61,
            "congress_number": 110,
            "party": party,
            "chamber": "H",
            "source": "stanford",
            "text": f"speech-{party}{n:02d} " + "word " * 60,
        }
        for party in ("D", "R")
        for n in range(per_party)
    ]
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class SlowScorer:
    """Answers after a short wait, tracking how many calls are open at once."""

    def __init__(self, fingerprint: str = "fp_a", delay: float = 0.02) -> None:
        self.calls: list[tuple[str, str]] = []
        self.open = 0
        self.most_open = 0
        self.fingerprint = fingerprint
        self.delay = delay

    async def score_one(self, spec, client, prompt) -> ScoreResult:
        speech_id = re.search(r"speech-(\w+)", prompt).group(1)
        self.calls.append((speech_id, spec.model))
        self.open += 1
        self.most_open = max(self.most_open, self.open)
        await asyncio.sleep(self.delay)
        self.open -= 1
        return ScoreResult(
            model=spec.model,
            ideology_score=0.5 if speech_id.startswith("R") else -0.5,
            tone_score=0.1,
            reasoning="r",
            prompt_tokens=100,
            completion_tokens=10,
            served_model=spec.model,
            system_fingerprint=self.fingerprint,
        )


@pytest.fixture
def scoring_env(tmp_path: Path, monkeypatch):
    corpus = _corpus(tmp_path)
    scores, metrics = tmp_path / "scores", tmp_path / "metrics"
    monkeypatch.setattr(pilot_run, "build_clients", lambda specs: dict.fromkeys(specs))
    monkeypatch.setattr(
        pilot_run, "count_prompt_tokens", lambda prompts: [900] * len(prompts)
    )
    monkeypatch.setattr("builtins.input", lambda _: "y")

    def run(*args: str, fake: SlowScorer) -> int:
        monkeypatch.setattr(pilot_run, "score_one", fake.score_one)
        argv = [
            "--corpus", str(corpus), "--output-dir", str(scores),
            "--metrics-dir", str(metrics), "--label", "t", *args,
        ]  # fmt: skip
        return asyncio.run(pilot_run.main(argv))

    return {"run": run, "scores": scores, "metrics": metrics}


def _run_id(directory: Path, prefix: str) -> str:
    (manifest,) = directory.glob(f"{prefix}*.json")
    return manifest.stem.removeprefix(prefix)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_concurrency_keeps_that_many_speeches_in_flight(scoring_env) -> None:
    fake = SlowScorer()

    code = scoring_env["run"]("--per-cell", "8", "--concurrency", "4", fake=fake)

    assert code == 0
    # Two models per speech, four speeches at once.
    assert 2 < fake.most_open <= 4 * len(ENSEMBLE_MODELS)
    run = _run_id(scoring_env["scores"], "t_manifest_")
    for model in ENSEMBLE_MODELS:
        rows = _rows(scoring_env["scores"] / f"t_{model}_{run}.jsonl")
        assert sorted(r["speech_id"] for r in rows) == sorted(
            f"{p}{n:02d}" for p in "DR" for n in range(8)
        )
    manifest = json.loads(
        (scoring_env["scores"] / f"t_manifest_{run}.json").read_text()
    )
    assert manifest["stages"][0]["concurrency"] == 4


def test_concurrency_one_scores_in_sample_order(scoring_env) -> None:
    fake = SlowScorer(delay=0)

    scoring_env["run"]("--per-cell", "4", fake=fake)

    speeches = [speech for speech, _ in fake.calls[:: len(ENSEMBLE_MODELS)]]
    assert speeches == sorted(speeches)
    assert fake.most_open <= len(ENSEMBLE_MODELS)


def test_a_run_can_score_with_deepseek_alone(scoring_env) -> None:
    fake = SlowScorer()

    code = scoring_env["run"](
        "--per-cell", "4", "--models", DEEPSEEK, "--concurrency", "3", fake=fake
    )

    assert code == 0
    assert {model for _, model in fake.calls} == {DEEPSEEK}
    run = _run_id(scoring_env["scores"], "t_manifest_")
    assert not list(scoring_env["scores"].glob(f"t_{GPT}_*.jsonl"))
    manifest = json.loads(
        (scoring_env["scores"] / f"t_manifest_{run}.json").read_text()
    )
    assert list(manifest["models"]) == [DEEPSEEK]
    ensemble = _rows(scoring_env["scores"] / f"t_ensemble_{run}.jsonl")
    assert {row["n_models"] for row in ensemble} == {1}
    summary = json.loads((scoring_env["metrics"] / f"t_summary_{run}.json").read_text())
    assert summary["models"] == [DEEPSEEK] and list(summary["per_model"]) == [DEEPSEEK]
    assert summary["partial_ensemble_rows"] == 0


def test_a_resume_keeps_the_runs_models(scoring_env, capsys) -> None:
    scoring_env["run"]("--per-cell", "2", "--models", DEEPSEEK, fake=SlowScorer())
    run = _run_id(scoring_env["scores"], "t_manifest_")

    resumed = SlowScorer()
    assert scoring_env["run"]("--resume", run, "--per-cell", "4", fake=resumed) == 0
    assert {model for _, model in resumed.calls} == {DEEPSEEK}

    refused = SlowScorer()
    code = scoring_env["run"]("--resume", run, "--models", DEEPSEEK, GPT, fake=refused)
    assert code == 2 and refused.calls == []
    assert "keeps its models" in capsys.readouterr().err


def test_a_run_stops_when_deepseek_answers_from_another_build(scoring_env) -> None:
    scoring_env["run"]("--per-cell", "2", "--models", DEEPSEEK, fake=SlowScorer("fp_a"))
    run = _run_id(scoring_env["scores"], "t_manifest_")

    new_build = SlowScorer("fp_b")
    code = scoring_env["run"](
        "--resume", run, "--per-cell", "8", "--concurrency", "3", fake=new_build
    )

    assert code == 3
    # Every worker stops after its current speech: at most 3 calls, not 12.
    assert 1 <= len(new_build.calls) <= 3
    rows = _rows(scoring_env["scores"] / f"t_{DEEPSEEK}_{run}.jsonl")
    rejected = [r for r in rows if r.get("rejected_answer")]
    assert len(rejected) == len(new_build.calls)
    assert all(r["error"].startswith(FINGERPRINT_ERROR) for r in rejected)
    assert all(r["ideology_score"] is None for r in rejected)
    assert all(r["rejected_answer"]["ideology_score"] is not None for r in rejected)
    ensemble = _rows(scoring_env["scores"] / f"t_ensemble_{run}.jsonl")
    assert sum(r["n_models"] for r in ensemble) == 4  # only the first build counts

    again = SlowScorer("fp_b")
    assert scoring_env["run"]("--resume", run, "--per-cell", "8", fake=again) == 3


def test_a_gpt_build_change_does_not_stop_a_run(scoring_env) -> None:
    scoring_env["run"]("--per-cell", "2", "--models", GPT, fake=SlowScorer("fp_1"))
    run = _run_id(scoring_env["scores"], "t_manifest_")

    code = scoring_env["run"](
        "--resume", run, "--per-cell", "4", fake=SlowScorer("fp_2")
    )

    assert code == 0


def test_concurrency_below_one_is_refused(scoring_env) -> None:
    fake = SlowScorer()

    assert scoring_env["run"]("--per-cell", "2", "--concurrency", "0", fake=fake) == 2
    assert fake.calls == []


# --- labelling ----------------------------------------------------------------------


class SlowLabeler:
    def __init__(self, fingerprint: str = "fp_a") -> None:
        self.calls: list[str] = []
        self.open = 0
        self.most_open = 0
        self.fingerprint = fingerprint

    async def label_one(self, spec, client, prompt) -> LabelResult:
        speech_id = re.search(r"speech-(\w+)", prompt).group(1)
        self.calls.append(speech_id)
        self.open += 1
        self.most_open = max(self.most_open, self.open)
        await asyncio.sleep(0.02)
        self.open -= 1
        return LabelResult(
            model=spec.model,
            speech_type="procedural",
            topic=None,
            reason="r",
            prompt_tokens=900,
            completion_tokens=30,
            served_model=spec.model,
            system_fingerprint=self.fingerprint,
        )


@pytest.fixture
def label_env(tmp_path: Path, monkeypatch):
    corpus = _corpus(tmp_path)
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(f"{p}{n:02d}" for p in "DR" for n in range(8)))
    labels, metrics = tmp_path / "labels", tmp_path / "metrics"
    monkeypatch.setattr(label_run, "build_clients", lambda specs: dict.fromkeys(specs))
    monkeypatch.setattr(
        label_run, "count_prompt_tokens", lambda prompts: [1000] * len(prompts)
    )
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    def run(*args: str, fake: SlowLabeler) -> int:
        monkeypatch.setattr(label_run, "label_one", fake.label_one)
        argv = [
            "--corpus", str(corpus), "--output-dir", str(labels),
            "--metrics-dir", str(metrics), "--label", "t", *args,
        ]  # fmt: skip
        return asyncio.run(label_run.main(argv))

    return {"run": run, "labels": labels, "ids_file": ids_file}


def test_labelling_runs_in_parallel(label_env) -> None:
    fake = SlowLabeler()

    code = label_env["run"]("--ids-file", str(label_env["ids_file"]),
                            "--concurrency", "5", fake=fake)  # fmt: skip

    assert code == 0
    assert 1 < fake.most_open <= 5
    assert sorted(fake.calls) == sorted(f"{p}{n:02d}" for p in "DR" for n in range(8))
    run = _run_id(label_env["labels"], "label_manifest_t_")
    manifest = json.loads(
        (label_env["labels"] / f"label_manifest_t_{run}.json").read_text()
    )
    assert manifest["stages"][0]["concurrency"] == 5


def test_labelling_stops_when_deepseek_answers_from_another_build(label_env) -> None:
    label_env["ids_file"].write_text("D00\nD01\n")
    label_env["run"]("--ids-file", str(label_env["ids_file"]), fake=SlowLabeler("a"))
    run = _run_id(label_env["labels"], "label_manifest_t_")
    label_env["ids_file"].write_text("\n".join(f"D{n:02d}" for n in range(8)))

    new_build = SlowLabeler("b")
    code = label_env["run"](
        "--resume", run, "--ids-file", str(label_env["ids_file"]),
        "--concurrency", "2", fake=new_build,
    )  # fmt: skip

    assert code == 3 and 1 <= len(new_build.calls) <= 2
    rows = _rows(label_env["labels"] / f"label_t_{LABEL_MODEL}_{run}.jsonl")
    rejected = [r for r in rows if r.get("rejected_answer")]
    assert rejected and all(r["speech_type"] is None for r in rejected)
    assert all(r["rejected_answer"]["speech_type"] == "procedural" for r in rejected)
