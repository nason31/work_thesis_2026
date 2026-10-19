"""Tests for running the labelling in stages: files, dry run, approval, resume.

`label_run.main` runs end to end on a tiny synthetic corpus. Only the provider
call (`label_one`), the client construction and the tokenizer are replaced --
selection, file writing, resume checks and the summary are the real ones.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import label_run

from src import labeling
from src.config import LABEL_MODEL
from src.labeling import LabelResult

# --- fixtures -------------------------------------------------------------------


def _corpus(tmp_path: Path) -> tuple[Path, list[str]]:
    """Eight speeches: two in each chamber x source cell."""
    rows = []
    for chamber in ("H", "S"):
        for source, congress in (("stanford", 110), ("govinfo", 116)):
            for n in range(2):
                speech_id = f"{chamber}{source[0]}{n}"
                rows.append(
                    {
                        "speech_id": speech_id,
                        "chamber": chamber,
                        "source": source,
                        "congress_number": congress,
                        "party": "D",
                        "text": f"speech-{speech_id} " + "word " * 60,
                    }
                )
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path, [row["speech_id"] for row in rows]


class FakeLabeler:
    """Labels every speech; optionally fails the first call on some speeches."""

    def __init__(self, fail_first: tuple[str, ...] = ()) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_first = set(fail_first)

    async def label_one(self, spec, client, prompt) -> LabelResult:
        speech_id = re.search(r"speech-(\w+)", prompt).group(1)
        self.calls.append((speech_id, spec.model))
        if speech_id in self.fail_first:
            self.fail_first.discard(speech_id)
            return LabelResult(model=spec.model, error="RateLimitError: slow down")
        policy = speech_id.startswith("H")
        return LabelResult(
            model=spec.model,
            speech_type="policy_debate" if policy else "ceremonial",
            topic="health" if policy else None,
            reason="r",
            prompt_tokens=900,
            completion_tokens=30,
            served_model=spec.model,
            system_fingerprint="fp_test",
        )


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """Paths plus a `run(*args)` that calls main with the fakes in place."""
    corpus, ids = _corpus(tmp_path)
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")
    labels, metrics = tmp_path / "labels", tmp_path / "metrics"
    labeler = FakeLabeler()
    answers = {"value": "yes"}

    monkeypatch.setattr(label_run, "build_clients", lambda specs: dict.fromkeys(specs))
    monkeypatch.setattr(
        label_run, "count_prompt_tokens", lambda prompts: [1000] * len(prompts)
    )
    monkeypatch.setattr("builtins.input", lambda _: answers["value"])

    def run(*args: str, fake: FakeLabeler | None = None) -> int:
        monkeypatch.setattr(label_run, "label_one", (fake or labeler).label_one)
        argv = [
            "--corpus", str(corpus),
            "--output-dir", str(labels),
            "--metrics-dir", str(metrics),
            *args,
        ]  # fmt: skip
        return asyncio.run(label_run.main(argv))

    return {
        "run": run,
        "ids": ids,
        "ids_file": ids_file,
        "labels": labels,
        "metrics": metrics,
        "labeler": labeler,
        "answers": answers,
        "tmp": tmp_path,
    }


def _only_run(labels: Path, label: str = "t") -> str:
    (manifest,) = labels.glob(f"label_manifest_{label}_*.json")
    return manifest.stem.removeprefix(f"label_manifest_{label}_")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _snapshot(directory: Path) -> dict[str, bytes]:
    if not directory.exists():
        return {}
    return {p.name: p.read_bytes() for p in directory.iterdir()}


# --- a new run ---------------------------------------------------------------------


def test_a_run_writes_its_rows_manifest_and_summary(env) -> None:
    assert env["run"]("--ids-file", str(env["ids_file"]), "--label", "t") == 0

    run = _only_run(env["labels"])
    rows = _rows(env["labels"] / f"label_t_{LABEL_MODEL}_{run}.jsonl")
    assert len(rows) == 8 and {r["stage"] for r in rows} == {1}
    assert set(rows[0]) >= {
        "speech_id", "model", "speech_type", "topic", "reason", "prompt_tokens",
        "completion_tokens", "served_model", "error", "attempted_at",
    }  # fmt: skip
    manifest = json.loads((env["labels"] / f"label_manifest_t_{run}.json").read_text())
    assert manifest["model"] == LABEL_MODEL
    assert manifest["stages"][0]["selection"]["ids_file"] == str(env["ids_file"])
    assert manifest["stages"][0]["speeches_in_selection"] == 8
    assert manifest["corpus_sha256"] and manifest["prompt_sha256"]
    summary = json.loads((env["metrics"] / f"label_summary_t_{run}.json").read_text())
    assert summary["speech_type_counts"]["policy_debate"] == 4
    assert summary["speech_type_counts"]["ceremonial"] == 4
    assert summary["failures"] == 0


def test_a_limit_labels_a_subset_spread_over_chamber_and_source(env) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--limit", "4", "--label", "t")

    called = {speech for speech, _ in env["labeler"].calls}
    assert len(called) == 4
    # One per chamber x source cell: the ids encode the cell in their first two
    # characters.
    assert {speech[:2] for speech in called} == {"Hs", "Hg", "Ss", "Sg"}


def test_a_dry_run_calls_nothing_writes_nothing_and_needs_no_key(
    env, monkeypatch
) -> None:
    def no_keys(specs):
        raise AssertionError("a dry run must not build clients")

    monkeypatch.setattr(label_run, "build_clients", no_keys)

    assert env["run"]("--ids-file", str(env["ids_file"]), "--dry-run") == 0

    assert env["labeler"].calls == []
    assert not env["labels"].exists() and not env["metrics"].exists()


def test_there_is_no_flag_to_skip_the_confirmation() -> None:
    with pytest.raises(SystemExit):
        label_run.parse_args(["--ids-file", "x", "--yes"])


@pytest.mark.parametrize("answer", ["y", "Y", "YES", "", "no"])
def test_anything_but_a_typed_yes_aborts(env, answer) -> None:
    env["answers"]["value"] = answer

    assert env["run"]("--ids-file", str(env["ids_file"]), "--label", "t") == 1

    assert env["labeler"].calls == []
    assert _snapshot(env["labels"]) == {}


def test_a_new_run_needs_a_source(env, capsys) -> None:
    assert env["run"]("--label", "t") == 2

    assert "--from-run" in capsys.readouterr().err


# --- resuming ------------------------------------------------------------------------


def test_a_resume_retries_only_the_failed_speech(env) -> None:
    flaky = FakeLabeler(fail_first=("Ss0",))
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t", fake=flaky)
    run = _only_run(env["labels"])

    second = FakeLabeler()
    assert env["run"]("--resume", run, fake=second) == 0

    assert second.calls == [("Ss0", LABEL_MODEL)]
    rows = _rows(env["labels"] / f"label_t_{LABEL_MODEL}_{run}.jsonl")
    assert [r["stage"] for r in rows if r["speech_id"] == "Ss0"] == [1, 2]
    summary = json.loads((env["metrics"] / f"label_summary_t_{run}.json").read_text())
    assert summary["failures"] == 0 and summary["failed_attempts"] == 1


def test_a_resume_with_a_larger_limit_labels_only_the_new_speeches(env) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--limit", "4", "--label", "t")
    run = _only_run(env["labels"])
    first = {speech for speech, _ in env["labeler"].calls}

    second = FakeLabeler()
    assert env["run"]("--resume", run, "--limit", "8", fake=second) == 0

    assert {speech for speech, _ in second.calls} == set(env["ids"]) - first


def test_a_resume_dry_run_changes_nothing(env) -> None:
    flaky = FakeLabeler(fail_first=("Ss0",))
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t", fake=flaky)
    run = _only_run(env["labels"])
    before = _snapshot(env["labels"])

    second = FakeLabeler()
    assert env["run"]("--resume", run, "--dry-run", fake=second) == 0

    assert second.calls == [] and _snapshot(env["labels"]) == before


def _refused(env, capsys, run: str, *args: str, message: str) -> None:
    before = _snapshot(env["labels"])
    second = FakeLabeler()

    assert env["run"]("--resume", run, *args, fake=second) == 2

    assert message in capsys.readouterr().err
    assert second.calls == []
    assert _snapshot(env["labels"]) == before


def test_a_resume_refuses_a_changed_prompt(env, capsys, monkeypatch) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t")
    run = _only_run(env["labels"])
    edited = env["tmp"] / "label_speech_v2.txt"
    edited.write_text("A revised codebook.\n\nSPEECH:\n{speech_text}\n")
    monkeypatch.setattr(label_run, "LABEL_PROMPT_PATH", edited)

    _refused(env, capsys, run, message="prompt_sha256")


def test_a_resume_refuses_a_changed_model_setting(env, capsys, monkeypatch) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t")
    run = _only_run(env["labels"])
    spec = replace(labeling.LABEL_MODEL_SPECS[LABEL_MODEL], max_output_tokens=512)
    monkeypatch.setitem(labeling.LABEL_MODEL_SPECS, LABEL_MODEL, spec)

    _refused(env, capsys, run, message="models")


def test_a_resume_refuses_a_changed_codebook_version(env, capsys, monkeypatch) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t")
    run = _only_run(env["labels"])
    monkeypatch.setattr(labeling, "CODEBOOK_VERSION", "v2")

    _refused(env, capsys, run, message="codebook_version")


def test_a_bare_resume_refuses_an_ids_file_that_changed(env, capsys) -> None:
    """A bare resume finishes the run as declared; an edited file is not that."""
    env["ids_file"].write_text("\n".join(env["ids"][:4]) + "\n")
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t")
    run = _only_run(env["labels"])
    env["ids_file"].write_text("\n".join(env["ids"][:5]) + "\n")

    _refused(env, capsys, run, message="changed")


def test_a_resume_refuses_a_selection_that_drops_labelled_speeches(env, capsys) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t")
    run = _only_run(env["labels"])
    fewer = env["tmp"] / "fewer.txt"
    fewer.write_text("\n".join(env["ids"][:3]) + "\n")

    _refused(
        env, capsys, run, "--ids-file", str(fewer), message="not in the new sample"
    )


def test_a_resume_refuses_another_model(env, capsys) -> None:
    env["run"]("--ids-file", str(env["ids_file"]), "--label", "t")
    run = _only_run(env["labels"])

    _refused(env, capsys, run, "--model", "gpt-4o-2024-11-20", message="one model")
