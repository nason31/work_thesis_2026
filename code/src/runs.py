"""A scoring run on disk: its files, which of its rows count, and what is left.

A run is one timestamp. `code/scripts/pilot_run.py` writes, in
``results/scores/``:

- ``<label>_manifest_<run>.json`` -- what defines the scores, and each stage
- ``<label>_<model>_<run>.jsonl`` -- one file per model, one row per attempt
- ``<label>_ensemble_<run>.jsonl`` -- one row per speech, rebuilt every stage

and ``<label>_summary_<run>.json`` in ``results/metrics/``. The label names the
kind of run (``pilot``, ``s8``); the timestamp identifies it.

A run is extended in stages rather than repeated (docs/decisions.md S13): a
later stage draws a larger sample from the same seed, which contains the
earlier one, and scores only what has no successful score yet. So a model's
file can hold two rows for one speech -- a failed attempt and its retry. Every
attempt is kept, because every attempt was billed; `effective_rows` says which
one counts.

A run is only resumed when nothing that defines a score has changed: corpus,
prompt, models and their settings, seed, length filter and strata. Anything
else would mix two instruments in one run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Label for runs that do not name one; the 2026-09-23 pilot carries it.
DEFAULT_LABEL = "pilot"

#: Manifest keys that decide what a score means. A run resumes only if every
#: one matches the current settings exactly.
SCORE_DEFINING_KEYS: tuple[str, ...] = (
    "corpus_sha256",
    "prompt_template",
    "models",
    "random_seed",
    "min_word_count",
    "stratify_by",
)

#: Score-defining settings other than the models, which a carry-over matches
#: model by model instead (S16).
RUN_SETTING_KEYS: tuple[str, ...] = tuple(
    key for key in SCORE_DEFINING_KEYS if key != "models"
)

#: Model settings that must match even across a declared name equivalence.
_MODEL_BEHAVIOUR_KEYS = ("provider", "effective_temperature", "max_output_tokens")

_MANIFEST = "_manifest_"

#: How many offending ids an error message lists.
_SHOW = 5


# --- file layout ---------------------------------------------------------


def manifest_path(scores_dir: Path, label: str, run: str) -> Path:
    """The run's manifest."""
    return Path(scores_dir) / f"{label}{_MANIFEST}{run}.json"


def model_path(scores_dir: Path, label: str, model: str, run: str) -> Path:
    """One model's raw rows for the run."""
    return Path(scores_dir) / f"{label}_{model.replace('/', '_')}_{run}.jsonl"


def ensemble_path(scores_dir: Path, label: str, run: str) -> Path:
    """The run's per-speech ensemble rows."""
    return Path(scores_dir) / f"{label}_ensemble_{run}.jsonl"


def summary_path(metrics_dir: Path, label: str, run: str) -> Path:
    """The run's summary: cost, failures, the party check."""
    return Path(metrics_dir) / f"{label}_summary_{run}.json"


def _manifests(scores_dir: Path) -> list[tuple[str, str]]:
    """(label, run) for every manifest in ``scores_dir``."""
    found = []
    for path in Path(scores_dir).glob(f"*{_MANIFEST}*.json"):
        label, _, run = path.stem.rpartition(_MANIFEST)
        found.append((label, run))
    return found


def find_label(scores_dir: Path, run: str) -> str:
    """The label of ``run``, read off its manifest's file name.

    Raises:
        FileNotFoundError: no manifest, or more than one, for ``run``.
    """
    labels = [label for label, found in _manifests(scores_dir) if found == run]
    if len(labels) != 1:
        raise FileNotFoundError(
            f"expected one manifest for run {run} in {scores_dir}, found {len(labels)}"
        )
    return labels[0]


def latest_run(scores_dir: Path) -> str:
    """Timestamp of the newest run in ``scores_dir``, whatever its label.

    Compared on the timestamp alone: sorting file names would rank every
    ``s8_`` run above every ``pilot_`` run.
    """
    runs = [run for _, run in _manifests(scores_dir)]
    if not runs:
        raise FileNotFoundError(f"No run manifest in {scores_dir}")
    return max(runs)


def run_models(scores_dir: Path, run: str) -> tuple[str, ...]:
    """The models ``run`` was scored with, from its manifest.

    Not the current ensemble: the ensemble changes (S15), and an older run must
    still be read with the models it actually used.

    Raises:
        FileNotFoundError: no manifest for ``run``.
        KeyError: the manifest lists no models.
    """
    label = find_label(scores_dir, run)
    manifest = json.loads(manifest_path(scores_dir, label, run).read_text())
    if not manifest.get("models"):
        raise KeyError(f"manifest of run {run} lists no models")
    return tuple(manifest["models"])


def read_rows(path: Path) -> list[dict[str, Any]]:
    """A JSONL file's rows; none if the file does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --- which rows count ------------------------------------------------------


def is_ok(row: Mapping[str, Any]) -> bool:
    """True when the row holds a validated score."""
    return row.get("error") is None and row.get("ideology_score") is not None


def effective_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per speech, the row that counts: its successful attempt, else its last.

    A success is never re-scored, so a speech has at most one; a failure is
    retried, and a speech that never succeeded is represented by its latest
    error.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        speech_id = str(row["speech_id"])
        if speech_id in chosen and is_ok(chosen[speech_id]):
            continue
        chosen[speech_id] = dict(row)
    return chosen


def pending_models(
    speech_ids: Iterable[str],
    rows_by_model: Mapping[str, Iterable[Mapping[str, Any]]],
    models: tuple[str, ...],
) -> dict[str, list[str]]:
    """Which models still owe each speech a score, in ``models`` order.

    Speeches every model has scored are left out, so an empty result means the
    run is complete.
    """
    done = {
        model: {
            speech_id
            for speech_id, row in effective_rows(rows_by_model.get(model, [])).items()
            if is_ok(row)
        }
        for model in models
    }
    pending: dict[str, list[str]] = {}
    for speech_id in speech_ids:
        missing = [model for model in models if speech_id not in done[model]]
        if missing:
            pending[speech_id] = missing
    return pending


# --- whether a run may be resumed -----------------------------------------


def check_resumable(
    manifest: Mapping[str, Any],
    current: Mapping[str, Any],
    keys: tuple[str, ...] = SCORE_DEFINING_KEYS,
) -> None:
    """Raise unless every score-defining setting matches the run's manifest.

    A setting the manifest does not record counts as changed: a run from before
    it was recorded cannot show that it matches.

    Raises:
        ValueError: naming every setting that differs.
    """
    changed = [
        key for key in keys if key not in manifest or manifest[key] != current[key]
    ]
    if changed:
        raise ValueError(
            "cannot resume: these settings differ from the run's manifest: "
            + ", ".join(changed)
            + ". Scores under different settings do not belong in one run -- "
            "start a new run instead."
        )


def check_nested(scored_ids: Iterable[str], sample_ids: Iterable[str]) -> None:
    """Raise if a speech the run already scored is not in the new sample.

    The larger sample from the same seed contains the smaller one, so this
    only fires when the sample shrank or was drawn differently.

    Raises:
        ValueError: listing some of the orphaned speech ids.
    """
    orphaned = sorted(set(scored_ids) - set(sample_ids))
    if orphaned:
        raise ValueError(
            f"cannot resume: {len(orphaned)} speech(es) the run already scored "
            f"are not in the new sample, e.g. {orphaned[:_SHOW]}. Use a sample "
            "at least as large as the run's last stage."
        )


# --- carrying answers from an earlier run into a new one (S16) ---------------


def match_models(
    old_models: Mapping[str, Mapping[str, Any]],
    current_models: Mapping[str, Mapping[str, Any]],
    equivalents: Mapping[str, str],
) -> dict[str, str]:
    """Map each current model to the earlier run's model whose answers it may take.

    A model under the same name must have identical settings; a manifest from
    before ``request_options`` existed counts as having sent none, which is what
    the code then did. A declared equivalent (an earlier name for the same
    model) must match on everything but its name and request options; whether
    it really was the same model is checked row by row (`check_served`).

    Raises:
        ValueError: the earlier run lacks a model, or its settings differ.
    """
    mapping: dict[str, str] = {}
    for name, current in current_models.items():
        old_name = equivalents.get(name, name)
        if old_name not in old_models:
            raise ValueError(f"cannot carry over: the earlier run has no {old_name!r}")
        old = {"request_options": {}, **old_models[old_name]}
        if old_name == name:
            same = old == dict(current)
        else:
            same = all(old.get(k) == current.get(k) for k in _MODEL_BEHAVIOUR_KEYS)
        if not same:
            raise ValueError(
                f"cannot carry over {name}: its settings differ from the earlier "
                f"run's {old_name} ({dict(old)} vs {dict(current)})"
            )
        mapping[name] = old_name
    return mapping


def check_served(rows: Iterable[Mapping[str, Any]], model: str) -> None:
    """Raise unless every row says ``model`` is what answered it.

    The evidence that an earlier model name was the same model.

    Raises:
        ValueError: naming the other served models found.
    """
    served = {row.get("served_model") for row in rows}
    if served - {model}:
        raise ValueError(
            f"cannot carry over as {model}: earlier rows were served by "
            f"{sorted(str(s) for s in served - {model})}"
        )


def carry_rows(
    rows: Iterable[Mapping[str, Any]], run: str, sample_ids: Iterable[str]
) -> list[dict[str, Any]]:
    """Rows of the sampled speeches, as stage 1 of the new run.

    Each keeps everything it recorded (including the model name it was
    requested under) and says where it came from.
    """
    wanted = set(sample_ids)
    carried = []
    for row in rows:
        if str(row["speech_id"]) not in wanted:
            continue
        copy = dict(row)
        copy["carried_from"] = {
            "run": run,
            "stage": row.get("stage"),
            "model": row.get("model"),
        }
        copy["stage"] = 1
        carried.append(copy)
    return carried
