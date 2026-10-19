"""Label congressional speeches by speech type and policy topic.

A pass separate from scoring (docs/decisions.md L1): one model reads the whole
speech, and only the speech, and answers from a closed codebook
(``prompts/label_speech.txt``). The labels separate procedural, ceremonial and
other non-substantive speech from policy debate, and let a trend be checked
within topic. They are covariates, strata and exclusion filters for the
analysis -- never a scoring result -- so the rule against binary
classification as the final output (M1) is not touched.

An answer is checked against the codebook exactly. A label outside it, a topic
on a non-policy speech, or a policy speech without one is recorded as a
failure, never mapped to the nearest code: a model ignoring the codebook is a
finding, as an out-of-range score is (S3).

A label run is one timestamp and one model. ``code/scripts/label_run.py``
writes, in ``results/labels/``:

- ``label_manifest_<label>_<run>.json`` -- what defines the labels, and each stage
- ``label_<label>_<model>_<run>.jsonl`` -- one row per attempt

and ``label_summary_<label>_<run>.json`` in ``results/metrics/``. Like a
scoring run (S13), a label run is extended or finished in stages: a resume
labels only speeches without a valid label, and refuses if the codebook, the
prompt, the corpus or the model's settings changed.

The provider call, retry policy, deadline and JSON extraction are scoring.py's,
reused as they are; scoring.py itself is unchanged, so nothing that scored S8
moved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import openai
import pandas as pd
import pyarrow.parquet as pq

from src import scoring
from src.config import (
    CODEBOOK_VERSION,
    CORPUS_PATH,
    LABEL_EXPECTED_OUTPUT_TOKENS,
    LABEL_LIMIT_STRATA,
    LABEL_MAX_OUTPUT_TOKENS,
    LABELS_DIR,
    MODEL_PRICING,
    POLICY_SPEECH_TYPE,
    SCORES_DIR,
    SPEECH_TYPES,
    TOPIC_BLOCKS,
    TOPICS,
)
from src.runs import ensemble_path, find_label, read_rows
from src.scoring import (
    MODEL_SPECS,
    CallTimeoutError,
    ModelSpec,
    RawResponse,
    ScoreParseError,
    TruncatedResponseError,
    estimate_cost,
    extract_json,
)

# --- models ----------------------------------------------------------------

#: The label pass's own specs, kept apart from MODEL_SPECS so the scoring specs
#: stay exactly as S8 used them. Derived from them, so the endpoint and key stay
#: in one place. DeepSeek runs with thinking DISABLED: the option is DeepSeek's
#: documented toggle (api-docs.deepseek.com/guides/thinking_mode, read
#: 2026-10-17), and `reasoning_effort` is not sent, because the docs do not say
#: what it does with thinking off (L3). GPT-4o keeps JSON mode.
LABEL_MODEL_SPECS: dict[str, ModelSpec] = {
    "deepseek-flash": replace(
        MODEL_SPECS["deepseek-flash"],
        max_output_tokens=LABEL_MAX_OUTPUT_TOKENS,
        request_options={"extra_body": {"thinking": {"type": "disabled"}}},
    ),
    "gpt-4o-2024-11-20": replace(
        MODEL_SPECS["gpt-4o-2024-11-20"],
        max_output_tokens=LABEL_MAX_OUTPUT_TOKENS,
    ),
}


def label_spec(model: str) -> ModelSpec:
    """The label spec for ``model``.

    Only OpenAI-compatible providers: scoring.py's Anthropic call enforces the
    scoring answer schema, which would reject a label answer.

    Raises:
        KeyError: no label spec for ``model``.
    """
    if model not in LABEL_MODEL_SPECS:
        raise KeyError(
            f"no label spec for {model!r}; add it to LABEL_MODEL_SPECS in "
            "code/src/labeling.py"
        )
    return LABEL_MODEL_SPECS[model]


def model_settings(spec: ModelSpec) -> dict[str, Any]:
    """What the run manifest records about a model -- and a resume compares."""
    return {
        "model": spec.model,
        "provider": spec.provider,
        "effective_temperature": spec.effective_temperature,
        "max_output_tokens": spec.max_output_tokens,
        "request_options": spec.request_options,
    }


# --- one answer ------------------------------------------------------------


@dataclass
class LabelResult:
    """One model's labels for one speech. Nulls on failure, never an exception."""

    model: str
    speech_type: str | None = None
    topic: str | None = None
    reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    served_model: str | None = None
    system_fingerprint: str | None = None
    #: Output tokens spent reasoning; with thinking disabled this should be
    #: none, so it is the evidence that the mode was actually off.
    reasoning_tokens: int | None = None

    @property
    def ok(self) -> bool:
        """True when the labels parsed and are in the codebook."""
        return self.error is None and self.speech_type is not None


class LabelParseError(ScoreParseError):
    """A response could not be turned into labels from the codebook."""


def validate_labels(payload: Mapping[str, Any]) -> tuple[str, str | None, str]:
    """Return (speech_type, topic, reason), raising unless they fit the codebook.

    Exact strings only: no case folding, no trimming, no mapping of near
    misses. ``topic`` must be present; it is a codebook topic for a policy
    debate and JSON null for every other type -- the string "null" is not null.

    Raises:
        LabelParseError: naming what is wrong.
    """
    if "speech_type" not in payload:
        raise LabelParseError("missing field 'speech_type'")
    speech_type = payload["speech_type"]
    if not isinstance(speech_type, str) or speech_type not in SPEECH_TYPES:
        raise LabelParseError(f"speech_type {speech_type!r} is not in the codebook")

    if "topic" not in payload:
        raise LabelParseError("missing field 'topic'")
    topic = payload["topic"]
    if speech_type == POLICY_SPEECH_TYPE:
        if not isinstance(topic, str) or topic not in TOPICS:
            raise LabelParseError(
                f"topic {topic!r} is not in the codebook ({POLICY_SPEECH_TYPE} "
                "needs one)"
            )
    elif topic is not None:
        raise LabelParseError(
            f"topic {topic!r} given for speech_type {speech_type!r}; only "
            f"{POLICY_SPEECH_TYPE} takes a topic"
        )

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return speech_type, topic, reason


async def label_one(
    spec: ModelSpec, client: openai.AsyncOpenAI, prompt: str
) -> LabelResult:
    """Label one speech with one model. Never raises -- failures become nulls.

    The same call, retries and deadline as `scoring.score_one` (S14): three
    attempts with back-off on transient errors, each abandoned after
    CALL_TIMEOUT_SECONDS. Token counts are kept on a truncation and on an
    answer outside the codebook, because those tokens were billed.
    """
    result = LabelResult(model=spec.model)

    def record(raw: RawResponse) -> None:
        result.prompt_tokens = raw.prompt_tokens
        result.completion_tokens = raw.completion_tokens
        result.served_model = raw.served_model
        result.system_fingerprint = raw.system_fingerprint
        result.reasoning_tokens = raw.reasoning_tokens

    try:
        async for attempt in scoring._retrying():
            with attempt:
                # Read through the module, so the deadline is the one scoring
                # uses (and the one tests shorten).
                timeout = scoring.CALL_TIMEOUT_SECONDS
                try:
                    raw = await asyncio.wait_for(
                        scoring._call_openai_compatible(spec, client, prompt),
                        timeout=timeout,
                    )
                except TimeoutError as error:
                    raise CallTimeoutError(
                        f"{spec.model} gave no answer within {timeout:g}s"
                    ) from error
    except TruncatedResponseError as error:
        record(error.response)
        result.error = f"{type(error).__name__}: {error}"
        return result
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        result.error = f"{type(error).__name__}: {error}"
        return result

    record(raw)
    try:
        speech_type, topic, reason = validate_labels(extract_json(raw.text))
    except ScoreParseError as error:
        result.error = f"{type(error).__name__}: {error}"
        return result

    result.speech_type = speech_type
    result.topic = topic
    result.reason = reason
    return result


# --- rows and which of them count --------------------------------------------


def label_row(
    speech_id: str, result: LabelResult, stage: int, attempted_at: str
) -> dict[str, Any]:
    """One raw JSONL record: one attempt at one speech."""
    return {
        "speech_id": speech_id,
        "model": result.model,
        "speech_type": result.speech_type,
        "topic": result.topic,
        "reason": result.reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "served_model": result.served_model,
        "system_fingerprint": result.system_fingerprint,
        "error": result.error,
        "stage": stage,
        "attempted_at": attempted_at,
    }


def label_ok(row: Mapping[str, Any]) -> bool:
    """True when the row holds labels that passed validation."""
    return row.get("error") is None and row.get("speech_type") is not None


def effective_label_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per speech, the row that counts: its successful attempt, else its last."""
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        speech_id = str(row["speech_id"])
        if speech_id in chosen and label_ok(chosen[speech_id]):
            continue
        chosen[speech_id] = dict(row)
    return chosen


def pending_ids(
    speech_ids: Iterable[str], rows: Iterable[Mapping[str, Any]]
) -> list[str]:
    """The speeches still without valid labels, in the order given."""
    done = {
        speech_id
        for speech_id, row in effective_label_rows(rows).items()
        if label_ok(row)
    }
    return [speech_id for speech_id in speech_ids if speech_id not in done]


# --- file layout -------------------------------------------------------------

_MANIFEST_PREFIX = "label_manifest_"


def label_rows_path(labels_dir: Path, label: str, model: str, run: str) -> Path:
    """The run's raw rows for ``model``."""
    return Path(labels_dir) / f"label_{label}_{model.replace('/', '_')}_{run}.jsonl"


def label_manifest_path(labels_dir: Path, label: str, run: str) -> Path:
    """The run's manifest."""
    return Path(labels_dir) / f"{_MANIFEST_PREFIX}{label}_{run}.json"


def label_summary_path(metrics_dir: Path, label: str, run: str) -> Path:
    """The run's summary: counts, failures, tokens and cost."""
    return Path(metrics_dir) / f"label_summary_{label}_{run}.json"


def find_label_run(labels_dir: Path, run: str) -> tuple[str, dict[str, Any]]:
    """The label and manifest of label run ``run``.

    Raises:
        FileNotFoundError: no manifest, or more than one, for ``run``.
    """
    found = sorted(Path(labels_dir).glob(f"{_MANIFEST_PREFIX}*_{run}.json"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one label manifest for run {run} in {labels_dir}, "
            f"found {len(found)}"
        )
    label = found[0].stem.removeprefix(_MANIFEST_PREFIX).removesuffix(f"_{run}")
    return label, json.loads(found[0].read_text(encoding="utf-8"))


def source_run(manifest: Mapping[str, Any]) -> str | None:
    """The scoring run the label run's speeches came from, if any (latest stage)."""
    for stage in reversed(manifest.get("stages", [])):
        run = stage.get("selection", {}).get("from_run")
        if run:
            return run
    return None


# --- which speeches ----------------------------------------------------------


def ids_from_run(run: str, scores_dir: Path = SCORES_DIR) -> list[str]:
    """The speech ids of scoring run ``run``, in its ensemble file's order.

    Raises:
        FileNotFoundError: no such run, or it has no ensemble rows.
    """
    label = find_label(scores_dir, run)
    rows = read_rows(ensemble_path(scores_dir, label, run))
    if not rows:
        raise FileNotFoundError(f"scoring run {run} has no ensemble rows")
    return [str(row["speech_id"]) for row in rows]


def ids_from_file(path: Path) -> list[str]:
    """Speech ids from a file: a CSV's ``speech_id`` column, else one per line.

    The CSV form takes a check sheet as it is, so the agreement model can label
    exactly the hand-checked speeches. Blank lines and ``#`` lines are skipped.

    Raises:
        ValueError: the file lists no ids, or lists one twice.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        column = pd.read_csv(path, usecols=["speech_id"], dtype={"speech_id": str})
        ids = column["speech_id"].dropna().str.strip().tolist()
    else:
        lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
        ids = [line for line in lines if line and not line.startswith("#")]
    if not ids:
        raise ValueError(f"{path} lists no speech ids")
    repeated = sorted(i for i, n in Counter(ids).items() if n > 1)
    if repeated:
        raise ValueError(
            f"{path} lists {len(repeated)} id(s) twice, e.g. {repeated[:5]}"
        )
    return ids


def ids_sha256(speech_ids: Iterable[str]) -> str:
    """Hash of an id list, independent of its order."""
    return hashlib.sha256("\n".join(sorted(speech_ids)).encode("utf-8")).hexdigest()


def read_corpus_rows(
    speech_ids: Sequence[str], columns: Sequence[str], corpus: Path = CORPUS_PATH
) -> pd.DataFrame:
    """``columns`` of the given speeches, without reading the whole corpus.

    Raises:
        FileNotFoundError: no corpus.
        ValueError: no ids, or ids the corpus does not hold.
    """
    if not speech_ids:
        raise ValueError("no speech ids to read")
    if not Path(corpus).exists():
        raise FileNotFoundError(f"Corpus not found at {corpus}. Build it: make corpus")
    frame = pq.read_table(
        corpus,
        columns=list(dict.fromkeys(("speech_id", *columns))),
        filters=[("speech_id", "in", list(speech_ids))],
    ).to_pandas()
    missing = sorted(set(speech_ids) - set(frame["speech_id"]))
    if missing:
        raise ValueError(
            f"{len(missing)} speech id(s) are not in {corpus}, e.g. {missing[:5]}"
        )
    return frame


def fetch_speeches(
    speech_ids: Sequence[str], corpus: Path = CORPUS_PATH
) -> list[dict[str, Any]]:
    """The speeches, in the order given: id, chamber, source, Congress and text."""
    frame = read_corpus_rows(
        speech_ids, ("chamber", "source", "congress_number", "text"), corpus
    ).set_index("speech_id")
    return [
        {
            "speech_id": speech_id,
            "chamber": frame.at[speech_id, "chamber"],
            "source": frame.at[speech_id, "source"],
            "congress_number": int(frame.at[speech_id, "congress_number"]),
            "text": frame.at[speech_id, "text"],
        }
        for speech_id in speech_ids
    ]


def even_quota(available: Mapping[Any, int], total: int) -> dict[Any, int]:
    """Split ``total`` as evenly as possible over the strata in ``available``.

    A stratum too small for its share gives the shortfall back to the others,
    so the total is reached wherever the strata allow. Strata are taken in
    sorted order, which keeps the split deterministic.
    """
    quota = dict.fromkeys(available, 0)
    remaining = min(total, sum(available.values()))
    open_strata = sorted(available)
    while remaining > 0 and open_strata:
        share, extra = divmod(remaining, len(open_strata))
        granted = 0
        still_open = []
        for index, stratum in enumerate(open_strata):
            want = share + (1 if index < extra else 0)
            take = min(want, available[stratum] - quota[stratum])
            quota[stratum] += take
            granted += take
            if available[stratum] > quota[stratum]:
                still_open.append(stratum)
        remaining -= granted
        if granted == 0:
            break
        open_strata = still_open
    return quota


def stratified_subset(
    frame: pd.DataFrame, n: int, strata: Sequence[str], seed: int
) -> list[str]:
    """``n`` speech ids from ``frame``, spread evenly over ``strata``, sorted.

    Seeded per stratum, and sorted by speech_id before drawing, so the same
    speeches, size and seed give the same ids whatever order they arrive in.
    Asking for at least as many as the frame holds returns all of them.
    """
    if n <= 0:
        raise ValueError(f"cannot draw {n} speeches")
    if n >= len(frame):
        return sorted(frame["speech_id"])
    frame = frame.sort_values("speech_id")
    groups = {key: group for key, group in frame.groupby(list(strata))}
    quota = even_quota({key: len(group) for key, group in groups.items()}, n)
    picked = [
        groups[key].sample(n=quota[key], random_state=seed)["speech_id"]
        for key in sorted(groups)
        if quota[key]
    ]
    return sorted(pd.concat(picked))


def selection_ids(
    selection: Mapping[str, Any],
    corpus: Path = CORPUS_PATH,
    scores_dir: Path = SCORES_DIR,
) -> list[str]:
    """The speech ids a run stage's selection names.

    A selection is ``from_run`` (a scoring run) or ``ids_file``, optionally cut
    to ``limit`` speeches stratified over ``strata`` with ``seed``.
    """
    if selection.get("from_run"):
        ids = ids_from_run(selection["from_run"], scores_dir)
    elif selection.get("ids_file"):
        ids = ids_from_file(Path(selection["ids_file"]))
    else:
        raise ValueError("a selection needs from_run or ids_file")
    if selection.get("limit") is not None:
        strata = selection.get("strata") or list(LABEL_LIMIT_STRATA)
        frame = read_corpus_rows(ids, strata, corpus)
        ids = stratified_subset(frame, selection["limit"], strata, selection["seed"])
    return ids


# --- cost --------------------------------------------------------------------


def count_prompt_tokens(prompts: list[str]) -> list[int]:
    """Input tokens of each rendered prompt, via tiktoken (OpenAI's tokenizer).

    It only approximates DeepSeek's: on the S8 prompts DeepSeek billed about 3%
    more input than tiktoken counted (S9). Used for the estimate only; the
    summary reports billed usage.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    return [len(encoding.encode(prompt)) for prompt in prompts]


def preflight(
    token_counts: Mapping[str, int],
    models: Sequence[str],
    output_per_speech: int = LABEL_EXPECTED_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """What labelling these speeches would cost, per model.

    Input is the tiktoken count of each rendered prompt; output is an assumed
    ``output_per_speech`` until a pilot measures it.
    """
    n = len(token_counts)
    prompt_tokens = sum(token_counts.values())
    completion_tokens = output_per_speech * n
    per_model = {}
    for model in models:
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        per_model[model] = {
            "speeches": n,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "cost_per_1000_speeches_usd": cost / n * 1000 if n else 0.0,
        }
    return {
        "speeches": n,
        "tiktoken_input_tokens": prompt_tokens,
        "assumed_output_tokens_per_speech": output_per_speech,
        "per_model": per_model,
    }


# --- the run manifest ----------------------------------------------------------

#: Manifest keys that decide what a label means. A run resumes only if every one
#: matches. Pricing is recorded but not compared: a price change does not
#: change a label.
LABEL_DEFINING_KEYS: tuple[str, ...] = (
    "codebook_version",
    "codebook",
    "prompt_sha256",
    "corpus_sha256",
    "models",
)


def prompt_sha256(template: str) -> str:
    """Hash of the prompt template text."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def label_settings(template: str, corpus_sha256: str, model: str) -> dict[str, Any]:
    """Everything that defines a label run's labels (LABEL_DEFINING_KEYS)."""
    return {
        "codebook_version": CODEBOOK_VERSION,
        "codebook": {"speech_types": list(SPEECH_TYPES), "topics": list(TOPICS)},
        "prompt_sha256": prompt_sha256(template),
        "corpus_sha256": corpus_sha256,
        "models": {model: model_settings(label_spec(model))},
    }


def new_manifest(
    run: str,
    label: str,
    model: str,
    settings: Mapping[str, Any],
    template: str,
    prompt_path: Path,
    corpus_path: Path,
    seed: int,
) -> dict[str, Any]:
    """A new run's manifest, before its first stage is appended."""
    return {
        "timestamp": run,
        "label": label,
        "model": model,
        **settings,
        "prompt_path": str(prompt_path),
        # Logged in full, as for every scoring run (CLAUDE.md reproducibility).
        "prompt_template": template,
        "corpus_path": str(corpus_path),
        "pricing": {model: MODEL_PRICING[model]},
        "random_seed": seed,
        "stages": [],
    }


def stage_record(
    stage: int,
    selection: Mapping[str, Any],
    speech_ids: Sequence[str],
    pending: Sequence[str],
    estimate: Mapping[str, Any],
    started_at: str,
    concurrency: int = 1,
) -> dict[str, Any]:
    """What one invocation of a run set out to do."""
    return {
        "stage": stage,
        "started_at": started_at,
        "selection": dict(selection),
        "speeches_in_selection": len(speech_ids),
        "ids_sha256": ids_sha256(speech_ids),
        "speeches_to_label": len(pending),
        "call_timeout_seconds": scoring.CALL_TIMEOUT_SECONDS,
        "concurrency": concurrency,
        "cost_estimate": dict(estimate),
    }


# --- reading a run back ---------------------------------------------------------

#: Topic -> its block, from TOPIC_BLOCKS.
BLOCK_OF_TOPIC: dict[str, str] = {
    topic: block for block, topics in TOPIC_BLOCKS.items() for topic in topics
}

#: The label columns a speech gets in the analysis.
LABEL_COLUMNS: tuple[str, ...] = ("speech_type", "topic", "topic_block")


def topic_block(topic: str | None) -> str | None:
    """The block of a topic; None for no topic."""
    return None if topic is None else BLOCK_OF_TOPIC[topic]


def load_labels(run: str, labels_dir: Path = LABELS_DIR) -> pd.DataFrame:
    """One row per speech the label run attempted: the labels that count.

    Columns: speech_id, model, speech_type, topic, topic_block, label_ok and
    error. A speech that never got valid labels has null labels and its last
    error.

    Raises:
        FileNotFoundError: no manifest for ``run``.
    """
    label, manifest = find_label_run(labels_dir, run)
    model = manifest["model"]
    rows = effective_label_rows(
        read_rows(label_rows_path(labels_dir, label, model, run))
    )
    records = []
    for speech_id, row in rows.items():
        ok = label_ok(row)
        topic = row.get("topic") if ok else None
        records.append(
            {
                "speech_id": speech_id,
                "model": model,
                "speech_type": row.get("speech_type") if ok else None,
                "topic": topic,
                "topic_block": topic_block(topic),
                "label_ok": ok,
                "error": row.get("error"),
            }
        )
    return pd.DataFrame(
        records,
        columns=["speech_id", "model", *LABEL_COLUMNS, "label_ok", "error"],
    )


# --- summary ---------------------------------------------------------------------

#: How much of an error message the summary keeps.
_ERROR_CHARS = 160


def _shares(
    ok: Mapping[str, Mapping[str, Any]],
    speeches: Sequence[Mapping[str, Any]],
    by: str,
) -> dict[str, dict[str, Any]]:
    """Speech-type shares among the labelled speeches, per value of ``by``."""
    out: dict[str, dict[str, Any]] = {}
    for value in sorted({str(speech[by]) for speech in speeches}):
        types = Counter(
            ok[speech["speech_id"]]["speech_type"]
            for speech in speeches
            if str(speech[by]) == value and speech["speech_id"] in ok
        )
        n = sum(types.values())
        out[value] = {
            "n": n,
            "speech_type": {
                t: types.get(t, 0) / n if n else None for t in SPEECH_TYPES
            },
        }
    return out


def summarize_labels(
    rows: Sequence[Mapping[str, Any]],
    speeches: Sequence[Mapping[str, Any]],
    model: str,
) -> dict[str, Any]:
    """Counts, failures, tokens and cost of a label run, from its raw rows.

    Cost counts every attempt, because every attempt was billed. Failures are
    speeches still without valid labels, so a retried failure is not one.
    """
    effective = effective_label_rows(rows)
    ok = {sid: row for sid, row in effective.items() if label_ok(row)}
    failed = [s["speech_id"] for s in speeches if s["speech_id"] not in ok]
    types = Counter(row["speech_type"] for row in ok.values())
    topics = Counter(row["topic"] for row in ok.values() if row["topic"])
    blocks = Counter(topic_block(t) for t in topics.elements())
    errors = Counter(
        str(effective[sid]["error"]).split(":", 1)[0]
        for sid in failed
        if sid in effective
    )
    prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in rows)
    completion_tokens = sum(r.get("completion_tokens") or 0 for r in rows)
    return {
        "codebook_version": CODEBOOK_VERSION,
        "model": model,
        "speeches": len(speeches),
        "speeches_labelled": len(ok),
        "failures": len(failed),
        "calls": len(rows),
        "failed_attempts": sum(1 for r in rows if not label_ok(r)),
        "errors_by_type": dict(errors.most_common()),
        "failed_speeches": [
            {
                "speech_id": sid,
                "error": (effective.get(sid, {}).get("error") or "not attempted")[
                    :_ERROR_CHARS
                ],
            }
            for sid in failed
        ],
        "speech_type_counts": {t: types.get(t, 0) for t in SPEECH_TYPES},
        "topic_counts": {t: topics.get(t, 0) for t in TOPICS},
        "topic_block_counts": {b: blocks.get(b, 0) for b in TOPIC_BLOCKS},
        "speech_type_shares_by_chamber": _shares(ok, speeches, "chamber"),
        "speech_type_shares_by_source": _shares(ok, speeches, "source"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "mean_completion_tokens_per_call": (
            completion_tokens / len(rows) if rows else None
        ),
        "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in rows),
        "calls_with_reasoning_tokens": sum(
            1 for r in rows if (r.get("reasoning_tokens") or 0) > 0
        ),
        "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        "served_models": sorted(
            {r["served_model"] for r in rows if r.get("served_model")}
        ),
        "system_fingerprints": sorted(
            {r["system_fingerprint"] for r in rows if r.get("system_fingerprint")}
        ),
        "topic_blocks": {block: list(t) for block, t in TOPIC_BLOCKS.items()},
    }


#: Words the prompt allows in a reason.
REASON_MAX_WORDS = 15


def odd_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Valid answers worth a look: type ``other``, an over-long reason, or a
    much longer answer than usual."""
    ok = [r for r in effective_label_rows(rows).values() if label_ok(r)]
    if not ok:
        return []
    mean_out = sum(r.get("completion_tokens") or 0 for r in ok) / len(ok)
    cases = []
    for row in ok:
        why = []
        if row["speech_type"] == "other":
            why.append("type other")
        if len((row.get("reason") or "").split()) > REASON_MAX_WORDS:
            why.append(f"reason over {REASON_MAX_WORDS} words")
        if (row.get("completion_tokens") or 0) > 2 * mean_out:
            why.append(f"{row.get('completion_tokens')} output tokens")
        if why:
            cases.append(
                {
                    "speech_id": row["speech_id"],
                    "speech_type": row["speech_type"],
                    "topic": row["topic"],
                    "reason": row.get("reason"),
                    "why": ", ".join(why),
                }
            )
    return cases
