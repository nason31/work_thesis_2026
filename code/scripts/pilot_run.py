"""Score sampled speeches through the model ensemble, in stages.

There is no full-corpus run (docs/decisions.md S11). The S8 sample is scored as
ONE run extended in stages (S13): a small first stage is the pilot gate, and
the same run is then resumed at the full size. A larger sample from the same
seed contains the smaller one, so a later stage scores only what is new, and a
failed call is retried rather than the run repeated. Nothing is paid for twice.

Usage (from the repo root):

    python code/scripts/pilot_run.py --label s8 --per-cell 4 --dry-run
    python code/scripts/pilot_run.py --label s8 --per-cell 4          # stage 1
    python code/scripts/pilot_run.py --resume <timestamp> --per-cell 100 --dry-run
    python code/scripts/pilot_run.py --resume <timestamp> --per-cell 100

`--dry-run` samples and prices what is left, then stops without calling
anything. `--resume` also picks up a run that crashed or had failed calls: run
it again with the same `--per-cell`. It refuses if anything that defines a
score has changed since the run began (corpus, prompt, models, seed, filters).

Needs API keys in .env (copy .env.example). The merged corpus must already be
built -- `make corpus`.

The party check at the end decides whether to continue. score_speech.txt
defines ideology as -1.0 liberal to +1.0 conservative, so Republicans should
average positive and Democrats negative. If that comes out flat or inverted, do
not extend the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CALL_TIMEOUT_SECONDS,
    CARRY_OVER_EQUIVALENTS,
    CORPUS_PATH,
    ENSEMBLE_DISAGREEMENT_THRESHOLD,
    ENSEMBLE_MODELS,
    METRICS_DIR,
    MIN_WORD_COUNT,
    PILOT_MEASURED_TOKENS,
    PILOT_PER_CELL,
    PILOT_TIKTOKEN_INPUT_PER_SPEECH,
    RANDOM_SEED,
    SCORE_PROMPT_PATH,
    SCORES_DIR,
    STRATIFY_BY,
)
from src.corpus import _fingerprint
from src.runs import (
    DEFAULT_LABEL,
    RUN_SETTING_KEYS,
    carry_rows,
    check_nested,
    check_resumable,
    check_served,
    effective_rows,
    ensemble_path,
    find_label,
    is_ok,
    manifest_path,
    match_models,
    model_path,
    pending_models,
    read_rows,
    summary_path,
)
from src.scoring import (
    ModelSpec,
    ScoreResult,
    build_clients,
    estimate_cost,
    load_prompt_template,
    render_prompt,
    score_one,
    specs_for,
)

#: Columns needed to choose the sample, beyond the stratification keys. `text` is
#: deliberately excluded -- see `load_sample`.
SAMPLING_COLUMNS = ("speech_id", "word_count")

BATCH_SIZE = 50_000


# --- sampling ----------------------------------------------------------


def allocate(
    strata: list[tuple[Any, ...]], available: dict[Any, int], total: int
) -> dict[Any, int]:
    """Split `total` as evenly as possible over `strata`.

    Any stratum too small for its share gives the shortfall back to the others,
    so the sample still reaches `total` where the corpus allows. Iterating over
    sorted strata keeps the split deterministic.
    """
    quota = {s: 0 for s in strata}
    remaining = total
    open_strata = sorted(strata)

    while remaining > 0 and open_strata:
        share, extra = divmod(remaining, len(open_strata))
        if share == 0 and extra == 0:
            break
        granted = 0
        still_open = []
        for index, stratum in enumerate(open_strata):
            want = share + (1 if index < extra else 0)
            room = available[stratum] - quota[stratum]
            take = min(want, room)
            quota[stratum] += take
            granted += take
            if room - take > 0:
                still_open.append(stratum)
        remaining -= granted
        if granted == 0:
            break
        open_strata = still_open
    return quota


def load_sample(
    corpus_path: Path,
    seed: int,
    min_words: int,
    per_cell: int | None = None,
    sample_size: int | None = None,
    strata: tuple[str, ...] = STRATIFY_BY,
) -> list[dict[str, Any]]:
    """Draw a stratified sample over `strata`, then fetch its text.

    Two allocation modes. `per_cell` takes a fixed number from every cell, which
    is what a balanced design wants: each cell contributes equally regardless of
    how large it is in the corpus. `sample_size` instead splits a total as evenly
    as it can, falling back on `allocate` when a cell is short. Exactly one must
    be given.

    Two passes on purpose. The corpus is ~540k rows, almost all of its bytes
    text; a single read to pick a few thousand speeches would pull several GB
    into memory. Pass one reads only the small columns, pass two streams batches and
    keeps the matched rows.
    """
    if (per_cell is None) == (sample_size is None):
        raise ValueError("give exactly one of per_cell or sample_size")
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Build it first: make corpus"
        )

    columns = list(dict.fromkeys(SAMPLING_COLUMNS + strata))
    frame = pq.read_table(corpus_path, columns=columns).to_pandas()
    # Currently a no-op: the corpus is already filtered at MIN_WORD_COUNT. Kept
    # because the dataset is provisional and may be rebuilt at a lower bound.
    frame = frame[frame["word_count"] >= min_words]
    if frame.empty:
        raise ValueError(f"No speeches with word_count >= {min_words} in {corpus_path}")

    # Explicit comprehension, not dict(frame.groupby(...)): pandas 3.0 raises
    # "'list' object is not callable" when a GroupBy over several keys is fed
    # straight to dict().
    groups = {key: group for key, group in frame.groupby(list(strata))}
    available = {key: len(group) for key, group in groups.items()}
    if per_cell is not None:
        short = {k: n for k, n in available.items() if n < per_cell}
        if short:
            raise ValueError(
                f"{len(short)} stratum/strata hold fewer than {per_cell} speeches: "
                + ", ".join(f"{k}={n}" for k, n in sorted(short.items())[:5])
                + ". Lower --per-cell, or use --sample-size to split a total."
            )
        quota = {k: per_cell for k in groups}
    else:
        quota = allocate(list(groups), available, sample_size)

    picked = []
    for stratum in sorted(groups):
        take = quota[stratum]
        if take:
            # Seeded per stratum so the draw is reproducible and independent of
            # dict ordering.
            picked.append(groups[stratum].sample(n=take, random_state=seed))
    sampled = pd.concat(picked).sort_values("speech_id")
    wanted = set(sampled["speech_id"])

    texts: dict[str, str] = {}
    for batch in pq.ParquetFile(corpus_path).iter_batches(
        batch_size=BATCH_SIZE, columns=["speech_id", "text"]
    ):
        ids = batch.column("speech_id").to_pylist()
        mask = [i in wanted for i in ids]
        if not any(mask):
            continue
        matched = batch.filter(pa.array(mask, type=pa.bool_()))
        texts.update(
            zip(
                matched.column("speech_id").to_pylist(),
                matched.column("text").to_pylist(),
            )
        )

    missing = wanted - texts.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} sampled speech_id(s) had no text row")

    return [
        {
            "speech_id": row.speech_id,
            "party": row.party,
            "chamber": row.chamber,
            "congress_number": int(row.congress_number),
            "text": texts[row.speech_id],
        }
        for row in sampled.itertuples()
    ]


# --- cost preflight ----------------------------------------------------


def count_prompt_tokens(prompts: list[str]) -> list[int]:
    """Input tokens of each rendered prompt, via tiktoken.

    OpenAI's tokenizer, so it only approximates DeepSeek and Anthropic. Used for
    the pre-flight estimate only; the closing summary reports billed usage.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    return [len(encoding.encode(prompt)) for prompt in prompts]


def preflight(
    token_counts: Mapping[str, int],
    pending: Mapping[str, list[str]],
    models: tuple[str, ...],
) -> dict[str, Any]:
    """Estimate what the pending calls will cost, per model and in total.

    Only what is still owed is priced: on a resumed run, speeches a model has
    already scored cost nothing. Grounded in measured usage (config.py
    PILOT_MEASURED_TOKENS) rather than a flat guess. tiktoken counts the pending
    prompts and the result is scaled against the measured tiktoken-per-speech
    figure, so longer
    or shorter speeches move the estimate. Each model then uses its own measured
    input and output tokens per speech -- the providers differ by ~30% on input
    for identical text because their tokenizers differ, and the reasoner emits
    ~7x the output.
    """
    per_model: dict[str, dict[str, Any]] = {}
    for model in models:
        measured = PILOT_MEASURED_TOKENS.get(model)
        if measured is None:
            raise KeyError(
                f"no measured token usage for {model!r}; add it to "
                "PILOT_MEASURED_TOKENS in config.py from a small staged run"
            )
        owed = [
            speech for speech, models_owed in pending.items() if model in models_owed
        ]
        n = len(owed)
        # How much longer or shorter these speeches are than the pilot's.
        length_ratio = (
            (sum(token_counts[s] for s in owed) / n) / PILOT_TIKTOKEN_INPUT_PER_SPEECH
            if n
            else 1.0
        )
        prompt_tokens = round(measured["input"] * length_ratio * n)
        completion_tokens = round(measured["output"] * length_ratio * n)
        per_model[model] = {
            "speeches": n,
            "length_ratio_vs_pilot": round(length_ratio, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        }
    return {
        "speeches": len(pending),
        "calls": sum(len(models_owed) for models_owed in pending.values()),
        "tiktoken_input_tokens": sum(token_counts[s] for s in pending),
        "per_model": per_model,
        "estimated_total_usd": sum(m["cost_usd"] for m in per_model.values()),
    }


def print_preflight(estimate: dict[str, Any]) -> None:
    """Show the estimate, with its basis stated rather than implied."""
    print(f"\n  speeches to score   {estimate['speeches']:,}")
    print(f"  calls to make       {estimate['calls']:,}")
    print()
    print(
        f"  {'model':<30}{'speeches':>10}{'length':>8}{'in tokens':>12}"
        f"{'out tokens':>12}{'cost':>10}"
    )
    for model, m in estimate["per_model"].items():
        print(
            f"  {model:<30}{m['speeches']:>10,}{m['length_ratio_vs_pilot']:>7.2f}x"
            f"{m['prompt_tokens']:>12,}{m['completion_tokens']:>12,}"
            f"{'$' + format(m['cost_usd'], '.2f'):>10}"
        )
    print(
        f"  {'TOTAL':<30}{'':>10}{'':>8}{'':>12}{'':>12}"
        f"{'$' + format(estimate['estimated_total_usd'], '.2f'):>10}"
    )
    print(
        "\n  Based on measured per-model usage (PILOT_MEASURED_TOKENS), scaled\n"
        "  by these speeches' length ('length' = x the measured average). Far\n"
        "  better than a flat guess, but still an estimate: reasoning length\n"
        "  varies per speech and list prices change."
    )


# --- scoring -----------------------------------------------------------


def _row(
    speech: dict[str, Any], result: ScoreResult, stage: int, scored_at: str
) -> dict[str, Any]:
    """One raw per-model JSONL record."""
    return {
        "speech_id": speech["speech_id"],
        "party": speech["party"],
        "chamber": speech.get("chamber"),
        "congress_number": speech["congress_number"],
        "ideology_score": result.ideology_score,
        "tone_score": result.tone_score,
        "reasoning": result.reasoning,
        "model": result.model,
        "served_model": result.served_model,
        "system_fingerprint": result.system_fingerprint,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "error": result.error,
        "stage": stage,
        "scored_at": scored_at,
    }


async def score_all(
    speeches: list[dict[str, Any]],
    template: str,
    handles: dict[str, TextIO],
    specs: dict[str, Any],
    clients: dict[str, Any],
    pending: Mapping[str, list[str]],
    stage: int,
) -> None:
    """Score each pending speech with the models that still owe it a score.

    Those models run concurrently for one speech; speeches run sequentially,
    to stay inside provider rate limits. Rows are appended and flushed as they
    land, so a crash keeps everything already paid for, and `--resume` carries
    on from there.
    """
    todo = [speech for speech in speeches if speech["speech_id"] in pending]
    for index, speech in enumerate(todo, start=1):
        models = pending[speech["speech_id"]]
        prompt = render_prompt(template, speech["text"])
        results = await asyncio.gather(
            *(score_one(specs[m], clients[m], prompt) for m in models)
        )
        scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for result in results:
            handle = handles[result.model]
            handle.write(json.dumps(_row(speech, result, stage, scored_at)) + "\n")
            handle.flush()
            if result.error:
                print(
                    f"  [{index}/{len(todo)}] {speech['speech_id']} "
                    f"{result.model}: {result.error}",
                    file=sys.stderr,
                )
        done = sum(1 for r in results if r.ok)
        print(
            f"  [{index}/{len(todo)}] {speech['speech_id']} "
            f"{done}/{len(models)} models ok",
            flush=True,
        )


# --- ensemble ----------------------------------------------------------


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and sample std. Std is None below two values.

    One model agreeing with itself is not agreement, so a single score gets no
    standard deviation rather than a misleading 0.0.
    """
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return statistics.fmean(values), statistics.stdev(values)


def results_by_speech(
    rows_by_model: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[ScoreResult]]:
    """The run's files as results: per speech, each model's row that counts."""
    by_speech: dict[str, list[ScoreResult]] = defaultdict(list)
    for model, rows in rows_by_model.items():
        for speech_id, row in effective_rows(rows).items():
            by_speech[speech_id].append(
                ScoreResult(
                    model=model,
                    ideology_score=row.get("ideology_score"),
                    tone_score=row.get("tone_score"),
                    reasoning=row.get("reasoning"),
                    error=row.get("error"),
                )
            )
    return by_speech


def build_ensemble(
    speeches: list[dict[str, Any]], by_speech: dict[str, list[ScoreResult]]
) -> list[dict[str, Any]]:
    """Average the models that succeeded, recording how many there were.

    `n_models` is written on every row so an average over fewer models is never mistaken
    for a full-ensemble one -- that would quietly bias the disagreement figures.
    """
    rows = []
    for speech in speeches:
        results = [r for r in by_speech.get(speech["speech_id"], []) if r.ok]
        ideology_mean, ideology_std = _mean_std(
            [r.ideology_score for r in results if r.ideology_score is not None]
        )
        tone_mean, tone_std = _mean_std(
            [r.tone_score for r in results if r.tone_score is not None]
        )
        rows.append(
            {
                "speech_id": speech["speech_id"],
                "party": speech["party"],
                "chamber": speech.get("chamber"),
                "congress_number": speech["congress_number"],
                "ideology_score_mean": ideology_mean,
                "ideology_score_std": ideology_std,
                "tone_score_mean": tone_mean,
                "tone_score_std": tone_std,
                "n_models": len(results),
            }
        )
    return rows


# --- summary -----------------------------------------------------------


def summarize(
    ensemble: list[dict[str, Any]],
    rows_by_model: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Token totals, cost, the party sanity check and the disagreement count.

    Cost counts every attempt, because every attempt was billed. Failures count
    speeches still without a score, so a failure that a later stage retried
    successfully is not one. `served_models` lists what each provider says
    answered -- more than one entry means the model changed during the run.
    """
    per_model: dict[str, dict[str, Any]] = {}
    per_stage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "cost_usd": 0.0}
    )
    for model in ENSEMBLE_MODELS:
        rows = rows_by_model.get(model, [])
        prompt_tokens = sum(r.get("prompt_tokens") or 0 for r in rows)
        completion_tokens = sum(r.get("completion_tokens") or 0 for r in rows)
        per_model[model] = {
            "calls": len(rows),
            "failed_attempts": sum(1 for r in rows if not is_ok(r)),
            "failures": sum(1 for r in effective_rows(rows).values() if not is_ok(r)),
            "served_models": sorted(
                {r["served_model"] for r in rows if r.get("served_model")}
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        }
        for r in rows:
            stage = per_stage[str(r.get("stage", 1))]
            stage["calls"] += 1
            stage["cost_usd"] += estimate_cost(
                model, r.get("prompt_tokens") or 0, r.get("completion_tokens") or 0
            )

    by_party: dict[str, dict[str, Any]] = {}
    for party in sorted({row["party"] for row in ensemble}):
        scores = [
            row["ideology_score_mean"]
            for row in ensemble
            if row["party"] == party and row["ideology_score_mean"] is not None
        ]
        mean, std = _mean_std(scores)
        by_party[party] = {"n": len(scores), "ideology_mean": mean, "ideology_std": std}

    disagreement = [
        row
        for row in ensemble
        if row["ideology_score_std"] is not None
        and row["ideology_score_std"] > ENSEMBLE_DISAGREEMENT_THRESHOLD
    ]
    return {
        "speeches_scored": sum(1 for row in ensemble if row["n_models"] > 0),
        "speeches_sampled": len(ensemble),
        "per_model": per_model,
        "per_stage": dict(sorted(per_stage.items(), key=lambda item: int(item[0]))),
        "total_cost_usd": sum(m["cost_usd"] for m in per_model.values()),
        "ideology_by_party": by_party,
        "disagreement_threshold": ENSEMBLE_DISAGREEMENT_THRESHOLD,
        "high_disagreement_speeches": len(disagreement),
        "partial_ensemble_rows": sum(
            1 for row in ensemble if 0 < row["n_models"] < len(ENSEMBLE_MODELS)
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print the summary, leading with the check that decides the next step."""
    print("\n" + "=" * 68)
    print(
        f"scored {summary['speeches_scored']:,} of {summary['speeches_sampled']:,} speeches"
    )
    if summary["partial_ensemble_rows"]:
        print(
            f"  {summary['partial_ensemble_rows']} row(s) averaged fewer than "
            f"{len(ENSEMBLE_MODELS)} models - see n_models"
        )

    print("\nper model (all stages; cost counts every attempt)")
    for model, stats in summary["per_model"].items():
        print(
            f"  {model:<30} in {stats['prompt_tokens']:>9,}  "
            f"out {stats['completion_tokens']:>8,}  ${stats['cost_usd']:7.3f}"
            + (f"  ({stats['failures']} unscored)" if stats["failures"] else "")
        )
        if len(stats["served_models"]) > 1:
            print(f"    *** served by several models: {stats['served_models']} ***")
    print(f"  {'TOTAL':<30} {'':>13} {'':>12}  ${summary['total_cost_usd']:7.3f}")
    for stage, stats in summary["per_stage"].items():
        print(f"  stage {stage}: {stats['calls']:,} calls, ${stats['cost_usd']:.3f}")

    print("\nsanity check - ideology by party (expect R positive, D negative)")
    for party, stats in summary["ideology_by_party"].items():
        mean = stats["ideology_mean"]
        std = stats["ideology_std"]
        mean_text = f"{mean:+.3f}" if mean is not None else "n/a"
        std_text = f"{std:.3f}" if std is not None else "n/a"
        print(f"  {party}  n={stats['n']:<4} mean {mean_text}  std {std_text}")

    means = summary["ideology_by_party"]
    if "D" in means and "R" in means:
        d, r = means["D"]["ideology_mean"], means["R"]["ideology_mean"]
        if d is not None and r is not None:
            verdict = "as expected" if r > d else "*** INVERTED - investigate ***"
            print(f"  R - D = {r - d:+.3f}  {verdict}")

    print(
        f"\n{summary['high_disagreement_speeches']} speech(es) with cross-model "
        f"ideology std > {summary['disagreement_threshold']}"
    )
    print("=" * 68)


# --- entry point -------------------------------------------------------


def model_manifest(specs: dict[str, ModelSpec]) -> dict[str, dict[str, object]]:
    """Per-model settings for the run manifest.

    ``max_output_tokens`` is recorded because it decides whether a reasoning
    model answers at all: its reasoning is billed from the same budget, and at
    a shared 1,024 cap the DeepSeek reasoner returned nothing (docs/decisions.md
    S10, O6). ``request_options`` (DeepSeek's thinking mode and effort) change
    what the model does. Both are part of what a resume must find unchanged.
    """
    return {
        name: {
            "model": spec.model,
            "provider": spec.provider,
            "effective_temperature": spec.effective_temperature,
            "max_output_tokens": spec.max_output_tokens,
            "request_options": spec.request_options,
        }
        for name, spec in specs.items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--per-cell",
        type=int,
        default=None,
        help=f"speeches per {' x '.join(STRATIFY_BY)} cell (default "
        f"{PILOT_PER_CELL}; when resuming, the run's current size)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="total speeches split evenly over cells, instead of --per-cell",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-words", type=int, default=MIN_WORD_COUNT)
    parser.add_argument("--output-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument(
        "--label",
        default=None,
        help=f"names a new run's files, e.g. s8 (default {DEFAULT_LABEL!r}); "
        "a resumed run keeps its own",
    )
    parser.add_argument(
        "--resume",
        metavar="TIMESTAMP",
        default=None,
        help="extend or finish that run: score only what has no score yet",
    )
    parser.add_argument(
        "--carry-over-from",
        metavar="TIMESTAMP",
        default=None,
        help="start a new run with that run's answers for the current models, "
        "scoring only what it lacks (S16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sample and estimate cost, then stop without calling any model",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the cost confirmation prompt (for unattended runs)",
    )
    return parser.parse_args(argv)


def load_manifest(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """The label and manifest of the run named by ``--resume``.

    Raises:
        FileNotFoundError: no manifest for the run.
        ValueError: ``--label`` names a different label than the run has.
    """
    run = args.resume
    label = find_label(args.output_dir, run)
    if args.label is not None and args.label != label:
        raise ValueError(
            f"run {run} is labelled {label!r}, not {args.label!r}; "
            "leave out --label when resuming"
        )
    return label, json.loads(manifest_path(args.output_dir, label, run).read_text())


def sample_design(
    args: argparse.Namespace, manifest: dict[str, Any] | None
) -> tuple[int | None, int | None]:
    """(per_cell, sample_size) to draw: as given, else the run's current size.

    A bare ``--resume`` therefore finishes the run as declared -- retrying its
    failures -- and only an explicit ``--per-cell`` extends it.
    """
    if args.sample_size is not None:
        return None, args.sample_size
    if args.per_cell is not None:
        return args.per_cell, None
    if manifest is None:
        return PILOT_PER_CELL, None
    if manifest.get("per_cell") is not None:
        return manifest["per_cell"], None
    return None, manifest["stages"][-1]["requested_sample_size"]


def check_run(
    args: argparse.Namespace,
    label: str,
    manifest: dict[str, Any],
    settings: dict[str, Any],
    speeches: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Check the resumed run may take this sample; return its rows per model.

    Raises:
        ValueError: a score-defining setting changed, or the new sample would
            leave out speeches the run already scored.
    """
    run = args.resume
    check_resumable(manifest, settings)
    rows_by_model = {
        model: read_rows(model_path(args.output_dir, label, model, run))
        for model in ENSEMBLE_MODELS
    }
    check_nested(
        {str(row["speech_id"]) for rows in rows_by_model.values() for row in rows},
        {speech["speech_id"] for speech in speeches},
    )
    return rows_by_model


def prepare_carry_over(
    args: argparse.Namespace,
    settings: dict[str, Any],
    speeches: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Check an earlier run's answers may enter this run; return them per model.

    Returns (mapping current model -> earlier model, carried rows per model).

    Raises:
        FileNotFoundError: no manifest for the earlier run.
        ValueError: a score-defining setting differs, a model's settings
            differ, an equivalent name was served by another model, or the
            earlier run scored speeches outside this sample.
    """
    old = args.carry_over_from
    old_label = find_label(args.output_dir, old)
    old_manifest = json.loads(
        manifest_path(args.output_dir, old_label, old).read_text()
    )
    check_resumable(old_manifest, settings, keys=RUN_SETTING_KEYS)
    mapping = match_models(
        old_manifest.get("models", {}), settings["models"], CARRY_OVER_EQUIVALENTS
    )
    old_rows = {
        model: read_rows(model_path(args.output_dir, old_label, old_model, old))
        for model, old_model in mapping.items()
    }
    for model, old_model in mapping.items():
        if model != old_model:
            check_served(old_rows[model], model)
    sample_ids = {speech["speech_id"] for speech in speeches}
    check_nested(
        {str(row["speech_id"]) for rows in old_rows.values() for row in rows},
        sample_ids,
    )
    return mapping, {
        model: carry_rows(rows, old, sample_ids) for model, rows in old_rows.items()
    }


def write_outputs(
    args: argparse.Namespace, label: str, run: str, speeches: list[dict[str, Any]]
) -> tuple[dict[str, Any], Path, Path]:
    """Rebuild the ensemble and summary from everything the run's files hold."""
    rows_by_model = {
        model: read_rows(model_path(args.output_dir, label, model, run))
        for model in ENSEMBLE_MODELS
    }
    ensemble = build_ensemble(speeches, results_by_speech(rows_by_model))
    ensemble_file = ensemble_path(args.output_dir, label, run)
    with ensemble_file.open("w", encoding="utf-8") as fh:
        for row in ensemble:
            fh.write(json.dumps(row) + "\n")

    summary = summarize(ensemble, rows_by_model)
    summary_file = summary_path(args.metrics_dir, label, run)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    return summary, ensemble_file, summary_file


async def main(argv: list[str] | None = None) -> int:
    """Sample, price what is left, confirm the spend, score and report."""
    args = parse_args(argv)
    if args.resume and args.carry_over_from:
        print(
            "pilot_run: --carry-over-from starts a new run; not with --resume",
            file=sys.stderr,
        )
        return 2

    label, manifest = args.label or DEFAULT_LABEL, None
    if args.resume:
        try:
            label, manifest = load_manifest(args)
        except (FileNotFoundError, ValueError) as error:
            print(f"pilot_run: {error}", file=sys.stderr)
            return 2
    per_cell, sample_size = sample_design(args, manifest)

    template = load_prompt_template(SCORE_PROMPT_PATH)
    specs = specs_for(ENSEMBLE_MODELS)
    speeches = load_sample(
        args.corpus,
        seed=args.seed,
        min_words=args.min_words,
        per_cell=per_cell,
        sample_size=sample_size,
    )
    # The path alone does not identify the data: corpus.parquet was the
    # Stanford-only build until 2026-09-29 and is the merged corpus since (D19).
    corpus_sha256 = _fingerprint(args.corpus)
    print(f"corpus  {args.corpus} (sha256 {corpus_sha256[:12]})")

    cells: dict[tuple[Any, ...], int] = defaultdict(int)
    for speech in speeches:
        cells[tuple(speech[k] for k in STRATIFY_BY)] += 1
    print(
        f"sample  {len(speeches):,} speeches across {len(cells)} "
        f"{' x '.join(STRATIFY_BY)} cells "
        f"({min(cells.values())}-{max(cells.values())} per cell, seed {args.seed})"
    )

    # Everything that decides what a score means. A run resumes only if all of
    # it is unchanged (S13).
    settings = {
        "corpus_sha256": corpus_sha256,
        "prompt_template": template,
        "models": model_manifest(specs),
        "random_seed": args.seed,
        "min_word_count": args.min_words,
        "stratify_by": list(STRATIFY_BY),
    }

    if manifest is not None:
        run = args.resume
        try:
            rows_by_model = check_run(args, label, manifest, settings, speeches)
        except ValueError as error:
            print(f"pilot_run: {error}", file=sys.stderr)
            return 2
    else:
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rows_by_model = {model: [] for model in ENSEMBLE_MODELS}

    carry_mapping: dict[str, str] = {}
    if args.carry_over_from:
        try:
            carry_mapping, rows_by_model = prepare_carry_over(args, settings, speeches)
        except (FileNotFoundError, ValueError) as error:
            print(f"pilot_run: {error}", file=sys.stderr)
            return 2
        print(
            f"carry   {sum(len(r) for r in rows_by_model.values()):,} answers from "
            f"run {args.carry_over_from}: "
            + ", ".join(
                f"{old} -> {new}" if old != new else new
                for new, old in carry_mapping.items()
            )
        )

    pending = pending_models(
        [speech["speech_id"] for speech in speeches], rows_by_model, ENSEMBLE_MODELS
    )
    attempted = {
        (str(row["speech_id"]), model)
        for model, rows in rows_by_model.items()
        for row in rows
    }
    retries = sum(
        (speech_id, model) in attempted
        for speech_id, models in pending.items()
        for model in models
    )
    print(
        f"run     {label} {run}"
        + (" (resumed)" if args.resume else " (new)")
        + f": {len(speeches) - len(pending):,} of {len(speeches):,} speeches fully "
        f"scored; {retries:,} failed call(s) to retry"
    )

    if not pending:
        print("\nNothing left to score.")
        if args.resume and not args.dry_run:
            summary, _, _ = write_outputs(args, label, run, speeches)
            print_summary(summary)
        return 0

    by_id = {speech["speech_id"]: speech for speech in speeches}
    owed = list(pending)
    token_counts = dict(
        zip(
            owed,
            count_prompt_tokens(
                [render_prompt(template, by_id[s]["text"]) for s in owed]
            ),
        )
    )
    estimate = preflight(token_counts, pending, ENSEMBLE_MODELS)
    print_preflight(estimate)

    if args.dry_run:
        print("\n--dry-run: stopping before any API call.")
        return 0

    # Check keys before asking anyone to approve a spend -- failing after the
    # confirmation, on a missing key, would be a needless round trip.
    clients = build_clients(specs)

    if not args.yes:
        try:
            answer = input("\nProceed and spend this? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "y":
            print("Aborted; nothing was called.")
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    if manifest is None:
        manifest = {
            "timestamp": run,
            "label": label,
            "corpus_path": str(args.corpus),
            **settings,
            "prompt_path": str(SCORE_PROMPT_PATH),
            "stages": [],
        }
        if carry_mapping:
            carried_ids = {
                str(row["speech_id"]) for rows in rows_by_model.values() for row in rows
            }
            manifest["carried_over_from"] = {
                "run": args.carry_over_from,
                "models": carry_mapping,
                "rows": {model: len(rows) for model, rows in rows_by_model.items()},
            }
            manifest["stages"].append(
                {
                    "stage": 1,
                    "carried_over_from": args.carry_over_from,
                    "speeches": len(carried_ids),
                    "rows": manifest["carried_over_from"]["rows"],
                }
            )
    stage = len(manifest["stages"]) + 1
    manifest["per_cell"] = per_cell
    manifest["sample_size"] = len(speeches)
    manifest["stages"].append(
        {
            "stage": stage,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "per_cell": per_cell,
            "requested_sample_size": sample_size,
            "speeches_in_sample": len(speeches),
            "speeches_to_score": len(pending),
            "calls_planned": estimate["calls"],
            "retries_planned": retries,
            "call_timeout_seconds": CALL_TIMEOUT_SECONDS,
            "cost_estimate": estimate,
        }
    )
    manifest_file = manifest_path(args.output_dir, label, run)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")

    handles: dict[str, TextIO] = {}
    try:
        for model in ENSEMBLE_MODELS:
            handles[model] = model_path(args.output_dir, label, model, run).open(
                "a", encoding="utf-8"
            )
        if carry_mapping:
            for model, rows in rows_by_model.items():
                for row in rows:
                    handles[model].write(json.dumps(row) + "\n")
                handles[model].flush()
        await score_all(speeches, template, handles, specs, clients, pending, stage)
    finally:
        for handle in handles.values():
            handle.close()

    summary, ensemble_file, summary_file = write_outputs(args, label, run, speeches)
    print_summary(summary)
    print(f"\nrun         {label} {run} (stage {stage})")
    print(f"raw scores  {args.output_dir}")
    print(f"ensemble    {ensemble_file}")
    print(f"manifest    {manifest_file}")
    print(f"summary     {summary_file}")
    print(
        f"\nTo extend or finish this run: python code/scripts/pilot_run.py "
        f"--resume {run} --per-cell <n> --dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
