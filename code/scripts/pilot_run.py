"""Pilot: score ~200 sampled speeches through the three-model ensemble.

This is the gate before the full run. It answers two questions: do the scores
look plausible, and what would the whole corpus cost?

Usage (from the repo root):

    python code/scripts/pilot_run.py --dry-run          # sample + cost, no calls
    python code/scripts/pilot_run.py --sample-size 6    # cheap end-to-end check
    python code/scripts/pilot_run.py                    # the real pilot

Needs API keys in .env (copy .env.example). The corpus must already be built --
see code/scripts/build_corpus.py.

The sanity check at the end is the point of the exercise. score_speech.txt
defines ideology as -1.0 liberal to +1.0 conservative, so Republicans should
average positive and Democrats negative. If that comes out flat or inverted, do
not proceed to the full run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CORPUS_PATH,
    ENSEMBLE_DISAGREEMENT_THRESHOLD,
    ENSEMBLE_MODELS,
    METRICS_DIR,
    MIN_WORD_COUNT,
    PILOT_SAMPLE_SIZE,
    RANDOM_SEED,
    SCORE_PROMPT_PATH,
    SCORES_DIR,
)
from src.scoring import (
    ScoreResult,
    build_clients,
    estimate_cost,
    load_prompt_template,
    render_prompt,
    score_one,
    specs_for,
)

#: Assumed answer length for the pre-flight estimate. The prompt asks for two
#: floats and 1-2 sentences; deepseek-reasoner also bills hidden reasoning
#: tokens this cannot see, which is why the estimate is a floor.
ASSUMED_COMPLETION_TOKENS = 250

#: Columns needed to choose the sample. `text` is deliberately excluded -- see
#: `load_sample`.
SAMPLING_COLUMNS = ("speech_id", "party", "congress_number", "word_count")

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
    corpus_path: Path, sample_size: int, seed: int, min_words: int
) -> list[dict[str, Any]]:
    """Draw a party x congress stratified sample, then fetch its text.

    Two passes on purpose. The corpus is 426,718 rows and ~618 MB with text; a
    single read to pick 200 speeches would pull several GB into memory. Pass one
    reads only the small columns, pass two streams batches and keeps the matched
    rows.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Build it first: make corpus"
        )

    frame = pq.read_table(corpus_path, columns=list(SAMPLING_COLUMNS)).to_pandas()
    # Currently a no-op: the corpus is already filtered at MIN_WORD_COUNT. Kept
    # because the dataset is provisional and may be rebuilt at a lower bound.
    frame = frame[frame["word_count"] >= min_words]
    if frame.empty:
        raise ValueError(f"No speeches with word_count >= {min_words} in {corpus_path}")

    groups = {key: group for key, group in frame.groupby(["congress_number", "party"])}
    available = {key: len(group) for key, group in groups.items()}
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
            "congress_number": int(row.congress_number),
            "text": texts[row.speech_id],
        }
        for row in sampled.itertuples()
    ]


# --- cost preflight ----------------------------------------------------


def count_prompt_tokens(prompts: list[str]) -> int:
    """Total input tokens across rendered prompts, via tiktoken.

    OpenAI's tokenizer, so it only approximates DeepSeek and Anthropic. Used for
    the pre-flight estimate only; the closing summary reports billed usage.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(prompt)) for prompt in prompts)


def preflight(prompts: list[str], models: tuple[str, ...]) -> dict[str, Any]:
    """Estimate what this run will cost, per model and in total."""
    prompt_tokens = count_prompt_tokens(prompts)
    completion_tokens = ASSUMED_COMPLETION_TOKENS * len(prompts)
    per_model = {
        model: estimate_cost(model, prompt_tokens, completion_tokens)
        for model in models
    }
    return {
        "speeches": len(prompts),
        "prompt_tokens_per_model": prompt_tokens,
        "assumed_completion_tokens_per_model": completion_tokens,
        "estimated_cost_usd": per_model,
        "estimated_total_usd": sum(per_model.values()),
    }


def print_preflight(estimate: dict[str, Any]) -> None:
    """Show the estimate, with its caveats stated rather than implied."""
    print(f"\n  speeches            {estimate['speeches']:,}")
    print(f"  input tokens/model  {estimate['prompt_tokens_per_model']:,}")
    print(
        f"  assumed output      {estimate['assumed_completion_tokens_per_model']:,}"
        f" tokens/model ({ASSUMED_COMPLETION_TOKENS}/speech)"
    )
    print()
    for model, cost in estimate["estimated_cost_usd"].items():
        print(f"  {model:<30} ~${cost:7.3f}")
    print(f"  {'TOTAL':<30} ~${estimate['estimated_total_usd']:7.3f}")
    print(
        "\n  This is a floor, not a forecast: the tokenizer is OpenAI's and only\n"
        "  approximates the other two, and deepseek-reasoner bills hidden\n"
        "  reasoning tokens as output that this cannot see."
    )


# --- scoring -----------------------------------------------------------


def _row(speech: dict[str, Any], result: ScoreResult) -> dict[str, Any]:
    """One raw per-model JSONL record."""
    return {
        "speech_id": speech["speech_id"],
        "party": speech["party"],
        "congress_number": speech["congress_number"],
        "ideology_score": result.ideology_score,
        "tone_score": result.tone_score,
        "reasoning": result.reasoning,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "error": result.error,
    }


async def score_all(
    speeches: list[dict[str, Any]],
    template: str,
    handles: dict[str, TextIO],
    specs: dict[str, Any],
    clients: dict[str, Any],
) -> dict[str, list[ScoreResult]]:
    """Score every speech with every model, writing each result as it lands.

    The three models for one speech run concurrently; speeches run sequentially,
    to stay inside provider rate limits. Rows are flushed immediately so a crash
    keeps everything already paid for.
    """
    by_speech: dict[str, list[ScoreResult]] = defaultdict(list)

    for index, speech in enumerate(speeches, start=1):
        prompt = render_prompt(template, speech["text"])
        results = await asyncio.gather(
            *(score_one(specs[m], clients[m], prompt) for m in ENSEMBLE_MODELS)
        )
        for result in results:
            handle = handles[result.model]
            handle.write(json.dumps(_row(speech, result)) + "\n")
            handle.flush()
            by_speech[speech["speech_id"]].append(result)
            if result.error:
                print(
                    f"  [{index}/{len(speeches)}] {speech['speech_id']} "
                    f"{result.model}: {result.error}",
                    file=sys.stderr,
                )
        done = sum(1 for r in results if r.ok)
        print(
            f"  [{index}/{len(speeches)}] {speech['speech_id']} "
            f"{done}/{len(ENSEMBLE_MODELS)} models ok",
            flush=True,
        )
    return by_speech


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


def build_ensemble(
    speeches: list[dict[str, Any]], by_speech: dict[str, list[ScoreResult]]
) -> list[dict[str, Any]]:
    """Average the models that succeeded, recording how many there were.

    `n_models` is written on every row so a two-model average is never mistaken
    for a three-model one -- that would quietly bias the disagreement figures.
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
    ensemble: list[dict[str, Any]], by_speech: dict[str, list[ScoreResult]]
) -> dict[str, Any]:
    """Token totals, cost, the party sanity check and the disagreement count."""
    per_model: dict[str, dict[str, Any]] = {}
    for model in ENSEMBLE_MODELS:
        results = [r for rs in by_speech.values() for r in rs if r.model == model]
        prompt_tokens = sum(r.prompt_tokens for r in results)
        completion_tokens = sum(r.completion_tokens for r in results)
        per_model[model] = {
            "calls": len(results),
            "failures": sum(1 for r in results if not r.ok),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": estimate_cost(model, prompt_tokens, completion_tokens),
        }

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

    print("\nper model")
    for model, stats in summary["per_model"].items():
        print(
            f"  {model:<30} in {stats['prompt_tokens']:>9,}  "
            f"out {stats['completion_tokens']:>8,}  ${stats['cost_usd']:7.3f}"
            + (f"  ({stats['failures']} failed)" if stats["failures"] else "")
        )
    print(f"  {'TOTAL':<30} {'':>13} {'':>12}  ${summary['total_cost_usd']:7.3f}")

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--sample-size", type=int, default=PILOT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-words", type=int, default=MIN_WORD_COUNT)
    parser.add_argument("--output-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
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


async def main(argv: list[str] | None = None) -> int:
    """Sample, confirm the spend, score, aggregate and report."""
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    template = load_prompt_template(SCORE_PROMPT_PATH)
    speeches = load_sample(args.corpus, args.sample_size, args.seed, args.min_words)
    prompts = [render_prompt(template, s["text"]) for s in speeches]

    strata = defaultdict(int)
    for speech in speeches:
        strata[(speech["congress_number"], speech["party"])] += 1
    print(
        f"sampled {len(speeches)} speeches across {len(strata)} strata (seed {args.seed})"
    )

    estimate = preflight(prompts, ENSEMBLE_MODELS)
    print_preflight(estimate)

    if args.dry_run:
        print("\n--dry-run: stopping before any API call.")
        return 0

    # Check keys before asking anyone to approve a spend -- failing after the
    # confirmation, on a missing key, would be a needless round trip.
    specs = specs_for(ENSEMBLE_MODELS)
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

    manifest = {
        "timestamp": timestamp,
        "corpus_path": str(args.corpus),
        "sample_size": len(speeches),
        "random_seed": args.seed,
        "min_word_count": args.min_words,
        "models": {
            name: {
                "model": spec.model,
                "provider": spec.provider,
                "effective_temperature": spec.effective_temperature,
            }
            for name, spec in specs.items()
        },
        "prompt_path": str(SCORE_PROMPT_PATH),
        "prompt_template": template,
        "cost_estimate": estimate,
    }
    manifest_path = args.output_dir / f"pilot_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    handles: dict[str, TextIO] = {}
    try:
        for model in ENSEMBLE_MODELS:
            safe = model.replace("/", "_")
            handles[model] = (args.output_dir / f"pilot_{safe}_{timestamp}.jsonl").open(
                "w", encoding="utf-8"
            )
        by_speech = await score_all(speeches, template, handles, specs, clients)
    finally:
        for handle in handles.values():
            handle.close()

    ensemble = build_ensemble(speeches, by_speech)
    ensemble_path = args.output_dir / f"pilot_ensemble_{timestamp}.jsonl"
    with ensemble_path.open("w", encoding="utf-8") as fh:
        for row in ensemble:
            fh.write(json.dumps(row) + "\n")

    summary = summarize(ensemble, by_speech)
    summary_path = args.metrics_dir / f"pilot_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print_summary(summary)
    print(f"\nraw scores  {args.output_dir}")
    print(f"ensemble    {ensemble_path}")
    print(f"manifest    {manifest_path}")
    print(f"summary     {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
