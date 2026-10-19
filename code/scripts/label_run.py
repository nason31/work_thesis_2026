"""Label speeches by speech type and policy topic, in stages.

A pass separate from scoring (docs/decisions.md L1-L5). One model per run,
deepseek-flash with thinking disabled by default; GPT-4o is the agreement
check on the hand-checked subsample. The speeches are those of a scoring run
(``--from-run``) or of a file (``--ids-file``: one id per line, or a CSV with a
``speech_id`` column such as a check sheet), optionally cut to ``--limit``
speeches stratified by chamber and source.

Usage (from the repo root):

    python code/scripts/label_run.py --from-run 20261003T110419Z --limit 200 \\
        --label s8 --dry-run
    python code/scripts/label_run.py --from-run 20261003T110419Z --limit 200 \\
        --label s8                                         # pilot
    python code/scripts/label_run.py --resume <timestamp> --dry-run
    python code/scripts/label_run.py --resume <timestamp> \\
        --from-run 20261003T110419Z                        # extend to all 5,200

`--concurrency N` keeps N speeches in flight at once (S17; default 1). The run
stops if DeepSeek answers from another model build than the run's earlier
answers (S18).

`--dry-run` selects and prices what is left, then stops: no API key needed, no
call, no file written. `--resume` alone finishes a run (retries its failures);
with a new selection it extends the run, which must contain every speech
already labelled. It refuses if the codebook, the prompt, the corpus or the
model's settings changed.

Spending always asks for a typed "yes". There is deliberately no flag to skip
it (docs/decisions.md E7).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CORPUS_PATH,
    DEFAULT_CONCURRENCY,
    FINGERPRINT_PINNED_MODELS,
    LABEL_LIMIT_STRATA,
    LABEL_MODEL,
    LABEL_PROMPT_PATH,
    LABELS_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    SCORES_DIR,
    TOPIC_BLOCKS,
)
from src.corpus import _fingerprint
from src.labeling import (
    LABEL_DEFINING_KEYS,
    LABEL_MODEL_SPECS,
    count_prompt_tokens,
    fetch_speeches,
    find_label_run,
    ids_sha256,
    label_manifest_path,
    label_one,
    label_row,
    label_rows_path,
    label_settings,
    label_spec,
    label_summary_path,
    new_manifest,
    odd_cases,
    pending_ids,
    preflight,
    selection_ids,
    stage_record,
    summarize_labels,
)
from src.runs import (
    DEFAULT_LABEL,
    FingerprintGuard,
    check_nested,
    check_resumable,
    read_rows,
)
from src.scoring import ModelSpec, build_clients, load_prompt_template, render_prompt

#: What has to be typed to approve the spend.
CONFIRMATION = "yes"


# --- arguments ---------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments. There is no option to skip the prompt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-run",
        metavar="TIMESTAMP",
        help="label the speeches of this scoring run",
    )
    source.add_argument(
        "--ids-file",
        type=Path,
        help="label these speeches: one id per line, or a CSV's speech_id column",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"only N of them, stratified by {' x '.join(LABEL_LIMIT_STRATA)} "
        f"(seed {RANDOM_SEED})",
    )
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
        help="finish or extend that label run: label only what has no valid label",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(LABEL_MODEL_SPECS),
        help=f"labelling model (default {LABEL_MODEL}; a resumed run keeps its own)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"speeches in flight at once (default {DEFAULT_CONCURRENCY}; S17)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and estimate cost, then stop without calling any model",
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--output-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args(argv)


def requested_selection(
    args: argparse.Namespace, manifest: Mapping[str, Any] | None
) -> tuple[dict[str, Any], bool]:
    """The selection to label, and whether it is the run's last one unchanged.

    A new source (``--from-run`` / ``--ids-file``) starts a fresh selection;
    ``--limit`` alone changes the last one's size; neither repeats the last
    one, which is how a bare ``--resume`` finishes a run.

    Raises:
        ValueError: a new run without a source.
    """
    if args.from_run or args.ids_file:
        selection: dict[str, Any] = {
            "from_run": args.from_run,
            "ids_file": str(args.ids_file) if args.ids_file else None,
            "limit": args.limit,
        }
        if args.limit is not None:
            selection |= {"strata": list(LABEL_LIMIT_STRATA), "seed": RANDOM_SEED}
        return selection, False
    if manifest is None:
        raise ValueError("give --from-run <timestamp> or --ids-file PATH")
    last = dict(manifest["stages"][-1]["selection"])
    if args.limit is None or args.limit == last.get("limit"):
        return last, True
    last |= {"limit": args.limit}
    last.setdefault("strata", list(LABEL_LIMIT_STRATA))
    last.setdefault("seed", RANDOM_SEED)
    return last, False


# --- printing ----------------------------------------------------------------


def print_preflight(estimate: Mapping[str, Any], model: str) -> None:
    """Show the estimate, with its basis stated rather than implied."""
    print(f"\n  speeches to label   {estimate['speeches']:,}")
    print(
        f"  input tokens        {estimate['tiktoken_input_tokens']:,}"
        "  (tiktoken on the rendered prompts)"
    )
    per_speech = estimate["assumed_output_tokens_per_speech"]
    first = next(iter(estimate["per_model"].values()))
    print(
        f"  output tokens       {first['completion_tokens']:,}"
        f"  (assumed {per_speech} per speech)"
    )
    print()
    print(
        f"  {'model':<34}{'in tokens':>12}{'out tokens':>12}{'cost':>10}"
        f"{'per 1,000':>11}"
    )
    for name, m in estimate["per_model"].items():
        role = "(this run)" if name == model else "(for reference)"
        print(
            f"  {name + ' ' + role:<34}{m['prompt_tokens']:>12,}"
            f"{m['completion_tokens']:>12,}"
            f"{'$' + format(m['cost_usd'], '.3f'):>10}"
            f"{'$' + format(m['cost_per_1000_speeches_usd'], '.3f'):>11}"
        )
    print(
        "\n  An estimate: tiktoken approximates DeepSeek's tokenizer (~3% under,\n"
        "  S9), output is assumed until a pilot measures it, and DeepSeek is\n"
        "  priced at its peak rate (off-peak is half)."
    )


def print_summary(summary: Mapping[str, Any]) -> None:
    """Print the label distribution, failures and what the run cost."""
    print("\n" + "=" * 68)
    print(
        f"labelled {summary['speeches_labelled']:,} of {summary['speeches']:,} "
        f"speeches with {summary['model']} (codebook {summary['codebook_version']})"
    )
    labelled = summary["speeches_labelled"] or 1

    print("\nspeech type")
    for name, n in summary["speech_type_counts"].items():
        print(f"  {name:<20}{n:>6,}  {n / labelled:6.1%}")
    policy = sum(summary["topic_counts"].values()) or 1
    print("\ntopic (policy_debate only)")
    for name, n in summary["topic_counts"].items():
        print(f"  {name:<20}{n:>6,}  {n / policy:6.1%}")
    print("\ntopic block")
    for name, n in summary["topic_block_counts"].items():
        topics = ", ".join(TOPIC_BLOCKS[name])
        print(f"  {name:<20}{n:>6,}  {n / policy:6.1%}  ({topics})")

    for by in ("chamber", "source"):
        print(f"\nspeech type share by {by}")
        for value, stats in summary[f"speech_type_shares_by_{by}"].items():
            shares = "  ".join(
                f"{t} {s:.0%}" for t, s in stats["speech_type"].items() if s
            )
            print(f"  {value:<10} n={stats['n']:<5} {shares}")

    print(
        f"\ncalls {summary['calls']:,}, failed attempts {summary['failed_attempts']}, "
        f"speeches without labels {summary['failures']}"
    )
    for kind, n in summary["errors_by_type"].items():
        print(f"  {kind}: {n}")
    print(
        f"tokens in {summary['prompt_tokens']:,}, out {summary['completion_tokens']:,}"
        f" (reasoning {summary['reasoning_tokens']:,} on "
        f"{summary['calls_with_reasoning_tokens']} calls), "
        f"cost ${summary['cost_usd']:.3f}"
    )
    print(f"served by {summary['served_models']}")
    print(f"system fingerprints {summary['system_fingerprints']}")
    if len(summary["system_fingerprints"]) > 1:
        print("  *** more than one system fingerprint: the model changed mid-run ***")
    print("=" * 68)


def print_cases(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    """The failed speeches, then valid answers that look odd."""
    if summary["failed_speeches"]:
        print("\nfailed")
        for case in summary["failed_speeches"]:
            print(f"  {case['speech_id']}  {case['error']}")
    cases = odd_cases(rows)
    if cases:
        print("\nworth a look")
        for case in cases:
            print(
                f"  {case['speech_id']}  {case['speech_type']}/{case['topic']}"
                f"  [{case['why']}]  {case['reason']}"
            )


# --- labelling -----------------------------------------------------------------


async def label_all(
    speeches: Sequence[Mapping[str, Any]],
    pending: Sequence[str],
    template: str,
    spec: ModelSpec,
    client: Any,
    handle: TextIO,
    stage: int,
    concurrency: int = DEFAULT_CONCURRENCY,
    guard: FingerprintGuard | None = None,
) -> bool:
    """Label each pending speech, appending each row as it lands.

    Up to ``concurrency`` speeches are in flight at once (S17), taken in order
    by that many workers; at 1 this is the sequential loop of the first runs.
    Rows are flushed one by one, so a crash keeps everything already paid for
    and ``--resume`` carries on from there.

    An answer the ``guard`` rejects -- the model answering from another build
    (S18) -- is written as a failure, its labels kept aside under
    ``rejected_answer``, and no further speech is started. Returns False if
    that happened.
    """
    owed = set(pending)
    todo = [speech for speech in speeches if speech["speech_id"] in owed]
    queue = iter(todo)
    finished = 0
    stopped: list[str] = []

    async def worker() -> None:
        nonlocal finished
        # One shared iterator: next() is synchronous, so no two workers take
        # the same speech.
        for speech in queue:
            if stopped:
                return
            prompt = render_prompt(template, speech["text"])
            result = await label_one(spec, client, prompt)
            attempted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            finished += 1
            rejected = None
            message = (
                guard.check(result.model, result.system_fingerprint) if guard else None
            )
            if message:
                rejected = {
                    "speech_type": result.speech_type,
                    "topic": result.topic,
                    "reason": result.reason,
                }
                result = replace(
                    result, speech_type=None, topic=None, reason=None, error=message
                )
                stopped.append(message)
            row = label_row(speech["speech_id"], result, stage, attempted_at)
            if rejected is not None:
                row["rejected_answer"] = rejected
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if result.ok:
                status = f"{result.speech_type}" + (
                    f" / {result.topic}" if result.topic else ""
                )
                print(
                    f"  [{finished}/{len(todo)}] {speech['speech_id']} {status}",
                    flush=True,
                )
            else:
                print(
                    f"  [{finished}/{len(todo)}] {speech['speech_id']} FAILED "
                    f"{result.error}",
                    file=sys.stderr,
                    flush=True,
                )

    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    return not stopped


def write_summary(
    args: argparse.Namespace,
    label: str,
    run: str,
    model: str,
    speeches: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Rebuild the summary from everything the run's file holds."""
    rows = read_rows(label_rows_path(args.output_dir, label, model, run))
    summary = summarize_labels(rows, speeches, model)
    summary |= {"run": run, "label": label}
    path = label_summary_path(args.metrics_dir, label, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, rows, path


# --- entry point -----------------------------------------------------------------


def _fail(message: str) -> int:
    print(f"label_run: {message}", file=sys.stderr)
    return 2


async def main(argv: list[str] | None = None) -> int:
    """Select, price what is left, confirm the spend, label and report."""
    args = parse_args(argv)
    if args.concurrency < 1:
        return _fail("--concurrency must be at least 1")

    label, manifest, run = args.label or DEFAULT_LABEL, None, None
    model = args.model or LABEL_MODEL
    if args.resume:
        try:
            label, manifest = find_label_run(args.output_dir, args.resume)
        except FileNotFoundError as error:
            return _fail(str(error))
        if args.label is not None and args.label != label:
            return _fail(
                f"run {args.resume} is labelled {label!r}, not {args.label!r}; "
                "leave out --label when resuming"
            )
        if args.model is not None and args.model != manifest["model"]:
            return _fail(
                f"run {args.resume} labels with {manifest['model']!r}, not "
                f"{args.model!r}; one model per run -- start a new run instead"
            )
        model, run = manifest["model"], args.resume

    try:
        selection, repeated = requested_selection(args, manifest)
        speech_ids = selection_ids(selection, args.corpus, args.scores_dir)
        template = load_prompt_template(LABEL_PROMPT_PATH)
        spec = label_spec(model)
        speeches = fetch_speeches(speech_ids, args.corpus)
    except (FileNotFoundError, KeyError, ValueError) as error:
        return _fail(str(error))

    corpus_sha256 = _fingerprint(args.corpus)
    print(f"corpus  {args.corpus} (sha256 {corpus_sha256[:12]})")
    settings = label_settings(template, corpus_sha256, model)

    rows: list[dict[str, Any]] = []
    if manifest is not None:
        try:
            check_resumable(manifest, settings, keys=LABEL_DEFINING_KEYS)
            rows = read_rows(label_rows_path(args.output_dir, label, model, run))
            check_nested({str(row["speech_id"]) for row in rows}, speech_ids)
        except ValueError as error:
            return _fail(str(error))
        last = manifest["stages"][-1]
        if repeated and ids_sha256(speech_ids) != last["ids_sha256"]:
            return _fail(
                f"the speeches of stage {last['stage']}'s selection have changed "
                f"since it ran ({selection}); give the selection explicitly to "
                "extend the run, or start a new run"
            )
    else:
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pending = pending_ids(speech_ids, rows)
    attempted = {str(row["speech_id"]) for row in rows}
    print(
        f"run     {label} {run}" + (" (resumed)" if args.resume else " (new)")
        + f": {model}, {len(speech_ids) - len(pending):,} of {len(speech_ids):,} "
        f"speeches labelled; {sum(s in attempted for s in pending):,} failed to retry"
    )  # fmt: skip

    if not pending:
        print("\nNothing left to label.")
        if args.resume and not args.dry_run:
            summary, rows, _ = write_summary(args, label, run, model, speeches)
            print_summary(summary)
        return 0

    by_id = {speech["speech_id"]: speech for speech in speeches}
    token_counts = dict(
        zip(
            pending,
            count_prompt_tokens(
                [render_prompt(template, by_id[s]["text"]) for s in pending]
            ),
        )
    )
    estimate = preflight(
        token_counts, [model, *(m for m in LABEL_MODEL_SPECS if m != model)]
    )
    print_preflight(estimate, model)

    if args.dry_run:
        print("\n--dry-run: stopping before any API call.")
        return 0

    # Keys are checked before anyone is asked to approve a spend.
    try:
        clients = build_clients({model: spec})
    except RuntimeError as error:
        return _fail(str(error))

    cost = estimate["per_model"][model]["cost_usd"]
    try:
        answer = input(
            f"\nLabel {len(pending):,} speeches with {model} for about ${cost:.2f}? "
            f'Type "{CONFIRMATION}" to proceed: '
        ).strip()
    except EOFError:
        answer = ""
    if answer != CONFIRMATION:
        print("Aborted; nothing was called.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    if manifest is None:
        manifest = new_manifest(
            run, label, model, settings, template, LABEL_PROMPT_PATH, args.corpus,
            RANDOM_SEED,
        )  # fmt: skip
    stage = len(manifest["stages"]) + 1
    manifest["stages"].append(
        stage_record(
            stage,
            selection,
            speech_ids,
            pending,
            estimate["per_model"][model],
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            args.concurrency,
        )
    )
    manifest_file = label_manifest_path(args.output_dir, label, run)
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows_file = label_rows_path(args.output_dir, label, model, run)
    guard = FingerprintGuard(FINGERPRINT_PINNED_MODELS, {model: rows})
    with rows_file.open("a", encoding="utf-8") as handle:
        completed = await label_all(
            speeches, pending, template, spec, clients[model], handle, stage,
            args.concurrency, guard,
        )  # fmt: skip

    summary, rows, summary_file = write_summary(args, label, run, model, speeches)
    print_summary(summary)
    print_cases(summary, rows)
    if not completed:
        print(
            f"\nlabel_run: STOPPED -- {model} answered from another build than this "
            "run's earlier answers (S18). Those answers are recorded as failures and "
            "not counted. A run is one instrument: start a new run.",
            file=sys.stderr,
        )
        return 3
    print(f"\nrun       {label} {run} (stage {stage})")
    print(f"labels    {rows_file}")
    print(f"manifest  {manifest_file}")
    print(f"summary   {summary_file}")
    print(
        f"\nTo finish or extend this run: python code/scripts/label_run.py "
        f"--resume {run} --dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
