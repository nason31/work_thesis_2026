"""Build data/processed/corpus_govinfo.parquet from the raw govinfo JSONL.

Usage (from the repo root):

    python code/scripts/build_govinfo_corpus.py --limit 20000   # smoke run first
    python code/scripts/build_govinfo_corpus.py                 # full run

Needs data/raw/govinfo/congress_speeches_2016_present.jsonl (team drive) and the
congress-legislators JSON in data/raw/congress_legislators/. What the build
repairs and why: docs/notes/2026-09-27_govinfo_data_reality.md.

All logic lives in code/src/govinfo.py; this is only the command line around it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    GOVINFO_BUILD_STATS_PATH,
    GOVINFO_CORPUS_PATH,
    GOVINFO_JSONL,
    MIN_WORD_COUNT,
)
from src.govinfo import build_govinfo_corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=GOVINFO_JSONL)
    parser.add_argument("--output", type=Path, default=GOVINFO_CORPUS_PATH)
    parser.add_argument("--min-words", type=int, default=MIN_WORD_COUNT)
    parser.add_argument(
        "--limit", type=int, default=None, help="read at most N raw rows"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stats-path", type=Path, default=GOVINFO_BUILD_STATS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the build and report what it produced."""
    args = parse_args(argv)

    # A --limit run is partial; keep it off the real path, as build_corpus does.
    if args.limit is not None and args.output == GOVINFO_CORPUS_PATH:
        args.output = args.output.with_suffix(".smoke.parquet")
        print(f"--limit run: writing partial corpus to {args.output}\n")

    try:
        stats = build_govinfo_corpus(
            src=args.input,
            dst=args.output,
            min_word_count=args.min_words,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"build_govinfo_corpus: {error}", file=sys.stderr)
        return 1
    stats_path = stats.write(args.stats_path) if args.limit is None else None

    unresolved = (
        stats.rows_dropped_unresolved_no_candidate
        + stats.rows_dropped_unresolved_ambiguous
    )
    print(f"read     {stats.rows_read:>10,} raw rows")
    print(f"dropped  {stats.rows_dropped_before_start:>10,} before {stats.start_date}")
    print(f"dropped  {stats.rows_dropped_after_end:>10,} after {stats.end_date}")
    print(
        f"dropped  {unresolved:>10,} unresolved speaker "
        f"({stats.rows_dropped_unresolved_ambiguous:,} ambiguous, "
        f"{stats.rows_dropped_unresolved_no_candidate:,} no member)"
    )
    print(f"dropped  {stats.rows_dropped_delegate:>10,} non-voting delegates")
    print(f"dropped  {stats.rows_dropped_excluded_member:>10,} excluded members")
    print(f"dropped  {stats.rows_dropped_short:>10,} under {args.min_words} words")
    print(f"dropped  {stats.rows_dropped_duplicate_id:>10,} duplicate speech_id")
    print(f"written  {stats.rows_written:>10,} rows -> {args.output}")
    print(f"cut at           {stats.rows_cut}")
    share = stats.words_removed_by_cleaning / max(stats.words_before_cleaning, 1)
    print(f"words removed    {stats.words_removed_by_cleaning:,} ({share:.1%})")
    print(f"unresolved       {stats.unresolved_by_chamber}")
    print(f"party            {stats.party_counts}")
    print(f"independents     {stats.independents_reassigned:,} reassigned to caucus")
    print(f"congresses       {stats.congress_counts}")
    print(f"date range       {stats.date_min} .. {stats.date_max}")
    print(f"icpsr missing    {stats.icpsr_missing:,}")
    if stats_path is not None:
        print(f"stats            {stats_path}")
    else:
        print("stats            not written (--limit run is a partial corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
