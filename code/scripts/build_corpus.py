"""Build data/processed/corpus.parquet from the raw Stanford parquet.

Usage (from the repo root):

    python code/scripts/build_corpus.py --limit 50000   # smoke run first
    python code/scripts/build_corpus.py                 # full run

The raw file is not in the repo. Download congress_speeches_2001_2017.parquet
from the team drive into data/raw/stanford/ first -- see
docs/notes/2026-09-15_stanford_dataset.md.

All logic lives in code/src/corpus.py; this is only the command line around it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    BUILD_STATS_PATH,
    CORPUS_PATH,
    MIN_WORD_COUNT,
    STANFORD_PARQUET,
)
from src.corpus import build_stanford_corpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, default=STANFORD_PARQUET, help="raw Stanford parquet"
    )
    parser.add_argument(
        "--output", type=Path, default=CORPUS_PATH, help="processed corpus parquet"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORD_COUNT,
        help=f"drop speeches shorter than this many words (default {MIN_WORD_COUNT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="read at most N raw rows; validates schema and roster in seconds",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=BUILD_STATS_PATH,
        help="where to write the build statistics JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the build and report what it produced."""
    args = parse_args(argv)

    # A --limit run produces a partial corpus. Never let it land on the real
    # corpus path, where a later reader would have no way to tell it apart
    # from a full build.
    if args.limit is not None and args.output == CORPUS_PATH:
        args.output = args.output.with_suffix(".smoke.parquet")
        print(f"--limit run: writing partial corpus to {args.output}\n")

    try:
        stats = build_stanford_corpus(
            src=args.input,
            dst=args.output,
            min_word_count=args.min_words,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        # These are all "a human has to decide something" failures -- a missing
        # download, an existing output, an independent outside the roster. The
        # message says what to do, so a traceback adds nothing.
        print(f"build_corpus: {error}", file=sys.stderr)
        return 1
    # A --limit run is a partial corpus; writing its counts over the real ones
    # would put misleading figures in a committed file.
    stats_path = stats.write(args.stats_path) if args.limit is None else None

    print(f"read     {stats.rows_read:>10,} raw rows")
    print(f"dropped  {stats.rows_dropped_delegate:>10,} non-voting delegates")
    print(f"dropped  {stats.rows_dropped_excluded_member:>10,} excluded members")
    print(f"dropped  {stats.rows_dropped_short:>10,} under {args.min_words} words")
    print(f"dropped  {stats.rows_dropped_empty_text:>10,} empty text")
    print(f"dropped  {stats.rows_dropped_duplicate_id:>10,} duplicate speech_id")
    print(f"written  {stats.rows_written:>10,} rows -> {args.output}")
    print(f"party            {stats.party_counts}")
    print(
        f"independents     {stats.independents_reassigned:,} reassigned to caucus party"
    )
    print(
        f"corrections      {stats.party_corrections_applied:,} party labels corrected"
    )
    print(f"congresses       {stats.congress_counts}")
    print(f"date range       {stats.date_min} .. {stats.date_max}")
    print(f"word_count check {stats.word_count_mismatch_rate:.1%} of rows disagree")
    if stats_path is not None:
        print(f"stats            {stats_path}")
    else:
        print("stats            not written (--limit run is a partial corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
