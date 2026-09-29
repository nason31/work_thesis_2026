"""Merge the Stanford and govinfo sides into data/processed/corpus.parquet.

Usage (from the repo root):

    python code/scripts/build_merged_corpus.py --overwrite   # what `make corpus` runs

Needs both processed sides first: `make stanford` and `make govinfo`. What the
merge checks, and why: docs/notes/2026-09-29_corpus_merge_design.md.

All logic lives in code/src/merge.py; this is only the command line around it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CORPUS_PATH,
    GOVINFO_BUILD_STATS_PATH,
    GOVINFO_CORPUS_PATH,
    MERGED_BUILD_STATS_PATH,
    STANFORD_BUILD_STATS_PATH,
    STANFORD_CORPUS_PATH,
)
from src.merge import merge_corpora


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stanford", type=Path, default=STANFORD_CORPUS_PATH)
    parser.add_argument("--govinfo", type=Path, default=GOVINFO_CORPUS_PATH)
    parser.add_argument(
        "--stanford-stats", type=Path, default=STANFORD_BUILD_STATS_PATH
    )
    parser.add_argument("--govinfo-stats", type=Path, default=GOVINFO_BUILD_STATS_PATH)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    parser.add_argument("--stats-path", type=Path, default=MERGED_BUILD_STATS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the merge and report what it produced."""
    args = parse_args(argv)
    try:
        stats = merge_corpora(
            stanford=args.stanford,
            govinfo=args.govinfo,
            dst=args.output,
            stanford_stats=args.stanford_stats,
            govinfo_stats=args.govinfo_stats,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        # Every check failure says what to do; a traceback adds nothing.
        print(f"build_merged_corpus: {error}", file=sys.stderr)
        return 1
    stats_path = stats.write(args.stats_path)

    for source, rows in stats.rows_by_source.items():
        print(f"{source:<8} {rows:>10,} rows")
    print(f"written  {stats.rows_written:>10,} rows -> {args.output}")
    print(
        f"seam             {stats.seam['stanford_last']} (stanford) | "
        f"{stats.seam['govinfo_first']} (govinfo)"
    )
    print(f"date range       {stats.date_min} .. {stats.date_max}")
    congresses = sorted(stats.cell_counts, key=int)
    print(f"congresses       {congresses[0]}..{congresses[-1]}")
    print(f"icpsr missing    {stats.icpsr_missing_by_source}")
    print(f"stats            {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
