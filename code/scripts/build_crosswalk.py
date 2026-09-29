"""Match every corpus member to a Voteview ICPSR: data/processed/member_crosswalk.parquet.

Usage (from the repo root):

    python code/scripts/build_crosswalk.py --overwrite   # what `make crosswalk` runs

Needs the merged corpus first (`make corpus`) and Voteview's HSall_members.csv
in data/raw/dw_nominate/. Writes match rates to
results/metrics/crosswalk_build_stats.json and the unmatched members, with
speech counts, to results/metrics/crosswalk_unmatched.csv. How members are
matched, and why: docs/decisions.md D21-D24.

All logic lives in code/src/crosswalk.py; this is only the command line around it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CORPUS_PATH,
    CROSSWALK_PATH,
    CROSSWALK_STATS_PATH,
    CROSSWALK_UNMATCHED_PATH,
    DW_NOMINATE_MEMBERS,
)
from src.crosswalk import build_crosswalk


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--voteview", type=Path, default=DW_NOMINATE_MEMBERS)
    parser.add_argument("--output", type=Path, default=CROSSWALK_PATH)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output file"
    )
    parser.add_argument("--stats-path", type=Path, default=CROSSWALK_STATS_PATH)
    parser.add_argument("--unmatched-path", type=Path, default=CROSSWALK_UNMATCHED_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the crosswalk and report its match rates."""
    args = parse_args(argv)
    try:
        stats = build_crosswalk(
            corpus=args.corpus,
            voteview=args.voteview,
            dst=args.output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        # Every check failure says what went wrong; a traceback adds nothing.
        print(f"build_crosswalk: {error}", file=sys.stderr)
        return 1
    stats_path = stats.write(args.stats_path)
    unmatched_path = stats.write_unmatched(args.unmatched_path)

    print(
        f"{'source':<9}{'units':>7}{'matched':>9}{'speeches':>10}{'matched':>10}"
        f"{'rate':>8}{'in validation':>15}"
    )
    for source, s in stats.summary.items():
        print(
            f"{source:<9}{s['units']:>7,}{s['units_matched']:>9,}"
            f"{s['speeches']:>10,}{s['speeches_matched']:>10,}"
            f"{s['speech_match_rate']:>8.2%}{s['speech_validation_rate']:>15.2%}"
        )
    print(f"party switchers     {len(stats.party_switch)} unit(s)")
    print(f"matched, no score   {len(stats.no_nominate)} unit(s)")
    print(f"icpsr disagreements {len(stats.icpsr_disagreements)} unit(s) (govinfo)")
    print(f"crosswalk           {args.output}")
    print(f"stats               {stats_path}")
    print(f"unmatched           {unmatched_path} ({len(stats.unmatched)} unit(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
