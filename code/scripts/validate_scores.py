"""Validate a scoring run's ideology scores against DW-NOMINATE.

Usage (from the repo root):

    python code/scripts/validate_scores.py                       # latest run
    python code/scripts/validate_scores.py --run 20260923T103556Z

Needs the run's per-model files in results/scores/, the merged corpus
(`make corpus`) and the member crosswalk (`make crosswalk`). Makes no API
calls. Writes results/metrics/validation_<run>.json.

Every combination is reported, because the choices are still open (O9): the
ensemble and each model; nominate_dim1 and nokken_poole_dim1; all speeches and
non-procedural ones; both parties together and each alone. Read the
within-party rows first -- across parties, r mostly re-measures the party gap.

All logic lives in code/src/validation.py; this is only the command line around it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CORPUS_PATH,
    CROSSWALK_PATH,
    ENSEMBLE_MODELS,
    METRICS_DIR,
    SCORES_DIR,
    VALIDATION_BENCHMARKS,
)
from src.corpus import _fingerprint
from src.validation import (
    ENSEMBLE_COLUMN,
    attach_benchmarks,
    correlate,
    load_run_scores,
    score_column,
)

SUBSETS = {
    "all_speeches": lambda frame: frame,
    "non_procedural": lambda frame: frame[~frame["procedural"]],
}


def latest_run(scores_dir: Path) -> str:
    """Timestamp of the newest run manifest in ``scores_dir``."""
    manifests = sorted(scores_dir.glob("pilot_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No run manifest in {scores_dir}")
    return manifests[-1].stem.removeprefix("pilot_manifest_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="run timestamp (default: the latest)")
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--crosswalk", type=Path, default=CROSSWALK_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args(argv)


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def _cell(stats: dict[str, object]) -> str:
    """'r [lo, hi] rho n' for the printed table."""
    if stats["pearson_r"] is None:
        return f"{'--':>24} n={stats['n']}"
    ci = stats["pearson_ci95"] or [float("nan"), float("nan")]
    return (
        f"{stats['pearson_r']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] "
        f"rho {stats['spearman_rho']:+.2f} n={stats['n']}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the validation and report it."""
    args = parse_args(argv)
    try:
        run = args.run or latest_run(args.scores_dir)
        scores = load_run_scores(args.scores_dir, run)
        frame = attach_benchmarks(scores, corpus=args.corpus, crosswalk=args.crosswalk)
    except (FileNotFoundError, ValueError) as error:
        print(f"validate_scores: {error}", file=sys.stderr)
        return 1

    score_columns = [ENSEMBLE_COLUMN] + [score_column(m) for m in ENSEMBLE_MODELS]
    results: dict[str, object] = {}
    for subset, select in SUBSETS.items():
        rows = select(frame)
        results[subset] = {
            benchmark: {
                column.removeprefix("ideology_"): correlate(rows, column, benchmark)
                for column in score_columns
            }
            for benchmark in VALIDATION_BENCHMARKS
        }

    validated = frame[frame["in_validation"]]
    report = {
        "run": run,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "scores": {
                model: _fingerprint(
                    args.scores_dir / f"pilot_{model.replace('/', '_')}_{run}.jsonl"
                )
                for model in ENSEMBLE_MODELS
            },
            "corpus": {"path": str(args.corpus), "sha256": _fingerprint(args.corpus)},
            "crosswalk": {
                "path": str(args.crosswalk),
                "sha256": _fingerprint(args.crosswalk),
            },
        },
        "counts": {
            "speeches": len(frame),
            "distinct_members": int(frame["member_id"].nunique()),
            "in_validation": len(validated),
            "excluded_unmatched": int(frame["unmatched_reason"].notna().sum()),
            "excluded_no_score": int(
                (frame["icpsr"].notna() & ~frame["in_validation"]).sum()
            ),
            "procedural": int(frame["procedural"].sum()),
            "party_switch_in_validation": int(validated["party_switch"].sum()),
            "by_party_in_validation": validated["party"].value_counts().to_dict(),
        },
        "results": results,
    }
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    out = args.metrics_dir / f"validation_{run}.json"
    out.write_text(json.dumps(_rounded(report), indent=2) + "\n")

    counts = report["counts"]
    print(
        f"run {run}: {counts['speeches']} speeches, {counts['in_validation']} in "
        f"validation, {counts['procedural']} procedural"
    )
    for subset in SUBSETS:
        for benchmark in VALIDATION_BENCHMARKS:
            print(f"\n{subset} vs {benchmark}   Pearson r [95% CI], Spearman rho")
            for column in score_columns:
                name = column.removeprefix("ideology_")
                cells = results[subset][benchmark][name]
                print(
                    f"  {name:<19}"
                    + " | ".join(
                        f"{group} {_cell(cells[group])}" for group in ("all", "D", "R")
                    )
                )
    print(f"\nreport {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
