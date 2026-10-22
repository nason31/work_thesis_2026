"""Compare the DW-NOMINATE validation designs; report the headline (decisions A5).

Usage (from the repo root):

    python code/scripts/validation_grid.py --run 20261018T122536Z \\
        --label-run 20261017T182223Z --compare-run 20261003T110419Z

Needs the run's score files, the label run, the merged corpus and the member
crosswalk. Makes no API calls. Takes about ten minutes, mostly the bootstrap
at speech level. Writes to results/metrics/:

    validation_grid_<run>.csv       every design: unit x threshold x procedural
                                    rule x benchmark, with member-bootstrap
                                    intervals and reliability
    validation_selection_<run>.csv  who the member units are at each threshold
    validation_selection_by_congress_<run>.csv
                                    the same per Congress, chamber and party,
                                    against every member Voteview scored
    validation_design_<run>.json    inputs (SHA-256), how the procedural rules
                                    differ, the headline (VALIDATION_HEADLINE),
                                    robustness, and the change diagnostic

All logic lives in code/src/validation_design.py; this is only the command line
around it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.analysis import attach_labels, load_scored_run, member_background
from src.config import (
    CORPUS_PATH,
    CROSSWALK_PATH,
    DW_NOMINATE_MEMBERS,
    LABELS_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    SCORES_DIR,
    VALIDATION_BENCHMARKS,
    VALIDATION_BOOTSTRAP,
    VALIDATION_HEADLINE,
    VALIDATION_MIN_SPEECHES,
    VALIDATION_MIN_UNITS_PER_PARTY,
    VALIDATION_SPLITS,
)
from src.corpus import _fingerprint
from src.crosswalk import load_voteview
from src.labeling import find_label_run, label_rows_path
from src.runs import find_label, model_path, run_models
from src.validation_design import (
    change_statistics,
    congress_changes,
    design_row,
    procedural_overlap,
    selection_by_congress,
    selection_profile,
    unit_table,
    validation_grid,
    validation_speeches,
)

#: The printed columns of a design: (label, statistic).
_PRINTED = (
    ("within D", "within_D_pearson"),
    ("within R", "within_R_pearson"),
    ("within cell", "within_cell_pearson"),
    ("cell, House", "within_cell_H"),
    ("cell, Senate", "within_cell_S"),
    ("overall", "overall_pearson"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="scoring run timestamp")
    parser.add_argument("--label-run", required=True, help="label run timestamp")
    parser.add_argument(
        "--compare-run",
        help="a second scoring run of a subset of the same speeches (robustness)",
    )
    parser.add_argument("--bootstrap", type=int, default=VALIDATION_BOOTSTRAP)
    parser.add_argument("--splits", type=int, default=VALIDATION_SPLITS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    return parser.parse_args(argv)


def _cell(row: dict | pd.Series, name: str) -> str:
    value = row[name]
    if value is None or not np.isfinite(value):
        return f"{'--':^22}"
    return f"{value:+.2f} [{row[name + '_ci_low']:+.2f},{row[name + '_ci_high']:+.2f}]"


def _print_row(row: dict) -> None:
    print(
        f"{row['procedural']:<10} {row['benchmark'][:9]:<9} {row['unit']:<15} "
        f">={row['min_speeches']:<2} n={row['n_units_D']:>5}/{row['n_units_R']:<5} "
        + " ".join(_cell(row, stat) for _, stat in _PRINTED),
        flush=True,
    )


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 4)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _design(
    frame: pd.DataFrame, args: argparse.Namespace, score: str = "ideology", **changes
) -> dict[str, object]:
    """The headline design on ``frame``, or counts only if a party is too thin."""
    design = {**VALIDATION_HEADLINE, **changes}
    speeches = validation_speeches(frame, design["procedural"], score)
    units = unit_table(
        speeches, design["unit"], design["benchmark"], score, design["min_speeches"]
    )
    counts = units["party"].value_counts().reindex(["D", "R"], fill_value=0)
    if counts.min() < VALIDATION_MIN_UNITS_PER_PARTY:
        return {
            **design,
            "score": score,
            "n_units_D": int(counts["D"]),
            "n_units_R": int(counts["R"]),
            "too_few_units": True,
        }
    row = design_row(
        speeches,
        design["unit"],
        design["benchmark"],
        design["min_speeches"],
        score,
        args.bootstrap,
        args.splits,
        args.seed,
    )
    return {**design, **row, "too_few_units": False}


def _inputs(args: argparse.Namespace, runs: list[str]) -> dict[str, object]:
    """SHA-256 of every file the result depends on."""
    files: dict[str, Path] = {}
    for run in runs:
        label = find_label(args.scores_dir, run)
        for model in run_models(args.scores_dir, run):
            files[f"scores {run} {model}"] = model_path(
                args.scores_dir, label, model, run
            )
    label, manifest = find_label_run(args.labels_dir, args.label_run)
    files[f"labels {args.label_run}"] = label_rows_path(
        args.labels_dir, label, manifest["model"], args.label_run
    )
    files["corpus"] = CORPUS_PATH
    files["crosswalk"] = CROSSWALK_PATH
    files["voteview"] = DW_NOMINATE_MEMBERS
    return {
        name: {"path": str(path), "sha256": _fingerprint(path)}
        for name, path in files.items()
    }


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and write it to results/metrics/."""
    args = parse_args(argv)
    try:
        frame = attach_labels(
            load_scored_run(args.run, args.scores_dir), args.label_run, args.labels_dir
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"validation_grid: {error}", file=sys.stderr)
        return 1
    print(f"run {args.run}: {len(frame):,} speeches, "
          f"{int(frame['in_validation'].sum()):,} in validation")  # fmt: skip
    print(f"\n{'rule':<10} {'benchmark':<9} {'unit':<15} {'min':<3} {'n D/R':<12} "
          + " ".join(f"{label:^22}" for label, _ in _PRINTED))  # fmt: skip

    grid = validation_grid(
        frame,
        VALIDATION_BENCHMARKS,
        VALIDATION_MIN_SPEECHES,
        n_draws=args.bootstrap,
        n_splits=args.splits,
        seed=args.seed,
        progress=_print_row,
    )

    labelled = validation_speeches(frame, "labels")
    voteview = load_voteview()
    selection = selection_profile(
        labelled, VALIDATION_MIN_SPEECHES, member_background(), voteview
    )
    by_congress = selection_by_congress(labelled, VALIDATION_MIN_SPEECHES, voteview)

    headline = grid[
        (grid["unit"] == VALIDATION_HEADLINE["unit"])
        & (grid["min_speeches"] == VALIDATION_HEADLINE["min_speeches"])
        & (grid["procedural"] == VALIDATION_HEADLINE["procedural"])
        & (grid["benchmark"] == VALIDATION_HEADLINE["benchmark"])
    ].iloc[0]

    print("\nrobustness ...", flush=True)
    switchers = frame["party_switch"].fillna(False).astype(bool)
    robustness: dict[str, object] = {
        "party_switch_speeches_in_validation": int(
            (switchers & frame["in_validation"]).sum()
        ),
        "without_party_switch": [
            _design(frame[~switchers], args, benchmark=benchmark)
            for benchmark in VALIDATION_BENCHMARKS
        ],
    }
    if args.compare_run:
        compare = attach_labels(
            load_scored_run(args.compare_run, args.scores_dir),
            args.label_run,
            args.labels_dir,
        )
        same = frame[frame["speech_id"].isin(compare["speech_id"])]
        scores = ["ideology"] + [
            f"ideology_{m}" for m in run_models(args.scores_dir, args.compare_run)
        ]
        robustness["compare_run"] = {
            "run": args.compare_run,
            "speeches": len(compare),
            "headline_design": [_design(compare, args, s) for s in scores]
            + [{**_design(same, args), "score": f"{args.run} on these speeches"}],
            "supplementary_career_5": [
                _design(compare, args, s, unit="career", min_speeches=5) for s in scores
            ]
            + [
                {
                    **_design(same, args, unit="career", min_speeches=5),
                    "score": f"{args.run} on these speeches",
                }
            ],
        }

    print("change diagnostic ...", flush=True)
    changes = {}
    for k in sorted({5, int(VALIDATION_HEADLINE["min_speeches"])}):
        units = unit_table(
            labelled, "member_congress", "nokken_poole_dim1", min_speeches=k
        )
        changes[f"min_speeches_{k}"] = change_statistics(
            congress_changes(units), args.bootstrap, args.seed
        )

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.metrics_dir / f"validation_grid_{args.run}.csv"
    selection_path = args.metrics_dir / f"validation_selection_{args.run}.csv"
    by_congress_path = (
        args.metrics_dir / f"validation_selection_by_congress_{args.run}.csv"
    )
    report_path = args.metrics_dir / f"validation_design_{args.run}.json"
    grid.round(4).to_csv(grid_path, index=False)
    selection.round(4).to_csv(selection_path, index=False)
    by_congress.round(4).to_csv(by_congress_path, index=False)
    report = {
        "run": args.run,
        "label_run": args.label_run,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": "python code/scripts/validation_grid.py " + " ".join(sys.argv[1:]),
        "settings": {
            "bootstrap_draws": args.bootstrap,
            "splits": args.splits,
            "seed": args.seed,
            "thresholds": list(VALIDATION_MIN_SPEECHES),
            "min_units_per_party": VALIDATION_MIN_UNITS_PER_PARTY,
        },
        "inputs": _inputs(
            args, [args.run] + ([args.compare_run] if args.compare_run else [])
        ),
        "procedural_overlap": procedural_overlap(frame),
        "headline_design": VALIDATION_HEADLINE,
        "headline": headline.to_dict(),
        "robustness": robustness,
        "change_diagnostic": changes,
        "outputs": [str(grid_path), str(selection_path), str(by_congress_path)],
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")

    stat = VALIDATION_HEADLINE["statistic"]
    print(
        f"\nheadline ({VALIDATION_HEADLINE['unit']}, >= "
        f"{VALIDATION_HEADLINE['min_speeches']}, {VALIDATION_HEADLINE['procedural']}, "
        f"{VALIDATION_HEADLINE['benchmark']}): {stat} {_cell(headline, stat)}"
    )
    for path in (grid_path, selection_path, by_congress_path, report_path):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
