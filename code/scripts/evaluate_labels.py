"""Evaluate a label run against the hand check and the S8 scores (L4).

Usage (from the repo root), once the check sheet is coded:

    python code/scripts/evaluate_labels.py \\
        --sheet data/processed/label_check/label_check_<label>_<run>.csv \\
        --run <DeepSeek label run> [--check-run <GPT-4o label run>]

Agreement -- accuracy, macro-F1 and Cohen's kappa, with confusion matrices --
for speech type, topic and topic block:

- the labelling model against coder 1, on every speech coder 1 coded;
- coder 1 against coder 2, on the double-coded rows;
- the labelling model against the check model, on every speech both labelled;
- the check model against coder 1.

Then checks against the scoring run the speeches came from (``--scores-run``,
by default the label run's own source), which cost nothing: speech type
against the "every model scored 0.0" procedural flag, scores by speech type,
and label shares by source and by Congress, the 114th marked as mixed-source.

Writes to results/metrics/ (``label_agreement_``, ``label_confusion_``,
``label_consistency_`` and ``label_eval_<run>``) and one figure to
results/plots/. No API call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import attach_labels, load_scored_run
from src.config import (
    CORPUS_PATH,
    CROSSWALK_PATH,
    LABELS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    SCORES_DIR,
    SPEECH_TYPES,
)
from src.label_eval import (
    coder_labels,
    compare_raters,
    label_shares,
    procedural_crosstab,
    scores_by_type,
)
from src.labeling import LABEL_COLUMNS, find_label_run, load_labels, source_run
from src.plots import plot_label_shares, save_figure

#: Note carried into every output built on the S8 sample.
SAMPLE_NOTE = (
    "S8 gives every Congress x party x chamber cell the same quota, so these are "
    "sample shares, not corpus shares (S11). Chambers are kept apart (O8)."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--run", required=True, metavar="TIMESTAMP")
    parser.add_argument("--check-run", metavar="TIMESTAMP", default=None)
    parser.add_argument(
        "--scores-run",
        metavar="TIMESTAMP",
        default=None,
        help="scoring run for the consistency checks (default: the label run's source)",
    )
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--crosswalk", type=Path, default=CROSSWALK_PATH)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    return parser.parse_args(argv)


def model_rater(run: str, labels_dir: Path) -> tuple[str, pd.DataFrame, list[str]]:
    """A label run as a rater: (model, labels by speech_id, failed speech ids)."""
    labels = load_labels(run, labels_dir)
    failed = labels.loc[~labels["label_ok"], "speech_id"].tolist()
    model = labels["model"].iloc[0] if len(labels) else run
    return model, labels.set_index("speech_id")[list(LABEL_COLUMNS)], failed


def raters_and_pairs(
    args: argparse.Namespace, sheet: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]], dict[str, Any]]:
    """Every rater's labels, the pairs to compare, and what each covers."""
    coder1 = coder_labels(sheet, "coder1")
    overlap = sheet.loc[sheet["overlap"] == 1, "speech_id"].astype(str)
    coder2 = coder_labels(sheet[sheet["overlap"] == 1], "coder2")
    model, labels, failed = model_rater(args.run, args.labels_dir)
    raters = {"coder1": coder1, "coder2": coder2, model: labels}
    pairs = [(model, "coder1"), ("coder1", "coder2")]
    coverage: dict[str, Any] = {
        "sheet_rows": len(sheet),
        "coded_by_coder1": len(coder1),
        "overlap_rows": len(overlap),
        "coded_by_coder2": len(coder2),
        model: {
            "run": args.run,
            "failed": len(failed),
            "failed_on_sheet": sorted(set(failed) & set(sheet["speech_id"])),
        },
    }
    if args.check_run:
        check, check_labels, check_failed = model_rater(args.check_run, args.labels_dir)
        raters[check] = check_labels
        pairs += [(model, check), (check, "coder1")]
        coverage[check] = {
            "run": args.check_run,
            "failed": len(check_failed),
            "failed_on_sheet": sorted(set(check_failed) & set(sheet["speech_id"])),
        }
    return raters, pairs, coverage


def consistency(
    args: argparse.Namespace, scores_run: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The free checks against the scoring run; also returns the plot's shares."""
    frame = load_scored_run(scores_run, args.scores_dir, args.corpus, args.crosswalk)
    frame = attach_labels(frame, args.run, args.labels_dir)
    crosstab = procedural_crosstab(frame)
    labelled = frame.dropna(subset=["speech_type"])
    procedural = labelled["speech_type"] == "procedural"
    agree = {
        "labelled_procedural": int(procedural.sum()),
        "flagged_all_zero": int(labelled["procedural"].sum()),
        "both": int((procedural & labelled["procedural"]).sum()),
        "label_only": int((procedural & ~labelled["procedural"]).sum()),
        "flag_only": int((~procedural & labelled["procedural"]).sum()),
    }
    by_congress = label_shares(frame, ["congress_number", "chamber"])
    out = {
        "scores_run": scores_run,
        "note": SAMPLE_NOTE,
        "speeches_in_scores_run": len(frame),
        "speeches_labelled": len(labelled),
        "procedural_label_vs_zero_flag": agree,
        "speech_type_vs_zero_flag": crosstab.reset_index().to_dict(orient="records"),
        "scores_by_speech_type": scores_by_type(frame).to_dict(orient="records"),
        "speech_type_shares_by_source": label_shares(
            frame, ["source", "chamber"]
        ).to_dict(orient="records"),
        "speech_type_shares_by_congress": label_shares(
            frame, ["congress_number", "source", "chamber"]
        ).to_dict(orient="records"),
        "topic_block_shares_by_congress": label_shares(
            frame, ["congress_number", "source", "chamber"], column="topic_block"
        ).to_dict(orient="records"),
    }
    return out, by_congress


def main(argv: list[str] | None = None) -> int:
    """Read the sheet and the runs, compute everything, write it out."""
    args = parse_args(argv)
    try:
        sheet = pd.read_csv(args.sheet, dtype={"speech_id": str})
        raters, pairs, coverage = raters_and_pairs(args, sheet)
        _, manifest = find_label_run(args.labels_dir, args.run)
    except (FileNotFoundError, ValueError) as error:
        print(f"evaluate_labels: {error}", file=sys.stderr)
        return 2

    stats, confusion = compare_raters(raters, pairs)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    agreement_file = args.metrics_dir / f"label_agreement_{args.run}.csv"
    confusion_file = args.metrics_dir / f"label_confusion_{args.run}.csv"
    stats.to_csv(agreement_file, index=False)
    confusion.to_csv(confusion_file, index=False)

    report: dict[str, Any] = {
        "label_run": args.run,
        "check_run": args.check_run,
        "sheet": str(args.sheet),
        "coverage": coverage,
        "agreement": stats.to_dict(orient="records"),
        "topic_rule": "topic and topic_block are compared only where both raters "
        "said policy_debate",
    }

    scores_run = args.scores_run or source_run(manifest)
    if scores_run:
        checks, by_congress = consistency(args, scores_run)
        consistency_file = args.metrics_dir / f"label_consistency_{args.run}.json"
        consistency_file.write_text(json.dumps(checks, indent=2, default=str) + "\n")
        by_congress.to_csv(
            args.metrics_dir / f"label_type_shares_{args.run}.csv", index=False
        )
        fig = plot_label_shares(
            by_congress,
            title="Speech types per Congress",
            subtitle=f"Label run {args.run} over scoring run {scores_run}. "
            "Sample shares (S8 quotas), not corpus shares.",
            order=SPEECH_TYPES,
        )
        figure = save_figure(fig, f"label_type_shares_{args.run}", args.plots_dir)
        report["consistency"] = str(consistency_file)
        report["figure"] = str(figure)
        print(f"consistency  {consistency_file}")
        print(f"figure       {figure}")
    else:
        print("no scoring run to check against; skipped the consistency checks")

    report_file = args.metrics_dir / f"label_eval_{args.run}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nagreement    {agreement_file}")
    print(f"confusion    {confusion_file}")
    print(f"report       {report_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
