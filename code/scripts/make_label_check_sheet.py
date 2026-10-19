"""Write a blind hand-coding sheet for a label run (docs/decisions.md L4).

Usage (from the repo root):

    python code/scripts/make_label_check_sheet.py --run <label timestamp>

Draws a seeded sample (default 200) from the speeches the label run attempted
-- failures included, so the check is not tilted toward speeches the model
could label -- stratified by chamber, source and presidential term. The first
50 rows (``overlap = 1``) are coded by both coders, for inter-coder kappa.

The model's labels are deliberately NOT in the sheet: the coders code blind.
Codes go into coder1_type / coder1_topic (and coder2_* on the overlap rows),
using exactly the strings of the codebook (code/src/prompts/label_speech.txt);
topic stays empty unless the type is policy_debate.

Writes ``data/processed/label_check/label_check_<label>_<run>.csv`` (git-ignored:
it holds full speech texts) and a ``.json`` beside it saying how it was drawn.
Never overwrites an existing sheet -- it may already hold someone's codes. No
API call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# No installed package yet, so put code/ on the path before importing src.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import add_presidency
from src.config import (
    CODEBOOK_VERSION,
    CORPUS_PATH,
    LABEL_CHECK_DIR,
    LABEL_CHECK_OVERLAP,
    LABEL_CHECK_SAMPLE_SIZE,
    LABEL_CHECK_STRATA,
    LABELS_DIR,
    RANDOM_SEED,
)
from src.label_eval import EXCEL_CELL_LIMIT, build_check_sheet
from src.labeling import (
    effective_label_rows,
    find_label_run,
    label_rows_path,
    read_corpus_rows,
)
from src.runs import read_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, metavar="TIMESTAMP")
    parser.add_argument("--size", type=int, default=LABEL_CHECK_SAMPLE_SIZE)
    parser.add_argument("--overlap", type=int, default=LABEL_CHECK_OVERLAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=LABEL_CHECK_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Draw the sample and write the sheet, refusing to overwrite one."""
    args = parse_args(argv)
    try:
        label, manifest = find_label_run(args.labels_dir, args.run)
    except FileNotFoundError as error:
        print(f"make_label_check_sheet: {error}", file=sys.stderr)
        return 2

    sheet_path = args.output_dir / f"label_check_{label}_{args.run}.csv"
    if sheet_path.exists():
        print(
            f"make_label_check_sheet: {sheet_path} exists and may hold codes; "
            "move it away first if you really want a new one",
            file=sys.stderr,
        )
        return 2

    rows = read_rows(
        label_rows_path(args.labels_dir, label, manifest["model"], args.run)
    )
    speech_ids = sorted(effective_label_rows(rows))
    frame = read_corpus_rows(
        speech_ids,
        ("party", "chamber", "source", "congress_number", "text"),
        args.corpus,
    )
    # The era: the presidential term each speech's Congress falls in.
    frame = add_presidency(frame)
    sheet = build_check_sheet(
        frame, args.size, args.overlap, LABEL_CHECK_STRATA, args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(sheet_path, index=False)

    drawn = frame[frame["speech_id"].isin(sheet["speech_id"])]
    cells = drawn.groupby(list(LABEL_CHECK_STRATA)).size()
    long = sheet[sheet["text"].str.len() > EXCEL_CELL_LIMIT]
    meta = {
        "label_run": args.run,
        "label": label,
        "model_in_run": manifest["model"],
        "codebook_version": CODEBOOK_VERSION,
        "drawn_from": len(speech_ids),
        "size": len(sheet),
        "overlap": int(sheet["overlap"].sum()),
        "strata": list(LABEL_CHECK_STRATA),
        "seed": args.seed,
        "cells": {" | ".join(map(str, key)): int(n) for key, n in cells.items()},
        "texts_over_excel_cell_limit": long["speech_id"].tolist(),
    }
    sheet_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"sheet   {sheet_path}")
    print(
        f"        {len(sheet)} speeches from {len(speech_ids)} in run {args.run}, "
        f"first {meta['overlap']} double-coded, seed {args.seed}"
    )
    for key, n in cells.items():
        print(f"        {' | '.join(map(str, key)):<40}{n:>4}")
    if not long.empty:
        # Excel cuts a cell off at 32,767 characters; give the coders the rest.
        long_dir = args.output_dir / f"{sheet_path.stem}_long"
        long_dir.mkdir(exist_ok=True)
        for row in long.itertuples():
            (long_dir / f"{row.speech_id}.txt").write_text(row.text, encoding="utf-8")
        print(
            f"        {len(long)} text(s) exceed Excel's {EXCEL_CELL_LIMIT:,}-character "
            f"cell limit; full texts in {long_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
