"""Figure (table): ideology agreement between the three models, one row per pair.

Pearson r, Spearman rho and mean absolute difference between the models' ideology
scores on the speeches all three scored in the 208-speech run (20261003T100851Z).
`--include-pilot-1` adds the 200-speech pilot (20260923T103556Z), 407 speeches in
all. Reads the raw per-model scores in results/scores/ (on the model-development
branch); makes no API calls.

    python code/scripts/plot_pilot_agreement_table.py [--scores-dir DIR] [--include-pilot-1]
"""

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib.pyplot as plt
import pandas as pd

from src.plots import (
    BASELINE,
    GRID,
    INK,
    INK_SECONDARY,
    SURFACE,
    save_figure,
)

MODELS = {
    "deepseek-reasoner": "DeepSeek",
    "gpt-4o-2024-11-20": "GPT-4o",
    "claude-sonnet-4-6": "Sonnet 4.6",
}
NAMES = tuple(MODELS.values())
PILOT_1 = ("pilot", "20260923T103556Z")
PILOT_2 = ("s8", "20261003T100851Z")  # the 208-speech run keeps the s8 file prefix

parser = argparse.ArgumentParser()
parser.add_argument("--scores-dir", type=Path, default=ROOT / "results" / "scores")
parser.add_argument(
    "--include-pilot-1", action="store_true", help="also pool the 200-speech pilot"
)
args = parser.parse_args()

frames = []
for prefix, run in (PILOT_1, PILOT_2) if args.include_pilot_1 else (PILOT_2,):
    frame = None
    for model, name in MODELS.items():
        d = pd.read_json(args.scores_dir / f"{prefix}_{model}_{run}.jsonl", lines=True)
        d = d.drop_duplicates("speech_id")
        d = d[d["error"].isna() & d["ideology_score"].notna()].set_index("speech_id")
        part = d[["ideology_score"]].rename(columns={"ideology_score": name})
        frame = part if frame is None else frame.join(part, how="inner")
    frames.append(frame)
data = pd.concat(frames)
assert data.index.is_unique, "the two pilots share a speech"

rows = []
for a, b in itertools.combinations(NAMES, 2):
    rows.append(
        (
            f"{a} – {b}",
            f"{data[a].corr(data[b]):.3f}",
            f"{data[a].corr(data[b], method='spearman'):.3f}",
            f"{(data[a] - data[b]).abs().mean():.3f}",
        )
    )
header = ("Model pair", "Pearson r", "Spearman ρ", "Mean |difference|")
table = pd.DataFrame(rows, columns=header)

xs = (0.0, 0.40, 0.58, 0.76)  # left edge of each column, in axes fractions
fig, ax = plt.subplots(figsize=(7.4, 2.3), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 1)
ax.set_ylim(0, len(rows) + 1)
ax.axis("off")
top = len(rows) + 0.5
for x, text in zip(xs, header):
    ax.text(x, top, text, ha="left", va="center", color=INK_SECONDARY, fontsize=10)
ax.plot([0, 1], [top - 0.5] * 2, color=BASELINE, lw=1)
for i, row in enumerate(rows):
    y = len(rows) - 0.5 - i
    for x, text in zip(xs, row):
        ax.text(x, y, text, ha="left", va="center", color=INK, fontsize=12)
    if i < len(rows) - 1:
        ax.plot([0, 1], [y - 0.5] * 2, color=GRID, lw=0.7)

scope = f"{len(data)} speeches" + (", both pilots" if args.include_pilot_1 else "")
fig.suptitle(
    "Agreement between the models' ideology scores", x=0.02, ha="left", color=INK
)
fig.text(
    0.02,
    0.88,
    f"Speeches all three models scored ({scope}). Ideology from −1 liberal to +1 conservative. Preliminary.",
    ha="left",
    va="top",
    color=INK_SECONDARY,
    fontsize=9,
)
fig.subplots_adjust(top=0.72, bottom=0.04)
print(
    save_figure(
        fig,
        (
            "pilot_agreement_table"
            if not args.include_pilot_1
            else "pilot_agreement_table_407"
        ),
    )
)
print(table.to_string(index=False))
