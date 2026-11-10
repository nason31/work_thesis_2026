"""Figure: how closely the three models agree on the 208 pilot speeches all three scored.

Pairwise scatter of ideology scores (run 20261003T100851Z). Reads the raw per-model scores in results/scores/ (on the
model-development branch); makes no API calls.

    python code/scripts/plot_pilot_agreement.py [--scores-dir DIR]
"""
import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.plots import (  # noqa: E402
    BASELINE, GRID, INK, INK_SECONDARY, PARTY_COLORS, PARTY_NAMES, SURFACE, save_figure,
)

RUNS = (
    ("s8", "20261003T100851Z", {"deepseek-reasoner": "DeepSeek", "gpt-4o-2024-11-20": "GPT-4o", "claude-sonnet-4-6": "Sonnet 4.6"}),
)
NAMES = ("DeepSeek", "GPT-4o", "Sonnet 4.6")

parser = argparse.ArgumentParser()
parser.add_argument("--scores-dir", type=Path, default=ROOT / "results" / "scores")
args = parser.parse_args()

frames = []
for prefix, run, models in RUNS:
    frame = None
    for model, name in models.items():
        d = pd.read_json(args.scores_dir / f"{prefix}_{model}_{run}.jsonl", lines=True)
        d = d.drop_duplicates("speech_id")
        d = d[d["error"].isna() & d["ideology_score"].notna()].set_index("speech_id")
        part = d[["ideology_score"]].rename(columns={"ideology_score": name})
        if frame is None:
            part["party"] = d["party"]
            frame = part
        else:
            frame = frame.join(part, how="inner")
    frames.append(frame)
data = pd.concat(frames)
assert data.index.is_unique

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE, sharex=True, sharey=True)
for ax, (x, y) in zip(axes, itertools.combinations(NAMES, 2)):
    ax.set_facecolor(SURFACE)
    ax.plot([-1, 1], [-1, 1], color=BASELINE, lw=1, zorder=1)
    for party in ("D", "R"):
        sub = data[data["party"] == party]
        ax.scatter(sub[x], sub[y], s=16, alpha=0.5, color=PARTY_COLORS[party],
                   edgecolor="none", label=PARTY_NAMES[party], zorder=2)
    ax.text(0.04, 0.95, f"r = {data[x].corr(data[y]):.3f}", transform=ax.transAxes,
            va="top", color=INK, fontsize=11)
    ax.set_xlabel(f"{x} ideology score", color=INK_SECONDARY)
    ax.set_ylabel(f"{y} ideology score", color=INK_SECONDARY)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.7)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
axes[0].legend(loc="lower right", frameon=False)
fig.suptitle("The three models score the pilot speeches almost alike", x=0.01, ha="left", color=INK)
fig.subplots_adjust(top=0.88)
print(save_figure(fig, "pilot_model_agreement"))
