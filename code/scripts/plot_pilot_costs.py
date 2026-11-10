"""Figure: what the 208-speech three-model pilot (P6, run 20261003T100851Z) cost per model.

Reads only the committed run summary; no API calls. Costs are at the prices
configured when the pilot ran, so DeepSeek is overstated (docs/decisions.md M3b).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib.pyplot as plt

from src.plots import (
    BASELINE,
    INK,
    INK_SECONDARY,
    MODEL_PALETTE,
    SURFACE,
    save_figure,
)

METRICS = ROOT / "results" / "metrics"
RUNS = (
    "s8_summary_20261003T100851Z.json",
)  # the 208-speech run keeps the s8 file prefix
MODELS = (
    ("deepseek-reasoner", "DeepSeek", MODEL_PALETTE[0]),
    ("gpt-4o-2024-11-20", "GPT-4o", MODEL_PALETTE[1]),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6", "#1baf7a"),
)

cost = {m: 0.0 for m, _, _ in MODELS}
speeches = 0
for name in RUNS:
    summary = json.loads((METRICS / name).read_text())
    speeches += summary["speeches_sampled"]
    for model in cost:
        cost[model] += summary["per_model"][model]["cost_usd"]
total = sum(cost.values())

fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for i, (model, label, color) in enumerate(MODELS):
    ax.barh(i, cost[model], color=color, height=0.55)
    ax.text(
        cost[model] + 0.03,
        i,
        f"${cost[model]:.2f}  ({cost[model] / total:.0%})",
        va="center",
        color=INK,
        fontsize=10,
    )
ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS], color=INK)
ax.invert_yaxis()
ax.set_xlim(0, max(cost.values()) * 1.35)
ax.set_xlabel("Cost in US dollars", color=INK_SECONDARY)
ax.tick_params(colors=INK_SECONDARY, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
fig.suptitle(
    f"What the {speeches} pilot speeches cost, by model", x=0.01, ha="left", color=INK
)
fig.subplots_adjust(top=0.86)
print(
    save_figure(fig, "pilot_costs_by_model"),
    {k: round(v, 3) for k, v in cost.items()},
    round(total, 3),
    speeches,
)
