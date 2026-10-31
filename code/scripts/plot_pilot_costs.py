"""Figure: what the two three-model pilots (P1: 200 speeches, P6: 208) cost per model.

Reads only the committed pilot summaries; no API calls. Costs are at the prices
configured when each pilot ran, so DeepSeek is overstated in P6 (docs/decisions.md M3b).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib.pyplot as plt  # noqa: E402

from src.plots import (  # noqa: E402
    BASELINE, INK, INK_SECONDARY, MODEL_PALETTE, SURFACE, save_figure,
)

METRICS = ROOT / "results" / "metrics"
RUNS = ("pilot_summary_20260923T103556Z.json", "s8_summary_20261003T100851Z.json")
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
    ax.text(cost[model] + 0.03, i, f"${cost[model]:.2f}  ({cost[model] / total:.0%})",
            va="center", color=INK, fontsize=10)
ax.set_yticks(range(len(MODELS)), [m[1] for m in MODELS], color=INK)
ax.invert_yaxis()
ax.set_xlim(0, max(cost.values()) * 1.35)
ax.set_xlabel("Cost in US dollars", color=INK_SECONDARY)
ax.tick_params(colors=INK_SECONDARY, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
fig.suptitle(f"What the {speeches} pilot speeches cost, by model", x=0.01, ha="left", color=INK)
fig.text(0.01, 0.89, f"Total ${total:.2f}; each speech scored by all three models. Prices as configured "
         "when each pilot ran;\nDeepSeek is overstated in the second pilot.",
         ha="left", va="top", color=INK_SECONDARY, fontsize=8.5)
fig.subplots_adjust(top=0.72)
print(save_figure(fig, "pilot_costs_by_model"), {k: round(v, 3) for k, v in cost.items()}, round(total, 3), speeches)
