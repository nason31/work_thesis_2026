"""Figure: how the DW-NOMINATE validation changes with the member-speech threshold (A5).

Reads only the committed validation grid (results/metrics/validation_grid_<run>.csv);
no API calls. Headline design: member x Congress units, A3's label rule for
procedural speech, Nokken-Poole benchmark, Pearson r within party x chamber x
Congress cells. Only the minimum number of speeches per unit varies. The
interval is the 95% bootstrap over members (A5); chamber lines have none.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

import matplotlib.pyplot as plt
import pandas as pd

from src.plots import (
    BAND_ALPHA,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    LINE_WIDTH,
    MARKER_SIZE,
    SURFACE,
    _style,
    save_figure,
)

RUN = "20261018T122536Z"
CHOSEN = 10  # A5: the member threshold the trend analysis uses (A3)
POOLED, HOUSE, SENATE = "#4a3aa7", "#eb6834", "#1baf7a"

grid = pd.read_csv(ROOT / "results" / "metrics" / f"validation_grid_{RUN}.csv")
rows = grid[
    (grid["unit"] == "member_congress")
    & (grid["procedural"] == "labels")
    & (grid["benchmark"] == "nokken_poole_dim1")
].sort_values("min_speeches")
x = list(range(len(rows)))
thresholds = rows["min_speeches"].tolist()

fig, ax = plt.subplots(figsize=(7, 4.6))
fig.patch.set_facecolor(SURFACE)
_style(ax)

ax.fill_between(
    x,
    rows["within_cell_pearson_ci_low"],
    rows["within_cell_pearson_ci_high"],
    color=POOLED,
    alpha=BAND_ALPHA,
    linewidth=0,
)
ax.plot(
    x,
    rows["within_cell_pearson"],
    color=POOLED,
    linewidth=LINE_WIDTH,
    marker="o",
    markersize=MARKER_SIZE + 1,
)
ax.plot(
    x,
    rows["within_cell_H"],
    color=HOUSE,
    linewidth=1.1,
    marker="o",
    markersize=MARKER_SIZE - 1,
)
ax.plot(
    x,
    rows["within_cell_S"],
    color=SENATE,
    linewidth=1.1,
    marker="o",
    markersize=MARKER_SIZE - 1,
)

# Direct labels at the right-hand end, so no legend is needed.
last = len(rows) - 1
for col, text, color, dy in (
    ("within_cell_S", "Senate", SENATE, 0.018),
    ("within_cell_pearson", "Both chambers (95% CI band)", POOLED, 0.0),
    ("within_cell_H", "House", HOUSE, -0.018),
):
    ax.annotate(
        text,
        (last, rows[col].iloc[-1]),
        xytext=(8, dy * 250),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        va="center",
    )

# Mark the chosen threshold.
ci = thresholds.index(CHOSEN)
ax.axvline(ci, color=INK_MUTED, linestyle="--", linewidth=0.9)
ax.text(ci, 0.275, " chosen (A3, A5)", color=INK_SECONDARY, fontsize=8, va="bottom")

units = [f"{int(d) + int(r):,}" for d, r in zip(rows["n_units_D"], rows["n_units_R"])]
ax.set_xticks(x)
ax.set_xticklabels([f"≥ {t}\n{u} units" for t, u in zip(thresholds, units)])
ax.set_xlim(-0.3, last + 1.55)
ax.set_ylim(0.25, 0.7)
ax.set_xlabel(
    "Minimum speeches per member × Congress  (units kept)",
    color=INK_SECONDARY,
    fontsize=9,
)
ax.set_ylabel(
    "Within-cell Pearson r with Nokken-Poole", color=INK_SECONDARY, fontsize=9
)
fig.suptitle(
    "The validation barely moves with the speech threshold",
    color=INK,
    fontsize=12,
    x=0.125,
    ha="left",
    y=1.0,
)
ax.set_title(
    "Raising the cut-off removes most units but r stays 0.48–0.54. DeepSeek, d100k.\n"
    "House falls at ≥ 20, where only 3–20 representatives per cell remain.",
    color=INK_SECONDARY,
    fontsize=8.5,
    loc="left",
)

print(save_figure(fig, "validation_threshold_sensitivity"))
