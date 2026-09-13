# Quantifying Political Polarization in the U.S. Congress Using LLMs

Master thesis project, Nova SBE — advisor: Yufei Shen. We measure how ideological
distance between Democrats and Republicans in U.S. congressional floor speeches
evolved from 2001 to 2025, using LLM-derived continuous scores validated against
DW-NOMINATE.

**Read [CLAUDE.md](CLAUDE.md) before writing code.** It holds the research
questions, the methodological decisions (and the ones still open), and the rules
this repo is organized around. This README only covers setup and layout.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in your API keys
```

`.env` is git-ignored. Never commit keys.

## Layout

```
code/
  src/          importable modules (production logic lives here)
    prompts/    prompt templates as files, so runs can log the exact text used
    config.py   all paths and shared constants — import, never hardcode
  scripts/      one-off pipeline / analysis scripts that call src/
  notebooks/    exploration only
data/
  raw/          stanford/ govinfo/ dw_nominate/ — read-only, git-ignored
  processed/    merged corpus — git-ignored
results/
  plots/        figures (committed)
  metrics/      tables, evaluation summaries (committed)
  scores/       raw per-model LLM output (git-ignored, can get large)
docs/notes/     meeting notes, design decisions
thesis/
  chapters/     01_introduction … 07_conclusion
  figures/      final figures, copied from results/plots/
  references/   bibliography
  appendix/
```

Data and raw model outputs stay out of git — share them via the team drive.
Folders are kept in git with `.gitkeep` placeholders.

## Naming

- Files and folders: lowercase with underscores
- Notes and dated artifacts: `YYYY-MM-DD_short_topic.md`
- Plots: descriptive names (`party_position_by_year.png`)
- Run outputs: include model and date so runs stay distinguishable

## Working rules (short version)

- `data/raw/` is read-only. Scripts never write there.
- Pilot on 100–500 speeches before any full API run; log tokens and cost.
- Save raw per-model scores to `results/scores/` *before* aggregating.
- Mark the 2017 source break (Stanford → govinfo) in every time-series plot.
- Format with `black`, lint with `ruff`.
