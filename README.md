# Quantifying Political Polarization in the U.S. Congress Using LLMs

Master thesis project, Nova SBE — advisor: Yufei Shen. We measure how ideological
distance between Democrats and Republicans in U.S. congressional floor speeches
evolved from 2001 to 2025, using LLM-derived continuous scores validated against
DW-NOMINATE.

**Read [CLAUDE.md](CLAUDE.md) before writing code.** It holds the research
questions, the methodological decisions (and the ones still open), and the rules
this repo is organized around. This README only covers setup and layout.

## Setup

Needs **Python 3.11+**. Check with `python3 --version` before you start —
macOS ships 3.9, which is too old.

```bash
make setup                         # creates .venv, installs pinned versions
cp .env.example .env               # then fill in your API keys
```

Or by hand, if you'd rather not use make:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt
```

**Always work inside the venv.** Installing into your base or conda
environment is how people end up with different versions of pandas and pyarrow
and results that don't reproduce. Every `make` target calls `.venv/bin/`
explicitly, so they work whether or not you activated first.

`.env` and `.venv/` are git-ignored. Never commit keys.

### Dependencies

Two files, on purpose:

- `requirements.txt` — the direct dependencies, with loose ranges. Says *what*
  the project needs. Edit this one.
- `requirements.lock.txt` — every package pinned to an exact version, generated
  from the above. Says what the team is actually *running*. Install from this.

Adding a dependency: add it to `requirements.txt`, run `make lock`, commit both.

### Everyday commands

```bash
make test        # pytest
make lint        # ruff + black --check
make format      # apply black, autofix ruff
make smoke       # corpus build on the first 50k rows (needs raw data)
make corpus      # full corpus build       (needs raw data)
```

`make test` and `make lint` need no data and should pass on a fresh clone.

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

## Building the corpus

The raw Stanford parquet is not in the repo. Download
`congress_speeches_2001_2017.parquet` (~681 MB) from the team drive into
`data/raw/stanford/` — see [docs/notes/2026-09-15_stanford_dataset.md](docs/notes/2026-09-15_stanford_dataset.md).

```bash
make smoke       # first 50k rows; seconds. Run this first.
make corpus      # full run
```

Writes `data/processed/corpus.parquet` and, for full runs,
`results/metrics/corpus_build_stats.json` — quote the stats file for any row
count that ends up in the thesis. The build refuses to guess: a `party == "I"`
speech by a member outside `CAUCUS_PARTY` in `code/src/config.py` stops it with
a message naming who to add.

Until the govinfo loader exists, this corpus stops at January 3, 2017. Check the
`source` column before reading a trend off it.

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
