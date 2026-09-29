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
make smoke       # Stanford build on the first 50k rows (needs raw data)
make stanford    # Stanford side           (needs raw data)
make govinfo     # govinfo side            (needs raw data)
make corpus      # merge both into data/processed/corpus.parquet
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

The corpus is built in three steps: each source separately, then the merge.
The raw files are not in the repo — download them from the team drive first:

- `congress_speeches_2001_2017.parquet` (~681 MB) into `data/raw/stanford/` —
  see [data/raw/stanford/README.md](data/raw/stanford/README.md)
- `congress_speeches_2016_present.jsonl` into `data/raw/govinfo/`, plus the
  congress-legislators JSON into `data/raw/congress_legislators/` — see
  [data/raw/govinfo/README.md](data/raw/govinfo/README.md)

```bash
make smoke       # Stanford, first 50k rows; seconds. Run this first.
make stanford    # -> data/processed/corpus_stanford.parquet (2001-01-03 .. 2016-09-09)
make govinfo     # -> data/processed/corpus_govinfo.parquet  (2016-09-12 .. 2025-12-19)
make corpus      # -> data/processed/corpus.parquet, both merged (538,804 rows)
```

Each step writes its stats to `results/metrics/` (`stanford_build_stats.json`,
`govinfo_build_stats.json`, `merged_build_stats.json`) — quote those files for
any row count that ends up in the thesis. The builds refuse to guess: a
`party == "I"` speech by a member outside `CAUCUS_PARTY` in
`code/src/config.py` stops the Stanford build with a message naming who to add,
and the merge stops on any seam overlap, repeated `speech_id` or stale input.

The data switches source on 2016-09-10, inside the 114th Congress. Every row
carries a `source` column; mark the break in every time-series plot.

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
