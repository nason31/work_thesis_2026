"""Central configuration: paths and shared constants.

CLAUDE.md rule: no hardcoded file paths anywhere else in the project.
Import from here instead:

    from src.config import RAW_GOVINFO, PROCESSED_DIR
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- roots -------------------------------------------------------------
# config.py lives in code/src/, so the repo root is two levels up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Data may live outside the repo (shared drive) via THESIS_DATA_DIR.
DATA_DIR: Path = Path(os.getenv("THESIS_DATA_DIR", REPO_ROOT / "data"))

# --- data --------------------------------------------------------------
RAW_DIR: Path = DATA_DIR / "raw"  # read-only, never written to
RAW_STANFORD: Path = RAW_DIR / "stanford"  # 2001-2016, Gentzkow/Shapiro/Taddy
RAW_GOVINFO: Path = RAW_DIR / "govinfo"  # 2017-2025, own pipeline
RAW_DW_NOMINATE: Path = RAW_DIR / "dw_nominate"  # VoteView validation scores
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Merged corpus: one row per speech, columns per CLAUDE.md ->
# speech_id, date, member_id, party, chamber, congress_number, text, source
CORPUS_PATH: Path = PROCESSED_DIR / "corpus.parquet"

# The Stanford parquet as published on the team drive. Not in the repo --
# download it into RAW_STANFORD before running the corpus build.
STANFORD_PARQUET: Path = RAW_STANFORD / "congress_speeches_2001_2017.parquet"

# --- results -----------------------------------------------------------
RESULTS_DIR: Path = REPO_ROOT / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"  # committed
METRICS_DIR: Path = RESULTS_DIR / "metrics"  # committed
SCORES_DIR: Path = RESULTS_DIR / "scores"  # raw per-model LLM output, git-ignored

# Corpus build statistics (row counts, drop counts, distributions). Committed,
# so the numbers quoted in the methodology chapter have a traceable source.
BUILD_STATS_PATH: Path = METRICS_DIR / "corpus_build_stats.json"

# --- thesis ------------------------------------------------------------
THESIS_DIR: Path = REPO_ROOT / "thesis"
FIGURES_DIR: Path = THESIS_DIR / "figures"  # final figures, copied from PLOTS_DIR

# --- analysis constants ------------------------------------------------
# The govinfo pipeline covers fewer speeches per year than the Stanford data.
# Mark this year in every time-series plot (CLAUDE.md: "do not ignore the gap").
SOURCE_BREAK_YEAR: int = 2017
YEAR_RANGE: tuple[int, int] = (2001, 2025)

# Stanford data covers the 107th-114th Congress. Primary aggregation is by
# Congress, not calendar year (CLAUDE.md "Decided -- data decisions").
CONGRESS_RANGE_STANFORD: tuple[int, int] = (107, 114)

# Corpus build: drop speeches shorter than this many words. Inclusive, so a
# 50-word speech is kept. Strips procedural filler ("I yield back the balance
# of my time") without biasing the corpus toward members who give long
# set-piece speeches. Stricter thresholds belong at the LLM-sampling step,
# where they cost nothing to change -- filtering harder here would require a
# full rebuild to undo.
MIN_WORD_COUNT: int = 50

# Independents -> the party they caucus with. CLAUDE.md "Decided" section.
# Keyed by (UPPERCASE last name, state) and applied ONLY to party == "I" rows,
# so e.g. Rep. Steve King (R-IA) is never touched. A party == "I" row that is
# not in here makes the build raise rather than guess -- extending this dict is
# a deliberate, documented act, and the methodology section must match it.
CAUCUS_PARTY: dict[tuple[str, str], str] = {
    ("SANDERS", "VT"): "D",
    ("KING", "ME"): "D",
    ("JEFFORDS", "VT"): "D",
}

# Ensemble composition decided — three models from distinct providers and
# training paradigms. Pin specific snapshot versions for reproducibility;
# update these constants (and CLAUDE.md) if a version is changed mid-project.
#   deepseek-reasoner  — DeepSeek R1, reasoning model, open-weight (RL-trained)
#   gpt-4o-2024-11-20  — GPT-4o snapshot, industry-standard baseline (OpenAI)
#   claude-3-5-sonnet  — Claude 3.5 Sonnet, Constitutional AI paradigm (Anthropic)
ENSEMBLE_MODELS: tuple[str, ...] = (
    "deepseek-reasoner",  # DeepSeek R1 — reasoning model, open-weight
    "gpt-4o-2024-11-20",  # GPT-4o — industry-standard baseline
    "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet — Constitutional AI
)

# Prompt templates live as files next to this module so every run can log the
# exact template text it used.
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"
