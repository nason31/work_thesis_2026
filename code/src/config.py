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

# --- results -----------------------------------------------------------
RESULTS_DIR: Path = REPO_ROOT / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"  # committed
METRICS_DIR: Path = RESULTS_DIR / "metrics"  # committed
SCORES_DIR: Path = RESULTS_DIR / "scores"  # raw per-model LLM output, git-ignored

# --- thesis ------------------------------------------------------------
THESIS_DIR: Path = REPO_ROOT / "thesis"
FIGURES_DIR: Path = THESIS_DIR / "figures"  # final figures, copied from PLOTS_DIR

# --- analysis constants ------------------------------------------------
# The govinfo pipeline covers fewer speeches per year than the Stanford data.
# Mark this year in every time-series plot (CLAUDE.md: "do not ignore the gap").
SOURCE_BREAK_YEAR: int = 2017
YEAR_RANGE: tuple[int, int] = (2001, 2025)

# OPEN DECISION (CLAUDE.md): ensemble composition is undecided until the pilot
# run. Models are listed in one place so nothing downstream hardcodes them.
CANDIDATE_MODELS: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-reasoner",
    "gemini-2.5-flash",
)

# Prompt templates live as files next to this module so every run can log the
# exact template text it used.
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"
