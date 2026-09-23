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

# --- corpus membership rules -------------------------------------------
# Keyed on `speakerid`, NOT on any name column. Names in this file are damaged
# by OCR: `speaker` carries an honorific plus noise on 99.7% of rows
# ("Mr. JEFFORDS", "LR. JEFFORDS", "Mr.. JEFFORDS"), and even the cleaner
# `last_name` has spelling variants on 30.6% of speakerids (JEFFORD/JEFFORDS,
# LIEBERMAN/LIEDERMAN/LISBERMAN/LIEBERMANN, SANDERS/SANDERR/SA.NDERS).
# `speakerid` is exact, and `state_map` is consistent within every speakerid
# (0 exceptions in 823,341 rows).
#
# speakerid encodes the congress in its first three digits, so a member serving
# several congresses needs one entry per congress. The lists below are complete
# for the 107th-114th: they were enumerated from the data, not guessed.
# See docs/notes/2026-09-21_stanford_data_reality.md.

# Independents -> the party they caucus with. CLAUDE.md "Decided" section.
# Applied ONLY to party == "I" rows. A party == "I" row matching no rule here
# makes the build raise rather than guess.
CAUCUS_PARTY: dict[str, str] = {
    # Jim Jeffords (VT, Senate) — left the GOP in May 2001, caucused with D
    "107113101": "D",
    "108113101": "D",
    "109113101": "D",
    # Bernie Sanders (VT) — House 107-109, Senate 110-114; caucuses with D
    "107118220": "D",
    "108118220": "D",
    "109118220": "D",
    "110118221": "D",
    "111118221": "D",
    "112118221": "D",
    "113118221": "D",
    "114118221": "D",
    # Joe Lieberman (CT, Senate) — Independent Democrat, caucused with Senate D
    "110116471": "D",
    "111116471": "D",
    "112116471": "D",
    # Angus King (ME, Senate) — caucuses with D
    "113122291": "D",
    "114122291": "D",
}

# Party labels in the raw file that are wrong, corrected here with the reason.
# Applied ONLY to party == "I" rows. Keep this as close to empty as possible:
# every entry overrides the source on outside knowledge.
PARTY_CORRECTIONS: dict[str, str] = {
    # Ander Crenshaw (FL-4, House, 107th) represented his district as a
    # Republican for his entire career (2001-2017); the "I" coding is a source
    # error. 34 rows.
    "107119090": "R",
}

# Members excluded entirely, because the caucus rule has no answer for them.
EXCLUDED_MEMBERS: frozenset[str] = frozenset(
    {
        # Dean Barkley (MN, Senate, 107th) — appointed Nov 2002 to fill
        # Wellstone's seat for ~2 months, Minnesota Independence Party,
        # caucused with neither party, so any assignment is arbitrary. 6 rows.
        "107113431",
    }
)

# Non-voting delegates and resident commissioners, dropped corpus-wide. They
# cannot vote on final passage, so DW-NOMINATE -- the validation anchor -- does
# not score them comparably, leaving their speeches with no benchmark. 5,665
# rows, 0.69%. Dropping them also removes the stray "A"/"P" party codes, which
# belong solely to Acevedo-Vila, Resident Commissioner of Puerto Rico.
# Keyed on state_map, which is clean and consistent for every speakerid.
DELEGATE_STATES: frozenset[str] = frozenset({"DC", "PR", "VI", "GU", "AS", "MP"})

# After the rules above, party must be exactly this set, or the build raises.
EXPECTED_PARTIES: frozenset[str] = frozenset({"D", "R"})

# Ensemble composition decided — three models from distinct providers and
# training paradigms. Pin specific snapshot versions for reproducibility;
# update these constants (and CLAUDE.md) if a version is changed mid-project.
#   deepseek-reasoner  — DeepSeek R1, reasoning model, open-weight (RL-trained)
#   gpt-4o-2024-11-20  — GPT-4o snapshot, industry-standard baseline (OpenAI)
#   claude-sonnet-4-6  — Claude Sonnet 4.6, Constitutional AI paradigm (Anthropic)
# Claude 3.5 Sonnet was the original choice but was RETIRED (404) before the
# first run; Sonnet 4.6 replaces it at the same tier and price. See
# docs/decisions.md M3a.
ENSEMBLE_MODELS: tuple[str, ...] = (
    "deepseek-reasoner",  # DeepSeek R1 — reasoning model, open-weight
    "gpt-4o-2024-11-20",  # GPT-4o — industry-standard baseline
    "claude-sonnet-4-6",  # Claude Sonnet 4.6 — Constitutional AI
)

# Prompt templates live as files next to this module so every run can log the
# exact template text it used.
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

# Prompt template for the two-dimensional scoring call. Loaded at runtime and
# logged verbatim with every run (CLAUDE.md reproducibility rule).
SCORE_PROMPT_PATH: Path = PROMPTS_DIR / "score_speech.txt"

# --- scoring run constants ---------------------------------------------
# Seed for every sampling step, logged with each run so a pilot can be redrawn.
RANDOM_SEED: int = 42

# Pilot size. CLAUDE.md: always run 100-500 speeches before any full run.
PILOT_SAMPLE_SIZE: int = 200

# NO TEMPERATURE IS SET. Not a preference -- the providers removed the control:
# `deepseek-reasoner` ignores it, and the anthropic SDK dropped the parameter
# entirely (current models return "`temperature` is deprecated for this model").
# Rather than set it on one model of three and imply the ensemble is uniformly
# configured, all three run at their provider default, and every run manifest
# records that. See docs/decisions.md S2a.
SCORING_TEMPERATURE = None

# Score bounds the prompt promises. Responses outside these are recorded as
# failures rather than clipped -- a model ignoring the scale is a finding.
IDEOLOGY_SCORE_RANGE: tuple[float, float] = (-1.0, 1.0)
TONE_SCORE_RANGE: tuple[float, float] = (0.0, 1.0)

# Cross-model standard deviation above which a speech is flagged as one the
# ensemble disagrees on. Provisional -- revisit once the pilot shows the
# distribution of disagreement.
ENSEMBLE_DISAGREEMENT_THRESHOLD: float = 0.3

# USD per 1M tokens, list prices as of September 2026. Approximate and provider
# pricing changes, so treat every cost figure as an estimate, not an invoice.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}
