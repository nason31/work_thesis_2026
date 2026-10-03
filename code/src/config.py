"""Central configuration: paths and shared constants.

CLAUDE.md rule: no hardcoded file paths anywhere else in the project.
Import from here instead:

    from src.config import RAW_GOVINFO, PROCESSED_DIR
"""

from __future__ import annotations

import datetime as dt
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
RAW_GOVINFO: Path = RAW_DIR / "govinfo"  # 2016-09-10 onward, Konsti's pipeline
RAW_DW_NOMINATE: Path = RAW_DIR / "dw_nominate"  # VoteView validation scores
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Merged corpus, both sources: one row per speech, columns per CLAUDE.md ->
# speech_id, date, member_id, party, chamber, congress_number, text, source
# (plus last_name, state, word_count, party_original, icpsr). Built by
# `make corpus` from the two processed sides below. See docs/decisions.md D19.
CORPUS_PATH: Path = PROCESSED_DIR / "corpus.parquet"

# Processed Stanford side, 2001-01-03 .. 2016-09-09. Built by `make stanford`.
STANFORD_CORPUS_PATH: Path = PROCESSED_DIR / "corpus_stanford.parquet"

# The Stanford parquet as published on the team drive. Not in the repo --
# download it into RAW_STANFORD before running the corpus build.
STANFORD_PARQUET: Path = RAW_STANFORD / "congress_speeches_2001_2017.parquet"

# The govinfo speeches as published on the team drive, built by Konsti's
# code/notebooks/02_govinfo_dataset.ipynb (Colab). Download into RAW_GOVINFO.
# Must be the 2026-10-02 re-fetch or later: the build requires its `state` key.
GOVINFO_JSONL: Path = RAW_GOVINFO / "congress_speeches_2016_present.jsonl"

# unitedstates/congress-legislators, JSON build (CC0). The govinfo file's own
# party/ICPSR columns are NOT trusted -- its lookup ignored chamber and term-level
# party switches -- so the build re-resolves every speaker against these.
# Download from https://unitedstates.github.io/congress-legislators/
RAW_LEGISLATORS: Path = RAW_DIR / "congress_legislators"
LEGISLATORS_FILES: tuple[Path, ...] = (
    RAW_LEGISLATORS / "legislators-current.json",
    RAW_LEGISLATORS / "legislators-historical.json",
)

# Processed govinfo side, 2016-09-10 onward. Same columns as the Stanford side
# plus `icpsr`. Built by `make govinfo`; merged into CORPUS_PATH by `make corpus`.
GOVINFO_CORPUS_PATH: Path = PROCESSED_DIR / "corpus_govinfo.parquet"

# Voteview member file, every Congress (https://voteview.com/data, "Member
# Ideology", HSall_members.csv). Read-only; the crosswalk filters it in code.
# data/processed/dw_nominate_107_119.csv is exactly that filter, written by hand
# before the crosswalk existed -- identical values, verified 2026-09-29.
DW_NOMINATE_MEMBERS: Path = RAW_DW_NOMINATE / "HSall_members.csv"

# Speech corpus member -> Voteview ICPSR, one row per source x member_id x
# congress x chamber x party_original. Built by `make crosswalk` from
# CORPUS_PATH; joins back onto the corpus on those five columns. Only rows with
# in_validation=True enter the DW-NOMINATE validation -- the trend analysis
# uses every speech regardless. See docs/decisions.md D21-D24.
CROSSWALK_PATH: Path = PROCESSED_DIR / "member_crosswalk.parquet"

# --- results -----------------------------------------------------------
RESULTS_DIR: Path = REPO_ROOT / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"  # committed
METRICS_DIR: Path = RESULTS_DIR / "metrics"  # committed
SCORES_DIR: Path = RESULTS_DIR / "scores"  # raw per-model LLM output, git-ignored

# Corpus build statistics (row counts, drop counts, distributions). Committed,
# so the numbers quoted in the methodology chapter have a traceable source.
STANFORD_BUILD_STATS_PATH: Path = METRICS_DIR / "stanford_build_stats.json"
GOVINFO_BUILD_STATS_PATH: Path = METRICS_DIR / "govinfo_build_stats.json"
MERGED_BUILD_STATS_PATH: Path = METRICS_DIR / "merged_build_stats.json"

# Crosswalk match rates (per source, Congress and chamber) and the members left
# unmatched, with speech counts, so gaps in the validation can be judged.
CROSSWALK_STATS_PATH: Path = METRICS_DIR / "crosswalk_build_stats.json"
CROSSWALK_UNMATCHED_PATH: Path = METRICS_DIR / "crosswalk_unmatched.csv"

# --- thesis ------------------------------------------------------------
THESIS_DIR: Path = REPO_ROOT / "thesis"
FIGURES_DIR: Path = THESIS_DIR / "figures"  # final figures, copied from PLOTS_DIR

# --- analysis constants ------------------------------------------------
# Where Stanford hands over to govinfo. govinfo yields about 20% fewer House and
# 38% fewer Senate speeches per year (measured, docs/decisions.md O8), so mark
# the break in every time-series plot (CLAUDE.md: "do not ignore the
# gap"). The break falls inside the 114th Congress, not on a Congress boundary:
# on a per-Congress axis, the 114th is the mixed-source point.
SOURCE_BREAK_DATE: dt.date = dt.date(2016, 9, 10)
SOURCE_BREAK_YEAR: int = SOURCE_BREAK_DATE.year
SOURCE_BREAK_CONGRESS: int = 114
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

# --- govinfo side --------------------------------------------------------
# The seam. Stanford's last speech is 2016-09-09 (verified), so govinfo starts
# the next day: no gap, no overlap. Rows outside this window are dropped and
# counted. The end is YEAR_RANGE's last day; the file runs into 2026, which is
# outside the thesis scope. See docs/decisions.md D15.
GOVINFO_START_DATE: dt.date = SOURCE_BREAK_DATE
GOVINFO_END_DATE: dt.date = dt.date(YEAR_RANGE[1], 12, 31)

# The state as the Record writes it beside a speaker ("Mr. SMITH of Texas.") ->
# the congress-legislators code. Keys are upper-case; lookup folds case. A
# string not listed here (misspellings, regex overruns, "of Japan") counts as no
# state, never as a guess. Territories are listed so their delegates resolve and
# are then dropped as delegates. See docs/decisions.md D25.
STATE_CODES: dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN",
    "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "AMERICAN SAMOA": "AS", "DISTRICT OF COLUMBIA": "DC", "GUAM": "GU",
    "NORTHERN MARIANA ISLANDS": "MP", "PUERTO RICO": "PR", "VIRGIN ISLANDS": "VI",
}  # fmt: skip

# Keyed on bioguide id, not speakerid: govinfo rows are resolved against
# congress-legislators, which has no speakerid. Applied only while the member's
# party *on that date* is not Democrat/Republican (congress-legislators records
# day-level affiliations). The build checks each entry against the source's own
# `caucus` field and raises on a disagreement or an unlisted independent.
GOVINFO_CAUCUS_PARTY: dict[str, str] = {
    "S000033": "D",  # Bernie Sanders (VT, Senate)
    "K000383": "D",  # Angus King (ME, Senate)
    "M001183": "D",  # Joe Manchin (WV) — Independent from 2024-05-31, caucused D
    "S001191": "D",  # Kyrsten Sinema (AZ) — Independent from 2022-12-09, caucused D
}

# Excluded only for the dates their party is neither D nor R -- their earlier
# Republican speeches stay in. Same rule as Barkley: no caucus, no assignment.
GOVINFO_EXCLUDED_MEMBERS: frozenset[str] = frozenset(
    {
        "M001201",  # Paul Mitchell (MI-10) — left the GOP 2020-12-14, no caucus
        "A000367",  # Justin Amash (MI-3) — Independent 2019-07-04, then Libertarian
    }
)

# --- DW-NOMINATE crosswalk ---------------------------------------------
# Congresses the corpus spans: 107th (Stanford, from 2001) to 119th (govinfo,
# to 2025). Voteview rows outside are ignored.
VOTEVIEW_CONGRESS_RANGE: tuple[int, int] = (107, 119)

# Corpus party code -> Voteview party_code, used ONLY to break a tie between
# same-surname candidates, and only on `party_original` -- the caucus rule
# rewrites `party` (Jeffords I -> D), while Voteview codes independents 328.
# Voteview gives a member who switches party a new ICPSR, so a switcher has two
# rows in that Congress; this picks the one matching the corpus's party label.
VOTEVIEW_PARTY_CODES: dict[str, int] = {"D": 100, "R": 200, "I": 328}

# Voteview scores the LLM ideology scores are validated against. Both are
# reported until one is chosen -- NOT YET DECIDED (docs/decisions.md O9):
# nominate_dim1 is constant over a member's career, nokken_poole_dim1 is
# estimated per Congress and so can show a member moving.
VALIDATION_BENCHMARKS: tuple[str, ...] = ("nominate_dim1", "nokken_poole_dim1")

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

# Pilot size. CLAUDE.md: always run 100-500 speeches before any larger run.
# There is no full-corpus run; the main results use the S8 sample below (S11).
# The first pilot (2026-09-23) used 200, stratified party x congress.
PILOT_SAMPLE_SIZE: int = 200

# Stratification keys for the sample, and how many speeches per cell.
# party x congress x chamber = 2 x 13 x 2 = 52 cells over the merged corpus
# (107th-119th); at 100 each that is 5,200 speeches. The smallest cell (119th,
# R, Senate) holds 2,627 rows, so the quota is never short.
# Chamber was added after the first pilot: House and Senate floor rhetoric differ
# in length and formality, and an unbalanced split would confound a per-Congress
# comparison with a drift in chamber mix.
STRATIFY_BY: tuple[str, ...] = ("congress_number", "party", "chamber")
PILOT_PER_CELL: int = 100

# Measured per-speech token usage from the 200-speech pilot
# (results/metrics/pilot_summary_20260923T103556Z.json). Used to estimate cost
# from observation rather than a flat guess: the providers differ by ~30% on
# input tokens for identical text because their tokenizers differ, and
# deepseek-reasoner emits ~7x the output of the other two because its reasoning
# is billed as output.
PILOT_TIKTOKEN_INPUT_PER_SPEECH: float = 903.5
PILOT_MEASURED_TOKENS: dict[str, dict[str, float]] = {
    "deepseek-reasoner": {"input": 926.8, "output": 529.0},
    "gpt-4o-2024-11-20": {"input": 909.1, "output": 69.1},
    "claude-sonnet-4-6": {"input": 1172.2, "output": 93.7},
}

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
