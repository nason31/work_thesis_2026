# CLAUDE.md — Master Thesis: Quantifying Political Polarization Using LLMs

This file is the primary reference for any coding agent working in this repository.
Read it fully before writing any code, making decisions, or suggesting architecture changes.

---

## Project Overview

**Thesis title:** Quantifying Political Polarization in the U.S. Congress Using LLMs
**Institution:** Nova School of Business and Economics (Nova SBE), Lisbon
**Advisor:** Yufei Shen (Assistant Professor, OTI Management)
**Timeline:** September 8 – December 8, 2026
**Next hard deadline:** Meeting 2 with Yufei — October 1–15, 2026 (preliminary findings required)
**GitHub:** https://github.com/nason31/work_thesis_2026
**Team:** 4–5 students; same grade for all; each person covers a specific analytical component

**Core goal:** Use LLMs to quantitatively measure how political polarization among U.S. Congress members has evolved from 2001 to 2025, based on floor speeches.

---

## Research Questions

Focus tightly. Yufei's explicit instruction: include findings only if they **add to or complement the main story**. No kitchen-sink approach.

### Primary (must answer)
1. How has ideological distance between Democrats and Republicans changed from 2001 to 2025?
2. **Which party actually moved?** Plot each party's position over time separately — this was Yufei's strongest emphasis in Meeting 1.

### Secondary (include only if they fit the narrative)
3. Does the trend differ across presidential terms?
4. Are there noticeable changes around election periods?
5. Has emotional tone become more hostile over time — and does it track ideology or move independently?
6. *(Stretch/RQ5)* Does the pattern in floor speeches also appear in official public communications (press releases, statements)?

---

## Data Sources

> Status marks: **[verified]** = confirmed against the actual dataset and
> documented in `docs/notes/`. **[assumed]** = planning estimate, not yet
> checked against data. Do not treat an [assumed] value as a fact in code or
> in the thesis — verify it first, then update this file and drop the mark.

### Jan 2001 – Jan 3, 2017: Stanford Dataset (Gentzkow, Shapiro & Taddy)

**[verified]** — built and documented by Konsti, see
[`docs/notes/2026-09-15_stanford_dataset.md`](docs/notes/2026-09-15_stanford_dataset.md)

- **Coverage:** 107th–114th Congress, January 2001 through **January 3, 2017**
  (end of the 114th Congress) — NOT through the end of calendar year 2017.
  The break falls on a clean Congress boundary (114th → 115th), which is
  convenient: govinfo can start at the 115th. But it is 3 days into calendar
  2017, so any year-level aggregation puts those 3 days of Stanford data into
  the 2017 bucket alongside govinfo data. Decide once whether to aggregate by
  Congress (clean) or by calendar year (3 mixed days in 2017) and say so.
- **Source:** `hein-daily.zip` from data.stanford.edu/congress_text, sessions
  107–114, speech text + metadata + speaker map merged per session on `speech_id`
- **Form:** one parquet file, ~650 MB, one row per speech. Not in the repo —
  stored on the team drive (link in the note above). `data/raw/stanford/` is
  where it goes locally.

**Actual columns** (raw, before renaming to the corpus schema):

| Column | Notes |
|---|---|
| `speech_id` | unique per speech |
| `speech` | full text → becomes `text` in the merged corpus |
| `chamber` | House / Senate |
| `date` | **YYYYMMDD**, needs parsing to a date type |
| `speaker` | last name as recorded |
| `first_name` | |
| `state` | |
| `gender` | as recorded in source |
| `word_count` | |
| `speakerid` | Gentzkow's own ID system — **not** ICPSR, see DW-NOMINATE below |
| `party` | **D / R / I** — independents are present, see open decision below |
| `congress` | 107–114 |

**Other facts:**

- **Known filter:** speeches with no matched speaker were dropped during the
  build (mostly procedural entries, e.g. the Clerk reading a bill title).
  **[assumed]** how many — the drop count is not yet recorded. Quantify it
  ("X of Y rows, Z%") before the methodology chapter is written.
- **Not yet reproducible:** the build script is announced as
  `code/scripts/build_stanford_dataset.py` but does not exist yet. Until it
  does, the parquet cannot be rebuilt or independently checked.
- Note: Yufei said we **can drop this** if our govinfo pipeline produces sufficient coverage

### Jan 2017 – 2025: Custom govinfo.gov Pipeline

**[assumed]** — nothing verified against data yet; this whole section is a plan.

- Path: `data/raw/govinfo/`
- Must pick up at **January 3, 2017** where the Stanford data ends, i.e. with
  the **115th Congress** — verify there is no gap and no double-counted overlap
  in the first days of January 2017
- **Known issue [assumed]:** fewer speeches per year than the Stanford dataset —
  this creates a discontinuity at the 2017 break. Address this in the
  methodology section; do NOT silently ignore it. Verify the actual per-year
  counts on both sides of the break and record them in `results/metrics/`.

### Processed / Combined

- Path: `data/processed/`
- Final merged corpus: one row per speech, with columns: `speech_id`, `date`,
  `member_id`, `party`, `chamber`, `congress_number`, `text`, `source`
- **Plus five retained columns** — `speaker`, `first_name`, `state`,
  `word_count`, `party_original`. Not decoration: the `speakerid` → ICPSR
  crosswalk needed for DW-NOMINATE validation is built from name + state +
  congress, and `party_original` preserves the pre-reassignment party so the
  independents mapping stays auditable and a robustness check can exclude them.
  Without these, either job means re-streaming the 681 MB raw file.
- Built by `code/scripts/build_corpus.py` (logic in `code/src/corpus.py`).
  Row counts, drop counts and distributions for every full build land in
  `results/metrics/corpus_build_stats.json` — quote that file, not a guess.

**This is a target schema, not what either source delivers.** The Stanford
side needs an explicit mapping step:

| Corpus column | Stanford source |
|---|---|
| `text` | `speech` |
| `member_id` | `speakerid` — see the ICPSR problem below |
| `congress_number` | `congress` |
| `date` | `date`, parsed from YYYYMMDD |
| `source` | **does not exist in either source — must be added at merge time** |

`source` is what keeps the 2017 break visible in the data itself. Every row
must carry `stanford` or `govinfo`. Do not merge without it.

### External Validation

- **DW-NOMINATE scores** — voting-based ideological measure from VoteView.com
- Used to validate LLM-derived ideological scores against a non-ML, voting-based benchmark
- Path: `data/raw/dw_nominate/`
- **Open blocker:** VoteView identifies members by **ICPSR** number; the Stanford
  data carries `speakerid`, Gentzkow's own identifier. There is currently no
  join path between our speeches and DW-NOMINATE. A crosswalk
  (`speakerid` → ICPSR, e.g. via the Stanford speaker map plus name/state/congress
  matching) has to be built and its match rate reported. Since CLAUDE.md treats
  the DW-NOMINATE check as the primary methodological defense, this blocks the
  validation entirely — resolve it early, not at the end.

### RQ5 Data (not yet committed to)
- Check govinfo.gov for press releases / official statements pre-2010 before starting
- Also check if Gentzkow-Shapiro-Taddy have a press release corpus
- Do NOT build RQ5 pipeline until coverage is confirmed

---

## Methodology

### Core design principles

**1. Continuous scoring, not binary classification**
The output must be a continuous ideological position score per speech, not a binary polarizing/non-polarizing label. Binary throws away too much information — a speech can be mildly or strongly partisan, and collapsing that to 0/1 loses the degree. Our entire analytical value comes from tracking position over time, which requires a scale.

**2. Two-dimensional scoring**
Each speech gets scored on two independent dimensions:
- **Ideological position** — continuous left-right score (this is the primary measure)
- **Tone / hostility** — emotional charge of the language, independent of ideological content

**3. Ensemble across multiple models — composition decided**
Run three LLMs and combine their outputs (average for continuous scores). This reduces single-model bias on a subjective task like ideological scoring, where no single model's internal calibration should be treated as ground truth. Log all individual model scores before aggregation — never discard the raw outputs.

The ensemble is fixed at three models from distinct providers and training paradigms:
- **DeepSeek R1** (`deepseek-reasoner`) — reasoning model, open-weight; selected for explicit chain-of-thought reasoning prior to scoring, well-suited to the multi-step ideological inference required
- **GPT-4o** (`gpt-4o-2024-11-20`) — industry-standard baseline (OpenAI); widely cited in NLP research, enables direct comparison with prior work
- **Claude 3.5 Sonnet** (`claude-3-5-sonnet-20241022`) — Constitutional AI paradigm (Anthropic); provides architectural and training diversity

This combination covers: one reasoning model + two instruction-following models; one open-weight model (reproducible checkpoint); three independent training approaches. Model constants live in `code/src/config.py` as `ENSEMBLE_MODELS` — change them there, nowhere else.

**4. Feed full speeches, not fragments**
Do not split speeches into short chunks unless a model's context window absolutely requires it. Chunking breaks cross-sentence rhetorical context, which matters for detecting ideological framing. Current frontier models (GPT-4o, Llama 4, Gemini 2.5, DeepSeek R1) all handle full congressional speeches comfortably. Verify context length per model before deciding.

**5. Prefer reasoning models for ideological scoring**
Scoring ideological position requires chaining inferences: "what policy positions does this language imply, and where do those sit on the political spectrum?" Reasoning models (o3, DeepSeek R1) are specifically better at this kind of multi-step judgment than standard instruction-following models. Prioritize them for the scoring step; cheaper models are fine for preprocessing and filtering.

**6. Structural validation via DW-NOMINATE**
LLM-derived scores must be cross-checked against DW-NOMINATE (voting-based ideological scores from VoteView.com). This is not optional — it is the primary methodological defense of our approach. Expect the correlation to be imperfect (floor speech rhetoric vs. actual votes are different signals), but it should be positive and meaningful. A weak or inverse correlation would be a red flag requiring investigation, not suppression.

### Decided — data decisions

- **Independents:** assign to the party they caucus with — Sanders → D, King → D,
  Jeffords → D. Applied once at corpus build time (`code/scripts/build_corpus.py`),
  never handled differently in any other script. Stated in the methodology.
- **Aggregation granularity:** by **Congress** (107th through 114th for the Stanford
  data). Clean 2-year periods aligned with the data structure. Calendar year is
  not used for primary aggregation.
- **Unlisted independents:** if a `party == "I"` speech belongs to a member not in
  `CAUCUS_PARTY` (e.g. Lieberman, who sat as an Independent Democrat 2007–2013),
  the build **fails** and names them. It does not guess and does not silently
  drop them. Extend the roster in `config.py` and record the addition here — that
  keeps the rule visible in one place instead of buried in a log.
- **Minimum speech length:** **50 words**, inclusive (`MIN_WORD_COUNT`). Strips
  procedural filler ("I yield back the balance of my time") without biasing the
  corpus toward members who give long set-piece speeches. Stricter thresholds
  belong at the LLM-sampling step, where they are cheap to change; filtering
  harder at build time would need a full rebuild to undo.
- **Text cleaning:** whitespace normalization only — collapse runs, trim ends.
  Boilerplate stripping ("Mr. Speaker,", procedural preambles) is **still open**:
  it changes what the model sees, so it is a methodological choice, not cleaning.
- **Duplicate `speech_id`:** first occurrence kept, count recorded in the build
  stats. `speech_id` is documented as unique, so a non-zero count is a signal
  worth investigating rather than a routine drop.

### What is NOT yet decided — do not hardcode

- **Prompting vs. fine-tuning:** Still open. Continuous scoring strongly favors prompting (zero-shot or few-shot); fine-tuning would lock us into a binary setup and requires labeled data. Do not build infrastructure that assumes one approach without flagging the trade-off in a comment.
- **Reinforcement fine-tuning:** Now more mature than a year ago and requires less labeled data than traditional fine-tuning. Worth evaluating if a supervised component turns out to be needed, but not the default path.

### If manual labeling is done
- Define a clear annotation rubric before labeling anything
- Use at least two independent annotators
- Report inter-annotator agreement using Cohen's kappa — this makes the annotation academically defensible
- Do not rely solely on "own judgment with peer review"

### Key boundaries
- The source discontinuity sits at **January 3, 2017** (end of the 114th Congress,
  where the Stanford data stops and the govinfo pipeline takes over). It must be
  addressed in the methodology section — flag it in code comments and make it
  visible in every time-series plot (e.g. vertical dashed line at the 2017 break).
  `SOURCE_BREAK_YEAR = 2017` in `code/src/config.py` is the year-level constant;
  use the exact date wherever daily resolution matters.
- Every row in the merged corpus carries a `source` column, so the break is
  recoverable from the data and not just from a note
- Do not fabricate or impute data to paper over coverage gaps

---

## Repository Structure

```
work_thesis_2026/
├── CLAUDE.md               ← this file
├── README.md               ← setup instructions for the team
├── Makefile                ← make setup / test / lint / corpus / lock
├── requirements.txt        ← direct dependencies (edit this)
├── requirements.lock.txt   ← pinned versions (generated by `make lock`)
├── pytest.ini              ← pythonpath = code
├── ruff.toml               ← src = ["code"], so src.* is first-party
├── .env.example            ← template for API keys (.env is git-ignored)
├── code/
│   ├── src/                ← production modules (importable, tested)
│   │   ├── config.py       ← ALL paths + shared constants; import, never hardcode
│   │   ├── corpus.py       ← corpus build: load, map, clean, write
│   │   └── prompts/        ← prompt templates as files, logged with every run
│   ├── scripts/            ← one-off analysis and pipeline scripts
│   ├── tests/              ← pytest; run with `pytest` from the repo root
│   └── notebooks/          ← exploratory work; not used in production
├── data/                   ← git-ignored (shared via team drive)
│   ├── raw/                ← original, never modified
│   │   ├── stanford/
│   │   ├── govinfo/
│   │   └── dw_nominate/
│   └── processed/          ← cleaned, merged, ready for models
├── results/
│   ├── plots/              ← all figures (named descriptively) — committed
│   ├── metrics/            ← tables, evaluation summaries (CSV/JSON) — committed
│   └── scores/             ← raw per-model LLM output — git-ignored (large)
├── docs/
│   └── notes/              ← meeting notes, design decisions, ideas
└── thesis/
    ├── chapters/           ← 01_introduction … 07_conclusion
    ├── figures/            ← final figures, copied from results/plots/
    ├── references/         ← bibliography
    └── appendix/
```

**Rules:**
- `data/raw/` is read-only — never write to it from scripts
- Processed outputs go to `data/processed/` or `results/`
- Figures used in the thesis go to `thesis/figures/` (copy from `results/plots/`)
- Notebooks are for exploration only — production logic belongs in `code/src/`
- Raw per-model LLM output goes to `results/scores/` before any aggregation
- Empty folders are held in git by `.gitkeep`; data and model outputs are not committed

---

## Code Conventions

### Language and environment
- Python 3.11+
- **Always work in the repo's `.venv`.** `make setup` creates it and installs
  the pinned versions. Never install into a base or conda environment — mixed
  pandas/pyarrow versions across the team mean results that do not reproduce.
- Two dependency files, do not conflate them:
  - `requirements.txt` — direct dependencies, loose ranges. Edit this.
  - `requirements.lock.txt` — every version pinned, generated by `make lock`.
    Install from this. Commit both whenever either changes.
- `make test` / `make lint` / `make corpus` call `.venv/bin/` explicitly, so
  they behave the same whether or not the venv is activated. Prefer them over
  bare `pytest` / `python`, which may resolve to the wrong interpreter.
- Lint config lives in `ruff.toml` (`src = ["code"]` so `src.*` resolves as
  first-party) and test config in `pytest.ini` (`pythonpath = code`). Both
  encode the same import root as the `sys.path` insert in `code/scripts/`.

### Style
- PEP 8 — use `black` for formatting, `ruff` for linting
- Type hints on all function signatures in `src/`
- Docstrings on every public function and class
- No hardcoded file paths — use `pathlib.Path` and a central config or env variable

### API keys and secrets
- Never commit API keys, credentials, or `.env` files
- `.gitignore` already covers `.env`; use `python-dotenv` to load at runtime
- LLM API keys: set via environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)

### Reproducibility
- Set and log random seeds where randomness is involved
- Log model name, version, temperature, and full prompt template with every run
- Save raw LLM outputs alongside derived scores — never only keep the aggregated result
- Store ensemble inputs and outputs separately so individual model behavior can be inspected

### Cost awareness
- The corpus spans 25 years and millions of speeches
- Always run on a small sample (100–500 speeches) before any full run
- Log token counts and estimated cost before and after large API calls
- Prefer batch APIs where available (OpenAI Batch API etc.) to reduce cost by ~50%
- Reasoning models cost more per call — use them only for the scoring step, not preprocessing

---

## What the Agent Should and Should Not Do

### DO
- Work within the existing folder structure
- Flag in a comment when a methodological decision is still open
- Log all key parameters and results to `results/metrics/`
- Write modular, reusable code in `src/` that scripts in `scripts/` call
- Mark 2017 discontinuity visibly in all time-series plots
- Always save raw per-model scores before aggregating into an ensemble score
- Validate LLM scores against DW-NOMINATE as part of every scoring experiment
- Prefer clarity over cleverness — this is academic code, not production software

### DO NOT
- Make irreversible changes to `data/raw/`
- Hardcode model names, prompt text, or thresholds in multiple places — centralize in config
- Build RQ5 pipeline without confirming govinfo coverage first
- Output binary classifications as the final scoring result — the goal is continuous scores
- Fragment speeches into short chunks unless a model's context window genuinely requires it
- Generate thesis text — students write first; AI polishes at the end only
- Ignore the 2017 coverage gap — flag it explicitly in outputs and plots
- Run full corpus through an LLM API without a sampled pilot first
- Commit API keys or large raw data files to git

---

## Priorities Until Meeting 2 (Oct 1–15, 2026)

1. **Literature Review** — concise, justifies novelty; Yufei will scrutinize this
2. **Finish Dataset** — merge Stanford + govinfo, document the 2017 gap
3. **Run models and summarize findings** — at least pilot results; per-party position plots
4. **Start written document** — chapter structure, table of contents, introduction draft
5. **Check formal requirements** — ~20 pages per person; Yufei says hitting the page limit is the actual risk, so be concise

---

## Key Contacts

| Person | Role |
|--------|------|
| Yufei Shen | Thesis advisor |
| Justus | Team member, technical lead on pipeline and LLM scoring |

---

*Last updated: September 20, 2026 — corpus build implemented
(`code/scripts/build_corpus.py`, logic in `code/src/corpus.py`, tests in
`code/tests/`); the data decisions it settles are recorded above. Everything
still marked **[assumed]** is a planning estimate awaiting verification —
including the Stanford speaker-match drop count, which this build cannot
recover because those rows were removed upstream. Update this file when major
decisions are made, and drop an [assumed] mark only once the value has been
checked against data.*
