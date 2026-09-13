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

### 2001–2016: Stanford Dataset (Gentzkow, Shapiro & Taddy)
- Pre-structured; speeches linked to member / party / chamber / date
- Path: `data/raw/stanford/`
- Note: Yufei said we **can drop this** if our govinfo pipeline produces sufficient coverage

### 2017–2025: Custom govinfo.gov Pipeline
- Path: `data/raw/govinfo/`
- **Known issue:** Fewer speeches per year than the Stanford dataset — this creates a potential discontinuity around 2017. Address this in the methodology section; do NOT silently ignore it.

### Processed / Combined
- Path: `data/processed/`
- Final merged corpus: one row per speech, with columns: `speech_id`, `date`, `member_id`, `party`, `chamber`, `congress_number`, `text`, `source`

### External Validation
- **DW-NOMINATE scores** — voting-based ideological measure from VoteView.com
- Used to validate LLM-derived ideological scores against a non-ML, voting-based benchmark
- Path: `data/raw/dw_nominate/`

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

**3. Ensemble across multiple models**
Run at least two or three LLMs and combine their outputs (e.g. average for continuous scores, or majority vote for any categorical decisions). This reduces single-model bias on a subjective task like ideological scoring, where no single model's internal calibration should be treated as ground truth. Log all individual model scores before aggregation — never discard the raw outputs.

**4. Feed full speeches, not fragments**
Do not split speeches into short chunks unless a model's context window absolutely requires it. Chunking breaks cross-sentence rhetorical context, which matters for detecting ideological framing. Current frontier models (GPT-4o, Llama 4, Gemini 2.5, DeepSeek R1) all handle full congressional speeches comfortably. Verify context length per model before deciding.

**5. Prefer reasoning models for ideological scoring**
Scoring ideological position requires chaining inferences: "what policy positions does this language imply, and where do those sit on the political spectrum?" Reasoning models (o3, DeepSeek R1) are specifically better at this kind of multi-step judgment than standard instruction-following models. Prioritize them for the scoring step; cheaper models are fine for preprocessing and filtering.

**6. Structural validation via DW-NOMINATE**
LLM-derived scores must be cross-checked against DW-NOMINATE (voting-based ideological scores from VoteView.com). This is not optional — it is the primary methodological defense of our approach. Expect the correlation to be imperfect (floor speech rhetoric vs. actual votes are different signals), but it should be positive and meaningful. A weak or inverse correlation would be a red flag requiring investigation, not suppression.

### What is NOT yet decided — do not hardcode

- **Prompting vs. fine-tuning:** Still open. Continuous scoring strongly favors prompting (zero-shot or few-shot); fine-tuning would lock us into a binary setup and requires labeled data. Do not build infrastructure that assumes one approach without flagging the trade-off in a comment.
- **Which models to include in the ensemble:** Candidates are GPT-4o, GPT-4o-mini, Llama 3.3 / Llama 4, DeepSeek R1, Gemini 2.5 Flash. The exact ensemble composition is to be decided after a pilot run comparing model outputs on the same sample.
- **Reinforcement fine-tuning:** Now more mature than a year ago and requires less labeled data than traditional fine-tuning. Worth evaluating if a supervised component turns out to be needed, but not the default path.

### If manual labeling is done
- Define a clear annotation rubric before labeling anything
- Use at least two independent annotators
- Report inter-annotator agreement using Cohen's kappa — this makes the annotation academically defensible
- Do not rely solely on "own judgment with peer review"

### Key boundaries
- The 2017 pipeline discontinuity must be addressed in the methodology section — flag it in code comments and make it visible in every time-series plot (e.g. vertical dashed line at 2017)
- Do not fabricate or impute data to paper over coverage gaps

---

## Repository Structure

```
work_thesis_2026/
├── CLAUDE.md               ← this file
├── THESIS_STRUCTURE.md     ← folder conventions
├── code/
│   ├── src/                ← production modules (importable, tested)
│   ├── scripts/            ← one-off analysis and pipeline scripts
│   └── notebooks/          ← exploratory work; not used in production
├── data/
│   ├── raw/                ← original, never modified
│   │   ├── stanford/
│   │   ├── govinfo/
│   │   └── dw_nominate/
│   └── processed/          ← cleaned, merged, ready for models
├── results/
│   ├── plots/              ← all figures (named descriptively)
│   └── metrics/            ← tables, evaluation summaries (CSV or JSON)
├── docs/
│   └── notes/              ← meeting notes, design decisions, ideas
└── thesis/
    └── chapters/           ← written thesis content (Markdown or LaTeX)
```

**Rules:**
- `data/raw/` is read-only — never write to it from scripts
- Processed outputs go to `data/processed/` or `results/`
- Figures used in the thesis go to `thesis/figures/` (copy from `results/plots/`)
- Notebooks are for exploration only — production logic belongs in `code/src/`

---

## Code Conventions

### Language and environment
- Python 3.11+
- Dependencies in `requirements.txt` (or `pyproject.toml` if project grows)
- Use virtual environments; do not assume global installs

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

*Last updated: September 2026. Update this file when major decisions are made.*
