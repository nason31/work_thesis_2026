# Decision log

Every choice that shapes the data, the models or the results, with the reasoning
that produced it. The thesis methodology chapter is written **from this file** —
if a decision is not here, by December nobody will remember why it was made.

**How to use this file**

- One entry per decision, newest section last. Never delete an entry: if a
  decision changes, mark the old one `SUPERSEDED` and add a new one that links
  back. The trail of what we tried and rejected is worth as much as the outcome.
- Every entry records what was **rejected** and why. A decision with no
  alternatives is a decision nobody actually made.
- Figures cited here come from `results/metrics/corpus_build_stats.json`, which
  is tied to a specific raw file by `source_sha256`. If the dataset changes, the
  figures change — re-run and update.
- Open questions live in the last section. Move them up when they are settled.

**Maintained by:** updated at the end of every working session (see the rule in
CLAUDE.md). Entries are written by whoever made the decision, human or agent.

---

## Methodology

### M1 — Continuous scoring, not binary classification
**Date:** 2026-09 (project setup) · **Status:** active

Output is a continuous ideological position per speech, not a polarizing/
not-polarizing label.

**Rejected:** binary classification. It throws away degree — a speech can be
mildly or strongly partisan — and the whole analytical contribution is tracking
*position over time*, which needs a scale. Binary would also have locked us into
fine-tuning and required labeled data.

---

### M2 — Two independent dimensions: ideology and tone
**Date:** 2026-09 (project setup) · **Status:** active

Each speech is scored on ideological position (left–right, the primary measure)
and on tone/hostility, separately.

**Why:** RQ5 asks whether hostility tracks ideology or moves independently. That
question is only answerable if the two are measured separately rather than
collapsed into one "extremeness" score.

---

### M3 — Ensemble of three models from different providers
**Date:** 2026-09-16 · **Status:** SUPERSEDED by [M3a] on the Anthropic member · **Commit:** 112343d

DeepSeek R1 (`deepseek-reasoner`), GPT-4o (`gpt-4o-2024-11-20`), Claude 3.5
Sonnet (`claude-3-5-sonnet-20241022`). Averaged for continuous scores, with all
individual model outputs logged before aggregation.

**Why these three:** one reasoning model plus two instruction-following models;
one open-weight model so the checkpoint is reproducible; three independent
training paradigms (RL-trained reasoning, standard RLHF, Constitutional AI).
Ideological scoring is subjective, so no single model's internal calibration
should be treated as ground truth.

**Rejected:** Gemini, dropped when the ensemble was fixed at three. The
`google-genai` dependency was removed later (E4).

**Consequence:** model IDs live only in `ENSEMBLE_MODELS` in `code/src/config.py`.

---

### M4 — Feed whole speeches, never chunks
**Date:** 2026-09 (project setup) · **Status:** active

**Why:** chunking breaks cross-sentence rhetorical context, which is exactly
what signals ideological framing. Current context windows handle full
congressional speeches comfortably, so there is no forcing constraint.

---

### M5 — Validate against DW-NOMINATE
**Date:** 2026-09 (project setup) · **Status:** active, **blocked**

LLM scores are cross-checked against VoteView DW-NOMINATE. This is the primary
methodological defense of the approach. A weak or inverse correlation is a
finding to investigate and report, not to suppress.

**Blocker:** VoteView keys on ICPSR; the speeches carry Gentzkow's `speakerid`.
The crosswalk does not exist yet. D9 keeps the columns needed to build it.

---

## Data

### D1 — Aggregate by Congress, not calendar year
**Date:** 2026-09-20 · **Status:** active

**Why:** Congresses are clean two-year units aligned with how the data is
organized. Calendar years would split the 114th awkwardly and, before D12 was
discovered, would have mixed three days of Stanford data into a 2017 bucket.

---

### D2 — Independents are assigned to the party they caucus with
**Date:** 2026-09-20 · **Status:** active

Sanders → D, Jeffords → D, King → D, later Lieberman → D (D6).

**Rejected:** dropping independents entirely (loses Sanders, a politically
distinctive and prolific speaker); keeping "I" as a third category (the primary
RQ is a D-vs-R comparison, and a third category would force every downstream
script to decide what to do with it — exactly the divergence we want to avoid).

**Consequence:** applied once, at corpus build time, nowhere else.
`party_original` preserves the pre-assignment label so this is auditable and
reversible.

---

### D3 — Minimum speech length: 50 words, inclusive
**Date:** 2026-09-20 · **Status:** active

**Why 50:** matches the Gentzkow/Shapiro/Taddy convention, and strips procedural
filler ("I yield back the balance of my time") without biasing the corpus toward
members who give long set-piece speeches.

**Rejected:** 100 and 200 words. Both cut substantive short floor remarks and
introduce a selection effect that would need defending. Cost control belongs at
the LLM-sampling step, where the threshold is one line and costs nothing to
change — filtering harder at build time needs a full rebuild to undo.

**Consequence:** drops 390,952 of 823,341 rows (47.5%).

---

### D4 — An independent no rule covers stops the build
**Date:** 2026-09-20 · **Status:** active

**Why:** CLAUDE.md fixes the handling of independents once, at build time. A
silent fallback would make the actual rule invisible and undocumentable. Failing
loudly forces a human decision that gets written down here.

**Rejected:** dropping unknown independents with a log line (shrinks the corpus
in a way only a log reveals); passing them through as "I" (pushes the decision
downstream into every script).

**Validated:** this fired on the first real run and caught four members nobody
had considered — Lieberman, Crenshaw, Sablan and Barkley (D6–D8, D10). The
design paid for itself immediately.

---

### D5 — Text cleaning is whitespace normalization only
**Date:** 2026-09-20 · **Status:** active

Collapse whitespace runs, trim ends. Nothing else.

**Why:** stripping boilerplate ("Mr. Speaker,", procedural preambles) changes
what the model sees, which makes it a methodological choice rather than cleaning.
It stays **open** — if we do it later, it needs its own entry and a robustness
check.

---

### D6 — Lieberman assigned to D
**Date:** 2026-09-21 · **Status:** active

1,255 speeches, 110th–112th Congress. He sat as an "Independent Democrat" and
caucused with Senate Democrats throughout, chairing a committee as part of that
caucus.

**Why:** consistent with D2. His caucus membership is unambiguous across all
three congresses.

**Rejected:** dropping him — more speeches than Jeffords and King combined, from
a politically distinctive figure, and breaking the stated rule for no reason.

---

### D7 — Crenshaw's party label corrected to R
**Date:** 2026-09-21 · **Status:** active

34 rows (31 after the word filter) coded `I` in the 107th Congress. Ander
Crenshaw represented FL-4 as a Republican for his entire career (2001–2017), so
the source label is wrong.

**Rejected:** dropping the rows. Discussed explicitly — dropping is the
lower-risk option, because if the party field is wrong the speaker attribution
might be too, and we cannot tell which field failed. **Chosen anyway** to keep
the speeches, on the judgment that this is a simple party-field glitch.

**Consequence:** this is the only place we override the source on outside
knowledge. `PARTY_CORRECTIONS` should stay as close to empty as possible; each
entry needs a documented reason. Count is reported as
`party_corrections_applied`.

---

### D8 — Barkley dropped
**Date:** 2026-09-21 · **Status:** active

6 speeches. Dean Barkley (MN) was appointed in November 2002 to fill Wellstone's
seat for about two months, from Minnesota's Independence Party, and caucused
with neither party.

**Why:** the caucus rule in D2 simply has no answer for him. With 6 speeches
there is nothing to gain by forcing one, and an arbitrary assignment is harder
to defend than an exclusion stated in one sentence.

---

### D9 — Retain four columns beyond the documented eight
**Date:** 2026-09-20, revised 2026-09-21 · **Status:** active

`last_name`, `state`, `word_count`, `party_original` on top of the CLAUDE.md
schema.

**Why:** the `speakerid` → ICPSR crosswalk that M5 is blocked on needs
name + state + congress + chamber. Without these, building it means re-streaming
the 681 MB raw file. `party_original` makes D2/D6/D7 auditable.

**`first_name` deliberately NOT retained:** it is the literal string `"Unknown"`
on 97.2% of rows, so it carries no information.

---

### D10 — Non-voting delegates dropped corpus-wide
**Date:** 2026-09-21 · **Status:** active

DC, PR, VI, GU, AS, MP — 5,665 rows, 0.69%.

**Why:** they cannot vote on final passage, so DW-NOMINATE does not score them
comparably and their speeches would have no validation benchmark (M5). The
primary RQ concerns D-vs-R positions among voting members.

**Rejected:** keeping them. They do give floor speeches carrying party rhetoric,
but including members the validation cannot cover weakens the central
methodological defense.

**Side effect:** removes the stray `A`/`P` party codes, which belong solely to
Acevedo-Vilá (Resident Commissioner of Puerto Rico), and removes Sablan without
needing a separate rule.

---

### D11 — Identify members by `speakerid`, not by name
**Date:** 2026-09-21 · **Status:** active · supersedes an earlier name-based key

All membership rules key on `speakerid`.

**Why:** the name columns are OCR-damaged. `speaker` carries an honorific plus
noise on 99.7% of rows — the same senator appears as `Mr. JEFFORDS`,
`LR. JEFFORDS`, `Mr. -JEFFORDS`, `Mr.. JEFFORDS`, `Mr. JEFFORD`. The cleaner
`last_name` looked safe (no nulls, no `"Unknown"`) but has spelling variants on
**30.6%** of speakerids (`LIEBERMAN`/`LIEDERMAN`/`LISBERMAN`,
`SANDERS`/`SANDERR`/`SA.NDERS`). `speakerid` is exact, and `state_map` is
consistent within every speakerid — 0 exceptions in 823,341 rows.

**What this cost:** a first attempt keyed on `(last_name, state_map)` failed on
the real data. Two rounds of failure were needed to find this; the lesson is that
"no nulls" is not the same as "clean".

**Consequence:** `speakerid` encodes the congress in its first three digits, so
it is unique per member *per congress*. A member serving several congresses needs
one entry per congress. The 18 independent ids were enumerated from the data.

---

### D12 — Coverage ends 2016-09-09, not 2017-01-03
**Date:** 2026-09-21 · **Status:** active, **and it opens O1**

Measured over all raw rows before any filtering. CLAUDE.md previously stated
January 3, 2017 and called the handover a clean Congress boundary; both were
wrong. The 114th Congress ran to January 2017 but the data stops nearly four
months early, so the 114th is **incomplete**.

**Why it matters:** if govinfo starts at the 115th Congress as planned, there is
a ~4 month hole from 2016-09-10 to 2017-01-02 — covering the run-up to the
November 2016 election, which RQ4 asks about directly. See O1.

---

### D13 — Use the clean column variants
**Date:** 2026-09-21 · **Status:** active

`last_name` over `speaker`, `state_map` over `state`, `chamber_map` over
`chamber`.

**Why:** the raw file carries both a dirty and a clean version of every speaker
field, and CLAUDE.md had documented the dirty ones. `state` is `"Unknown"` on
many rows; `chamber` has 97 nulls; `state_map` and `chamber_map` have none.

---

### D14 — Duplicate `speech_id`: keep the first, count it
**Date:** 2026-09-20 · **Status:** active

**Why:** `speech_id` is documented as unique, so a non-zero count is a signal
worth investigating rather than a routine drop. Reported in the build stats.

**Outcome:** 0 duplicates across 823,341 rows — the property holds.

---

## Engineering

### E1 — Pinned virtualenv and a lockfile
**Date:** 2026-09-20 · **Status:** active

`requirements.txt` holds direct dependencies with loose ranges;
`requirements.lock.txt` pins all 140 packages exactly. `make setup` installs
from the lock.

**Why:** open ranges with no venv gave each team member whatever pip resolved
that day. This bit us concretely: ruff installed into a base conda environment
and resolved to 0.16 while `requirements.txt` permitted 0.5, so two people would
get different lint results on the same code. Lint noise is cosmetic, but a
pandas or pyarrow difference can change *results*.

---

### E2 — pandas 3.0 kept, not pinned back to 2.x
**Date:** 2026-09-20 · **Status:** active

The resolver chose pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1 — nobody picked
these. The concern was that the pinned seaborn 0.13.2 and statsmodels 0.15.0
both predate pandas 3.0 and are exactly what the results chapters depend on.

**Tested before deciding:** a miniature of the real analysis path — `read_parquet`,
the per-party-by-Congress panel, D–R distance, a statsmodels OLS, a seaborn
lineplot with the break marker. All of it worked, no warnings. So pandas 3.0
stays; the reproducibility win came from pinning at all, not from the version.

**Two gotchas it exposed:** `date` reads back as `object` (Python `datetime.date`),
not `datetime64`, so `.dt`/`resample`/dated axes need `pd.to_datetime` first; and
`text` has pandas 3's `str` dtype, so `select_dtypes(include="object")` finds no
text columns.

---

### E3 — Corpus build streams with pyarrow, in two passes
**Date:** 2026-09-20 · **Status:** active

Pass 1 reads metadata only and fails fast; pass 2 streams text in 50k-row
batches through one writer.

**Why:** the raw file is 681 MB compressed and decompresses to several GB, almost
all text, against 16 GB of RAM. `pd.read_parquet()` on the whole thing is not
viable. Splitting the passes means validation failures (D4) cost seconds and
happen before any output exists.

**Outcome:** full build 41 s, peak RSS 1.66 GB.

---

### E4 — `google-genai` removed
**Date:** 2026-09-20 · **Status:** active

The ensemble was fixed to three models in M3, so Gemini is not in the design.
`openai` covers GPT-4o and DeepSeek (OpenAI-compatible API); `anthropic` covers
Claude.

---

### E5 — A `--limit` run never writes to the corpus path
**Date:** 2026-09-21 · **Status:** active

**Why:** a smoke run left a partial 24,570-row file at `CORPUS_PATH` that nothing
downstream could distinguish from a full build. Limited runs write
`corpus.smoke.parquet` and do not write the stats JSON, since partial counts in a
committed file would mislead.

---

### E6 — Every build records the raw file's SHA-256
**Date:** 2026-09-21 · **Status:** active

**Why:** the current Stanford parquet is provisional — a starting point for
building the pipeline, not the final corpus. `source_sha256` ties every count in
the stats file to the exact raw file it came from, so a swapped dataset is
visible rather than silent.

**When the dataset is replaced:** re-run `make smoke`, then `make corpus`, and
re-check D3, D10, D12 and the figures in
`docs/notes/2026-09-21_stanford_data_reality.md`. The build fails loudly on any
independent, party code or column it does not recognize, so a swap cannot drift
through unnoticed.

---

## Scoring

### S1 — Pilot sample: 200 speeches, stratified party × Congress, seed 42
**Date:** 2026-09-21 · **Status:** active

Equal allocation over 2 parties × 8 Congresses = 16 strata, 12–13 each, exactly
100 D and 100 R.

**Why stratified rather than random:** a simple random draw would over-represent
the 110th (66,727 rows) against the 114th (36,661) and Democrats against
Republicans, so a per-Congress or per-party comparison on the pilot would be
measuring sample composition rather than rhetoric. Every stratum holds ≥17,354
rows, so equal allocation costs nothing.

**Why 200:** CLAUDE.md requires a 100–500 speech pilot before any full run.

**Consequence:** seed 42 is logged in the run manifest, and the draw is
reproducible — verified identical across runs and different under another seed.

---

### S2 — `deepseek-reasoner` runs without a temperature setting
**Date:** 2026-09-21 · **Status:** SUPERSEDED by [S2a] — no model gets a temperature

GPT-4o and Claude get `temperature=0.1`. The parameter is **omitted** for
`deepseek-reasoner`.

**Why:** DeepSeek documents that the reasoner ignores `temperature`, and some API
versions reject it outright. A 400 there would have lost all 200 calls for that
model.

**Consequence — this one matters for the write-up.** The three models are not
identically configured, so the methodology chapter must not claim they are.
`ModelSpec.effective_temperature` reports `0.1` for two models and
`"provider default"` for the reasoner, and the run manifest records it per model.

---

### S3 — Out-of-range scores are failures, not clipped
**Date:** 2026-09-21 · **Status:** active

An `ideology_score` outside [−1, 1] or a `tone_score` outside [0, 1] is recorded
as null with the error, not clamped to the boundary.

**Why:** a model returning 1.5 has ignored the scale the prompt defined. Clipping
would turn that into a plausible-looking maximum score and hide a real problem
with the instrument. The pilot exists to surface exactly this.

---

### S4 — A failed model averages over the survivors, with `n_models` recorded
**Date:** 2026-09-21 · **Status:** active

If one of the three models fails on a speech, the ensemble row is built from the
two that succeeded, and every row carries `n_models`.

**Rejected:** nulling the whole row (throws away two valid scores over one
failure, and a flaky provider silently shrinks the pilot); averaging with no
marker (a 2-model average becomes indistinguishable from a 3-model one, which
would quietly bias the cross-model disagreement statistic).

**Related:** standard deviation is `null` below two models rather than 0.0 — one
model agreeing with itself is not agreement.

---

### S5 — Spending requires confirmation
**Date:** 2026-09-21 · **Status:** active

The pilot estimates cost with `tiktoken`, prints it, and waits for `y` before
calling anything. `--dry-run` stops after the estimate; `--yes` skips the prompt
for unattended runs.

**Why:** 600 calls including a reasoning model is real money, and CLAUDE.md
requires logging cost before and after large API calls. Estimated at **~$2.45**
for the 200-speech pilot.

**Caveat recorded in the output:** the estimate is a floor. The tokenizer is
OpenAI's and only approximates the other two providers, and `deepseek-reasoner`
bills hidden reasoning tokens as output that the estimate cannot see. Extrapolate
the full run from the *billed* usage in the closing summary, not from this.

---

### S6 — Scoring logic lives in `src/`, not in the script
**Date:** 2026-09-21 · **Status:** active

`code/src/scoring.py` holds the provider clients, prompt rendering, JSON
extraction, validation and retry; `code/scripts/pilot_run.py` holds sampling,
orchestration and reporting.

**Why:** the full run needs the same clients. CLAUDE.md puts production logic in
`src/` with scripts calling it, and `corpus.py`/`build_corpus.py` already follow
that split.

---

### M3a — Claude Sonnet 4.6 replaces the retired Claude 3.5 Sonnet
**Date:** 2026-09-23 · **Status:** active · supersedes the Anthropic half of [M3]

`claude-3-5-sonnet-20241022` returns **404 — the model has been retired**. This
was found on the very first API call, not by reading a deprecation notice.

**Chosen:** `claude-sonnet-4-6`. Same Sonnet tier and the same $3/$15 per 1M as
the retired model, so neither the budget nor M3's "architectural diversity"
rationale changes. It is also the newest model that still accepts a temperature
at the API level — though see [S2a], where that turned out not to matter.

**Rejected:** `claude-sonnet-5` (cheaper per token at $2/$10, but adaptive
thinking is on by default and used roughly twice the output tokens in a probe,
so the saving largely washes out, and it rejects temperature outright);
`claude-opus-5` (most capable but $5/$25, ~1.7× the old Anthropic cost, and no
temperature either).

**Lesson worth keeping:** a pinned model ID is not a guarantee of availability.
`make smoke` verifies every model answers, for free, before a run that spends.

---

### S2a — No model gets a temperature; all three run at provider default
**Date:** 2026-09-23 · **Status:** active · supersedes [S2]

Temperature is not set anywhere. **This is not a methodological preference — the
providers removed the control.**

- `deepseek-reasoner` ignores it (the original [S2] finding).
- The anthropic SDK 1.7.0 has **no `temperature` parameter at all**; passing it
  is a `TypeError` before any request leaves the machine. Forced through
  `extra_body`, Sonnet 4.6 accepts it but Sonnet 5 answers
  `400 — "temperature is deprecated for this model"`.

So the only model that could still take 0.1 was GPT-4o. Setting it on one model
of three would imply an ensemble tuned alike, which would be false.

**Rejected:** keeping 0.1 on GPT-4o and Sonnet 4.6 via an `extra_body`
passthrough. It works today, but it forces a parameter the SDK deliberately
removed and that the API already calls deprecated — it would likely break during
the thesis, and it buys consistency on two models out of three.

**Consequence for the write-up:** do not claim the ensemble was run at a
controlled temperature. Every run manifest records `"provider default"` for all
three, and that is what the methodology chapter should state. Repeat runs will
be noisier than a temperature-0.1 design would have been; if that matters for a
robustness claim, it has to be measured, not assumed.

---

### S7 — Anthropic returns JSON via structured outputs, not assistant prefill
**Date:** 2026-09-23 · **Status:** active

`output_config.format` with a server-enforced JSON schema.

**Why not prefill:** seeding the assistant turn with `{` was the standard way to
force JSON before structured outputs existed. It returns a **400 on all current
Claude models**. The bug was invisible until the retired-model 404 was fixed,
because the 404 came first.

**Consequence:** Claude's replies are schema-valid by construction, so the
brace-depth parser is a safety net for that model rather than the mechanism.
DeepSeek still needs it — the reasoner emits reasoning before its answer.

---

## Open questions

Move these up into a numbered entry once decided.

### O1 — Where does the govinfo pipeline start?
**Raised:** 2026-09-21 (from D12) · **Blocks:** the merged corpus, RQ4

Stanford data ends 2016-09-09. Two options, both defensible:

- **Start at 2016-09-10** — closes the gap, but the handover no longer sits on a
  Congress boundary, which complicates D1's per-Congress aggregation at the seam.
- **Start at the 115th Congress (2017-01-03)** — keeps the clean boundary, but
  leaves a ~4 month hole over the 2016 election run-up that must be declared as a
  limitation.

Decide **before** building the govinfo pipeline, not after.

### O2 — Prompting or fine-tuning?
**Raised:** project setup

Continuous scoring (M1) strongly favors prompting, zero- or few-shot.
Fine-tuning needs labeled data and would push toward a binary setup. Reinforcement
fine-tuning is a middle path worth evaluating if a supervised component turns out
to be needed. Do not build infrastructure that assumes one without flagging it.

### O3 — Does boilerplate get stripped from speech text?
**Raised:** 2026-09-20 (from D5)

Currently no. If it changes, it needs its own entry and a robustness check
showing scores do not move much.

### O4 — How many speeches did the upstream build drop?
**Raised:** 2026-09-15 · **Status:** unanswerable here

Speeches with no matched speaker were dropped when Konsti built the parquet.
Those rows are not in the file, so no analysis of it recovers the count. Only the
missing `code/scripts/build_stanford_dataset.py` can close this, and the
methodology chapter needs the figure.
