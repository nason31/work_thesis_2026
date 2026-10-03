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
- Figures cited here come from the build stats in `results/metrics/`
  (`stanford_build_stats.json`, `govinfo_build_stats.json`,
  `merged_build_stats.json`), each tied to the exact file it describes by
  SHA-256. If the dataset changes, the
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
**Date:** 2026-09 (project setup) · **Status:** active — **unblocked 2026-09-29** by the crosswalk (D21–D24)

LLM scores are cross-checked against VoteView DW-NOMINATE. This is the primary
methodological defense of the approach. A weak or inverse correlation is a
finding to investigate and report, not to suppress.

**Blocker:** VoteView keys on ICPSR; the speeches carry Gentzkow's `speakerid`.
The crosswalk does not exist yet. D9 keeps the columns needed to build it.

**Update 2026-09-29 — the crosswalk looks feasible.** A rough probe matched each
Stanford `speakerid` to congress-legislators (already in the repo for D16) on
state + chamber + date + most common `last_name`: 4,191 speakerids matched
exactly and 31 loosely (compound surnames), covering **415,792 of 426,718
speeches (97.4%)** with an ICPSR. 94 speakerids (6,923 speeches) were ambiguous
and 18 (3,645) unmatched. Not validated — no miss was inspected. It is the next
piece of work after the merge (D19), and would also let `member_id` become the
bioguide id on both sides of the break.

**Update 2026-09-29 — built.** `data/processed/member_crosswalk.parquet`
(`make crosswalk`, D21) matches **99.82% of Stanford speeches and 100% of
govinfo speeches** to a Voteview ICPSR; 538,039 of 538,804 speeches (99.86%)
can enter the validation. The probe above matched against congress-legislators
on the most common spelling only; the build matches against Voteview directly
and uses every spelling, which is why it does better. *(Rebuilt 2026-10-03 on
the re-fetched govinfo file, D25: 579,242 of 580,007, 99.87%.)*

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
**Date:** 2026-09-21 · **Status:** active; the O1 it opened is settled by D15

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

### D15 — govinfo covers 2016-09-10 to 2025-12-31 (settles O1)
**Date:** 2026-09-27 · **Status:** active

`GOVINFO_START_DATE` / `GOVINFO_END_DATE` in `config.py`. Konsti's file already
starts at 2016-09-12 (the first sitting day after Stanford's last, 2016-09-09),
so the seam has **no gap and no overlap**. The build still enforces the window
and counts anything outside it (0 before, 15,283 after).

**Rejected:** starting at the 115th Congress — it would throw away the 2016
election run-up that RQ4 asks about, in order to keep a boundary D1 does not
actually need (the 114th is simply built from two sources). Keeping 2026 —
outside the thesis scope (`YEAR_RANGE`), and a partial year that would read as a
trend in any per-year plot.

**Consequence:** the 114th Congress is mixed-source (Stanford to 2016-09-09,
govinfo after). The `source` column keeps that visible. Note October 2016 has
zero speeches — a normal pre-election recess (Stanford has zero for October
2006, 2010 and 2014 too), not a data gap.

---

### D16 — govinfo speakers are re-resolved against congress-legislators
**Date:** 2026-09-27 · **Status:** active

The file's `party` and `icpsr` columns are ignored. Every speaker is resolved
again against unitedstates/congress-legislators, filtered by **chamber and
date**, and only a unique match is accepted.

**Why:** the upstream lookup ignored chamber, so a surname counted as ambiguous
whenever *anyone* in Congress shared it — every Senate speech by Mike Lee (518),
Sherrod Brown (401), Chris Murphy, Gary Peters, Ron Johnson and Tina Smith came
out with no party. It also used each term's end-of-term party, so mid-term
switchers were mislabelled: Van Drew's 2019 speeches as R (he switched
2019-12-19), Mitchell's and Amash's Republican-era speeches as Independent.
congress-legislators records day-level `party_affiliations`, which fixes both.

**Result:** unresolved rows in the window fall from 29,013 (13.8%, file) to
24,762 (11.8%); Senate unresolved from ~4,100 to 1,196, which are mostly the two
Senators Scott (genuinely ambiguous) and non-members (trial counsel, video
clips). ICPSR agrees with the file on every row where both have one.

**Rejected:** trusting the file's labels (wrong in the ways above); guessing
among same-name candidates (would put speeches in the wrong party).

**Remaining limitation (not fixable from this file):** the Record header
`Mr. SMITH of Texas.` names the state, but the upstream parser kept only the
surname, so 23,075 House rows stay ambiguous. They lean **Republican** — about
13,900 R vs 9,100 D by candidate weight — so their loss under-represents House
Republicans. Fixing it needs the parser to keep the state and a re-fetch.
7,029 of them have candidates who all share one party. See O7. *(Fixed by D25,
2026-10-03: the re-fetched file keeps the state; 47 rows remain ambiguous.)*

**ICPSR gap:** congress-legislators has no ICPSR for many members first elected
2021 or later (missing on 12% of 117th, 29% of 118th, 34% of 119th rows). The
DW-NOMINATE join for those needs VoteView's member file (name/state/congress).

---

### D17 — govinfo text is cut to the member's own words
**Date:** 2026-09-27 · **Status:** active

The upstream parser split speeches only when a *member* began speaking, so
presiding-officer turns, Record narration and whole bill texts (up to 29k
words) were glued onto the previous speech. `clean_speech` cuts each speech at
the first of: an officer turn (`The PRESIDING OFFICER.`, `The SPEAKER pro
tempore.` …), a fixed narration phrase opening a paragraph (`The Clerk read the
title of the bill.`), or a long `____________________` rule between Record
items. It then strips page markers, HTML residue, `{time}` stamps and centered
heading lines. The 50-word filter is applied **after** cleaning.

**Result:** 19.5% of words removed; 92k of 210k in-window rows cut (56k at an
officer turn, 27k at a separator, 8.5k at narration).

**Checked, not assumed:** a first version also cut at the short `____` rule and
at rules framing `{time}` stamps; both occur *inside* a member's turn, and the
check below caught it. After the fix, the removed text contains a paragraph
addressed to the chair ("Mr. Speaker, I …") in 31 of ~92k cut rows (0.03%) —
mostly amendment text, or a member resuming after the Record moved their
remarks. Accepted.

**Rejected:** leaving the text as is (other speakers' words and bill text in
front of the model, and inflated word counts letting procedural exchanges pass
the 50-word filter); a word cap like upstream's 30,000 (it cannot catch
officer text, and drops the member's real words along with the bill).

**Not O3:** this is separating speakers and removing typesetting, which the
Stanford source already does. Boilerplate inside a member's own words ("Mr.
Speaker, I yield back") is untouched; O3 stays open.

---

### D18 — govinfo independents
**Date:** 2026-09-27 · **Status:** active

Keyed on bioguide id (`GOVINFO_CAUCUS_PARTY`, `GOVINFO_EXCLUDED_MEMBERS`) and
applied only on dates when the member's party is neither D nor R. Same rules as
D2/D8, and the build raises on anyone unlisted (D4). Each caucus entry is
checked against congress-legislators' own `caucus` field.

- Sanders, King → D (as D2)
- Manchin → D from 2024-05-31; Sinema → D from 2022-12-09 (both caucused D)
- Mitchell (from 2020-12-14) and Amash (from 2019-07-04) → **excluded** for
  those dates only: neither caucused with a party. 9 rows. Their earlier
  Republican speeches stay in as R.

Kiley (Independent caucusing R from 2026-03-09) falls outside D15's window; if
the window is extended the build will stop and ask.

---

### D19 — Stanford and govinfo merged into one corpus file
**Date:** 2026-09-29 · **Status:** active ·
**Design:** `docs/notes/2026-09-29_corpus_merge_design.md`

`data/processed/corpus.parquet` now holds both sources: **538,804 rows**
(426,718 Stanford + 112,086 govinfo), 2001-01-03 → 2025-12-19, Congresses
107–119. *(Since 2026-10-03, D25: 580,007 rows, 153,289 of them govinfo.)* `make corpus` builds it from `corpus_stanford.parquet` (the Stanford
build, which used to write `corpus.parquet`; `make stanford`) and
`corpus_govinfo.parquet` (`make govinfo`). Stats:
`results/metrics/merged_build_stats.json`; the Stanford stats file is renamed
`stanford_build_stats.json`.

- **The merge drops nothing.** Every input row is written or the build fails.
  It refuses inputs that cross the seam, repeat a `speech_id`, carry a null or
  an unexpected party/chamber/source, leave a Congress with no rows, or do not
  match their committed build stats (a smoke file or a stale file).
- **`member_id` stays source-native** — Stanford `speakerid`, govinfo bioguide
  id. The formats cannot collide. `icpsr` is null on Stanford rows until the
  crosswalk (M5) fills it. *(Revised by D21: the crosswalk does not fill it —
  it is a separate table, and the govinfo `icpsr` is not the join key either.)*
- **Row order:** Stanford (date-sorted), then govinfo (its file order, not
  date-sorted). Nothing downstream depends on it.

**Verified on the real data:** the rebuilt `corpus_stanford.parquet` is
byte-identical to the old `corpus.parquet`; every column of the merged file is
identical to its inputs; all 200 speeches of the 2026-09-23 pilot resolve in it.

**Rejected:** a merge done at read time, with no file (every consumer would
need a helper, and there is no single file to fingerprint); one build that
re-runs both sources (couples them); naming the output `corpus_merged.parquet`
(`config.py` already documented `corpus.parquet` as the merged corpus);
matching Stanford members to legislators in the same step (delays the sampled
run, which needs only party).

**Consequence:** `corpus.parquet` changed meaning on 2026-09-29. The
2026-09-23 pilot manifest's `corpus_path` refers to the Stanford-only file of
that date. Run manifests now record `corpus_sha256`, so the file a run used is
identifiable regardless of its name.

---

### D20 — govinfo `chamber` is the member's chamber, not the Record section's
**Date:** 2026-09-29 · **Status:** active

House impeachment managers speaking at the Senate trials of January 2020 and
February 2021 (Schiff, Jeffries, Crow, Demings, Lofgren, Nadler, Raskin and 6
others) were written with `chamber = S`, because the build copied the Record
section. The build already resolved them correctly — the Record heads their
turns `Manager SCHIFF`, and D16 looks those up among House terms in
congress-legislators — it just wrote the wrong column. `chamber` is now the
chamber of the resolved member's term on that date, as on the Stanford side
(`chamber_map`). Counted as `rows_chamber_reassigned` in
`govinfo_build_stats.json`.

**Effect, verified by rebuilding:** exactly **425 rows** *(478 on the
2026-10-02 file, D25)* move S → H (373 in the
116th, 52 in the 117th, all Democrats); nothing else in the govinfo or merged
stats changes. The smallest S8 sampling cell this touches, 116th/S/D, falls from
2,353 to 1,980 speeches — still far above 100.

**Why:** a House member at a trial is still a House member. With `S`, these 13
members had no Voteview record to match (Voteview files them under the House),
and they inflated the Senate cells of any chamber-stratified sample.

**Rejected:** leaving the build alone and letting the crosswalk fall back to
the other chamber — the corpus would keep a wrong label that every
chamber-level plot and the S8 sampler read.

---

### D21 — The DW-NOMINATE crosswalk: a separate member-level table
**Date:** 2026-09-29 · **Status:** active · unblocks M5

`data/processed/member_crosswalk.parquet`, built by `make crosswalk`
(`code/src/crosswalk.py`) from the merged corpus and Voteview's raw
`HSall_members.csv`. One row per `source` × `member_id` × `congress_number` ×
`chamber` × `party_original` — the key it joins back onto the corpus with — so
7,176 rows, not 538,804 (7,530 since D25). Match rates:
`results/metrics/crosswalk_build_stats.json`; unmatched members with speech
counts: `results/metrics/crosswalk_unmatched.csv`.

| Source | Units | Matched | Speeches | Matched | In validation |
|---|---|---|---|---|---|
| Stanford | 4,334 | 4,322 | 426,718 | 425,971 (99.82%) | 425,971 |
| govinfo | 2,842 | 2,842 | 112,086 | 112,086 (100%) | 112,068 |
| govinfo, 2026-10-02 file (D25) | 3,196 | 3,196 | 153,289 | 153,289 (100%) | 153,271 |

- **Stanford: surname + state + Congress + chamber.** Matched against the
  Voteview surname (the part of `bioname` before the comma), normalized:
  uppercase, accents stripped, hyphens split, non-letters dropped. Every
  spelling a speakerid carries is tried, since OCR variants differ within one
  speakerid (HODES/RHODES). If no surname equals, a shared word is accepted:
  Stanford keeps only the last word of compound surnames (JACKSON LEE → LEE,
  WASSERMAN SCHULTZ → SCHULTZ) — 28 units, 3,196 speeches, all checked by eye.
  No fuzzy matching: after this, 4 units (5 speeches) are left, all OCR
  garbage (`WIU`/`WVU` for Wu).
- **govinfo: bioguide id + Congress + chamber**, not the corpus `icpsr`. That
  column is congress-legislators' ICPSR, the member's *original* number, while
  Voteview gives a party switcher a new one. Joined directly, Van Drew's
  Republican speeches in the 116th would land on his Democratic record, and his
  117th–119th speeches, Sinema's as an independent and Dold's after his return
  would not land at all. The corpus `icpsr` disagrees with the match on 7 units, every one
  explained by Voteview's renumbering; they are listed in the stats. `member_id`
  is the bioguide id on every govinfo row, so the name-based fallback is never
  needed there.
- **Voteview is read from the raw file**, filtered in code to Congresses
  107–119, House and Senate. This reproduces the hand-made
  `data/processed/dw_nominate_107_119.csv` exactly (all values identical, row
  order aside), so that file has a recipe now; the build records the raw file's
  SHA-256.
- **Fails loudly** on two members matched to one ICPSR in one Congress and
  chamber, a speakerid person part pointing to two people (D22), an unknown
  `source`, or a repeated Voteview key.

**Rejected:** writing `icpsr` into the corpus (the merge design's plan, D19) —
it would rewrite an 820 MB file for a 7k-row fact, and govinfo's existing
`icpsr` is the wrong key anyway (above); matching the Stanford side against
congress-legislators, as the M5 probe did — it adds a hop, since Voteview
carries names and states itself; fuzzy matching — nothing left for it to do.

---

### D22 — Same-surname, same-party namesakes: speakerid person part, then elimination
**Date:** 2026-09-29 · **Status:** active

Surname, state and party cannot separate Loretta and Linda Sánchez (CA), Gene
and Al Green (TX), Carolyn and Sean Patrick Maloney (NY), Lincoln and Mario
Diaz-Balart (FL), Dan and Jeff Miller (FL, 107th) or Julia and André Carson
(IN). Two steps resolve them, repeated until nothing changes:

1. **speakerid person part.** A speakerid is the Congress (3 digits) plus a
   person part that stays the same across Congresses: over the 1,100 persons
   matched uniquely somewhere, each person part maps to exactly one bioguide id
   (0 exceptions). So a member matched uniquely in one Congress (Loretta
   Sanchez alone in the 107th) is recognized in the Congresses where a namesake
   joins. 26 units, 2,163 speeches.
2. **Elimination.** If every candidate but one is already taken by another
   member of that Congress and chamber, the last one is assigned. 16 units,
   703 speeches.

**Verified independently:** for all 42 units, the raw Stanford `speaker`
headers name the assigned person — `Ms. LINDA T. SANCHEZ of California` on the
unit assigned to Linda, `Mr. GENE GREEN of Texas` on Gene's, and so on. 42 of
42 agree. The headers were not used by the matching.

**Guarded:** the build stops if a person part points to two bioguide ids, or if
a speakerid does not start with its Congress — a replacement dataset that
breaks the layout cannot slip through.

**Rejected:** leaving all namesakes unmatched (2,866 speeches, all from 12
House members, 8 of them Democrats — a systematic gap, not noise).

---

### D23 — Party breaks ties only via `party_original`; switchers are flagged
**Date:** 2026-09-29 · **Status:** active

When several candidates remain, the one whose Voteview `party_code` matches
`party_original` (D 100, R 200, I 328; `VOTEVIEW_PARTY_CODES`) wins. Never
`party`: the caucus rule (D2) turns Jeffords' `I` into `D`, while Voteview codes
him 328. 51 Stanford units (3,876 speeches) are settled this way, mostly
different-party namesakes (George and Gary Miller, CA), and 7 govinfo units.

**Switchers within a Congress.** Voteview gives a member who switches party a
new ICPSR, so they have two records in that Congress. On the govinfo side
`party_original` is recorded by day, so each speech gets the right record (Van
Drew's 116th splits 82 D / 44 R). On the Stanford side a speakerid carries one
party for the whole Congress, so every speech of Jeffords (107th, 260), Goode
(107th, 15), Hall (108th, 24), Specter (111th, 222) and Griffith (111th, 17)
gets the record of the party Stanford lists — Specter's speeches from before
April 2009 get his Democratic score. Accepted, and every such unit carries
`party_switch = True` (12 in all, both sides), so a robustness check can drop
them.

**Rejected:** splitting Stanford switchers by speech date (needs switch dates
from outside the data, for 538 speeches); leaving them unmatched (drops
Specter and Jeffords, two of the most-cited cases of ideological movement).

---

### D24 — Who enters the validation
**Date:** 2026-09-29 · **Status:** active

`in_validation = True` only for a matched member with a DW-NOMINATE score.
Everyone else is excluded **from the validation only** — every speech stays in
the trend analysis.

- **Tom and Jo Ann Davis (VA)** stay unmatched: both Republicans, both in the
  House 107th–110th, same surname and state. 8 units, 742 speeches (0.17%).
  Nothing the build uses separates them. *(Noted for later: the raw file does —
  in every Congress one speakerid is headed `Mrs. JO ANN DAVIS of Virginia` and
  the other `Mr. TOM DAVIS of Virginia`, and the raw `gender` column agrees —
  if these 742 are ever worth a dependency on the raw file.)*
- **4 OCR-garbage units** (5 speeches) stay unmatched.
- **Matched but unscored:** Kwanza Hall (GA, 116th, 18 speeches) served a month
  and cast too few votes for a score. If a member cannot be scored on votes,
  there is nothing to validate against.
- **Delegates** never reach the crosswalk: D10 removes them on both sides.

**Rejected:** guessing the Davis pair; imputing a score for Hall.

---

### D25 — govinfo: the Record's state must agree with the speaker (settles O7)
**Date:** 2026-10-03 · **Status:** active · **Data:** raw file SHA-256
`58bcc59b…`, `results/metrics/govinfo_build_stats.json`

Konsti's re-fetch of 2026-10-02 (276,513 rows, 2016-09-12 → 2026-09-30) follows
the API's pagination, which fixes O8's cause, and keeps the state the Record
writes beside a speaker (`Mr. SMITH of Texas.`) as a new `state` column. It is
present on 40,642 rows (39,838 House, 804 Senate) as a full name, mapped to the
congress-legislators code by `STATE_CODES` in `config.py`.

**Rule:** when the state is recognized, the resolved member must be from it.
Among same-surname candidates it picks the one; if no candidate is from that
state, the row is unresolved (`state_mismatch`) and dropped. An unrecognized
string counts as no state, so the row falls back to D16's surname-only match.
`state` is a required key: a file without it predates the pagination fix and
the build refuses it.

**Result, measured on the full build:**

| | Before (old file) | After |
|---|---|---|
| ambiguous House + Senate rows dropped | 23,075 | **47** |
| settled by the state | — | **23,606** (22,903 House, 703 Senate) |
| dropped because the state contradicts | — | 5 |
| govinfo speeches written | 112,086 | **153,289** |
| Republican share, House | 48.8% | 50.9% |

(The before/after compares two raw files, so the gain mixes this rule with the
re-fetch. The 23,606 is this rule's own contribution; it is counted in the
build as `rows_resolved_by_state`.) The House Republican under-representation
that O7 predicted (~60/40 R among the ambiguous) is gone. 47 rows stay
ambiguous: 42 with no usable state, 5 where namesakes share a state (Carolyn and
Sean Maloney, NY).

**Why strict, not a tie-breaker only.** The first design used the state only
when the name was ambiguous. The data said otherwise: of the 13,629 rows where
the name already matched exactly one member *and* a state was given, 4
disagree, and none of them is that member speaking. "Mr. BROWN of Maryland" in
the Senate section (2020-05-04) is a House message naming Anthony Brown, not
Sherrod Brown; "Ms. LEE of California" (2021-02-12, twice) is the January 6
joint-session transcript played at the impeachment trial, not Mike Lee; "Mr.
SMITH of New Jersey" (2021-02-22) is an appointments notice, not Tina Smith. A
tie-breaker would have kept all four in the wrong senator's name.

**Unrecognized states (9 rows), treated as no state:** `Virgina` (5),
`Tennesse`, `Massachusetts Mr` (the notebook's regex overran), `Dr`, `Japan`.
All are non-members or procedural lines under 50 words, so a typo table would
change nothing. They are listed in `state_unrecognized` in the stats; if a
future file has more, add the spellings to `STATE_CODES` rather than loosening
the rule.

**Rejected:** trusting the file's own `party` — the notebook's lookup now uses
the state but still ignores chamber, the bug D16 fixed; state as a tie-breaker
only (keeps the 4 misattributions above); typo correction (nothing to gain on
this file); keeping the old file as an input (it carries O8's fetch cap).

**Consequence:** every downstream figure changed. The merged corpus is
**580,007** rows (426,718 + 153,289); the crosswalk matches 100% of govinfo
speeches (3,196 units) and 579,242 of 580,007 speeches (99.87%) enter the
validation. D16's limitation paragraph, D19's row count and D21's table
describe the old file; the figures here supersede them. D20's reassigned rows
are now 478.

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

**When the dataset is replaced:** re-run `make smoke`, then `make stanford`,
then `make corpus` (the merge, D19), and
re-check D3, D10, D12 and the figures in
`docs/notes/2026-09-21_stanford_data_reality.md`. The build fails loudly on any
independent, party code or column it does not recognize, so a swap cannot drift
through unnoticed.

---

### E7 — No external API call without a human's go-ahead
**Date:** 2026-09-29 · **Status:** active · extends [S5] from the pilot script
to every agent and every API

Before any call to an LLM provider (OpenAI, DeepSeek, Anthropic) or a data API
(govinfo / api.data.gov, congress.gov), an agent states what it will call, how
many requests and the estimated cost, and waits for an explicit yes from a
human in the current session. Cheap test calls and model-availability checks
included; an approval covers one run; `--yes` is never passed to skip
`pilot_run.py`'s prompt. Code that makes no calls needs no approval. The rule
is in CLAUDE.md under DO NOT.

**Why:** calls cost money (the full run was estimated at ~$4,100, P3; there is
no full run since S11), and they
reach outside the machine — data API keys have daily limits (`DEMO_KEY`: about
50 calls a day, O8). S5 already made `pilot_run.py` ask before spending, but
that prompt is one `--yes` away from being skipped, and it does not cover other
scripts or ad-hoc calls an agent writes itself.

**Rejected:** a cost threshold under which calls are allowed (a threshold
invites "just a small test" to add up unseen, and data-API calls cost quota,
not dollars); relying on S5's prompt alone (it guards one script).

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
the full run *(since S11: any larger run)* from the *billed* usage in the
closing summary, not from this.

---

### S6 — Scoring logic lives in `src/`, not in the script
**Date:** 2026-09-21 · **Status:** active

`code/src/scoring.py` holds the provider clients, prompt rendering, JSON
extraction, validation and retry; `code/scripts/pilot_run.py` holds sampling,
orchestration and reporting.

**Why:** the full run *(since S11: the S8 run)* needs the same clients.
CLAUDE.md puts production logic in
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
*(Corrected 2026-09-29: it does not — `make smoke` is the Stanford corpus build
and contacts no model, and no other tool checks availability. The check is a
cheap real run, `pilot_run.py --sample-size 6`, which needs a human's go-ahead
under E7.)*

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

## Pilot findings (2026-09-23)

### P1 — The pilot passed its gate; the instrument works
**Date:** 2026-09-23 · **Status:** finding, not a decision
**Data:** `results/metrics/pilot_summary_20260923T103556Z.json`

200 speeches, 600 calls, **$1.911** (under the $2.45 estimate). 199 of 200 scored
by all three models.

**The party check passes decisively.** Ensemble R − D = **+0.590**
(D −0.279, R +0.311; t=11.3, p=3e-23). Each model separates the parties on its
own — R − D of +0.558 (Sonnet 4.6), +0.557 (DeepSeek), +0.658 (GPT-4o) — so the
result does not depend on one member carrying the ensemble.

**The scale is genuinely used**, not bunched at zero: deciles −0.65 / −0.33 /
0.00 / +0.34 / +0.73, and 72 of 200 speeches score beyond ±0.5. The 41 speeches
scored exactly 0.0 are procedural, which is what the prompt instructs. Excluding
them raises separation to **+0.744** — worth a robustness check later, and note
the procedural rate is similar across parties (D 18, R 23), so it is not a
confound.

**Tone**: D 0.289 vs R 0.231. Small, and the opposite direction from what a
naive reading might expect. Not interpretable at this n.

---

### P2 — The three models agree at r ≈ 0.95, which cuts both ways
**Date:** 2026-09-23 · **Status:** finding, feeds [S12] (was O5)

Mean pairwise Pearson r on ideology = **0.949** (DeepSeek–GPT-4o 0.935,
DeepSeek–Sonnet 0.953, GPT-4o–Sonnet 0.961). Mean cross-model standard deviation
0.082; only 2 of 200 speeches exceed 0.3.

**Good news:** strong convergent validity. Three models from different providers
and training paradigms measure substantially the same construct, which is
evidence the construct is real and not a single model's artefact.

**Bad news for [M3]'s rationale:** the ensemble exists to "reduce single-model
bias". At r = 0.95 there is little independent error left to average away, so the
ensemble buys less than the design assumed — while costing 3× to run. See [S12].

**The 2 disagreements are substantive, not bugs.** Both are Democratic speeches
attacking defence spending and corporate welfare on fiscal-restraint grounds.
DeepSeek read the fiscal conservatism as right-leaning (+0.20, +0.40); GPT-4o
read the criticism as left-leaning (−0.70, −0.70). That is a real ambiguity in
the construct, and exactly the kind of case worth quoting in the thesis.

---

### P3 — Full-corpus cost is ~$4,100, not a rounding error
**Date:** 2026-09-23 · **Status:** finding, fed O5 (closed by S12) · **moot since S11**
(2026-10-03): there is no full run

At $0.00955 per speech, all three models over 426,718 speeches extrapolates to
**~$4,077**, or **~$2,038** with batch APIs at roughly half price.

Per model, extrapolated: Sonnet 4.6 **$2,100**, GPT-4o **$1,265**, DeepSeek
**$712**. The reasoner is the cheapest of the three despite emitting ~529 output
tokens per speech against 69 and 94 — its per-token price is far lower.

This is the number to budget from, and it is large enough that the ensemble
question in O5 is a financial decision, not only a methodological one.
*(O5 closed by S12 once S11 removed the full run.)*

---

### P4 — RQ2 preview: the Democratic mean moved, the Republican mean did not
**Date:** 2026-09-23 · **Status:** **indicative only — do not cite**

OLS of speech-level ensemble score on Congress number, 107th–114th:

| Party | Slope / congress | Change over 107→114 | p |
|---|---|---|---|
| D | **−0.0383** | −0.268 | **0.012** |
| R | +0.0148 | +0.104 | 0.379 (ns) |

Signed extremity (each score oriented toward its own party) rises +0.0266 per
congress, p=0.019. Per-congress D–R distance runs 0.33 → 0.43 → 0.59 → 0.52 →
0.69 → **0.92** (112th) → 0.65 → 0.63.

**This is the question Yufei cared most about in Meeting 1**, and the preliminary
answer is that Democratic floor rhetoric moved left while Republican rhetoric
held roughly flat.

**Why it cannot be cited yet:** n = 12–13 per party-congress cell. The design was
built to prove the pipeline works, not to estimate a trend. Treat the direction
as a hypothesis the full run *(since S11: the S8 run)* must test, and resist
the temptation to put this
table in a slide before then.

---

### P5 — First DW-NOMINATE validation: clear across parties, weak within them
**Date:** 2026-09-29 (on the 2026-09-23 pilot) · **Status:** finding, feeds [O9]
**Data:** `results/metrics/validation_20260923T103556Z.json`
(`python code/scripts/validate_scores.py --run 20260923T103556Z`, no API calls)

All 200 pilot speeches (Stanford, 107th–114th) join to a scored member through
the crosswalk. Ensemble ideology score against `nominate_dim1`, speech level:

| Speeches | Both parties | Within D | Within R |
|---|---|---|---|
| all (n = 200) | **+0.64** [0.55, 0.72] | **+0.27** [0.07, 0.44] | +0.14 [−0.06, 0.33] |
| non-procedural (n = 160) | **+0.71** [0.62, 0.78] | +0.23 [0.02, 0.43] | +0.12 [−0.11, 0.33] |

Pearson r with 95% CI; Spearman agrees within 0.05 everywhere.
`nokken_poole_dim1` gives nearly the same figures (within 0.04).

- **Positive, as M5 requires — no red flag.** Every model, every benchmark and
  every subset is positive.
- **But the overall r is mostly the party gap.** In this sample party alone
  correlates **r = 0.947** with `nominate_dim1`, and the ensemble only 0.63
  with party. An overall r of 0.64 therefore says little more than P1 did.
- **Within party the signal is weak:** significant for Democrats, not for
  Republicans. Within-party variation in DW-NOMINATE is small (SD 0.13 D, 0.15
  R, against 0.44 overall), and a single speech is a noisy reading of a member —
  179 of the 189 members contributed exactly one speech. Weak speech-level
  correlation is what attenuation predicts; it does not show the member-level
  correlation is weak. This pilot cannot tell the two apart.
- **Models:** GPT-4o is highest within Democrats (+0.32), DeepSeek lowest
  (+0.17); every CI overlaps, so the pilot cannot rank them (see S12).
- Dropping procedural speeches raises the overall r (+0.64 → +0.71), because
  they sit at 0.0 regardless of party, but not the within-party r.

**Consequence:** the validation that defends the method needs several speeches
per member, averaged, then correlated at member level — see O9. The 5,200-speech
S8 sample would not provide that either: it stratifies by cell, not by member.

---

### S8 — Sampling stratifies by party × congress × chamber, 100 per cell
**Date:** 2026-09-23 · **Status:** active · extends [S1]

32 cells (2 parties × 8 congresses × 2 chambers) at 100 each = **3,200 speeches**,
seed 42. Estimated **$30.86**.

*Update 2026-10-03:* over the merged corpus (D19, D25) the same rule gives 52
cells (13 congresses) = **5,200 speeches**; `pilot_run.py --dry-run` estimates
**$52.96**. The smallest cell is 119th / R / Senate with 2,627 speeches (1,128
before the re-fetch). `config.py` already describes the 52-cell design.

**Why chamber was added:** House and Senate floor rhetoric differ in length and
formality, and the chamber mix drifts across congresses in the corpus (Senate
share falls from ~52% in the 107th to ~35% in the 114th). Without balancing on
it, a per-Congress comparison would partly measure that drift rather than
rhetoric. Every cell holds at least 5,865 speeches, so 100 each costs nothing.

**Consequence:** this is no longer a pilot in CLAUDE.md's sense (it specifies
100–500 speeches) — it is a first measurement run. The 200-speech pilot [P1]
remains the pipeline gate.

---

### S9 — Cost is estimated from measured usage, not a flat assumption
**Date:** 2026-09-23 · **Status:** active · supersedes the estimator in [S5]

The pre-flight estimate uses each model's **measured** per-speech token counts
from the 200-speech pilot, scaled by the current sample's length, instead of a
flat 250-token guess for every model.

**Why it matters:** the flat assumption was wrong per model, not just in total.
Providers differ by ~30% on input tokens for identical text because their
tokenizers differ, and `deepseek-reasoner` emits ~7× the output of the other two
because its reasoning is billed as output. The old estimator put the 3,200-speech
run near $40 and misattributed the split; the measured one says $30.86, with the
cost concentrated in Claude ($15.90) rather than the reasoner ($5.39).

**Maintenance:** `PILOT_MEASURED_TOKENS` in `config.py` is pinned to the
2026-09-23 pilot. Re-measure if the prompt, the models or the corpus change.

---
### S10 — The output cap is per model: 8,192 for the reasoner, 1,024 for the rest
**Date:** 2026-09-23 · **Status:** active · **Commit:** 586a3bd

A reasoning model spends its reasoning tokens from the **same** budget as its
answer. Under a shared 1,024-token cap, `deepseek-reasoner` spent the whole cap
reasoning about a 623-word speech and returned empty content with
`finish_reason="length"` — the full pilot died on its first speech. The two
instruction-following models were nowhere near the cap.

`MAX_OUTPUT_TOKENS = 1024` therefore applies to GPT-4o and Claude, and
`REASONING_MAX_OUTPUT_TOKENS = 8192` to `deepseek-reasoner`
(`code/src/scoring.py`, set per model on `ModelSpec`).

**Measured on the 200-speech pilot** (`results/scores/pilot_*_20260923T103556Z.jsonl`),
output tokens per speech:

| Model | median | mean | max | over the old 1,024 cap |
|---|---|---|---|---|
| `deepseek-reasoner` | 427 | 532 | 3,792 | **16 of 200 (8%)** |
| `claude-sonnet-4-6` | 94 | 94 | 152 | 0 |
| `gpt-4o-2024-11-20` | 66 | 69 | 123 | 0 |

So the old cap would have lost roughly one speech in twelve from the reasoner,
not one in two hundred. It is a **length-dependent** failure, which is why the
6-speech smoke test passed and the 200-speech run did not: small samples hide it.

**Raising the cap is also cheaper, not merely more correct.** Billing is on
tokens *used*, not on the cap. Truncated at 1,024 the reasoner burned all 1,024
and produced nothing to score; given room it stops naturally at a median of 427.
A larger cap buys a usable answer for fewer tokens than a small cap wastes.

**Rejected:**

- *One uniform cap for all three* — the configuration that broke. It treats
  reasoning tokens and answer tokens as the same resource across models that
  bill them differently.
- *Raising all three to 8,192* — free in billing terms, but the measured maxima
  for GPT-4o and Claude are 123 and 152, so 1,024 is already ~7× headroom and a
  cheap guard against a runaway response. There is no reason to remove it.
- *Splitting long speeches to fit the cap* — forbidden by [M4]: chunking breaks
  the cross-sentence context the ideology score depends on.
- *Keeping whatever partial text came back* — a truncated reasoner emits
  reasoning, not an answer. Scoring it would invent a number the model never
  gave, the same error [S3] rejects for out-of-range scores.

**On truncation:** `TruncatedResponseError` names the model and its cap, instead
of surfacing as `no JSON object in response: ''` three layers from the cause. It
subclasses `ScoreParseError`, so a truncation becomes a null row under [S4] and
the surviving models still average — it never ends a run that has already been
paid for. One speech still hits the raised cap; see [O6].

**Follow-up:** the run manifest records `effective_temperature` per model but not
`max_output_tokens`, so the cap that decides whether a model answers at all is
not yet in the run record. Add it to the manifest in `pilot_run.py` before the
full run. *(Done 2026-10-03: `model_manifest()` in `pilot_run.py` writes each model's
`max_output_tokens` into the manifest, tested in `test_scoring.py`. The
2026-09-23 manifest predates it; that run used the caps in the table above.)*

---

### S11 — No full-corpus scoring run; the main results come from the S8 sample
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus · corrects
the full run that P3, P4, O5 (now S12), O6 and S10 took for granted

The 580,007 speeches are **not** scored in full. They are the sampling frame.
The main scoring run is the S8 stratified sample: party × Congress × chamber,
52 cells × 100 = **5,200 speeches**, about **$53** for all three models
(`pilot_run.py --dry-run`, 2026-10-03). If its results turn out too imprecise
for RQ1/RQ2 or for the DW-NOMINATE validation (O9), a somewhat larger sample is
reconsidered, as a new entry here. It is still a sample, not a full run.

**Why:**
1. **Cost.** All three models over the full corpus come to at least ~$5,540
   (P3's per-speech cost × 580,007; govinfo speeches are longer, so it would be
   more), against ~$53 for S8.
2. **A sample is enough.** RQ1 and RQ2 ask about party positions per Congress.
   A sample balanced across those cells estimates them directly, and scoring
   every speech would add precision those comparisons do not need.

**Rejected:**
- *The full run.* Too expensive for the precision it adds (above).
- *The middle path once raised for trimming the ensemble: one model over
  the full corpus, all three on a subsample.* That is still a full run, ~$968 with DeepSeek alone, and it would
  make the headline numbers single-model.

**Consequences:**
- **P3 is moot.** It stays as the record of why.
- **The ensemble question (O5) loses its cost case.** Settled in S12:
  keep all three.
- **O6's truncation rate means ~26 speeches in S8,** not ~2,100.
- **S10's follow-up is done** (2026-10-03): the manifest now records each
  model's `max_output_tokens`.
- **Equal quotas per cell mean a pooled figure is not a corpus average.** A
  mean over both chambers, for example, weights the Senate as heavily as the
  House. Report per chamber, as O8 requires anyway, or weight the cells by
  their size in the corpus.
- **The CLAUDE.md pilot rule (100–500 speeches first) remains the gate** before
  the S8 run.

---

### S12 — The ensemble stays at three models (closes O5)
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus · reaffirms
[M3]

O5 asked whether to trim the ensemble, because the three models agree at
r ≈ 0.95 (P2), so averaging buys little error reduction. A full run would also
have cost ~$4,077, against ~$712 for DeepSeek alone (P3). S11 removed the full
run, and with it the cost case: on the 5,200-speech S8 sample all three models
cost ~$53, DeepSeek alone ~$9.

**Why three:**
- M3's cross-provider design is the defence against one model's calibration
  being taken as ground truth.
- The agreement between three independent providers is itself a validity
  result worth reporting (P2).
- `validate_scores.py` checks each model against DW-NOMINATE separately, which
  needs all three scored.

**Rejected:**
- *DeepSeek alone.* It saves ~$44 on S8 and loses all of the above.
- *The cheapest two.* The same trade, for even less.
- *One model over the full corpus, three on a subsample.* It is a full run
  (S11).

---

## Open questions

Move these up into a numbered entry once decided.

### O6 — DeepSeek still truncates on long speeches
**Raised:** 2026-09-23 · **Status:** 1 speech in 200 · **From:** [S10]

Speech `1110041041` (732 words) hit even the raised 8,192-token cap set in [S10]
and returned no answer, so its ensemble row has `n_models=2`. Raising the cap
further is cheap (billing is on tokens used), but 0.5% at pilot scale is ~2,100
speeches over the full corpus *(~26 in the S8 run; no full run since S11)*.

The pilot gives the headroom to reason from: across the other 199 speeches the
reasoner's output peaked at 3,792 tokens, well under 8,192, so this is a tail
case rather than a cap set too low across the board. Note the failure does not
scale with word count in any simple way — 732 words is unremarkable in a corpus
filtered at 50 words — so it is the reasoning that runs long, not the input.
Measure output tokens against the word-count distribution before the S8 run
rather than guessing again.

### O7 — Recover the ~23k ambiguous House speeches?
**Raised:** 2026-09-27 (from D16) · **Settled:** 2026-10-03 by [D25] — option
(a): Konsti's re-fetch keeps the state, and 23,606 rows are now resolved by it

The govinfo parser dropped the state from `Mr. SMITH of Texas.`, leaving 23,075
House speeches unattributable. They lean Republican (~60/40 by candidate
weight). Options: (a) ask Konsti to keep the state and re-fetch — the only full
fix; (b) keep the 7,029 whose candidates all share a party, with party but no
member (enough for party-level plots, useless for DW-NOMINATE); (c) accept the
loss and report it. Konsti's notebook is not in the repo, so (a) starts with
committing it.

### O8 — Why does the Senate lose 71% of its speeches at the break?
**Raised:** 2026-09-29 (from D19) · **Cause found:** 2026-09-29, an upstream
fetch cap · **Re-fetched:** 2026-10-02 (Konsti), verified 2026-10-03 — the cap
is gone, a smaller drop remains · **Blocks:** pooling the chambers in a trend
until the residual is understood

**Update 2026-10-03, after the re-fetch** (2017–2025 averages per year against
Stanford 2011–15, from the rebuilt `corpus.parquet`):

| Step | House | Senate |
|---|---|---|
| Stanford, ≥ 50 words | 14,488 | 8,082 |
| new raw file, ≥ 50 words, before our code | 13,535 (−7%) | 6,759 (−16%) |
| written, old file (before re-fetch) | 9,830 (−32%) | 2,330 (−71%) |
| **written, new file** | **11,533 (−20%)** | **5,050 (−38%)** |

The Senate share of govinfo speeches is now 25–37% a year (was 11.5–27.5%),
against 34–38% in Stanford's last years; 2016's 34.6% sits right beside
Stanford's 33.7%. **The fetch cap is fixed.** What remains is no longer mostly
in the raw file: of the Senate's 38%, 16 points are already there (real
change, or the notebook's extraction — e.g. it only recognizes `Mr./Ms./Mrs.`,
never `Miss`), and the rest is our own processing, where D17's cut leaves a
speech under 50 words. That share is larger in the Senate, plausibly because
its unanimous-consent exchanges with the chair are short once the chair's
turn is cut. **Not yet checked:** whether Stanford's own speaker splitting
treats those exchanges the same way, which would make the remaining gap a
property of the sources rather than of the Senate. Until then, keep the
chambers apart in trend plots, as below. The original analysis follows.

Speeches per year from `merged_build_stats.json`, Stanford 2011–2015 average
against govinfo 2017–2025 average:

| Chamber | Stanford | govinfo | Change |
|---|---|---|---|
| House | 14,488 | 9,830 | −32% |
| Senate | 8,082 | 2,330 | **−71%** |

*(Updated 2026-09-29 after D20 moved 425 impeachment-trial speeches from the
Senate to the House; the drop is unchanged.)*

The House drop is mostly O7: about 2,500 ambiguous House speeches a year are
dropped. The Senate loses only about 130 unresolved speeches a year (1,196 in
total). The Senate share of speeches falls from 33.7–51.5% a year in Stanford
to 11.5–27.5% in govinfo.

**Cause found 2026-09-29: the upstream fetch kept only the first 100 granules
of each day's Congressional Record.** The table above was re-checked against
`corpus.parquet` on the same day and still holds. Each daily Record package
(`CREC-YYYY-MM-DD`) lists its granules House first, then Senate, then Extensions
and Daily Digest. The notebook apparently read one page of 100 and never
followed `nextPage`, so whatever sits past position 100 (usually the Senate) is
not in the file. Evidence, from the govinfo API:

- **2021-09-30** (193 granules, Senate at positions 59–146): every Senate
  granule with a speaker turn at positions ≤ 99 is in the file; none at ≥ 102
  is (positions 100–101 have no speaker turn). Lost from that day, for example:
  "Infrastructure Investment and Jobs Act", every "Introductory Statement on
  S. …", "Tribute to Dr. Mark J. Cochran".
- **Six randomly drawn days with busy House speeches (>150 rows) but no
  Senate speeches in the file:** four
  had a real Senate session of 54–80 granules, starting at positions 101–125
  (2019-07-23, 2023-11-07, 2024-12-04, 2025-12-10). The other two were real
  recess days: 2021-06-30 had no Senate granules and 2018-09-13 had 3.
- **Such days are no longer rare:** days with House speeches and no Senate
  speeches number 0–30 a year in Stanford (2001–2016) and 57–111 a year in
  govinfo (2017–2025). The Senate has speeches on 117–176 days a year in
  Stanford (full years) but only 71–130 in govinfo.
- **The Senate shrinks when the House is busy:** on days both chambers appear
  in the file, the Senate has 54 raw rows per day when the House is quietest
  (bottom quarter of days) and 30–31 when it is busiest (top half). On days with
  no House speeches it has 60 rows and 32k words a day, much closer to
  Stanford's 36–43k (2011–15) than the 20k it has on shared days.

Where the Senate speeches go (per-year averages, Stanford 2011–15 against
govinfo 2017–25, by Record section):

| Step | Senate speeches/yr |
|---|---|
| Stanford, ≥ 50 words | 8,082 |
| govinfo raw file, ≥ 50 words, before our code | 3,140 (−61%) |
| govinfo after D16/D17 (written) | 2,377 |

So about 87% of the drop is already in the raw file. Of the rest, about 680 a
year fall below 50 words only after D17's cut, and about 130 a year are
unresolved. Of the three candidate causes once listed, splitting and cleaning
account for little, and no real change in Senate activity is needed to explain
the drop.

**What it means for the analysis.** The govinfo Senate is not only smaller but
a different sample. Days when the House was busy are missing entirely. On
other days, the speeches that survive are from early in the Senate day (leader
remarks, the main debate, votes), and the late-day statements, tributes and
bill introductions are lost. That shift in composition is concentrated
exactly at the break, so it can masquerade as a trend in tone or ideology. The
House is exposed too: any House speech past position 100 is lost. On the days
checked, House speeches ended well before 100 (by position 66 on 2019-07-23),
with only end-of-day listings after them, but the House loss is **not
measured**.

**Fix:** re-fetch 2016-09-10 → 2025-12-31 following the API's `nextPage` /
`offsetMark` pagination (or `pageSize=1000`, the maximum). The same re-fetch can
keep the House member's state and settle O7 (a). It needs an api.data.gov key
(`DEMO_KEY` allows about 50 calls a day), and Konsti's notebook is not in the
repo. Until the re-fetch: do not pool chambers in a trend, and treat any
govinfo-era Senate trend as unreliable. Plot the chambers separately or
stratify (the S8 sampler already stratifies by chamber).

### O9 — How exactly is the DW-NOMINATE validation done?
**Raised:** 2026-09-29 (from P5) · **Blocks:** the validation that goes in the
thesis

`code/src/validation.py` reports every option side by side
(`VALIDATION_BENCHMARKS` in `config.py`) until these are settled:

1. **Benchmark:** `nominate_dim1` (constant over a member's career) or
   `nokken_poole_dim1` (per Congress). RQ2 is about movement over time, which
   only Nokken-Poole can show; in the pilot they give the same answer.
2. **Unit:** speech level, or member (× Congress) level after averaging each
   member's speeches. P5 suggests speech level is dominated by noise within
   party; member level needs a sample with several speeches per member, which
   neither the pilot nor the S8 design provides. How many speeches per member,
   and how many members, is a cost question (P3, S9). *(Since S11: this is the
   likeliest reason to extend the S8 sample.)*
3. **Headline statistic:** overall r mostly re-measures the party gap (P5), so
   the within-party r — or a regression of DW-NOMINATE on party plus the LLM
   score — is the defensible one.
4. **Procedural speeches:** keep or drop (P1 and P5 both show they change the
   overall figure, not the within-party one).

### O10 — govinfo compound surnames the Record shortens
**Raised:** 2026-10-03 (from D25's build) · **Blocks:** nothing; ~620 speeches

The largest unresolved govinfo speakers after D25 are `RODGERS` (House, 508)
and `PEREZ` (House, 112): Cathy McMorris Rodgers and Marie Gluesenkamp Perez,
whom congress-legislators files under the whole compound surname while the
Record writes only its last word. This is the reverse of JACKSON LEE, which
D16 handles. Both are from Washington and almost always carry the state, so a
fallback to the last word of a compound surname, *only when the state agrees*,
would recover them. Do not add it without the state: a bare last-word match
would put Sheila Jackson Lee among the candidates for every `LEE`. Not yet
checked: the next names down the list (`FROST`, House, 27).

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
**Raised:** 2026-09-15 · **Status:** answerable since 2026-10-02, not yet answered

Speeches with no matched speaker were dropped when Konsti built the parquet.
Those rows are not in the file, so no analysis of it recovers the count. Only the
missing `code/scripts/build_stanford_dataset.py` can close this, and the
methodology chapter needs the figure.

**Update 2026-10-03:** the recipe is now committed as
`code/notebooks/01_stanford_dataset.ipynb`. It keeps
`merged["speakerid"].notna()` per session but prints counts only after that
filter (823,341 in total), so the figure is still not recorded. It also reads
all three per-session files with `on_bad_lines="skip"` and `quoting=3`,
which drops malformed lines **without counting them** — a second upstream
filter nobody had listed. Closing O4 needs one re-run of that step that counts
rows before the `speakerid` filter and lines skipped by the parser. It needs
the 2.6 GB `hein-daily.zip` from stacks.stanford.edu, an external download, so
it waits for a human's go-ahead (E7).
