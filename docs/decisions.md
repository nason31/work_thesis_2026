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
**Date:** 2026-09-16 · **Status:** SUPERSEDED by [M3a] on the Anthropic member, by [M3b] on the DeepSeek member, and by [S15] on the number of models in the S8 run · **Commit:** 112343d

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
*Re-measured 2026-10-03* on the 208-speech S8 stage ([P6]), after the corpus
merge and with DeepSeek served by `deepseek-flash` ([M3b]): tiktoken 967.7
per speech; input/output per speech DeepSeek 997.3/498.9, GPT-4o 972.8/69.3,
Sonnet 1,241.9/95.6.

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
**Date:** 2026-10-03 · **Status:** **SUPERSEDED by [S15]** the same day (two
models for the S8 run) · **Decided by:** Justus · reaffirms [M3]

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

### S13 — The S8 run is one run, scored in stages; nothing is paid for twice
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus · implements
[S8]/[S11]; keeps the CLAUDE.md pilot rule as its first stage

*Update 2026-10-03:* the first run under this plan, `20261003T100851Z`,
completed stage 1 ([P6]) but cannot be resumed after the DeepSeek switch
([M3b]); the S8 run restarts with the same plan under a new timestamp.

Justus wants the S8 sample scored **once**, as the final run, and run again only
if the analysis shows it is not enough. The S8 sample is therefore one run
(label `s8`, one timestamp) extended in stages:

1. **Stage 1, the gate:** `pilot_run.py --label s8 --per-cell 4` — 4 per cell,
   208 speeches, 624 calls, **$2.13** estimated. It is the CLAUDE.md pilot over
   the merged corpus: the 2026-09-23 pilot scored Stanford speeches only, and no
   govinfo speech (115th–119th, D17-cut text) had been through the models.
2. **Stage 2:** `pilot_run.py --resume <timestamp> --per-cell 100` — the
   same run at the full 5,200. Only the 4,992 speeches not yet scored are
   called, ~$50.83, so the whole run costs the same ~$53 as scoring S8 directly.
3. **Later, only if needed:** extend the same run (e.g. `--per-cell 150`).
   The first 100 per cell are reused; only new speeches are billed.

**Why this works:** a sample drawn with the same seed and a larger per-cell
quota *contains* the smaller one — pandas draws a seeded permutation per cell
and takes its first n. Verified on the merged corpus (sha256 `f49c7bb2…`): all
208 stage-1 speeches are in the 5,200 draw. The draw depends on the corpus
file, so a rebuilt corpus breaks this, and the resume refuses (below).

**What `--resume` does** (`code/src/runs.py`, `code/scripts/pilot_run.py`):

- Scores only (speech, model) pairs with no successful score — so it also
  finishes a crashed run and retries failed calls. A bare `--resume <ts>`
  keeps the run's current size; only an explicit `--per-cell` extends it.
- **Refuses** if anything that defines a score differs from the manifest:
  corpus SHA-256, prompt text, models and their settings (incl. output caps),
  seed, minimum words, strata. Also refuses if the new sample would leave out
  speeches the run already scored. Scores under two instruments do not belong
  in one run.
- Appends to the per-model files; every attempt is kept (each was billed).
  A failed attempt and its retry are two rows; the successful one counts
  (`effective_rows`), in the ensemble and in `validate_scores.py` alike.
- Rebuilds the ensemble and summary from the files after each stage. The
  summary bills every attempt, splits cost by stage, and counts as failures
  only speeches still unscored.
- Each row records `stage`, `scored_at`, and the model the provider says
  answered (`served_model`, plus `system_fingerprint` for OpenAI-compatible
  APIs). `deepseek-reasoner` is an alias the provider can repoint; the summary
  lists every served model and flags a change mid-run.

**Fixed on the way:** a truncated call was logged with 0 tokens although the
provider billed it (8,192 output tokens for the reasoner). It now records what
was billed. The 2026-09-23 pilot under-reported by that one call, ~$0.02 of P1's
$1.911.

**Also decided (Justus):**
- **Live calls, not batch APIs.** Batch would save ~$22 on GPT-4o and Sonnet
  but is code that has never run; stage 1 would not test the path that scores
  the other 96%.
- **DeepSeek keeps the 8,192 cap** ([S10]); stage 1 tests it. See [O6].
- **What would justify extending the run** is left to the analysis — [O11].

**Rejected:**
- *Scoring S8 straight through* — no gate on the govinfo side, and a crash or a
  bad setting costs the whole ~$53 instead of ~$2.
- *A separate pilot, then S8 from scratch* — pays for the 208 speeches twice.
- *Batch APIs* — above.

---

### M3b — DeepSeek member: `deepseek-flash` (DeepSeek-V4.1-Flash, thinking mode) replaces `deepseek-reasoner`
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus · amends
[M3], as [M3a] did for Claude

`ENSEMBLE_MODELS`' DeepSeek member is now `deepseek-flash`, called with thinking
mode on and `reasoning_effort="high"`, both pinned explicitly
(`request_options` on its `ModelSpec`, recorded in every run manifest).

**Why:**
- **"DeepSeek R1" was not what we were calling.** The first S8 stage recorded
  the served model for the first time: all 208 `deepseek-reasoner` calls were
  answered by `deepseek-flash` (one system fingerprint). The 2026-09-23 pilot
  did not record it, so its DeepSeek scores cannot be attributed to a model.
  V4.1-Flash was released 2026-09-10 (secondary sources), so the pilot ran on
  V4-Flash or V4.1-Flash, not R1.
- **The alias is on its way out.** DeepSeek's API docs (checked 2026-10-03)
  list only `deepseek-flash` (DeepSeek-V4.1-Flash) and `deepseek-v4-pro`, and
  no longer mention `deepseek-reasoner`; secondary sources report it was
  announced for retirement on 2026-07-24. It still answered on 2026-10-03 but
  could stop at any time: mid-run, or before a later extension ([O11]), which
  could then not continue the run, because the model is score-defining ([S13]).
- **Thinking mode at high effort** are `deepseek-flash`'s documented defaults
  ("Thinking mode is enabled by default, with the default effort being
  `high`"), and the alias was routed to thinking mode. Pinning both keeps the
  instrument fixed if DeepSeek changes a default.

M3's rationale still holds: a reasoning model from a third provider.
**Open weights:** secondary sources report V4.1-Flash as MIT-licensed on
Hugging Face — verify the release before the methodology says so.
`deepseek-flash` is not a dated snapshot (DeepSeek repoints names: its docs say
the retired `deepseek-v4-flash` is now served by V4.1-Flash), so every row
records `served_model`, `system_fingerprint` and `reasoning_tokens`, and the
run summary flags a change of served model mid-run.

**Consequences:**
- **A new S8 run.** The model is score-defining, so run `20261003T100851Z`
  cannot be resumed under the new name; it stands as a pilot over the merged
  corpus ([P6]). *Changed by [S16]: no second stage 1 — the new run carries
  over the old run's GPT-4o and DeepSeek answers.* The new run starts at 4 per cell again (~$2): the same 208
  speeches, so its DeepSeek scores against the old run's check that the alias
  and the new name behave as one instrument. Then resume to 100 per cell.
- **Prices:** `deepseek-flash` lists $0.30 / $1.20 per 1M tokens (peak, cache
  miss; $0.15 / $0.60 off-peak). `config.py` carried R1's $0.55 / $2.19 until
  now, so the DeepSeek cost in the 2026-09-23 and `20261003T100851Z` summaries
  is overstated.
- JSON mode stays off for DeepSeek: the answer is parsed from the content, as
  in every run so far.
- **Thesis:** name the model "DeepSeek-V4.1-Flash, thinking mode (API name
  `deepseek-flash`)", not R1.

**Rejected:**
- *Keep `deepseek-reasoner`* — free, and today's run continues; but it is past
  its announced retirement and undocumented, and if it stops the run can be
  neither finished nor extended. ~$2 buys the stable name.
- *`deepseek-v4-pro`* — a different, larger model: breaks comparability with
  every DeepSeek score so far, at about twice the price.
- *Leave thinking mode and effort at their defaults* — identical today, but a
  changed default would change the instrument unseen.

---

### S14 — Every provider call has a hard 300-second deadline, then a retry
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus

`score_one` wraps each call in `asyncio.wait_for` with `CALL_TIMEOUT_SECONDS`
(300, `config.py`). A call past it raises `CallTimeoutError`, retried like a
rate limit (3 attempts in all); after the third it is a failed row, which
`--resume` retries ([S13]).

**Why:** in the first S8 stage one call hung for 20 minutes, on speech 147 of
208 (`gov-6853a24a8dbe839f`, 377 words, scored normally on resume), and the run
sat still until stopped by hand. The SDK's own timeout never fired. Answered
speeches took 3.75 s on average and at most 21 s, so 300 s is ~14× headroom.
At one stall per ~150 speeches, the 5,200-speech run would otherwise stall ~30
times.

**Not score-defining:** it decides whether an answer arrives, not what it says,
so a resume does not compare it; each stage records the value it ran with
(`call_timeout_seconds` in the manifest).

**Caveats:** tokens of an abandoned call are unknown and not recorded, though a
provider may still bill them. Which model hung on 2026-10-03 is unknown — all
three calls for the speech were in flight.

**Rejected:**
- *The SDK's timeout setting* — the SDK timeout is what did not fire.
- *A short deadline (e.g. 60 s)* — the reasoner's longest answers need room,
  and a tight deadline would turn long reasoning into failures, biasing which
  speeches DeepSeek scores.
- *Supervising by hand* — a ~5-hour run.

---

### P6 — First S8 stage (run `20261003T100851Z`): the gate passed; now a pilot
**Date:** 2026-10-03 · **Status:** finding · per-Congress figures
**indicative only — do not cite**
**Data:** `results/metrics/s8_summary_20261003T100851Z.json`,
`results/metrics/validation_20261003T100851Z.json`

208 speeches (4 per Congress × party × chamber, 107th–119th), 624 calls,
**0 failures, 0 truncations** (DeepSeek's longest answer 3,914 of 8,192
tokens — [O6]'s cap held). $2.07 at the prices then configured (estimate
$2.13; DeepSeek overstated, [M3b]). Run in two invocations: stopped by hand
after the 20-minute stall ([S14]) and resumed — the resume worked as designed,
re-scoring nothing.

- **Party check:** ensemble R − D = **+0.683** (D −0.380, R +0.303; Welch
  p ≈ 1e-28). It holds in both sources (Stanford +0.55, govinfo +0.90), both
  chambers, and per model (DeepSeek +0.67, GPT-4o +0.71, Sonnet +0.67).
- **Agreement:** pairwise r 0.952 (DeepSeek–GPT-4o), 0.952 (DeepSeek–Sonnet),
  0.969 (GPT-4o–Sonnet); P2 had 0.949 on average.
- **DW-NOMINATE** (ensemble vs `nominate_dim1`): overall +0.72; **within D
  +0.26** [0.07, 0.43], **within R +0.39** [0.22, 0.54] — P5 had R +0.14, not
  significant. `nokken_poole_dim1` within D: +0.16, CI includes 0.
- **Procedural** (all three at 0.0): 41 of 208.
- **govinfo text** (D17's cut) reads as the member's own words in all four
  speeches spot-checked: opens at "Mr. Speaker"/"Mr. President", ends at the
  member's own close.
- **Per Congress** (n = 8 per party): the gap is ~0.45 in the 107th–110th,
  0.69–0.80 in the 112th–113th, 0.79–1.05 in the govinfo Congresses, and in
  this sample the rise is mostly Republican. **That contradicts [P4]**, where
  Democrats moved and Republicans did not. Both samples hold 8–13 speeches per
  party-Congress; neither can answer RQ2, which is what the S8 run is for.
- **Source break:** govinfo separates the parties more than Stanford (+0.90
  vs +0.55). The rise begins inside Stanford (112th), so it is not only the
  source change, but a level shift at 2016-09-10 cannot be excluded. The
  114th Congress holds both sources; the full run allows a within-Congress
  comparison there.

Superseded as the S8 run by [M3b]; it stays as the pilot over the merged
corpus, and its 208 speeches are re-scored first in the new run.

---

### S15 — The S8 run scores with two models: DeepSeek and GPT-4o; Sonnet is dropped
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus ·
**supersedes [S12]**, amends [M3]

`ENSEMBLE_MODELS` is now `deepseek-flash` and `gpt-4o-2024-11-20`. Claude
Sonnet 4.6 scores nothing in the S8 run. Its spec and price stay in the code,
so the three-model runs remain readable and re-validatable
(`validate_scores.py` now reads each run's models from its manifest).

**Why:**
1. **Sonnet adds the least.** It is the model GPT-4o predicts best: r = 0.969
   on the 208 speeches of [P6] and 0.961 on the 200 of [P2] — the closest
   pair both times (DeepSeek–GPT-4o 0.952 / 0.935, DeepSeek–Sonnet 0.952 /
   0.953). Each model alone reproduces the party gap per Congress ([P6]), and
   against DW-NOMINATE within party Sonnet and GPT-4o are indistinguishable (P6:
   D +0.25 vs +0.26, R +0.42 vs +0.38; every CI overlaps).
2. **Sonnet costs the most.** It was ~56% of stage 1's cost at current prices
   ($1.07 of ~$1.91). Over the 5,200-speech run: ~$21 for two models against
   ~$48 for three.
3. **Why Sonnet and not GPT-4o:** the two are near-duplicates, so dropping
   either keeps most of the information. GPT-4o is cheaper and is the baseline
   prior work used ([M3]). DeepSeek stays as the only reasoning model and the
   one least correlated with the others, so it carries the most independent
   signal.

**What is lost — state it in the methodology:**
- Provider diversity falls from three training paradigms to two; the
  Constitutional AI member ([M3a]) is gone from the main results.
- With two members, the cross-model spread of a speech is one difference
  (`ideology_score_std` = |a − b| / √2), and a speech the two disagree on has
  no third opinion. The disagreement threshold (0.3) was set for three.
- The per-model robustness check in the thesis has two lines, not three.

**The thesis argument rests on evidence already scored:** three-model scores on
408 speeches ([P2]: 200 Stanford speeches; [P6]: 208 over the merged corpus,
all 13 Congresses). That is where the redundancy is shown; the S8 run is two
models from its first call, because the model list is score-defining ([S13]).

**Rejected:**
- *Keep three ([S12])* — ~$27 more for a member whose scores GPT-4o predicts at
  r ≈ 0.97.
- *Drop GPT-4o instead* — keeps the more expensive of two near-duplicates.
- *DeepSeek alone* — loses the cross-provider check entirely (S12's argument
  still holds against one model).
- *Sonnet on a subsample of the S8 run* — a run has one model list ([S13]); the
  three-model evidence already exists in P2 and P6.

**Consequences:** the new S8 run's first stage (4 per cell, the same 208
speeches as [P6]) costs ~$0.85 instead of ~$2; the full run ~$21. *[S16]:
that stage is carried over from [P6] instead — ~$20 for the remaining 4,992.* [M3b]'s
"new S8 run" is this two-model run.

---

### S16 — The S8 run starts from stage 1's answers instead of re-scoring them
**Date:** 2026-10-03 · **Status:** active · **Decided by:** Justus · changes
the consequences of [M3b] and [S15]

The S8 run is started with

    pilot_run.py --label s8 --per-cell 100 --carry-over-from 20261003T100851Z

Its stage 1 is the 416 GPT-4o and DeepSeek answers to [P6]'s 208 speeches,
carried over; stage 2 scores only the other 4,992 speeches (~$20). Sonnet's
208 answers stay in the old run as pilot evidence ([S15]).

**Why:** those answers come from the models the run uses. GPT-4o: same snapshot,
same settings. DeepSeek: requested as `deepseek-reasoner`, but every one of the
208 rows records `deepseek-flash` as the model that answered (one system
fingerprint); DeepSeek documents that name as Flash's thinking mode, and
thinking at high effort is Flash's default — the settings [M3b] pins. A
second stage 1 would have cost ~$0.84 and bought a test-retest comparison;
Justus judged that not worth another run.

**Assumption — state it in the methodology:** the old DeepSeek calls relied on
the default effort instead of sending it, and did not record reasoning tokens,
so "high effort" for those 208 answers is inferred from the documented
default, not observed. **Check after stage 2:** DeepSeek's output tokens per
speech on the carried speeches (mean 499) against the new ones; a clear shift
would put the equivalence in doubt.

**Guards** (`--carry-over-from`, `code/src/runs.py`):
- corpus, prompt, seed, length filter and strata must be identical;
- a model under the same name must have identical settings (a manifest from
  before request options were recorded counts as having sent none, which is
  what the code did);
- a renamed model is allowed only as a declared pair
  (`CARRY_OVER_EQUIVALENTS` in `config.py`, today only `deepseek-reasoner` →
  `deepseek-flash`) and only if every carried row's served model is the new
  name;
- the old run's speeches must lie inside the new sample.

Carried rows keep everything they recorded — including the model name they
were requested under — plus `carried_from` (run, stage, model); the new
manifest records `carried_over_from`.

**Rejected:**
- *A second stage 1 under the new name* ([M3b]'s plan) — ~$0.84 and 15 minutes
  for answers already in hand.
- *Resuming the old run with Sonnet dropped* — its manifest and file names
  record three models and the alias; continuing it would make the run's own
  record wrong.
- *A general switch to ignore model changes on resume* — would let any change
  of instrument through. The pair is declared, and checked row by row.

---

### S17 — Speeches are scored and labelled in parallel: `--concurrency N`
**Date:** 2026-10-18 · **Status:** active · **Decided by:** Justus (N = 10 for
the [S19] run); implemented by the agent

`pilot_run.py` and `label_run.py` keep up to N speeches in flight at once: N
workers take speeches from the sample in order, each writing and flushing its
rows the moment they land. **The default stays 1**, the sequential loop of every
run so far, so the S8 path and every earlier test behave as before.

**Why:** at S8's ~3.8 s per speech, 100,000 speeches take ~105 hours in
sequence and ~10.5 hours at N = 10. DeepSeek's rate-limit docs (read
2026-10-18) allow **2,500 concurrent requests per account** for
`deepseek-flash` and answer HTTP 429 beyond that; 429s are retried as before
([S14]), and a speech still failing is retried by `--resume`.

**Not score-defining**, like the deadline ([S14]): it decides when an answer
arrives, not what it says, so a resume does not compare it; each stage
records the value it ran with (`concurrency` in the manifest). Rows land in
completion order rather than sample order; nothing reads them by order
(`effective_rows` keys on `speech_id`, and the ensemble file is rebuilt in
sample order). An interruption loses at most the N calls in flight, which the
provider may still bill.

**Rejected:** one task per speech gathered at once (100,000 open tasks for no
gain over a fixed pool); threads (the SDKs are already async); making N > 1
the default (it would change the behaviour of every existing command
silently).

---

### S18 — A run stops when DeepSeek answers from another model build
**Date:** 2026-10-18 · **Status:** active · **Decided by:** Justus ("address the
open questions"); design by the agent

On 2026-10-17 DeepSeek moved `deepseek-flash` to DeepSeek-V4.1-Flash-1017
under the same name ([L3]). The API still reports `deepseek-flash` as the
served model, so the served-model check of [S13]/[M3b] cannot see it; only
`system_fingerprint` changed (`aeb56401…` on all 5,202 S8 answers,
`e72d3c7f…` on every label since). A resume compares the *requested* settings,
so extending S8 would have mixed two models without a word.

**Now** (`runs.FingerprintGuard`, both scripts): for the models in
`FINGERPRINT_PINNED_MODELS` (`config.py`: `deepseek-flash`), a run's build is
the fingerprint of its earlier answers (carried-over rows included), or of its
first new answer. An answer from another build is written as a failure
(`FingerprintChangedError`, its scores or labels kept aside under
`rejected_answer`, never counted), no further speech is started, and the
script exits with code 3. Rejected rows never define the build, so the run
stays stopped on every resume. The summaries list `system_fingerprints` per
model.

**Consequence: the S8 run can no longer be extended with DeepSeek** — its
first new answer would come from the 1017 build and stop it. More DeepSeek
scores mean a new run ([S19]).

**Why not for GPT-4o:** OpenAI rotates fingerprints across the backend builds
of one dated snapshot — three within the S8 run alone — so for a dated
snapshot a changed fingerprint says nothing, and stopping on it would stop
every long run.

**Rejected:** a warning only (a 10-hour run would finish mixed, and the
warning would be read afterwards); checking before the run (the build is only
known from an answer); treating a missing fingerprint as a change (nothing to
compare; it passes).

---

### S19 — A ~100,000-speech run with DeepSeek alone (label `d100k`)
**Date:** 2026-10-18 · **Status:** active — not started; each stage needs a
go-ahead ([E7]) · **Decided by:** Justus · amends [S11] (a larger sample) and
[S15] (one model instead of two)

A new run, label `d100k`: the S8 design (party × Congress × chamber, seed 42,
≥ 50 words) at **1,923 speeches per cell = 99,996 speeches**, scored by
**`deepseek-flash` alone** (thinking mode, high effort, the [M3b] settings) at
`--concurrency 10` ([S17]):

    pilot_run.py --label d100k --per-cell 4 --models deepseek-flash --concurrency 10      # stage 1
    pilot_run.py --resume <timestamp> --per-cell 1923 --concurrency 10                    # stage 2

**Why DeepSeek alone (Justus):** cost, and the two models' results are similar
enough for the analysis. On the 5,200 S8 speeches DeepSeek and GPT-4o agree at
r = 0.94 (0.88 within Democrats, 0.91 within Republicans), and each model on
its own tells the same story per party and Congress (notebook 04, section 8).
GPT-4o was ~77% of S8's cost ($16.19 of $21.08); over 99,996 speeches it would
add roughly $300.

**Why a larger sample — proposed by the agent, to confirm:** the within-topic
gaps of [A2] rest on 16–73 speeches per party and period in some blocks; the
member-level validation ([O9]) needs several speeches per member; per-cell
intervals narrow by up to √19 ≈ 4.4× (less in practice: they are clustered
by member, and frequent speakers recur). This answers [O11].

**Why a new run, not an S8 extension:** the DeepSeek model changed on
2026-10-17 ([S18]); S8's DeepSeek answers come from the earlier build.

**Built-in snapshot check:** the 99,996 draw contains all 5,200 S8 speeches
(checked 2026-10-18), and stage 1 (4 per cell) is exactly [P6]'s 208. So the
1017 build's scores can be compared with the old build's on the same speeches
at no extra cost — first on 208 after stage 1, then on all 5,200.

**Cost and time (dry runs, 2026-10-18):** stage 1, 208 speeches, **$0.19**;
the whole run **$89.36** at peak list price (99.3M input, 49.7M output tokens
by S8's measured usage), about half off-peak. About 10.5 hours at
concurrency 10, if DeepSeek's per-call time holds under load. After stage 1,
re-measure `PILOT_MEASURED_TOKENS` for the new build ([S9]): DeepSeek reports
shorter output for it, which would lower the cost. *Done 2026-10-18 ([P7]):
output 358.7 tokens per speech (was 498.9); stage 2 now prices at **$72.46**
at peak for 99,788 speeches, and stage 1 ran at ~0.27 s per speech at
concurrency 10, so ~7.5 hours.*

**What is lost — state it in the methodology:** the cross-provider check
([M3], [S12], [S15]). The main results become single-model; the evidence
that a second model would tell the same story is the S8 comparison above and
the three-model pilots ([P2], [P6]). The ensemble mean is DeepSeek's score
(`n_models` = 1), and there is no per-speech disagreement measure.
`ENSEMBLE_MODELS` in `config.py` still names the S8 pair; this run chooses its
model with `--models`, and its manifest records it.

**Still to approve separately:** labelling the run ([L1]; ~$43 at peak by the
label pilot's usage) and any GPT-4o subsample.

**Rejected:**
- *Extending S8 to 99,996* — mixes two DeepSeek builds; [S18] would stop it.
- *Keeping GPT-4o* — ~$300 for a model that agrees at r = 0.94.
- *GPT-4o on a subsample of this run* — a run has one model list ([S13]); the
  S8 run already holds GPT-4o on 5,200 of these speeches.
- *A full-corpus run* — still ruled out ([S11]).

---

### P7 — Stage 1 of `d100k` (run `20261018T122536Z`): the new DeepSeek build scores like the old one
**Date:** 2026-10-18 · **Status:** finding, not a decision · run by Justus
**Data:** `results/metrics/d100k_summary_20261018T122536Z.json`

`pilot_run.py --label d100k --per-cell 4 --models deepseek-flash
--concurrency 10`: [P6]'s 208 speeches, DeepSeek-V4.1-Flash-1017 alone.
**208 of 208 scored, 0 failures, 0 truncations** (longest answer 6,352 of
8,192 tokens), one build throughout (`e72d3c7f…`), **$0.152** (estimate
$0.187), **56 seconds** at concurrency 10.

**Against the old build (`aeb56401…`, the S8 answers) on the same 208:**

| | new vs old DeepSeek | old DeepSeek vs GPT-4o | new DeepSeek vs GPT-4o |
|---|---|---|---|
| ideology r, all | **0.984** | 0.952 | 0.944 |
| ideology r, within D / R | 0.973 / 0.972 | 0.884 / 0.947 | 0.871 / 0.924 |
| tone r | 0.985 | 0.923 | 0.919 |

- Mean ideology shift new − old **+0.002** (paired t, p = 0.74; D +0.009,
  R −0.005); mean |difference| 0.053, 1% of speeches differ by more than 0.3.
  Tone shifts −0.016.
- R − D gap 0.652 (old 0.666; GPT-4o 0.711). Speeches at exactly 0.0: 72 (old
  71, 67 in both).
- Output per speech **358.7 tokens, against 498.9** (−28%): the new build
  reasons less. `PILOT_MEASURED_TOKENS` updated ([S9]).

**Reading:** the build change moves scores far less than the difference
between providers does, and does not shift either party. Since no temperature
is set ([S2a]), part of the remaining difference is run-to-run noise that a
same-build retest would also show; the two cannot be separated here. The 208
are not a test of the 1017 build on govinfo-era speech in particular; the
comparison on all 5,200 S8 speeches comes free with stage 2.

---

### P8 — The `d100k` run analysed (notebook 05): both parties moved, Republicans about twice as far
**Date:** 2026-10-19 · **Status:** finding, not a decision — preliminary; labels
not yet hand-checked ([L4]) · analysed as in [A1]/[A2]
**Data:** `code/notebooks/05_d100k_analysis.ipynb`;
`results/metrics/d100k_*_20261018T122536Z.csv`,
`validation_20261018T122536Z.json`; `results/plots/d100k_*.png`

99,993 of 99,996 speeches scored (DeepSeek-V4.1-Flash-1017 alone, [S19]),
99,967 labelled ([L8]); 1,923 per Congress × party × chamber; chambers apart,
intervals clustered by member.

- **The new build against the old, on all 5,200 S8 speeches:** r = 0.978
  (0.959 within D, 0.970 within R; GPT-4o against either ≈ 0.94). A small
  systematic shift: Democrats 0.003–0.023 and Republicans 0.003–0.007 closer
  to the centre, so the gap is ~0.01–0.03 narrower. **Levels from `d100k`
  and S8 are not interchangeable; trends within a run are.**
- **RQ1:** the gap widened in both chambers, not linearly — House 0.42
  (107th) → 0.78 (112th) → 0.57–0.63 (114th–116th) → 0.72–0.76
  (117th–119th); Senate 0.31 → ~0.55 (111th–114th) → 0.71 (115th) → 0.75
  (119th), the second Senate step at the source change.
- **RQ2, 2001–2016 (one source):** Republicans +0.16 (House) and +0.19
  (Senate), Democrats −0.09 and −0.06 — **both parties moved, Republicans
  about twice as far.** S8 could not establish the Democratic move; this run
  can (House p < 10⁻⁶, Senate p = 0.01). Policy debate alone: steeper (R +0.20
  / +0.28, D −0.10 / −0.09). Positions swing with the presidency: each party
  is most extreme in opposition, Senate Democrats most of all under Trump
  (−0.46, −0.47).
- **114th by source:** govinfo speeches sit closer to the centre than the same
  Congress's Stanford speeches (House D −0.24 vs −0.33, R +0.20 vs +0.29) —
  no sign that govinfo pushes scores outward, but source and moment (Sept–Dec
  2016) are confounded.
- **DW-NOMINATE:** speech level r = 0.65 across parties, 0.24 (D) / 0.29 (R)
  within; **member level r = 0.61 (D) / 0.62 (R) on 1,347 members** with ≥ 5
  speeches (S8: 0.51 / 0.55 on 298).
- **Tone (RQ5):** higher in opposition in all four party-chamber cells;
  highest for Democrats in 2025 (House 0.44, Senate 0.53).
- **Within topic:** the gap widens inside topic blocks (clearly in economic,
  governance and social-cultural in both chambers, security/foreign in the
  Senate), and the agenda shifts toward the most divided blocks
  (social-cultural 10.5% → 18.0% of House policy debate, governance up,
  economic down). Not decomposed yet.

---

## Analysis

### A1 — How the S8 run is analysed: chambers apart, intervals clustered by member
**Date:** 2026-10-04 · **Status:** active · **Decided by:** Justus (design
approved for the Meeting 2 notebook)

`code/notebooks/04_score_analysis.ipynb` analyses run `20261003T110419Z`; its
logic is in `code/src/analysis.py` and `code/src/plots.py` (tested). Figures go
to `results/plots/s8_*.png`, the tables behind them to
`results/metrics/s8_*_20261003T110419Z.csv`.

- **House and Senate are never pooled** — O8's thinner govinfo Senate, and S8's
  equal quotas per cell (S11).
- **Intervals are clustered by member** (CR1, normal 95%). Many speeches come
  from one member, and treating them as independent overstates the evidence.
  A member is their career-long ICPSR from the crosswalk (5,196 of 5,200
  speeches); the 4 unmatched fall back to the source-native id. Not
  Stanford's `speakerid`, which changes every Congress.
- **Trends are OLS slopes of the speech-level score on the Congress number**,
  reported for 2001–2016 (Stanford only, no source change) and 2001–2025. A
  straight line is a summary, not a claim that change was linear.
- **The party gap's error** adds the two parties' errors in quadrature — their
  speeches come from different members.
- **Presidency per Congress** (`PRESIDENTS_BY_CONGRESS` in `config.py`): each
  Congress goes to the president who held office for nearly all of it (a
  Congress starts 3 January, a presidency 20 January).
- **Member-level validation is exploratory** (members with ≥ 5 sampled speeches,
  each member's mean against their Voteview score) until O9 is settled.
- Procedural speeches stay in every figure; a robustness check without them is
  still to do. *(Done 2026-10-18 on the speech-type labels, [A2].)*

**Rejected:** pooling chambers (O8); speech-level standard errors (overstate
the evidence); clustering by `member_id` (Stanford's changes every Congress, so
a career would count as several members).

---

### A2 — Speech type and topic in the S8 analysis (notebook section 10)
**Date:** 2026-10-18 · **Status:** active · **Decided by:** Justus (labels into
notebook 04); the cuts below by the agent

`04_score_analysis.ipynb` section 10 joins the labels of run `20261017T182223Z`
([L7]) onto the S8 run with `analysis.attach_labels`. Tables:
`results/metrics/s8_{label_type_shares,party_gap_policy_only,ideology_trends_policy_only,party_gap_by_topic_block}_20261003T110419Z.csv`;
figures `s8_label_type_shares.png`, `s8_party_gap_policy_only.png`.

- **Speech-type shares per Congress, chambers apart**, source break marked.
- **The procedural label against the 0.0 proxy** ([O12]): 688 of the 796
  speeches labelled procedural carry the proxy; the proxy's 1,599 also hold
  468 ceremonial, 366 policy-debate and 67 bill-introduction speeches.
- **The robustness check [A1] left open:** gap and trends on speeches labelled
  policy_debate only. Same course and same 2001–2016 answer to RQ2
  (Republicans right in both chambers, Democrats not clearly left), with a
  larger gap throughout.
- **Within topic:** gap per topic block in two periods, **107th–114th vs
  115th–119th**. The split falls after the mixed-source 114th, so "period"
  also means "source"; a finer split leaves governance and "other" cells
  with too few speeches (they already hold 16–73 per party and period).
  Both readings hold: the gap widens within blocks, and the agenda moves
  toward the most divided ones (social-cultural, governance).

**Not done:** a decomposition of the gap's change into within-topic and
topic-mix parts, with intervals. **All of it rests on labels not yet
hand-checked ([L4]).**

**Rejected:** dropping non-policy speeches from the main figures — the
exclusion rule is open ([O12]), so they stay in sections 2–9 and section 10
shows the policy-only version beside them.

---

### A3 — The overview leaves out procedural and ceremonial speeches by label; members and topics over time
**Date:** 2026-10-19 · **Status:** active · **Decided by:** Justus (exclusion by
label, procedural and ceremonial; members over time; topic polarization); the
criteria below by the agent · settles [O12] for the overview analysis

`code/notebooks/06_overview.ipynb`, on the `d100k` run ([S19], [P8]) and its
labels ([L8]). Figures `results/plots/overview_*.png`, tables
`results/metrics/overview_*_20261018T122536Z.csv`.

- **Exclusion:** speeches labelled `procedural` or `ceremonial` are left out
  (`EXCLUDED_SPEECH_TYPES` in `config.py`, `analysis.substantive`), and so
  are the 29 unlabelled ones; 71,121 of 99,993 scored speeches remain. Bill
  introductions, constituent items and "other" stay in. **Rejected:** the
  "every model scored 0.0" proxy — with one model it also drops 19% of the
  policy debates ([O12], [P8]); keeping all speeches — the left-out share
  grows in the House after 2016 (L7), so an all-speech gap understates the
  widening (the difference grows from 0.10 to 0.28 in the House).
- **Trend of the gap:** `analysis.gap_trend`, OLS of the speech score on
  party × Congress, clustered by member; the interaction is the change in
  R − D per Congress. Reported for 2001–2016 and 2001–2025.
- **Members over time:** a member is followed in a Congress where they have
  **≥ 10 substantive speeches** (`member_trajectories`), and gets a trend if
  followed in **≥ 5 Congresses** (`member_trends`: 170 members). Each
  trajectory is shown beside the party's mean in the member's chamber; the
  trend is also taken relative to that mean (`slope_relative`) and set against
  the member's Nokken-Poole slope. Sixteen members are featured by ICPSR,
  chosen for coverage and range (leaders, moderates, firebrands, chamber
  switchers). **Rejected:** a lower threshold (single Congresses on 3–5
  speeches are noise at this spread); per-member clustering (one member is
  one cluster).
- **Topic polarization:** the gap per **topic block** and Congress (≥ 275
  speeches per party cell in the 05 period split; per Congress governance
  and "other" are thinner), and one trend per **topic** over 2001–2025
  (`gap_trend` by topic) — single topics hold as few as 6 speeches in a
  party cell per Congress, too few for a line per Congress.

---

### A4 — Age, generation and tenure in the overview (notebook 06, section 5)
**Date:** 2026-10-19 · **Status:** active · **Decided by:** Justus (asked whether
younger members polarize more); measures and design by the agent

**Data:** Voteview's `born` (birth year) and every Congress each ICPSR served,
from the raw `HSall_members.csv` back to 1789 (`analysis.member_background`);
99.91% of substantive `d100k` speeches have a birth year. Age = the speech's
year − `born` (±1). Groups in `config.py`: age `under 45 / 45–54 / 55–64 /
65–74 / 75+`; generations by birth year as Pew defines them; tenure
`1st Congress / 2–3 / 4–6 / 7–10 / 11+` Congresses served.

**Measures:** *extremity* = ideology toward the party's own side (D
sign-flipped, `toward_party`), and *tone*; both taken relative to the mean of
the speaker's party, chamber and Congress (`relative_to_cell`), so a group
difference is not the 2001–2025 trend; group means clustered by member; then
age and tenure together by OLS within those cells (`within_cell_ols`).

**Findings (preliminary):** younger is *not* further out (age differences
≤ ~0.05, mostly within intervals; House Democrats under 45 slightly less far
out). **Newer is further out for Republicans:** House Republicans in their
first three Congresses +0.05–0.06, with 11+ Congresses −0.09; −0.015 per
Congress served with age held fixed (p < 10⁻¹²); Senate Republicans in their
first Congress +0.09. Republican newcomers are also harsher in tone,
Democratic newcomers milder. Consistent with replacement ([A3], section 3).

**Not separable:** within a Congress, age and generation are the same thing;
long tenure selects re-elected members and leadership roles.

**Rejected:** raw group means without the cell adjustment (would mix the
trend over time with age, since the Senate aged by 11 years); exact
birthdays from congress-legislators (bioguide-keyed, would need a second join
for the Stanford side, for a ±1-year gain).

---

## Labels

A second pass that gives each speech a **speech type** and, for policy debate,
a **topic**. Added 2026-10-17. Code: `code/src/labeling.py`,
`code/src/label_eval.py`, `code/scripts/label_run.py`,
`make_label_check_sheet.py`, `evaluate_labels.py`; prompt
`code/src/prompts/label_speech.txt`.

### L1 — Speech labels: zero-shot LLM classification with a closed codebook, separate from scoring
**Date:** 2026-10-17 · **Status:** active · **Decided by:** Justus

Each scored speech gets two labels from one model call, reading only the speech
text (no party, name, date, chamber or Congress) and the whole speech ([M4]):

- `speech_type`: procedural, ceremonial, bill_introduction, constituent_local,
  policy_debate, other
- `topic`: one of 11 codes, **only** for policy_debate; null otherwise

**What they are for:** to separate non-substantive speeches (procedural,
ceremonial, bill introductions, constituent items) from policy debate, and to
test whether the polarization trend survives controlling for what Congress talks
about. The topic mix shifts over time, and the govinfo Senate loses late-day
statements, tributes and bill introductions at the 2016-09-10 break ([O8]), so
the speech-type mix alone could fake a trend there.

**Labels are covariates, strata and exclusion filters — never a scoring
result.** The ideology and tone scores stay continuous ([M1]); a categorical
label used to filter or stratify them does not make the result binary, so M1
and the CLAUDE.md rule against binary final output are not touched.

**A separate pass, not part of the scoring prompt.** `score_speech.txt` is
unchanged, so the S8 scores stay comparable and the S8 run stays resumable
([S13]: the prompt is score-defining).

**Rejected:**
- *Topic models (BERTopic, LDA)* — topics are unstable across runs and seeds,
  have no fixed taxonomy to report or hand-check against, and need naming after
  the fact.
- *Keyword dictionaries* — brittle on floor speech, where procedural phrases
  ("I yield", "Mr. Speaker") open substantive speeches too and topics are
  named in many ways.
- *Adding the labels to the scoring prompt* — would change the scoring
  instrument and break comparability with every S8 score.
- *Fine-tuning a classifier* — needs a labelled training set we do not have
  ([O2]); the hand check ([L4]) is sized to validate, not to train.

---

### L2 — Codebook v1 and the topic blocks
**Date:** 2026-10-17 · **Status:** active · **Decided by:** Justus

The codebook is the prompt `code/src/prompts/label_speech.txt`, saved as given
(the brief's quoting indent removed). `CODEBOOK_VERSION = "v1"`; the label
strings are `SPEECH_TYPES` and `TOPICS` in `config.py`, and a test reads them
back out of the prompt so the two cannot drift. **Any change to the codebook
is a new version and a new entry here**, and a label run never resumes across
versions.

Topics roll up into five blocks (`TOPIC_BLOCKS` in `config.py`), applied after
the fact, so the map can change without relabelling:

| Block | Topics |
|---|---|
| economic | economy_tax_budget, welfare_labor, health, environment_energy |
| social_cultural | social_cultural, immigration, education |
| security_foreign | defense_foreign, law_crime |
| governance | government_process |
| other | other_policy |

**Judgement calls — state them in the methodology:**
- **Immigration under social_cultural**, although it is argued in economic
  terms (labour markets, fiscal cost) as often as cultural ones.
- **Health under economic**: insurance, Medicare/Medicaid and drug prices are
  mostly argued as spending and markets.
- **law_crime under security_foreign**, beside defence, rather than under
  social_cultural.
- **Guns under social_cultural** (in the prompt's topic list), although the
  Comparative Agendas Project files them under law and crime.
- **Voting rights split by framing:** voting rights as a civil-rights matter →
  social_cultural; election administration, campaign finance and redistricting
  → government_process. Expect coders and models to disagree most on this line.

**Risk to watch in the pilot:** the prompt's JSON example writes the topic as
`"<one of the eleven topics, or null>"`, with null inside the quotes. A model
may return the string `"null"` for non-policy speeches; under [L3] that is a
failure, not null. The pilot's failure count will show whether it happens.

---

### L3 — Labelling model: DeepSeek-V4.1-Flash, thinking disabled; GPT-4o only as the agreement check
**Date:** 2026-10-17 · **Status:** active · **Decided by:** Justus (model,
mode, temperature, cap); option and wiring checked by the agent

- **Labelling model:** `deepseek-flash` with **thinking disabled**.
  Labelling is preprocessing, and CLAUDE.md keeps reasoning models for the
  scoring step. This does not contradict [M3b]: the same model scores with
  thinking on and labels with it off. The option is DeepSeek's documented
  toggle (`api-docs.deepseek.com/guides/thinking_mode`, read 2026-10-17):
  `extra_body={"thinking": {"type": "disabled"}}`. `reasoning_effort` is **not
  sent**: the docs do not say what it does with thinking off. Pricing is the
  same in both modes ($0.30 / $1.20 per 1M at peak). Every row records
  `reasoning_tokens`, the evidence that thinking was actually off.
- **Separate spec:** `LABEL_MODEL_SPECS` in `labeling.py`, derived from the
  scoring specs with only the cap and request options changed. `MODEL_SPECS`,
  which scored S8, is untouched (a test checks it still has thinking on).
- **Agreement-check model:** `gpt-4o-2024-11-20`, JSON mode, only on the
  hand-check subsample ([L4]).
- **No temperature.** As in [S2a], for consistency with scoring. **Caveat for
  the methodology:** S2a's DeepSeek reason (the model ignores temperature) is
  true of *thinking* mode only; non-thinking DeepSeek and GPT-4o both accept a
  temperature. Leaving it unset is a choice here, not a constraint, and repeat
  runs will be noisier than a temperature-0 design.
- **Output cap 256** (`LABEL_MAX_OUTPUT_TOKENS`): two label strings and a reason
  of at most 15 words. **Same 300-second deadline and retry** as scoring ([S14]),
  reusing scoring.py's call, retry and JSON extraction.
- **Labels outside the codebook are failures** (the [S3] rule): an unknown
  type, wrong case, a topic on a non-policy speech, a policy speech without a
  topic, or the string "null". The tokens are still recorded. Nothing is mapped
  to a nearest code.
- **Cost (dry run, 2026-10-17):** 200 speeches $0.084, all 5,200 S8 speeches
  $2.24 (tiktoken input, 45 output tokens per speech assumed until the pilot
  measures it). GPT-4o on 200 for reference: $0.70.

**Snapshot note — record it in the methodology:** on 2026-10-17 DeepSeek
upgraded `deepseek-flash` to **DeepSeek-V4.1-Flash-1017** under the same name
(DeepSeek changelog, 2026-10-17). The S8 scores (2026-10-03) came from the
earlier V4.1-Flash; labels made from today on come from the 1017 snapshot. The
API returns `deepseek-flash` as the served model either way, so only
`system_fingerprint` can show the change (all 5,202 S8 DeepSeek rows share one
fingerprint, `aeb56401…`). Label rows record it, and the label summary flags
more than one fingerprint.

**Rejected:**
- *Thinking on* — a reasoning model for a preprocessing step. On the scoring
  prompt, thinking mode spent ~499 output tokens per speech ([S9]), against the
  45 assumed here for a non-thinking answer.
- *GPT-4o as the labelling model* — ~8× DeepSeek's cost for a step that does
  not need it; it stays as the independent check.
- *Temperature 0* — see above; decided against for consistency with scoring.

---

### L4 — How the labels are validated
**Date:** 2026-10-17 · **Status:** active · **Decided by:** Justus

1. **Pilot:** `label_run.py --from-run 20261003T110419Z --limit 200 --label s8`
   — 200 S8 speeches, 50 per chamber × source, seed 42. Run only after a
   human's go-ahead ([E7]).
2. **Hand check:** `make_label_check_sheet.py` draws 200 speeches (stratified
   by chamber, source and era, [L5]) into a blind CSV in
   `data/processed/label_check/`. The model's labels are not in it. **The first
   50 rows are double-coded.** Coders use the codebook's exact strings.
3. **Agreement** (`evaluate_labels.py`): accuracy, macro-F1 and Cohen's kappa,
   with confusion matrices, for speech type, topic and block —
   - model vs coder 1 (all coded rows),
   - coder 1 vs coder 2 (the 50; the CLAUDE.md manual-labelling rule asks for
     two coders and kappa),
   - DeepSeek vs GPT-4o on the same speeches (GPT-4o run on the sheet's ids,
     its own approval),
   - GPT-4o vs coder 1.
4. **Free consistency checks against S8** (no API): speech_type == procedural
   against the "both models scored ideology exactly 0.0" flag (1,599 of the
   5,200), mean |ideology| and tone by speech type and chamber, and label
   shares by source and by Congress with the 114th marked as mixed-source.
   Shares are sample shares (S8's equal quotas, [S11]), chambers apart ([O8]).

Each later step needs its own approval: labelling all 5,200, GPT-4o on the
check subsample, labelling any larger scoring run.

---

### L5 — Label runs on disk, and the judgement calls in the pipeline
**Date:** 2026-10-17 · **Status:** active · **Decided by:** agent (to confirm
by Justus)

**Run mechanics** (mirroring [S13]): a label run is one timestamp and **one
model**. `results/labels/label_<label>_<model>_<run>.jsonl` holds one row per
attempt, and `label_manifest_<label>_<run>.json` holds the codebook version
and labels, prompt path, hash and full text, corpus path and hash, per-model
settings and pricing, seed, and per stage the selection, id count and id-list
hash. `results/metrics/label_summary_<label>_<run>.json` holds counts, shares,
failures, tokens, cost, served models and fingerprints. `results/labels/` is
git-ignored like `results/scores/`. A resume labels only speeches without valid
labels and **refuses** if the codebook version, the codebook labels, the
prompt hash, the corpus hash or the model's settings changed, or if a new
selection drops speeches already labelled. A bare `--resume` repeats the last
selection and refuses if its ids changed (e.g. an edited ids file).
`label_run.py` has **no `--yes`**: spending needs a typed "yes" ([E7]).

**Judgement calls:**
- **`--limit` strata: chamber × source**, equal quotas, seed 42 (the brief).
  The pilot's 200 are 50 per cell; Congresses fall where they fall (8–23 each).
- **"Era" for the check sheet = presidential term** (`term` from
  `analysis.add_presidency`: Bush, Obama, Trump 2017–20, Biden, Trump 2025–26).
  The brief named era without defining it; terms are already in `config.py`
  and tie to RQ3.
- **The 50 double-coded rows are a stratified sub-draw** of the 200, placed
  first, with rows shuffled within each part so the order reveals no stratum.
- **Failed speeches stay eligible for the sheet**; excluding them would tilt
  the check toward speeches the model could label. They are listed separately
  in the evaluation.
- **Topic and block agreement are computed only where both raters said
  policy_debate**, so a disagreement on type is not counted twice.
- **Coders' codes are trimmed of whitespace; anything else that is not an exact
  codebook string stops the evaluation** and lists the rows (no silent drop).
- **Extra row fields:** `reasoning_tokens`, `system_fingerprint` and `stage`, in
  addition to the brief's list.
- **`--ids-file` takes a CSV's `speech_id` column**, so a check sheet can be fed
  straight back for the GPT-4o run.
- **Sheets are never overwritten**, and texts longer than Excel's
  32,767-character cell limit (11 of the 5,200) are also written out in full.

---

### L6 — Label pilot (run `20261017T182223Z`): 200 of 200 labelled, no failures
**Date:** 2026-10-17 · **Status:** finding, not a decision — shares
**indicative only, do not cite** (n = 200, sample shares)
**Data:** `results/metrics/label_summary_s8_20261017T182223Z.json`

`label_run.py --from-run 20261003T110419Z --limit 200 --label s8`: 50 speeches
per chamber × source, `deepseek-flash` with thinking off. Approved by Justus in
the session ([E7]).

- **Instrument:** 200 calls, **0 failures, 0 retries**. No string `"null"`
  topics, so [L2]'s risk did not show. 246,100 input tokens (tiktoken counted
  242,627, +1.4%) and 8,136 output tokens (40.7 per speech, against the 45
  assumed), **$0.084** (estimate $0.084). About 1 s per call.
- **Thinking was off:** 0 reasoning tokens on every call.
- **The model changed since S8:** every row carries system fingerprint
  `e72d3c7f…`; all S8 DeepSeek rows carry `aeb56401…`. This confirms the
  2026-10-17 switch to V4.1-Flash-1017 ([L3]), and that only the fingerprint
  shows it (`served_model` is `deepseek-flash` in both).
- **Speech types:** policy_debate 136 (68%), procedural 34 (17%), ceremonial 23
  (11.5%), bill_introduction 5, constituent_local 2, other 0. Procedural is
  25% in the Senate against 9% in the House. All 5 bill introductions are
  govinfo Senate speeches.
- **Topics** (of 136 policy debates): economy_tax_budget 28, defense_foreign 21,
  government_process 19, health 17, environment_energy 17, immigration 11,
  other_policy 7, social_cultural 5, welfare_labor 4, law_crime 4, education 3.
- **Not yet known:** whether the labels are right. That is the hand check
  ([L4]); the blind sheet is
  `data/processed/label_check/label_check_s8_20261017T182223Z.csv` (200 rows,
  first 50 double-coded).

---

### L7 — All 5,200 S8 speeches labelled (run `20261017T182223Z`, stage 2): 7 failures
**Date:** 2026-10-17 · **Status:** finding, not a decision — labels not yet
hand-checked ([L4]); shares are S8 sample shares ([S11])
**Data:** `results/metrics/label_summary_s8_20261017T182223Z.json`

Run by Justus as stage 2 of the pilot's run
(`label_run.py --resume 20261017T182223Z --from-run 20261003T110419Z`), so the
pilot's 200 were not relabelled. 5,000 calls, 18:37–20:04 UTC (1 h 27 min),
on a Saturday, so DeepSeek's off-peak rate applies.

- **Cost:** the whole run (both stages) is $2.24 at peak list price (6,603,748
  input, 212,070 output tokens; estimate $2.24); off-peak about half.
  Thinking stayed off (0 reasoning tokens). **One fingerprint (`e72d3c7f…`)
  across all 5,200 rows**: no model change mid-run.
- **Failures: 7 of 5,200 (0.13%)**, all recorded as nulls, none mapped ([L3]):
  four put a topic in the type field (`speech_type: "government_process"`),
  one invented the topic `agriculture` (other_policy in the codebook), and two
  returned broken JSON (curly quotes; an object cut off without a length
  stop). *Retried 2026-10-18 (stage 3, bare `--resume`, approved by Justus,
  $0.004):* 6 of 7 now have valid labels (all policy_debate: 4
  government_process, 1 welfare_labor, 1 other_policy). `1130107165` failed
  the same way again (`speech_type: "government_process"`) and stays
  unlabelled: **5,199 of 5,200 labelled**, the whole run $2.24 at peak list
  price.
- **Speech types (5,193 labelled):** policy_debate 3,491 (67.2%), procedural
  796 (15.3%), ceremonial 702 (13.5%), bill_introduction 187 (3.6%),
  constituent_local 16 (0.3%), other 1. constituent_local and other are
  close to unused.
- **Topics (3,491 policy debates):** economy_tax_budget 801 (22.9%),
  defense_foreign 568 (16.3%), health 424 (12.1%), government_process 376
  (10.8%), environment_energy 348 (10.0%), other_policy 260 (7.4%),
  social_cultural 187 (5.4%), immigration 162 (4.6%), welfare_labor 161
  (4.6%), law_crime 127 (3.6%), education 77 (2.2%). Blocks: economic 49.7%,
  security_foreign 19.9%, social_cultural 12.2%, governance 10.8%, other 7.4%.
- **Across the source break, chambers apart** (share of labelled speeches,
  Stanford → govinfo): the **Senate mix barely moves** (policy 58.7 → 59.4%,
  procedural 21.6 → 21.6%, ceremonial 14.4 → 12.7%, bill introductions
  4.9 → 6.1%). The **House shifts:** ceremonial 9.8 → 18.8%, policy
  78.2 → 71.1%, procedural 10.1 → 7.5%. Source coincides with era here
  (2001–16 vs 2016–25), so this cannot separate a real change from the
  govinfo extraction (D17). It is not the Senate composition shift [O8]
  feared, and it is a reason to control for speech type in the House trend.

---

### L8 — The `d100k` speeches labelled (label run `20261017T182223Z`, stage 4): 188 failures
**Date:** 2026-10-19 · **Status:** finding, not a decision — labels not yet
hand-checked ([L4]); shares are sample shares ([S11]) · run by Justus
**Data:** `results/metrics/label_summary_s8_20261017T182223Z.json`

The label run was extended to the speeches of scoring run `d100k`
(`20261018T122536Z`, [S19]) rather than started anew: its 5,200 S8 speeches lie
inside the new draw and were labelled by the same DeepSeek build, so they were
not paid for twice ([S13]'s rule). Hence the run's label stays `s8` though it
now covers all 99,996 `d100k` speeches; each stage's selection is in the
manifest.

- **Stage 4:** 94,797 calls (94,796 new speeches plus the one S8 failure,
  now labelled), 05:17–08:05 UTC on a weekday, so mostly at **peak** price;
  2.8 h at concurrency 10 ([S17]). Whole run: **$42.99** at peak list price
  (127.0M input, 4.1M output tokens; 40.7 output tokens per call, as in the
  pilot), stage 4 about $40.75 (estimate $40.76). One build throughout
  (`e72d3c7f…`), 0 reasoning tokens.
- **99,808 of 99,996 labelled; 188 failures (0.19%),** all kept as nulls
  ([L3]): 168 put a topic or an invented code in the type field
  (`government_process` 144, `procedure` 11, `defense_foreign` 7,
  `procedure_debate` 3, others 3), 8 invented a topic (`agriculture` 3,
  `foreign_policy`, `energy`, `labor`, a free-text one), 2 gave a topic to a
  non-policy type, 10 returned broken JSON. *Retried 2026-10-19 (stage 5, bare
  `--resume`, approved by Justus, $0.086): 159 of 188 now labelled; **29 stay
  unlabelled** (all a topic in the type field, mostly `government_process`):
  **99,967 of 99,996 labelled**, whole run $43.08 at peak list price.* **For a codebook v2:** most
  failures are a topic code written into the type field, a format slip a
  stricter answer format would prevent.
- **Speech types (99,808):** policy_debate 67,032 (67.2%), procedural 14,736
  (14.8%), ceremonial 14,075 (14.1%), bill_introduction 3,648 (3.7%),
  constituent_local 249 (0.25%), other 68. Senate procedural 20.3% vs House
  9.3%; policy debate 60.3% vs 74.0%.
- **Topics (67,032 policy debates):** economy_tax_budget 21.9%,
  defense_foreign 16.8%, health 11.9%, government_process 11.3%,
  environment_energy 9.8%, other_policy 7.4%, social_cultural 5.3%,
  immigration 4.5%, welfare_labor 4.5%, law_crime 3.8%, education 2.7%.
  Blocks: economic 48.1%, security_foreign 20.6%, social_cultural 12.5%,
  governance 11.3%, other 7.4%. All within about a point of the 5,200-speech
  S8 shares ([L7]).

---

## Open questions

Move these up into a numbered entry once decided.

### O12 — Should `speech_type == procedural` replace the "all models scored 0.0" proxy?
**Raised:** 2026-10-17 (from [L1]) · **To be decided by:** Justus, after the
hand check ([L4]) · **Settled 2026-10-19 for the overview analysis by [A3]**:
by label, procedural and ceremonial left out. Whether the hand check confirms
the labels is still open

The analysis flags a speech as procedural when every model scored its ideology
at exactly 0.0, as the scoring prompt asks for procedural speech (1,599 of the
5,200 S8 speeches; [A1] keeps them in every figure, and their exclusion is a
pending robustness check). Once the labels exist, the procedural label is a
direct alternative, and ceremonial, bill_introduction and constituent_local
are further candidates for exclusion. `evaluate_labels.py` cross-tabulates the
label against the flag. Which one defines the exclusion, and which speech types
it covers, is a methodological choice for Justus, not something the code
decides.

*Update 2026-10-19 ([P8]):* with one model the proxy is worse. In `d100k`,
where DeepSeek alone decides, a 0.0 ideology score flags 38,874 of 99,996
speeches: 91% of those labelled procedural, but also 76% of the ceremonial
ones and **19% of the policy debates** (12,907; 10% in S8, where both models
had to score 0.0).

### O13 — Arlen Specter's party is wrong in the Stanford data for 2005–2009
**Raised:** 2026-10-19 (from [A3]) · **Status:** open · **Affects:** about
1,100 corpus speeches, ~90 of them in `d100k` (after the exclusion)

The Stanford parquet codes Specter (PA, Senate) as a Democrat from the 109th
Congress (`speakerid` 109114511, 110114511; `party` and `party_original` both
`D`), but he left the Republican party only in April 2009, in the 111th. His
109th–110th speeches therefore sit in the Senate Democratic cells. [D23]
already flags him as a party switcher for the DW-NOMINATE join, but not this
misdating. **Options:** (a) a `PARTY_CORRECTIONS`-style override in the corpus
build — but it changes the corpus SHA-256, so no existing scoring or label run
could be resumed or extended ([S13], [L5]); (b) an override at analysis time,
keyed on the two speakerids; (c) leave it and state it. The 111th is mixed
(R until 2009-04-28) and would need the date. Not fixed; the notebooks state it.

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

**Measured 2026-10-03** (the 2026-09-23 pilot files): output length barely
follows speech length — Spearman 0.37 between word count and the reasoner's
output tokens. Of the 199 answers, 99% stay under 2,003 tokens, the maximum is
3,792, and the median rises only from 299 (shortest quarter of speeches) to 562
(longest). The one failure spent all 8,192 tokens on an unremarkable 732-word
speech: a runaway, not a long input. A higher cap costs little (billing is on
tokens used, ≤ ~$0.07 per runaway at 32,768), but whether it would rescue a
runaway is unknown. **Decision (Justus, 2026-10-03): keep 8,192** and see what
stage 1 of the S8 run ([S13]) shows; changing it later means a new run, because
the cap is a score-defining setting.

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

**Update 2026-10-03 — the S8 draw supports a member-level check after all**,
at career level. P5 expected it not to, because S8 balances cells, not
members; but frequent speakers recur. In the 5,200-speech draw, 5,196 speeches
are `in_validation`, from 1,010 members (ICPSR): **297 members have ≥ 5
speeches** (3,793 speeches) and 131 have ≥ 10. Per member × Congress — what
`nokken_poole_dim1` needs — it is thin: 160 member-Congresses with ≥ 5, 502
with ≥ 3. Caveat: members with many sampled speeches are the frequent
speakers, not a random draw of members.

### O11 — What would justify extending the S8 run?
**Raised:** 2026-10-03 (from [S13]) · **To be decided:** during the analysis ·
**Addressed 2026-10-18 by [S19]**: a new ~100,000-speech DeepSeek-only run
rather than an extension of S8 (the model changed, [S18]); its rationale for
the size is proposed there and awaits Justus's confirmation

S13 makes extending cheap and waste-free, which is exactly why the trigger
should be written down before the results are read: otherwise a second run is a
reaction to numbers someone disliked. Candidates: party × Congress means too
imprecise to tell whether each party moved (RQ2); too few members with several
speeches for the member-level validation (O9); a chamber stratum too noisy for
the separate-chamber trends O8 requires.

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
