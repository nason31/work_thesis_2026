# What the Stanford parquet actually contains (2026-09-21)

**The dataset is provisional** — a starting point for building the pipeline, not
the final corpus. Every figure here was measured against one specific file
(681,029,375 bytes; `source_sha256` in
`results/metrics/corpus_build_stats.json`). If Konsti rebuilds or replaces it,
re-run `make smoke` and `make corpus` and re-check these numbers before quoting
any of them. What should survive a swap is the *method*: key on `speakerid`, use
the `_map` columns, and let the build fail loudly on anything new.

Written after the first real build. The earlier note
(`2026-09-15_stanford_dataset.md`) and CLAUDE.md's **[verified]** Stanford
section describe the file from the build recipe rather than from the file, and
several of those descriptions are wrong. This note records what 823,341 rows
actually look like. Where the two disagree, this one was measured.

## The schema has 20 columns, not 12

Undocumented until now: `number_within_file`, `last_name`, `line_start`,
`line_end`, `file`, `char_count`, `state_map`, `chamber_map`.

That matters because **the file carries both a dirty and a clean version of the
speaker fields**, and the documented 12 are the dirty ones:

| Documented (dirty) | Actual quality | Use instead |
|---|---|---|
| `speaker` | honorific + OCR damage on **99.7%** of rows | `last_name` |
| `state` | literal string `"Unknown"` on many rows | `state_map` |
| `first_name` | `"Unknown"` on **97.2%** of rows | — (dropped) |
| `chamber` | 97 nulls | `chamber_map` |

Examples of the damage in `speaker`, all the same senator:
`Mr. JEFFORDS`, `LR. JEFFORDS`, `Mr. -JEFFORDS`, `Mr.. JEFFORDS`, `Mr. JEFFORD`.

## Even `last_name` is not clean — key on `speakerid`

`last_name` has no nulls and no `"Unknown"`, which made it look safe. It is not:
**30.6% of speakerids have more than one spelling of the same name**
(`JEFFORD`/`JEFFORDS`, `LIEBERMAN`/`LIEBERMANN`/`LIEDERMAN`/`LISBERMAN`,
`SANDERS`/`SANDERR`/`SA.NDERS`).

`speakerid` is exact, and `state_map` is consistent within **every** speakerid
(0 exceptions in 823,341 rows). So all membership rules in `config.py` are
keyed on `speakerid`. Note it encodes the congress in its first three digits,
so a member serving several congresses has one id per congress — the rules need
one entry each.

## Other dtype and value surprises

- `word_count` and `date` are **strings**, not integers.
- `party` has **five** values, not three: `D`, `R`, `I`, plus `A` (17 rows) and
  `P` (37 rows). Both of the latter belong solely to Acevedo-Vilá, Resident
  Commissioner of Puerto Rico.
- The corpus is **823,341 rows** — not the "millions of speeches" CLAUDE.md's
  cost-awareness section assumes. That changes LLM cost estimates by roughly an
  order of magnitude.

## Coverage ends 2016-09-09, not 2017-01-03

The single most consequential correction. Measured over **all raw rows**, before
any filtering:

```
RAW date range: 2001-01-03 .. 2016-09-09
114th Congress: 2015-01-06 .. 2016-09-09   <- incomplete
```

CLAUDE.md states coverage runs "January 2001 through **January 3, 2017** (end of
the 114th Congress)" and calls the break a clean Congress boundary. It is not.
The 114th Congress ran to January 3, 2017, but the data stops nearly four months
early.

**Consequence:** if the govinfo pipeline starts at the 115th Congress as
planned, there is a **~4 month hole from 2016-09-10 to 2017-01-02** — which
covers the run-up to the November 2016 election, directly relevant to RQ4
("changes around election periods"). Either govinfo starts at 2016-09-10 instead
of the Congress boundary, or the gap is documented as a known limitation.
**This is an open decision.**

## The independents, enumerated

19 speakerids carry `party == "I"`, resolving to 7 people. One (Sablan) is a
delegate and drops out before the party rules apply:

| Member | Speeches | Rule |
|---|---|---|
| Sanders (VT, House 107–109, Senate 110–114) | 2,164 | caucus → D |
| Lieberman (CT, Senate 110–112) | 1,255 | caucus → D |
| Jeffords (VT, Senate 107–109) | 1,004 | caucus → D |
| King (ME, Senate 113–114) | 238 | caucus → D |
| Crenshaw (FL, House 107) | 34 | **corrected → R** |
| Sablan (MP, House 114) | 29 | dropped as delegate |
| Barkley (MN, Senate 107) | 6 | dropped, caucused with neither |

Crenshaw is a source error: Ander Crenshaw represented FL-4 as a Republican for
his entire career (2001–2017). Barkley was appointed for about two months after
Wellstone's death and caucused with neither party, so no assignment is
defensible.

## Non-voting delegates are dropped corpus-wide

5,665 rows (0.69%) from DC, PR, VI, GU, AS, MP. They cannot vote on final
passage, so DW-NOMINATE — the validation anchor — does not score them
comparably, leaving their speeches with no benchmark. Dropping them also removes
the stray `A`/`P` party codes.

## First full build

```
read        823,341 raw rows
dropped       5,665 non-voting delegates
dropped           6 excluded members (Barkley)
dropped     390,952 under 50 words
dropped           0 empty text
dropped           0 duplicate speech_id
written     426,718 rows
party       D 232,498 | R 194,220
chamber     H 249,286 | S 177,432
date range  2001-01-03 .. 2016-09-09
```

41 seconds, peak RSS 1.66 GB. `speech_id` is unique across all 823,341 rows —
zero duplicates, so that documented property holds.

The source `word_count` agrees with a recomputed whitespace token count on
**100%** of retained rows, so it can be trusted for filtering.

Per-congress counts are in `results/metrics/corpus_build_stats.json`. Note the
114th is the smallest (36,661) partly because its coverage stops in September
2016.

## Still not resolved

CLAUDE.md's `[assumed]` mark on *"speeches with no matched speaker were dropped
— how many"* stands. Those rows were removed by the upstream build and are not
in this parquet, so no amount of analysis here recovers the count. Only the
missing `code/scripts/build_stanford_dataset.py` can close it.
