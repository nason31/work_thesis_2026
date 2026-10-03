# govinfo data reality — what the file actually contains

**Date:** 2026-09-27 · **File:** `data/raw/govinfo/congress_speeches_2016_present.jsonl`
(572,540,799 bytes, 225,564 rows; sha256 `822d1c80…`)
· **Built by:** Konsti's `02_govinfo_dataset.ipynb` (not in the repo at the time)

> **Superseded file, 2026-10-03.** This note describes the original file. It
> was replaced by Konsti's 2026-10-02 re-fetch (276,513 rows, sha256
> `58bcc59b…`), which follows the API's pagination and adds a `state` column.
> The missing-state problem below is settled by `docs/decisions.md` D25 and the
> Senate fetch cap by the re-fetch (O8, updated). The text problems (D17) and
> the chamber-blind party lookup (D16) are unchanged in the new file. The
> notebook is now in the repo at `code/notebooks/02_govinfo_dataset.ipynb`.

Measured against the file, not the README. The README on `main`
(`data/raw/govinfo/README.md`) is accurate about coverage and columns, but its
explanation of the missing party labels is wrong, and it does not mention the
text problems below. Decisions taken in response: `docs/decisions.md` D15–D18, O7.

## Shape

| | |
|---|---|
| Columns | `date`, `speaker`, `party`, `icpsr`, `chamber`, `speech` — no id, state, congress or word count |
| Dates | 2016-09-12 → 2026-09-24. Stanford ends 2016-09-09: **no gap, no overlap** |
| `party` | Democrat 96,658 · Republican 96,612 · null 31,302 · Independent 972 · Libertarian 20 |
| `chamber` | HOUSE 181,030 · SENATE 44,534 |
| Duplicates | 0 on (date, chamber, speaker, speech). 13,996 repeated *texts* are boilerplate ("I demand the yeas and nays") said on different occasions |

October 2016 has no rows. That is the pre-election recess, not a hole —
Stanford has none for October 2006, 2010 and 2014 either.

## Four upstream defects

**1. Party lookup ignores chamber.** A surname counted as ambiguous if anyone in
Congress shared it. Every Senate speech by Mike Lee (518/518), Sherrod Brown
(401/401), Chris Murphy (287/287), Gary Peters, Ron Johnson and Tina Smith has no
party, although each was the only senator of that name. The README attributes
the nulls to the Record not disambiguating; for these it does not need to.

**2. The state is thrown away.** The Record writes `Mr. SMITH of Texas.` when
surnames collide. No `speaker` value contains " of ", so the parser drops the
state — which is why SMITH (2,819), THOMPSON, CARTER, SCOTT, JOHNSON … are
unresolved in the House.

**3. Speeches split only at member turns.** Officer turns are glued onto the
preceding speech: SPEAKER pro tempore 36,896, PRESIDING OFFICER 31,148, Acting
CHAIR 18,073 occurrences. Record narration too, and with it whole bill texts —
the largest single case is 29,474 words of H.R. 1301 attached to a 39-word "I
call up the bill". (Member turns *are* split correctly: 0 member headers inside
speeches.)

**4. Markup left in.** `</pre></body></html>` on 56,374 rows, `[[Page S3236]]`
markers on 49,760, centered section headings, `{time} 1430` stamps. Stanford has
none of these (0 page markers; officer text on 821 of 823,341 rows).

Also: labels are end-of-term, so mid-term switchers are wrong for part of their
term (Van Drew labelled R through 2019; Mitchell and Amash labelled Independent
for their Republican months). And the 2020–21 impeachment trials put House
managers, counsel and video clips under `SENATE` — "Counsel CASTOR" (Steve
Castor, staff) is labelled Democrat, matched to Rep. Kathy Castor.

## What the build repairs

`make govinfo` (`code/src/govinfo.py`) fixes 1, 3 and 4 and the switchers, and
counts everything in `results/metrics/govinfo_build_stats.json`:

| | rows |
|---|---|
| read | 225,564 |
| after 2025-12-31 (out of scope) | −15,283 |
| speaker unresolved (23,075 ambiguous, 1,687 no member) | −24,762 |
| non-voting delegates | −1,602 |
| excluded (Mitchell, Amash while independent) | −9 |
| under 50 words **after** cleaning | −71,822 |
| **written** | **112,086** (D 57,278 · R 54,808) |

Cleaning removes 19.5% of words. The 50-word filter drops far more here than it
would on the raw text, because the officer text and bill text no longer count
toward the length.

## What it cannot repair

Defect 2. The 23,075 ambiguous House rows lean Republican (~13,900 R vs ~9,100 D
by candidate weight), so the processed file under-represents House Republicans.
The fix is upstream: keep `of <State>` and re-fetch. See O7.

## Volume at the seam

Written speeches per year: 2017 15,048, 2018–2019 ≈ 13.5–13.9k, 2020–2025
≈ 10.6–11.8k. The Stanford 114th has 36,661 over ~20 months (Jan 2015 – Sep
2016), ≈ 22k per year — so **govinfo yields about a third fewer speeches per year
at the seam**. Before cleaning the two looked similar; the difference was hidden
by officer and bill text inflating govinfo word counts past the 50-word filter.
This is the discontinuity CLAUDE.md warns about, now measured: compare
`congress_counts` in both stats files before reading any trend across 2017.
