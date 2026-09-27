# govinfo.gov Dataset — Congressional Record Speeches (2016–present)

## What this is

Congress floor speeches (House and Senate), extracted from the
Congressional Record via the official govinfo.gov API. This is the
second half of the combined dataset, covering the period the Stanford
dataset does not (see `data/raw/stanford/README.md`): September 10,
2016 onward. The September 10 – December 31, 2016 portion closes the
gap found between Stanford's documented range and the actual end date
present in its released data (confirmed: September 9, 2016).

## Where the actual file lives

Not committed to this repo (per the `data/raw/` policy — this folder
stays read-only and git-ignored for actual data files). The dataset is
produced and saved by `02_govinfo_dataset.ipynb`.

- **Working copy** (written automatically by the notebook):
  https://drive.google.com/drive/u/1/folders/1Eqq2K7dM9gFAldVEVSAZh_vLs9dOkrEB
  File: `congress_speeches_2016_present.jsonl`
- **Shared copy** (promoted manually once validated, for use by
  collaborators):
  https://drive.google.com/drive/u/1/folders/1VLrhAbY5-09EoppbqigAaMcxbXNQacRs
  File: `congress_speeches_2016_present.jsonl`

## Time period covered

September 10, 2016 onward. The most recent year at any given time
should be treated as partial (still in progress at the time of
collection) rather than a complete year, in any year-over-year
analysis.

The September 10 – December 31, 2016 period was collected using a
pipeline validated only against 2017 and later data. A confirmed check
of two representative weeks within this period (see
`03_validation.ipynb`, Section 5) showed party-match rates of 81.0%
and 89.7% — the lower figure is within a plausible range given the
small sample size, but somewhat below the roughly 86% typically seen
elsewhere in this dataset, and worth re-checking on any future rerun
of this period.

## Columns

| Column | Description |
|---|---|
| `date` | YYYY-MM-DD |
| `speaker` | Extracted last name (or full name where needed for disambiguation) |
| `party` | Democrat / Republican / null (see limitations below) |
| `icpsr` | ICPSR identifier for the speaker, where resolved / null |
| `chamber` | HOUSE / SENATE |
| `speech` | Full text of the speech |

## Method summary

- Speaker names extracted via pattern-matching on the Congressional
  Record's "Mr./Ms./Mrs. NAME." convention (see `02_govinfo_dataset.ipynb`,
  Section 2, for the full set of edge cases this handles).
- Party and ICPSR both assigned via date-aware lookup against the
  `congress-legislators` project
  (github.com/unitedstates/congress-legislators), trying first+last
  name, then full compound surname, then last name alone (Section 3).
- Speeches under 50 characters (procedural fragments) or over 30,000
  words (likely mis-extractions or bill-text contamination) excluded.
- "Extensions of Remarks" excluded, for consistency with the Stanford
  dataset's methodology.
- Fetching from govinfo.gov retries on both rate-limit (429) responses
  and connection-level failures (timeouts, dropped connections), up to
  8 attempts with increasing delay (Section 4).

## Known limitations

- **A meaningful share of speeches have no party or ICPSR assigned**
  (`party` and `icpsr` are `null`). This is by design: when a speaker's
  surname is shared by multiple members serving at the same time, and
  the Congressional Record text itself doesn't disambiguate (no first
  name/state given), no party or ICPSR is guessed. A confirmed run of
  the complete dataset showed an 86.1% party-match rate.
- **ICPSR coverage (75.8% in a confirmed run) is meaningfully lower
  than the party-match rate, and this is expected.** ICPSR identifiers
  are assigned only to members who cast recorded floor votes;
  non-voting territorial delegates (e.g. the District of Columbia,
  American Samoa, the U.S. Virgin Islands) structurally never receive
  one, regardless of how reliably their name resolves to a party.
  These delegates are disproportionately active on the floor relative
  to their small share of all members (often their only effective
  legislative tool, since they cannot cast votes), which plausibly
  explains why the ICPSR gap is wider than the party-match gap. A
  speaker resolving to a party but not an ICPSR id should not, by
  itself, be treated as a resolution failure.
- **Rare mid-term party switches are not reflected.** The underlying
  legislator data records party per full term, not per day, so a
  member who changed party mid-term (a rare event) shows their
  end-of-term party for their whole term.
- **The most recent year is partial.** See "Time period covered" above.
- **The September–December 2016 gap-filling period** was collected
  with a pipeline validated only against 2017 onward; see "Time period
  covered" above for the confirmed match-rate check.

## Validation

A confirmed run of the complete dataset (September 2016 onward)
produced 225,564 total records, 0 exact duplicates, an 86.1%
party-match rate, and a longest speech of 29,910 words (under the
30,000-word cap). See `03_validation.ipynb` for the full set of
checks: per-year record counts, duplicate detection, party/ICPSR-match
rate, speech length distribution, gap-period-specific checks, and
manual spot-checks of random speech text for coherence and
readability.
