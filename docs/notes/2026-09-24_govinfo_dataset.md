# govinfo.gov Dataset — Congressional Record Speeches (2017–2026)

## What this is

Congress floor speeches (House and Senate), extracted from the
Congressional Record via the official govinfo.gov API. This is the
second half of our combined dataset, covering the period the Stanford
dataset does not (see `2026-09-15_stanford_dataset.md`): 2017 onward.

## Where the actual file lives

Not committed to this repo (per our data policy — `data/raw/` stays
read-only and git-ignored). The processed file is here:
https://drive.google.com/drive/u/1/folders/1Eqq2K7dM9gFAldVEVSAZh_vLs9dOkrEB
File: `congress_speeches_2017_2025.jsonl` (~530 MB)

## Time period covered

January 1, 2017 – September 17, 2026 (data collection date: September
23, 2026). 2026 is a partial year, since the year itself was still in
progress at the time of collection.

## Columns

| Column | Description |
|---|---|
| `date` | YYYY-MM-DD |
| `speaker` | Extracted last name (or full name where needed for disambiguation) |
| `party` | Democrat / Republican / null (see limitations below) |
| `chamber` | HOUSE / SENATE |
| `speech` | Full text of the speech |

## Method summary

- Speaker names extracted via pattern-matching on the Congressional
  Record's "Mr./Ms./Mrs. NAME." convention, refined over many rounds
  of testing (hyphenated names, middle initials, disambiguation by
  full name, procedural false positives, mid-sentence false positives).
- Party assigned via date-aware lookup against the `congress-legislators`
  project (github.com/unitedstates/congress-legislators), trying
  first+last name, then full compound surname, then last name alone.
- Speeches under 50 characters (procedural fragments) or over 30,000
  words (likely mis-extractions or bill-text contamination) excluded.
- "Extensions of Remarks" excluded, for consistency with the Stanford
  dataset's methodology.

## Known limitations

- **~13.9% of speeches have no party assigned** (`party` is `null`).
  This is by design: when a speaker's surname is shared by multiple
  members serving at the same time, and the Congressional Record text
  itself doesn't disambiguate (no first name/state given), we
  deliberately don't guess rather than risk an incorrect assignment.
- **Rare mid-term party switches are not reflected.** The underlying
  legislator data records party per full term, not per day, so a
  member who changed party mid-term (a rare event) shows their
  end-of-term party for their whole term.
- **Three years have small unexplained gaps** between originally
  reported and finally saved record counts, traced to a Drive write-
  sync timing issue found during collection (not a data extraction
  problem): 2017 (−439 records), 2024 (−325), 2026 (−4,058, the
  largest gap, on top of 2026 already being a partial year). Not yet
  re-collected — flagged for the next dataset rebuild.
- **2026 is a partial year** and should not be compared directly
  against full years in any year-over-year analysis without noting
  this.
- 3,568 exact duplicate records (from repair/refetch cycles during
  collection) were identified and removed before finalizing this file.

## Validation

Full dataset validated against: per-year record counts, duplicate
detection, party-match rate (86.1%, consistent across 20+ diverse
test weeks run before the full collection), speech length distribution
(max 29,910 words, under the 30,000 cap), and manual spot-checks of
random speech text for coherence and readability.
