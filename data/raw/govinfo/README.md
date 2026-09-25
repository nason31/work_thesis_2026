# govinfo.gov Dataset — Congressional Record Speeches (2017–present)

## What this is

Congress floor speeches (House and Senate), extracted from the
Congressional Record via the official govinfo.gov API. This is the
second half of the combined dataset, covering the period the Stanford
dataset does not (see `data/raw/stanford/README.md`): 2017 onward.

## Where the actual file lives

Not committed to this repo (per the `data/raw/` policy — this folder
stays read-only and git-ignored for actual data files). The dataset is
produced and saved by `02_govinfo_dataset.ipynb`.

- **Working copy** (written automatically by the notebook):
  `Work Project 2026: Datasets (raw)/congress_speeches_2017_present.jsonl`
- **Shared copy** (promoted manually once validated, for use by
  collaborators):
  `Work Project 2026: Datasets (shared)/congress_speeches_2017_present.jsonl`

## Time period covered

January 1, 2017 onward. The most recent year at any given time should
be treated as partial (still in progress at the time of collection)
rather than a complete year, in any year-over-year analysis.

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
  name/state given), no party or ICPSR is guessed. The exact current
  rate is checked in `03_validation.ipynb` and should be recorded here
  after each full collection run.
- **Rare mid-term party switches are not reflected.** The underlying
  legislator data records party per full term, not per day, so a
  member who changed party mid-term (a rare event) shows their
  end-of-term party for their whole term.
- **The most recent year is partial.** See "Time period covered" above.
- Prior versions of this dataset showed small, unexplained gaps
  between the number of records reported during collection and the
  number ultimately saved, traced to a Drive write-sync timing issue.
  The current pipeline (Section 9 of `02_govinfo_dataset.ipynb`)
  automatically records any day affected by a collection-time warning
  and reprocesses it before the dataset is considered final; this
  should be re-confirmed via `03_validation.ipynb` after each full run
  rather than assumed resolved.

## Validation

See `03_validation.ipynb` for the full set of checks run against this
file: per-year record counts, duplicate detection, party/ICPSR-match
rate, speech length distribution, and manual spot-checks of random
speech text for coherence and readability.
