# Stanford Congressional Record Dataset (2001–2017)

## What this is

Congress floor speeches (House and Senate), from the Gentzkow, Shapiro
& Taddy Congressional Record dataset published by Stanford. This is
the first half of the combined dataset, covering sessions 107–114
(2001 through the confirmed end date below).

## Where the actual file lives

Not committed to this repo (per the `data/raw/` policy — this folder
stays read-only and git-ignored for actual data files). The dataset is
produced and saved by `01_stanford_dataset.ipynb`.

- **Working copy** (written automatically by the notebook):
  https://drive.google.com/drive/u/1/folders/1Eqq2K7dM9gFAldVEVSAZh_vLs9dOkrEB
  File: `congress_speeches_2001_2017.parquet`
- **Shared copy** (promoted manually once validated, for use by
  collaborators):
  https://drive.google.com/drive/u/1/folders/1VLrhAbY5-09EoppbqigAaMcxbXNQacRs
  File: `congress_speeches_2001_2017.parquet`

## Time period covered

107th–114th Congress: **2001-01-03 to 2016-09-09** (confirmed against
both the processed output and the raw session 114 source file — see
`01_stanford_dataset.ipynb`, Section 4). This is earlier than
Stanford's documentation states (sessions 97–114, 1981–2017); the
released data for session 114 stops several months before the
session's actual close. This gap is closed by the govinfo dataset's
gap-filling period (September 10 – December 31, 2016; see
`data/raw/govinfo/README.md`).

## Columns

Documented columns:

| Column | Description |
|---|---|
| `speech_id` | Unique speech identifier |
| `speech` | Full text of the speech |
| `chamber` | HOUSE / SENATE |
| `date` | YYYYMMDD |
| `speaker` | Raw speaker tag text (e.g. "Mr. GREGG") - not a bare surname; see `last_name` |
| `last_name` | Speaker's cleanly parsed last name |
| `speaker_first_name` | Speaker's first name, pulled from the SpeakerMap file - the reliable field for first-name matching |
| `first_name` | An unpopulated placeholder column ("Unknown" for every row checked) - not usable; use `speaker_first_name` instead |
| `state` | Speaker's state (full name, e.g. "Illinois") |
| `gender` | Speaker's gender |
| `word_count` | Word count of the speech |
| `speakerid` | Stanford-internal speaker identifier (not a cross-dataset ID) |
| `party` | See "Party values" below |
| `congress` | Congress session number (107–114) |
| `icpsr` | ICPSR identifier, resolved against `congress-legislators` (see Method summary) |
| `party_crosscheck` | Independently-derived party value, for comparison against Stanford's own `party` column |

The raw session files also carry additional, currently unused columns:
`number_within_file`, `line_start`, `line_end`, `file`, `char_count`,
`state_map`, `chamber_map`. These pass through unchanged in the saved
file.

**Party values are not limited to D/R/I.** A confirmed run of the full
dataset showed: D (441,536), R (377,021), I (4,730), P (37), A (17).
The rarer codes (P, A) correspond to minor historical parties. Any
downstream analysis that assumes only three party values, or that maps
party codes to full names, should account for these.

## Method summary

Downloaded and merged from Stanford's `hein-daily.zip` (speeches,
descr, and SpeakerMap files per session), sessions 107 through 114.
Speeches with no matched speaker (mostly procedural entries) are
dropped.

ICPSR and `party_crosscheck` are resolved against the same
`congress-legislators` project used for the govinfo dataset
(github.com/unitedstates/congress-legislators), matching on last name,
first name, and date (which term, if any, covers the speech date),
with state used as an additional tie-breaker when a last name is
shared by multiple sitting members. Full processing steps are
documented in `01_stanford_dataset.ipynb`, Section 5.

## Known limitations

- **Actual coverage ends 2016-09-09**, earlier than Stanford's
  documentation states. Closed by the govinfo dataset's gap-filling
  period; see "Time period covered" above.
- Encoding is latin-1 (historical OCR-derived text), not UTF-8.
- **`icpsr` and `party_crosscheck` are `null`** where no legislator
  could be matched unambiguously - most commonly when a surname is
  shared by multiple sitting members and neither first name nor state
  disambiguate it.
- **`party_crosscheck` disagrees with Stanford's own `party` column in
  90 of 810,632 rows where both are available (99.99% agreement)**,
  confirmed concentrated in two individuals rather than spread across
  the dataset:
  - **Acevedo-Vilá** (53 rows), Puerto Rico's non-voting Resident
    Commissioner (2001–2005): non-voting territorial delegates can
    have complex or non-standard party representations in the
    reference data; one specific term entry appears to contain a data
    quality error (a state abbreviation in the party field) in the
    upstream reference source itself.
  - **Crenshaw** (34 rows): Stanford's own `party` column records "I"
    (Independent) throughout, despite him being a Republican for his
    entire tenure - this looks like a genuine pre-existing error in
    Stanford's own party coding, correctly surfaced by the crosscheck.
  - A small remainder (3 rows, a "Lee" from New York) was not further
    investigated given how small a share of the total it represents.

## Validation

A confirmed run produced 823,341 rows with 0 duplicate speech IDs,
97.7% ICPSR coverage, and the party-agreement rate described above;
see `03_validation.ipynb`, Section 2, for the full set of checks
(session coverage, party distribution, date range, duplicate
detection, ICPSR/crosscheck, and manual spot-checks of sample rows).
