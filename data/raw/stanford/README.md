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
| `speaker` | Speaker's last name |
| `first_name` | Speaker's first name |
| `state` | Speaker's state |
| `gender` | Speaker's gender |
| `word_count` | Word count of the speech |
| `speakerid` | Stanford-internal speaker identifier (not a cross-dataset ID) |
| `party` | See "Party values" below |
| `congress` | Congress session number (107–114) |

The raw session files also carry additional columns not listed above:
`number_within_file`, `last_name`, `line_start`, `line_end`, `file`,
`char_count`, `state_map`, `chamber_map`. These pass through unchanged
in the saved file; they are not currently used by any downstream step
but are available if needed.

**Party values are not limited to D/R/I.** A confirmed run of the full
dataset showed: D (441,536), R (377,021), I (4,730), P (37), A (17).
The rarer codes (P, A) correspond to minor historical parties. Any
downstream analysis that assumes only three party values, or that maps
party codes to full names, should account for these.

Note: this dataset does not include ICPSR identifiers — Stanford's
release does not provide one. The govinfo dataset (see
`data/raw/govinfo/README.md`) does include an ICPSR column where a
match could be resolved.

## Method summary

Downloaded and merged from Stanford's `hein-daily.zip` (speeches,
descr, and SpeakerMap files per session), sessions 107 through 114.
Speeches with no matched speaker (mostly procedural entries) are
dropped. Full processing steps are documented in
`01_stanford_dataset.ipynb`.

## Known limitations

- **Actual coverage ends 2016-09-09**, earlier than Stanford's
  documentation states. Closed by the govinfo dataset's gap-filling
  period; see "Time period covered" above.
- Encoding is latin-1 (historical OCR-derived text), not UTF-8.

## Validation

A confirmed run produced 823,341 rows with 0 duplicate speech IDs; see
`03_validation.ipynb`, Section 2, for the full set of checks (session
coverage, party distribution, date range, duplicate detection, and
manual spot-checks of sample rows).
