# Stanford Congressional Record Dataset (2001–2017)

## What this is

Congress floor speeches (House and Senate), from the Gentzkow, Shapiro
& Taddy Congressional Record dataset published by Stanford. This is
the first half of the combined dataset, covering sessions 107–114
(2001 through the actual end date confirmed below).

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

107th–114th Congress. Stanford's documentation describes the daily
edition as covering sessions 97–114 (1981–2017). A discrepancy has
been reported between this documented range and the actual latest date
present in the released data for session 114, possibly ending several
months earlier than January 2017. `01_stanford_dataset.ipynb` includes
a diagnostic step (Section 4) that checks the real date range directly
— both in the processed output and in the raw session 114 source file
— rather than relying on the documentation. Results of that check
should be recorded here once available.

## Columns

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
| `party` | D / R / I |
| `congress` | Congress session number (107–114) |

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

- **Possible end-date discrepancy**: see "Time period covered" above —
  not yet independently confirmed against the actual file.
- Encoding is latin-1 (historical OCR-derived text), not UTF-8.
