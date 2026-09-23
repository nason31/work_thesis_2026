# Stanford Congressional Record Dataset (2001–2017)

## What this is

Congress floor speeches (House and Senate), from the Gentzkow, Shapiro
& Taddy Congressional Record dataset published by Stanford. This is
the first half of our combined dataset, covering sessions 107–114
(roughly 2001 through early 2017 — see known limitations below for a
flagged discrepancy).

## Where the actual file lives

Not committed to this repo (per our data policy — `data/raw/` stays
read-only and git-ignored). The processed file is here:
https://drive.google.com/drive/u/1/folders/1Eqq2K7dM9gFAldVEVSAZh_vLs9dOkrEB
File: `congress_speeches_2001_2017.parquet` (~650 MB)

## Time period covered

107th–114th Congress. Stanford's own coverage is reported to end
January 3, 2017 (end of the 114th Congress) — though a team member
has separately reported the actual data may end around October 2016.
This discrepancy is flagged and not yet resolved (see limitations).

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
| `speakerid` | Unique speaker identifier |
| `party` | D / R / I |
| `congress` | Congress session number (107–114) |

## Method summary

Downloaded and merged from Stanford's `hein-daily.zip` (speeches,
descr, and SpeakerMap files per session), sessions 107 through 114.
Speeches with no matched speaker (mostly procedural entries) dropped.

## Known limitations

- **Reported end date discrepancy**: a team member reports the actual
  data may end around October 2016, not January 2017 as Stanford's
  documentation states. Not yet independently re-verified against the
  actual file's date range — flagged for follow-up.
- Encoding is latin-1 (historical OCR-derived text), not UTF-8.
