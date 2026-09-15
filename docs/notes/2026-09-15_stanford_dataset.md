# Stanford dataset (2001–2017)

## What this is

Congressional Record speeches from the 107th through 114th Congress
(January 2001 – January 3, 2017), compiled by Gentzkow, Shapiro and
Taddy (Stanford Libraries). Each row is one individual speech, already
linked to the speaking member, their party, chamber, and the date.

## Where the actual file lives

Not committed to this repo (per our data policy — `data/raw/` stays
read-only and git-ignored). The processed file is here:

https://drive.google.com/drive/u/1/folders/1Eqq2K7dM9gFAldVEVSAZh_vLs9dOkrEB

File: `congress_speeches_2001_2017.parquet` (~650 MB)

## Columns

| Column | Description |
|---|---|
| `speech_id` | unique speech identifier |
| `speech` | full text of the speech |
| `chamber` | House or Senate |
| `date` | date of the speech (YYYYMMDD) |
| `speaker` | speaker's last name as recorded |
| `first_name` | speaker's first name |
| `state` | state represented |
| `gender` | as recorded in source data |
| `word_count` | length of the speech |
| `speakerid` | unique speaker identifier (Gentzkow dataset's own ID system) |
| `party` | D / R / I |
| `congress` | Congress session number (107–114) |

## How it was built

1. Downloaded `hein-daily.zip` from Stanford (data.stanford.edu/congress_text)
2. Loaded sessions 107 through 114 (one Congress = one two-year session)
3. Merged the raw speech text, metadata, and speaker-map files for each
   session on `speech_id`
4. Dropped speeches with no matched speaker (mostly procedural entries
   like "The Clerk" reading a bill title, not real member speeches)
5. Combined all sessions into one table, saved as parquet

The reproducible build script for this will be added to
`code/scripts/build_stanford_dataset.py`.

## Known limitation

Coverage ends January 3, 2017 (end of the 114th Congress), not December
2017. Everything from 2017 onward comes from our own govinfo.gov-based
pipeline instead — see the corresponding note in `docs/notes/` once
that's finalized. This is the "2017 source break" mentioned in
CLAUDE.md and README.md, and must be marked in every time-series plot.
