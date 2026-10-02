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

## Columns

| Column | Description |
|---|---|
| `date` | YYYY-MM-DD |
| `speaker` | Extracted last name (or full name where needed for disambiguation) |
| `party` | Democrat / Republican / null (see limitations below) |
| `icpsr` | ICPSR identifier for the speaker, where resolved / null |
| `state` | State named alongside the speaker in the source text (e.g. "Texas" in "Mr. SMITH of Texas."), where present / null |
| `chamber` | HOUSE / SENATE |
| `speech` | Full text of the speech |

## Method summary

- Speaker names extracted via pattern-matching on the Congressional
  Record's "Mr./Ms./Mrs. NAME." convention (see `02_govinfo_dataset.ipynb`,
  Section 2, for the full set of edge cases this handles), including
  the state named alongside a speaker when present.
- Party and ICPSR both assigned via date-aware lookup against the
  `congress-legislators` project
  (github.com/unitedstates/congress-legislators), trying first+last
  name, then full compound surname, then last name alone, with the
  captured state used as an additional tie-breaker when a name match
  is still ambiguous (Section 3).
- A full day's document listing is fetched completely, following the
  govinfo API's pagination rather than stopping at the first page
  (Section 4) — see "Known limitations (resolved)" below.
- Speeches under 50 characters (procedural fragments) or over 30,000
  words (likely mis-extractions or bill-text contamination) excluded.
- "Extensions of Remarks" excluded, for consistency with the Stanford
  dataset's methodology.
- Fetching from govinfo.gov retries on both rate-limit (429) responses
  and connection-level failures (timeouts, dropped connections), up to
  8 attempts with increasing delay.

## Validation

A confirmed run of the complete dataset produced:

- **276,513 total records, 0 exact duplicates.**
- **96.5% party-match rate, 86.4% ICPSR-match rate.**
- **Senate share of speeches, by year: 27.9%–41.0%**, consistent with
  Stanford's own declining trend over its final sessions (Stanford's
  last session, 114, shows 35.7%; govinfo's first year, 2016, shows
  35.2% — a close match at the handover point).
- Longest speech: 29,910 words (at the `MAX_SPEECH_WORDS` cap, as
  expected).
- Two representative weeks within the September–December 2016
  gap-filling period: 98.0% and 99.0% party-match rates respectively.

See `03_validation.ipynb` for the full set of checks: per-year record
counts, duplicate detection, party/ICPSR/state-match rates, chamber
balance, speech length distribution, gap-period-specific checks, and
manual spot-checks of random speech text for coherence and readability.

## Known limitations

- **A meaningful share of speeches have no party or ICPSR assigned**
  (`party` and `icpsr` are `null`). This is by design: when a speaker's
  surname is shared by multiple members serving at the same time, and
  neither the source text nor a captured state disambiguate it, no
  party or ICPSR is guessed.
- **ICPSR coverage is somewhat lower than the party-match rate.** ICPSR
  identifiers are assigned only to members who cast recorded floor
  votes; non-voting territorial delegates structurally never receive
  one, regardless of how reliably their name resolves to a party.
- **Rare mid-term party switches are not reflected.** The underlying
  legislator data records party per full term, not per day.
- **The most recent year is partial.** See "Time period covered" above.
- **A small number of records have trailing HTML markup** (e.g.
  `</pre></body></html>`) appended at the end of the speech text,
  apparently left over at a document boundary not fully cleaned by the
  extraction pattern. Appears limited to trailing boilerplate at the
  end of a document's final speech, not substantive content; not
  currently addressed.

## Known limitations (resolved in this version, documented for history)

- **A prior version of this pipeline requested only the first page
  (up to 100 documents) of each day's document listing**, silently
  dropping anything beyond it. Since the Congressional Record lists
  House documents before Senate documents within a day's package, this
  disproportionately affected the Senate — confirmed, before the fix,
  as a Senate share of only 11.5%–30.8% per year, well below Stanford's
  own historical trend. Fixed by following the API's pagination fully
  (`02_govinfo_dataset.ipynb`, Section 4); see "Validation" above for
  the confirmed, corrected figures.
