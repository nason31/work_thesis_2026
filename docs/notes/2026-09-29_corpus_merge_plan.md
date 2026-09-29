# Corpus merge — implementation plan

> Executed inline (native) in the session that wrote it. Steps use checkboxes.
> **No commits** — the result is reviewed together before anything is committed.
> **Status 2026-09-29:** all 8 tasks done; Task 8 ran before Task 7 so the docs quote measured figures.

**Goal:** one `data/processed/corpus.parquet` holding both sources (538,804
rows, 2001-01-03 → 2025-12-19), built by `make corpus`.

**Architecture:** a new `code/src/merge.py` reads the two processed files,
validates them in a metadata-only first pass, computes all statistics there,
then streams both into a temp file renamed onto the output. The Stanford build
moves to `corpus_stanford.parquet` so the merged file can take `corpus.parquet`.

**Tech stack:** Python 3.11, pyarrow 25, pytest, black/ruff.

**Spec:** [2026-09-29_corpus_merge_design.md](2026-09-29_corpus_merge_design.md)

## Global constraints

- All paths come from `code/src/config.py`; nothing hardcoded elsewhere.
- Type hints and docstrings on everything in `src/`.
- Never write to `data/raw/`.
- `make test` and `make lint` clean at the end.
- No commits.

## Review focus

Inputs the spec implies but a happy-path test would not exercise:

1. **Output path equal to an input** — must be refused before anything is
   written (the merge would replace a file it is still reading). Test in Task 2.
2. **A merge that fails with `overwrite=True`** — the previous `corpus.parquet`
   must survive and no `.tmp` file may remain. Test in Task 2.
3. **Old local layout** (Stanford data still at `corpus.parquet`, no
   `corpus_stanford.parquet`) — `make corpus` must fail naming `make stanford`
   and leave `corpus.parquet` untouched. Test in Task 3.
4. **An empty input file** — a named "no rows" error, not a crash in
   `min()`/`max()` over nothing. Test in Task 3.
5. **Nulls in key columns** (party, date, source) — a named error, not a
   `TypeError` from comparing `None`. Test in Task 3.

---

### Task 1: Stanford build writes to the Stanford paths

**Files:** modify `code/src/config.py`, `code/src/corpus.py`,
`code/scripts/build_corpus.py`, `Makefile`; test in `code/tests/test_corpus.py`.

**Produces:** `STANFORD_CORPUS_PATH`, `STANFORD_BUILD_STATS_PATH`,
`MERGED_BUILD_STATS_PATH` in config; `make stanford`.

- [x] Test: `build_stanford_corpus`'s `dst` default is `STANFORD_CORPUS_PATH`,
      `BuildStats.write`'s `path` default is `STANFORD_BUILD_STATS_PATH`, and
      `STANFORD_CORPUS_PATH != CORPUS_PATH`. Run → fails on import.
- [x] config: add `STANFORD_CORPUS_PATH = PROCESSED_DIR / "corpus_stanford.parquet"`;
      rewrite the `CORPUS_PATH` comment (merged, built by `make corpus`, D19);
      rename `BUILD_STATS_PATH` → `STANFORD_BUILD_STATS_PATH`
      (`stanford_build_stats.json`); add `MERGED_BUILD_STATS_PATH`
      (`merged_build_stats.json`); fix the `GOVINFO_CORPUS_PATH` comment
      ("merging … is a separate, later step"); update the sampling comment to
      2 × 13 × 2 = 52 cells / 5,200, smallest cell 1,128.
- [x] corpus.py / build_corpus.py: switch defaults and imports; the smoke
      redirect compares against `STANFORD_CORPUS_PATH`; update the module
      docstring's "2017 source break" paragraph and the `BuildStats` docstring.
- [x] Makefile: `stanford` target runs `build_corpus.py`; `smoke` unchanged;
      help comment lists `stanford`, `govinfo`, `corpus` (merge).
- [x] Run `make test` → all pass.

### Task 2: merge module — happy path and output safety

**Files:** create `code/src/merge.py`, `code/tests/test_merge.py`.

**Produces:**

```python
MERGED_SCHEMA: pa.Schema  # == GOVINFO_SCHEMA: 12 corpus columns + icpsr int32
@dataclass(frozen=True)
class MergeInput: source: str; path: Path; stats_path: Path; schema: pa.Schema; make_target: str
@dataclass
class MergedBuildStats: output_path, built_at, inputs, rows_written, rows_by_source,
    cell_counts, year_counts, seam, icpsr_missing_by_source, date_min, date_max
    def write(self, path: Path = MERGED_BUILD_STATS_PATH) -> Path
def merge_corpora(stanford=STANFORD_CORPUS_PATH, govinfo=GOVINFO_CORPUS_PATH,
                  dst=CORPUS_PATH, *, stanford_stats=STANFORD_BUILD_STATS_PATH,
                  govinfo_stats=GOVINFO_BUILD_STATS_PATH,
                  overwrite=False) -> MergedBuildStats
```

Fixtures: `_stanford_row`/`_govinfo_row` builders; default inputs mirror the
real seam — Stanford rows in the 113th and 114th (last 2016-09-09), govinfo
rows in the 114th (2016-09-12) and 115th; `_write_input` writes a parquet plus
a stats JSON `{"limit": null, "rows_written": n, "source_sha256": "raw-<name>"}`;
`_merge` writes both and merges.

- [x] Tests:
  - every row of both inputs is written, Stanford first (`to_pylist()` of the
    output equals Stanford rows + govinfo rows, with `icpsr=None` added to
    Stanford rows) — covers order, pass-through and null `icpsr`
  - output schema equals `MERGED_SCHEMA`, whose names are
    `CORPUS_SCHEMA.names + ["icpsr"]`
  - an existing output is kept without `overwrite` (`FileExistsError`) and
    replaced with it
  - output equal to an input → `ValueError` "own input" (review focus 1)
  - `_stream` monkeypatched to raise → the previous output survives, no
    `corpus.parquet.tmp` remains (review focus 2)
- [x] Run → fails (no module).
- [x] Implement: `_check_output`, `_read_metadata`, `_stream` (appends a null
      `icpsr` column to Stanford batches, rebuilds each batch against
      `MERGED_SCHEMA`), and `merge_corpora` writing to `<dst>.tmp`, deleting it
      on any exception, `replace()`-ing onto `dst` on success.
- [x] Run → pass.

### Task 3: input checks 1–8

**Files:** modify `code/src/merge.py`, `code/tests/test_merge.py`.

- [x] Tests (each asserts the message names the problem):
  - missing Stanford / govinfo input → `FileNotFoundError` naming
    `make stanford` / `make govinfo`; an existing `corpus.parquet` is left
    untouched even with `overwrite=True` (review focus 3)
  - a missing column is named (`missing last_name`); a wrong type is named
    (`icpsr is int64, expected int32`)
  - stats JSON absent → `FileNotFoundError` "build stats"; stats with
    `limit` set → "--limit"; `rows_written` ≠ file rows → "different builds";
    stats lacking the keys → "cannot vouch"
  - a govinfo row tagged `stanford` → "unexpected source"
  - a Stanford date of 2016-09-10, a govinfo date of 2016-09-09, a govinfo date
    of 2026-01-05 → each refused, message contains the date
  - a `speech_id` present in both inputs is named
  - party `I` / chamber `J` → "unexpected party/chamber … I/J"
  - a null party → "null value(s) in 'party'" (review focus 5)
  - Congresses 113, 114, 116 → names `[115]`
  - an empty govinfo input → "no rows" (review focus 4)
- [x] Run → fail.
- [x] Implement `_check_exists`, `_check_schema`, `_check_full_build` (returns
      the input's stats dict), `_check_values` (empty, nulls, source, party,
      chamber), `_check_seam` (returns the seam dict), `_check_unique_ids`,
      `_check_congress_coverage`; call them in pass 1 in that order, all before
      the temp file is opened.
- [x] Run → pass.

### Task 4: merge statistics

**Files:** modify `code/src/merge.py`, `code/tests/test_merge.py`.

- [x] Tests on the default fixture:
  - `cell_counts == {"113": {"stanford": {"H": {"R": 1}, "S": {"D": 1}}},
    "114": {"govinfo": {"H": {"D": 1}}, "stanford": {"S": {"D": 1}}},
    "115": {"govinfo": {"S": {"R": 1}}}}`
  - `year_counts == {"2013": {"stanford": {"H": 1, "S": 1}},
    "2016": {"govinfo": {"H": 1}, "stanford": {"S": 1}},
    "2017": {"govinfo": {"S": 1}}}`
  - `seam == {"stanford_last": "2016-09-09", "govinfo_first": "2016-09-12"}`;
    date range 2013-01-10 … 2017-06-05
  - `rows_by_source == {"stanford": 3, "govinfo": 2}`;
    `icpsr_missing_by_source == {"stanford": 3, "govinfo": 1}`
  - `inputs["stanford"]["sha256"]` equals `hashlib.sha256` of the processed
    file, `rows == 3`, `raw_source_sha256 == "raw-stanford"`
  - `write()` produces JSON with `rows_written == 5`
- [x] Run → fail.
- [x] Implement `_collect_stats` from the pass-1 metadata (Counters keyed by
      `(congress, source, chamber, party)` and `(year, source, chamber)`,
      nested by `_nest`), plus provenance via `corpus._fingerprint`.
- [x] Run → pass.

### Task 5: command line and `make corpus`

**Files:** create `code/scripts/build_merged_corpus.py`; modify `Makefile`;
tests in `code/tests/test_merge.py`.

- [x] Tests: `main([...tmp paths...])` returns 0 and writes the stats JSON;
      with the Stanford input deleted it returns 1 and prints `make stanford`
      to stderr.
- [x] Implement the script in the pattern of `build_govinfo_corpus.py`
      (`--stanford`, `--govinfo`, `--stanford-stats`, `--govinfo-stats`,
      `--output`, `--overwrite`, `--stats-path`); Makefile `corpus` target runs
      it with `--overwrite`.
- [x] Run → pass.

### Task 6: scoring run records which corpus it used

**Files:** modify `code/scripts/pilot_run.py`.

- [x] After sampling, compute `corpus_sha256 = _fingerprint(args.corpus)`,
      print it, and add it to the manifest. Update the docstrings (merged
      corpus, `make corpus`, row count). Verified by the `--dry-run` in Task 8,
      which prints the fingerprint.

### Task 7: documentation

**Files:** `CLAUDE.md`, `README.md`,
`docs/notes/2026-09-21_stanford_data_reality.md`, `docs/decisions.md`.

- [x] Replace `corpus_build_stats.json` → `stanford_build_stats.json` and the
      `make smoke` / `make corpus` sequence → `make smoke`, `make stanford`,
      `make corpus` wherever they describe the Stanford build.
- [x] CLAUDE.md: govinfo "Not yet merged" → merged (D19); Processed/Combined
      section names the merged build; DW-NOMINATE blocker points to the M5
      probe; repo tree lists `govinfo.py` and `merge.py`.
- [x] README: rewrite "Building the corpus" for the three targets.
- [x] decisions.md: add D19 (merge, with rejected alternatives and the
      `corpus.parquet` meaning change), O8 (Senate share, with the corrected
      House/Senate figures), a dated note under M5 (the 97.4% probe).

### Task 8: verify on real data (stop before commit)

- [x] Move `results/metrics/corpus_build_stats.json` →
      `stanford_build_stats.json`; run `build_corpus.py`; diff old vs new stats
      (only `built_at`/`output_path` may differ); compare the new
      `corpus_stanford.parquet`'s SHA-256 with the old `corpus.parquet`.
- [x] `make corpus` → 538,804 rows, Congresses 107–119, seam
      2016-09-09 / 2016-09-12.
- [x] Every `speech_id` in the 2026-09-23 pilot output resolves in the merged
      corpus.
- [x] `pilot_run.py --dry-run` → 52 cells, 5,200 speeches, cost estimate, no
      API calls.
- [x] `make test`, `make lint` clean.
- [x] Stop and review together. No commit.
