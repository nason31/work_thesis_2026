"""Merge the processed Stanford and govinfo sides into one corpus file.

The two sources are built separately (``corpus.py``, ``govinfo.py``) because
their raw data needs different repairs. This module only concatenates the
results. It drops nothing: every input row is written, or the merge fails. All
filtering, and all counting of what was filtered, belongs to the source builds.

The seam
--------
Stanford ends 2016-09-09 and govinfo starts 2016-09-12, inside the 114th
Congress (docs/decisions.md D15). Every row keeps its ``source`` tag, so the
break stays recoverable from the data itself.

member_id is source-native
--------------------------
Stanford rows carry Gentzkow's ``speakerid`` (unique per member *per
Congress*), govinfo rows the bioguide id. The formats cannot collide, but the
same person has different ids on either side of the break. Party-level
analysis is unaffected; member-level analysis across the break goes through
the member crosswalk (``crosswalk.py``, docs/decisions.md D21), a separate
table. ``icpsr`` stays null on Stanford rows, and on govinfo rows it is not the
DW-NOMINATE join key -- use the crosswalk for both.

Why a temp file
---------------
Every scoring run reads ``corpus.parquet``. The output is written next to it
as ``corpus.parquet.tmp`` and renamed only on success, so a merge that fails
halfway leaves the previous corpus in place instead of half a new one.

Design: docs/notes/2026-09-29_corpus_merge_design.md.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import (
    CORPUS_PATH,
    EXPECTED_PARTIES,
    GOVINFO_BUILD_STATS_PATH,
    GOVINFO_CORPUS_PATH,
    GOVINFO_END_DATE,
    MERGED_BUILD_STATS_PATH,
    SOURCE_BREAK_DATE,
    STANFORD_BUILD_STATS_PATH,
    STANFORD_CORPUS_PATH,
)
from src.corpus import CORPUS_SCHEMA, _fingerprint
from src.corpus import SOURCE_TAG as STANFORD_TAG
from src.govinfo import GOVINFO_SCHEMA
from src.govinfo import SOURCE_TAG as GOVINFO_TAG

#: The 12 corpus columns plus ``icpsr`` -- identical to the govinfo schema.
#: Stanford rows get a null ``icpsr``.
MERGED_SCHEMA = GOVINFO_SCHEMA

#: Pass 1 reads only these small columns; ``text`` is left for pass 2.
KEY_COLUMNS: tuple[str, ...] = (
    "speech_id",
    "date",
    "party",
    "chamber",
    "congress_number",
    "source",
)

EXPECTED_CHAMBERS: frozenset[str] = frozenset({"H", "S"})

#: Rows per streaming batch. Sets peak memory: only ``text`` is large.
BATCH_SIZE = 50_000

#: How many offending values an error message lists.
_SHOW = 5


@dataclass(frozen=True)
class MergeInput:
    """One side of the merge: a processed corpus file and what it must be."""

    source: str  # the value every row's ``source`` must carry
    path: Path  # the processed parquet
    stats_path: Path  # the stats JSON its build wrote
    schema: pa.Schema  # the schema its build writes
    make_target: str  # named in error messages, so the fix is one command


@dataclass
class MergedBuildStats:
    """Provenance and distributions for one merge.

    Serialized to ``results/metrics/merged_build_stats.json``. ``inputs`` ties
    the merged file to the processed files it was built from and, through
    their own stats, to the raw files behind those.
    """

    output_path: str
    built_at: str
    inputs: dict[str, dict[str, object]] = field(default_factory=dict)
    rows_written: int = 0
    rows_by_source: dict[str, int] = field(default_factory=dict)
    # congress -> source -> chamber -> party -> speeches
    cell_counts: dict[str, dict[str, dict[str, dict[str, int]]]] = field(
        default_factory=dict
    )
    # year -> source -> chamber -> speeches
    year_counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    seam: dict[str, str] = field(default_factory=dict)
    icpsr_missing_by_source: dict[str, int] = field(default_factory=dict)
    date_min: str | None = None
    date_max: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the stats."""
        return asdict(self)

    def write(self, path: Path = MERGED_BUILD_STATS_PATH) -> Path:
        """Write the stats to ``path`` as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


# --- checks ------------------------------------------------------------


def _check_output(dst: Path, inputs: tuple[MergeInput, ...], overwrite: bool) -> None:
    """Refuse an output that is one of the inputs, or exists without overwrite."""
    for spec in inputs:
        if dst.resolve() == spec.path.resolve():
            raise ValueError(
                f"Output {dst} is the {spec.source} input; the merge would "
                "overwrite its own input. Choose another output path."
            )
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"{dst} already exists; pass overwrite=True to replace it."
        )


def _check_exists(spec: MergeInput) -> None:
    """Check 1: the input file is there."""
    if not spec.path.exists():
        raise FileNotFoundError(
            f"The {spec.source} corpus is not at {spec.path}. "
            f"Build it first: make {spec.make_target}"
        )


def _check_schema(spec: MergeInput) -> None:
    """Check 2: the input has exactly the schema its build writes."""
    actual = pq.ParquetFile(spec.path).schema_arrow
    if actual.equals(spec.schema):
        return
    want = {f.name: f.type for f in spec.schema}
    have = {f.name: f.type for f in actual}
    problems = [f"missing {name}" for name in want if name not in have]
    problems += [f"unexpected {name}" for name in have if name not in want]
    problems += [
        f"{name} is {have[name]}, expected {want[name]}"
        for name in want
        if name in have and have[name] != want[name]
    ]
    if not problems:
        problems = ["column order or nullability differs"]
    raise ValueError(
        f"{spec.path} does not have the {spec.source} corpus schema: "
        f"{'; '.join(problems)}. Rebuild it: make {spec.make_target}"
    )


def _check_full_build(spec: MergeInput) -> dict[str, object]:
    """Check 3: the file is the full build its committed stats describe.

    Returns those stats, for provenance. Catches a smoke file copied onto the
    real path, and a file rebuilt without its stats being updated.
    """
    if not spec.stats_path.exists():
        raise FileNotFoundError(
            f"No build stats for the {spec.source} corpus at {spec.stats_path}; "
            f"a full build writes them. Rebuild: make {spec.make_target}"
        )
    build = json.loads(spec.stats_path.read_text())
    if "limit" not in build or "rows_written" not in build:
        raise ValueError(
            f"{spec.stats_path} has no 'limit'/'rows_written', so it cannot vouch "
            f"for {spec.path}. Rebuild: make {spec.make_target}"
        )
    if build["limit"] is not None:
        raise ValueError(
            f"{spec.stats_path} records a --limit (smoke) build; only a full "
            f"build may be merged. Rebuild: make {spec.make_target}"
        )
    rows = pq.ParquetFile(spec.path).metadata.num_rows
    if rows != build["rows_written"]:
        raise ValueError(
            f"{spec.path} has {rows:,} rows but {spec.stats_path} says "
            f"{build['rows_written']:,}: the file and its stats come from "
            f"different builds. Rebuild: make {spec.make_target}"
        )
    return build


def _check_values(spec: MergeInput, keys: pa.Table) -> None:
    """Checks 4 and 7: rows exist, no nulls, and only the expected codes."""
    if keys.num_rows == 0:
        raise ValueError(
            f"{spec.path} has no rows. Rebuild it: make {spec.make_target}"
        )
    for column in KEY_COLUMNS:
        nulls = keys.column(column).null_count
        if nulls:
            raise ValueError(f"{spec.path}: {nulls:,} null value(s) in {column!r}")
    allowed = {
        "source": {spec.source},
        "party": EXPECTED_PARTIES,
        "chamber": EXPECTED_CHAMBERS,
    }
    for column, expected in allowed.items():
        found = set(pc.unique(keys.column(column)).to_pylist())
        unexpected = sorted(found - expected)
        if unexpected:
            raise ValueError(
                f"{spec.path}: unexpected {column} value(s) {unexpected[:_SHOW]}; "
                f"expected only {sorted(expected)}"
            )


def _check_seam(
    stanford: MergeInput,
    stanford_keys: pa.Table,
    govinfo: MergeInput,
    govinfo_keys: pa.Table,
) -> dict[str, str]:
    """Check 5: Stanford ends before the break, govinfo runs from it to the end.

    Returns the seam -- the last Stanford day and the first govinfo day.
    """
    stanford_last = pc.max(stanford_keys.column("date")).as_py()
    govinfo_first = pc.min(govinfo_keys.column("date")).as_py()
    govinfo_last = pc.max(govinfo_keys.column("date")).as_py()
    if stanford_last >= SOURCE_BREAK_DATE:
        raise ValueError(
            f"{stanford.path} runs to {stanford_last}, past the source break "
            f"{SOURCE_BREAK_DATE}; govinfo covers from there (docs/decisions.md D15)"
        )
    if govinfo_first < SOURCE_BREAK_DATE:
        raise ValueError(
            f"{govinfo.path} starts {govinfo_first}, before the source break "
            f"{SOURCE_BREAK_DATE}; Stanford covers until then (docs/decisions.md D15)"
        )
    if govinfo_last > GOVINFO_END_DATE:
        raise ValueError(
            f"{govinfo.path} runs to {govinfo_last}, past GOVINFO_END_DATE "
            f"{GOVINFO_END_DATE}"
        )
    return {
        "stanford_last": stanford_last.isoformat(),
        "govinfo_first": govinfo_first.isoformat(),
    }


def _check_unique_ids(keys: list[pa.Table]) -> None:
    """Check 6: no speech_id occurs twice, within or across the inputs."""
    ids = pa.chunked_array(
        [chunk for table in keys for chunk in table.column("speech_id").chunks],
        type=pa.string(),
    )
    counts = pc.value_counts(ids)
    repeated = counts.filter(pc.greater(counts.field("counts"), 1))
    if len(repeated):
        shown = repeated.field("values").to_pylist()[:_SHOW]
        raise ValueError(
            f"{len(repeated):,} speech_id(s) occur more than once across the "
            f"inputs, e.g. {shown}"
        )


def _check_congress_coverage(keys: list[pa.Table]) -> None:
    """Check 8: every Congress from the first to the last present has rows."""
    present: set[int] = set()
    for table in keys:
        present |= set(pc.unique(table.column("congress_number")).to_pylist())
    missing = sorted(set(range(min(present), max(present) + 1)) - present)
    if missing:
        raise ValueError(
            f"No speeches for Congress(es) {missing}, between the "
            f"{min(present)}th and the {max(present)}th: a source build is "
            "incomplete"
        )


# --- statistics --------------------------------------------------------


def _nest(counts: Counter[tuple[object, ...]]) -> dict[str, object]:
    """``{(a, b, c): n}`` -> ``{"a": {"b": {"c": n}}}``, keys sorted, as strings."""
    nested: dict[str, object] = {}
    for key in sorted(counts):
        node = nested
        for part in key[:-1]:
            node = node.setdefault(str(part), {})
        node[str(key[-1])] = counts[key]
    return nested


def _collect_stats(
    stats: MergedBuildStats,
    inputs: tuple[MergeInput, ...],
    keys: list[pa.Table],
    builds: list[dict[str, object]],
) -> None:
    """Fill provenance and distributions from the pass-1 columns.

    Everything is counted here, before streaming, so the stats describe
    exactly the rows the checks approved.
    """
    cells: Counter[tuple[object, ...]] = Counter()
    years: Counter[tuple[object, ...]] = Counter()
    dates: list[dt.date] = []
    for spec, table, build in zip(inputs, keys, builds):
        stats.inputs[spec.source] = {
            "path": str(spec.path),
            "bytes": spec.path.stat().st_size,
            "sha256": _fingerprint(spec.path),
            "rows": table.num_rows,
            "stats_path": str(spec.stats_path),
            "raw_source_sha256": build.get("source_sha256"),
        }
        stats.rows_by_source[spec.source] = table.num_rows
        stats.icpsr_missing_by_source[spec.source] = (
            table.column("icpsr").null_count
            if "icpsr" in table.column_names
            else table.num_rows
        )
        chambers = table.column("chamber").to_pylist()
        cells.update(
            (congress, spec.source, chamber, party)
            for congress, chamber, party in zip(
                table.column("congress_number").to_pylist(),
                chambers,
                table.column("party").to_pylist(),
            )
        )
        years.update(
            (year, spec.source, chamber)
            for year, chamber in zip(
                pc.year(table.column("date")).to_pylist(), chambers
            )
        )
        dates += [
            pc.min(table.column("date")).as_py(),
            pc.max(table.column("date")).as_py(),
        ]
    stats.cell_counts = _nest(cells)
    stats.year_counts = _nest(years)
    stats.date_min = min(dates).isoformat()
    stats.date_max = max(dates).isoformat()


# --- streaming ---------------------------------------------------------


def _stream(spec: MergeInput, writer: pq.ParquetWriter) -> int:
    """Copy one input into ``writer`` batch by batch; return rows written."""
    written = 0
    for batch in pq.ParquetFile(spec.path).iter_batches(batch_size=BATCH_SIZE):
        if "icpsr" not in batch.schema.names:
            batch = batch.append_column("icpsr", pa.nulls(batch.num_rows, pa.int32()))
        # Rebuilt against MERGED_SCHEMA so a type drift fails here, loudly,
        # instead of being written.
        writer.write_batch(
            pa.RecordBatch.from_arrays(batch.columns, schema=MERGED_SCHEMA)
        )
        written += batch.num_rows
    return written


# --- merge -------------------------------------------------------------


def merge_corpora(
    stanford: Path = STANFORD_CORPUS_PATH,
    govinfo: Path = GOVINFO_CORPUS_PATH,
    dst: Path = CORPUS_PATH,
    *,
    stanford_stats: Path = STANFORD_BUILD_STATS_PATH,
    govinfo_stats: Path = GOVINFO_BUILD_STATS_PATH,
    overwrite: bool = False,
) -> MergedBuildStats:
    """Concatenate the processed Stanford and govinfo corpora into ``dst``.

    Args:
        stanford: Processed Stanford parquet (``make stanford``).
        govinfo: Processed govinfo parquet (``make govinfo``).
        dst: Output parquet, ``MERGED_SCHEMA``. Stanford rows first.
        stanford_stats: Stats JSON of the Stanford build.
        govinfo_stats: Stats JSON of the govinfo build.
        overwrite: Required to replace an existing ``dst``.

    Returns:
        Provenance and counts for the merge. Caller decides where to persist
        them; see ``MergedBuildStats.write``.

    Raises:
        FileNotFoundError: an input, or its build stats, is missing.
        FileExistsError: ``dst`` exists and ``overwrite`` is False.
        ValueError: ``dst`` is one of the inputs, or an input fails a check:
            wrong schema, not its full build, no rows, a null or unexpected
            code, dates across the seam, a repeated ``speech_id``, or a gap in
            the Congresses covered. Every check runs before anything is
            written.
    """
    dst = Path(dst)
    inputs = (
        MergeInput(
            STANFORD_TAG,
            Path(stanford),
            Path(stanford_stats),
            CORPUS_SCHEMA,
            "stanford",
        ),
        MergeInput(
            GOVINFO_TAG, Path(govinfo), Path(govinfo_stats), GOVINFO_SCHEMA, "govinfo"
        ),
    )
    _check_output(dst, inputs, overwrite)

    # Pass 1: small columns only. Everything that can fail, fails here, before
    # a byte of output exists.
    keys: list[pa.Table] = []
    builds: list[dict[str, object]] = []
    for spec in inputs:
        _check_exists(spec)
        _check_schema(spec)
        builds.append(_check_full_build(spec))
        # icpsr is read too where it exists, for the missing-ICPSR count; it
        # may be null, so it is not one of the checked KEY_COLUMNS.
        columns = list(KEY_COLUMNS) + (
            ["icpsr"] if "icpsr" in spec.schema.names else []
        )
        table = pq.read_table(spec.path, columns=columns)
        _check_values(spec, table)
        keys.append(table)
    seam = _check_seam(inputs[0], keys[0], inputs[1], keys[1])
    _check_unique_ids(keys)
    _check_congress_coverage(keys)

    stats = MergedBuildStats(
        output_path=str(dst),
        built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        seam=seam,
    )
    _collect_stats(stats, inputs, keys, builds)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    writer = pq.ParquetWriter(tmp, MERGED_SCHEMA, compression="snappy")
    try:
        for spec in inputs:
            stats.rows_written += _stream(spec, writer)
    except BaseException:
        writer.close()
        tmp.unlink(missing_ok=True)
        raise
    writer.close()
    tmp.replace(dst)
    return stats
