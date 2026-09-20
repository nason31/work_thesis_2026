"""Build the processed speech corpus from the raw Stanford parquet.

One row per speech, mapped onto the corpus schema documented in CLAUDE.md.
Called by ``code/scripts/build_corpus.py``; the logic lives here so a govinfo
loader can be added alongside it later and the two concatenated.

Why two passes instead of one read
----------------------------------
``speech`` is the only large column: the raw file is ~680 MB compressed and
decompresses to several GB, almost all of it text. Everything needed to
*validate* the file and to count rows lives in the small columns. So pass 1
reads metadata only and fails fast -- in seconds, before a single byte of
output is written -- and pass 2 streams the text row group by row group
through a single writer, keeping peak memory at roughly one batch.

The 2017 source break
---------------------
This loader covers the Stanford side only: the 107th-114th Congress, ending
January 3, 2017. Every row it writes carries ``source == "stanford"`` so the
break stays recoverable from the data itself rather than only from a note.
Until the govinfo loader exists, the corpus written here stops in 2017 --
check the ``source`` column before reading a trend off it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import (
    BUILD_STATS_PATH,
    CAUCUS_PARTY,
    CORPUS_PATH,
    MIN_WORD_COUNT,
    STANFORD_PARQUET,
)

# --- schemas -----------------------------------------------------------

#: Raw Stanford columns, in the order the source file carries them.
RAW_COLUMNS: tuple[str, ...] = (
    "speech_id",
    "speech",
    "chamber",
    "date",
    "speaker",
    "first_name",
    "state",
    "gender",
    "word_count",
    "speakerid",
    "party",
    "congress",
)

#: Columns the build cannot proceed without. ``gender`` is deliberately absent:
#: it is carried by the source but unused here.
REQUIRED_RAW_COLUMNS: tuple[str, ...] = tuple(c for c in RAW_COLUMNS if c != "gender")

#: Pass 1 reads these and skips ``speech`` -- that is what makes it cheap.
METADATA_COLUMNS: tuple[str, ...] = (
    "speech_id",
    "date",
    "speaker",
    "state",
    "word_count",
    "speakerid",
    "party",
    "congress",
)

#: Value of the ``source`` column for every row this loader writes.
SOURCE_TAG = "stanford"

#: The first eight fields are the corpus schema from CLAUDE.md. The rest are
#: retained deliberately: the speakerid -> ICPSR crosswalk needed for
#: DW-NOMINATE validation is built from name + state + congress, and
#: ``party_original`` keeps the caucus reassignment auditable. Without them,
#: either job means re-streaming the whole raw file.
CORPUS_SCHEMA = pa.schema(
    [
        pa.field("speech_id", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("member_id", pa.string()),
        pa.field("party", pa.string()),
        pa.field("chamber", pa.string()),
        pa.field("congress_number", pa.int16()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("speaker", pa.string()),
        pa.field("first_name", pa.string()),
        pa.field("state", pa.string()),
        pa.field("word_count", pa.int32()),
        pa.field("party_original", pa.string()),
    ]
)

#: Rows per streaming batch. Sets peak memory, and makes output independent of
#: how the source file happens to be split into row groups.
BATCH_SIZE = 50_000


# --- statistics --------------------------------------------------------


@dataclass
class BuildStats:
    """Counts and parameters for one corpus build.

    Serialized to ``results/metrics/corpus_build_stats.json`` so any figure
    quoted in the methodology chapter has a traceable source.
    """

    source_path: str
    source_bytes: int
    output_path: str
    built_at: str
    min_word_count: int
    limit: int | None
    source_tag: str = SOURCE_TAG
    raw_schema: dict[str, str] = field(default_factory=dict)
    rows_read: int = 0
    rows_written: int = 0
    rows_dropped_short: int = 0
    rows_dropped_empty_text: int = 0
    rows_dropped_duplicate_id: int = 0
    independents_reassigned: int = 0
    party_counts: dict[str, int] = field(default_factory=dict)
    party_counts_original: dict[str, int] = field(default_factory=dict)
    congress_counts: dict[str, int] = field(default_factory=dict)
    chamber_counts: dict[str, int] = field(default_factory=dict)
    date_min: str | None = None
    date_max: str | None = None
    word_count_mismatch_rows: int = 0
    word_count_mismatch_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the stats."""
        return asdict(self)

    def write(self, path: Path = BUILD_STATS_PATH) -> Path:
        """Write the stats to ``path`` as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


# --- helpers -----------------------------------------------------------


def _probe_schema(src: Path) -> dict[str, str]:
    """Return the raw file's arrow schema, raising if a needed column is absent.

    The exact dtypes of the Stanford file are not documented anywhere, so they
    are recorded in the stats rather than assumed.
    """
    schema = pq.ParquetFile(src).schema_arrow
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in schema.names]
    if missing:
        raise ValueError(
            f"{src} is missing required column(s): {', '.join(missing)}. "
            f"Expected the raw Stanford layout: {', '.join(RAW_COLUMNS)}"
        )
    return {name: str(schema.field(name).type) for name in schema.names}


def _iter_batches(
    src: Path, columns: tuple[str, ...], limit: int | None
) -> Iterator[pa.RecordBatch]:
    """Stream ``columns`` from ``src`` in fixed-size batches, honouring ``limit``.

    Both passes read in the same order, so a ``limit`` selects the same rows in
    each of them.
    """
    remaining = limit
    for batch in pq.ParquetFile(src).iter_batches(
        batch_size=BATCH_SIZE, columns=list(columns)
    ):
        if remaining is not None:
            if remaining <= 0:
                return
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            remaining -= batch.num_rows
        yield batch


def _parse_yyyymmdd(column: pa.Array) -> pa.Array:
    """Parse a YYYYMMDD column to ``date32``, accepting integer or string input."""
    if not pa.types.is_string(column.type) and not pa.types.is_large_string(
        column.type
    ):
        column = pc.cast(column, pa.string())
    return pc.cast(pc.strptime(column, format="%Y%m%d", unit="s"), pa.date32())


def _normalize_text(column: pa.Array) -> pa.Array:
    """Collapse whitespace runs to single spaces and trim the ends.

    Whitespace only, on purpose. Stripping boilerplate ("Mr. Speaker,",
    procedural preambles) would change what the model sees, which is a
    methodological choice rather than cleaning -- OPEN DECISION, not settled
    here. CLAUDE.md principle 4 also says feed full speeches.
    """
    return pc.utf8_trim_whitespace(
        pc.replace_substring_regex(column, pattern=r"\s+", replacement=" ")
    )


def _resolve_parties(
    party: list[str | None], speaker: list[str | None], state: list[str | None]
) -> tuple[list[str | None], int]:
    """Map independents to the party they caucus with.

    Only ``party == "I"`` rows are touched, so a Republican named King is left
    alone. Returns the resolved parties and how many rows were reassigned.
    Callers must have validated the roster in pass 1; anything unmatched here
    is passed through unchanged rather than guessed at.
    """
    resolved: list[str | None] = []
    reassigned = 0
    for row_party, row_speaker, row_state in zip(party, speaker, state):
        if row_party == "I":
            caucus = CAUCUS_PARTY.get(_roster_key(row_speaker, row_state))
            if caucus is not None:
                resolved.append(caucus)
                reassigned += 1
                continue
        resolved.append(row_party)
    return resolved, reassigned


def _roster_key(speaker: str | None, state: str | None) -> tuple[str, str]:
    """Normalize a (last name, state) pair to the CAUCUS_PARTY key form."""
    return ((speaker or "").strip().upper(), (state or "").strip().upper())


def _true_count(mask: pa.Array) -> int:
    """Number of true values in a boolean mask, treating null as false."""
    return pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py() or 0


def _count_values(column: pa.Array) -> Counter[str]:
    """Value counts as strings, so nulls cannot break JSON keys or sorting."""
    return Counter(
        "UNKNOWN" if value is None else str(value) for value in column.to_pylist()
    )


# --- pass 1: metadata ---------------------------------------------------


@dataclass
class _MetadataScan:
    """Result of the cheap metadata pass."""

    rows_read: int
    duplicate_id_candidates: set[str]


def _scan_metadata(src: Path, limit: int | None) -> _MetadataScan:
    """Validate the independents roster and find duplicate ids, without text.

    Raises before any output exists if the file contains an independent that
    CLAUDE.md's roster does not cover -- see ``_raise_unrostered``.
    """
    rows_read = 0
    seen_ids: set[str] = set()
    duplicates: set[str] = set()
    unrostered: dict[tuple[str, str], set[tuple[int | None, str]]] = {}

    for batch in _iter_batches(src, METADATA_COLUMNS, limit):
        rows_read += batch.num_rows

        ids = pc.cast(batch.column("speech_id"), pa.string()).to_pylist()
        for speech_id in ids:
            if speech_id in seen_ids:
                duplicates.add(speech_id)
            else:
                seen_ids.add(speech_id)

        independents = batch.filter(
            pc.fill_null(pc.equal(batch.column("party"), "I"), False)
        )
        if independents.num_rows == 0:
            continue
        speakers = independents.column("speaker").to_pylist()
        states = independents.column("state").to_pylist()
        congresses = independents.column("congress").to_pylist()
        member_ids = pc.cast(independents.column("speakerid"), pa.string()).to_pylist()
        for speaker, state, congress, member_id in zip(
            speakers, states, congresses, member_ids
        ):
            key = _roster_key(speaker, state)
            if key not in CAUCUS_PARTY:
                unrostered.setdefault(key, set()).add((congress, member_id or ""))

    if unrostered:
        _raise_unrostered(unrostered)

    return _MetadataScan(rows_read=rows_read, duplicate_id_candidates=duplicates)


def _raise_unrostered(
    unrostered: dict[tuple[str, str], set[tuple[int | None, str]]],
) -> None:
    """Fail with the members that need adding to CAUCUS_PARTY.

    Deliberately fatal rather than a silent fallback: CLAUDE.md fixes the
    handling of independents once, at build time, so an unlisted independent is
    a decision for a human to make and document -- not something to guess.
    """
    lines = []
    for (speaker, state), occurrences in sorted(unrostered.items()):
        congresses = sorted(
            {congress for congress, _ in occurrences if congress is not None}
        )
        member_ids = sorted({member_id for _, member_id in occurrences if member_id})
        lines.append(
            f"  {speaker} ({state}) — congress(es) "
            f"{', '.join(str(c) for c in congresses) or 'unknown'}; "
            f"speakerid(s) {', '.join(member_ids) or 'unknown'}"
        )
    raise ValueError(
        "Found party == 'I' speeches by members not in CAUCUS_PARTY:\n"
        + "\n".join(lines)
        + "\n\nAdd each one to CAUCUS_PARTY in code/src/config.py with the party "
        "they caucus with, and record the addition in CLAUDE.md's 'Decided — "
        "data decisions' section. The build refuses to guess."
    )


# --- pass 2: stream and write ------------------------------------------


def _project_batch(batch: pa.RecordBatch) -> tuple[pa.RecordBatch, int]:
    """Map a raw batch onto CORPUS_SCHEMA. Returns the batch and reassign count."""
    party_original = batch.column("party")
    resolved, reassigned = _resolve_parties(
        party_original.to_pylist(),
        batch.column("speaker").to_pylist(),
        batch.column("state").to_pylist(),
    )
    projected = pa.RecordBatch.from_arrays(
        [
            pc.cast(batch.column("speech_id"), pa.string()),
            _parse_yyyymmdd(batch.column("date")),
            pc.cast(batch.column("speakerid"), pa.string()),
            pa.array(resolved, type=pa.string()),
            pc.cast(batch.column("chamber"), pa.string()),
            pc.cast(batch.column("congress"), pa.int16()),
            pc.cast(batch.column("text_clean"), pa.string()),
            pa.array([SOURCE_TAG] * batch.num_rows, type=pa.string()),
            pc.cast(batch.column("speaker"), pa.string()),
            pc.cast(batch.column("first_name"), pa.string()),
            pc.cast(batch.column("state"), pa.string()),
            pc.cast(batch.column("word_count"), pa.int32()),
            pc.cast(party_original, pa.string()),
        ],
        schema=CORPUS_SCHEMA,
    )
    return projected, reassigned


def _count_words(column: pa.Array) -> pa.Array:
    """Whitespace-token count per row, used to audit the source word_count."""
    return pc.list_value_length(pc.utf8_split_whitespace(column))


def build_stanford_corpus(
    src: Path = STANFORD_PARQUET,
    dst: Path = CORPUS_PATH,
    min_word_count: int = MIN_WORD_COUNT,
    limit: int | None = None,
    overwrite: bool = False,
) -> BuildStats:
    """Build the processed corpus from the raw Stanford parquet.

    Args:
        src: Raw Stanford parquet. Read only; never modified.
        dst: Output parquet, one row per retained speech.
        min_word_count: Speeches shorter than this are dropped (inclusive
            bound, so ``min_word_count`` words is kept).
        limit: Read at most this many raw rows. For smoke runs -- it validates
            the real schema and the roster in seconds.
        overwrite: Required to replace an existing ``dst``.

    Returns:
        Counts and parameters for the run. Caller decides where to persist
        them; see ``BuildStats.write``.

    Raises:
        FileNotFoundError: ``src`` does not exist.
        FileExistsError: ``dst`` exists and ``overwrite`` is False.
        ValueError: a required raw column is missing, or an independent is not
            covered by ``CAUCUS_PARTY``.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(
            f"Raw Stanford parquet not found at {src}. Download "
            "congress_speeches_2001_2017.parquet from the team drive into "
            "data/raw/stanford/ first (see docs/notes/2026-09-15_stanford_dataset.md)."
        )
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"{dst} already exists; pass overwrite=True to replace it."
        )

    stats = BuildStats(
        source_path=str(src),
        source_bytes=src.stat().st_size,
        output_path=str(dst),
        built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        min_word_count=min_word_count,
        limit=limit,
    )

    # Pass 1: cheap, and fatal before anything is written.
    stats.raw_schema = _probe_schema(src)
    scan = _scan_metadata(src, limit)
    stats.rows_read = scan.rows_read

    party_counts: Counter[str] = Counter()
    party_counts_original: Counter[str] = Counter()
    congress_counts: Counter[str] = Counter()
    chamber_counts: Counter[str] = Counter()
    date_min: dt.date | None = None
    date_max: dt.date | None = None
    seen_duplicates: set[str] = set()

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(dst, CORPUS_SCHEMA, compression="snappy")
    try:
        # Pass 2: stream the text through, filtering before it is materialized.
        for batch in _iter_batches(src, REQUIRED_RAW_COLUMNS, limit):
            long_enough = pc.fill_null(
                pc.greater_equal(batch.column("word_count"), min_word_count), False
            )
            stats.rows_dropped_short += batch.num_rows - _true_count(long_enough)
            batch = batch.filter(long_enough)
            if batch.num_rows == 0:
                continue

            cleaned = _normalize_text(batch.column("speech"))
            has_text = pc.fill_null(pc.not_equal(cleaned, ""), False)
            stats.rows_dropped_empty_text += batch.num_rows - _true_count(has_text)
            batch = batch.append_column("text_clean", cleaned).filter(has_text)
            if batch.num_rows == 0:
                continue

            # Only ids pass 1 saw more than once can collide, so the set stays
            # small even though the corpus does not.
            if scan.duplicate_id_candidates:
                keep = []
                for speech_id in pc.cast(
                    batch.column("speech_id"), pa.string()
                ).to_pylist():
                    if speech_id in scan.duplicate_id_candidates:
                        if speech_id in seen_duplicates:
                            keep.append(False)
                            continue
                        seen_duplicates.add(speech_id)
                    keep.append(True)
                dropped = keep.count(False)
                if dropped:
                    stats.rows_dropped_duplicate_id += dropped
                    batch = batch.filter(pa.array(keep, type=pa.bool_()))
                    if batch.num_rows == 0:
                        continue

            mismatches = pc.fill_null(
                pc.not_equal(
                    _count_words(batch.column("text_clean")), batch.column("word_count")
                ),
                True,
            )
            stats.word_count_mismatch_rows += _true_count(mismatches)

            projected, reassigned = _project_batch(batch)
            stats.independents_reassigned += reassigned
            stats.rows_written += projected.num_rows

            party_counts += _count_values(projected.column("party"))
            party_counts_original += _count_values(projected.column("party_original"))
            congress_counts += _count_values(projected.column("congress_number"))
            chamber_counts += _count_values(projected.column("chamber"))
            span = pc.min_max(projected.column("date"))
            batch_min, batch_max = span["min"].as_py(), span["max"].as_py()
            if batch_min is not None:
                date_min = batch_min if date_min is None else min(date_min, batch_min)
                date_max = batch_max if date_max is None else max(date_max, batch_max)

            writer.write_batch(projected)
    except BaseException:
        # Never leave a half-written corpus behind for a later run to trust.
        writer.close()
        dst.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    stats.party_counts = dict(sorted(party_counts.items()))
    stats.party_counts_original = dict(sorted(party_counts_original.items()))
    stats.congress_counts = dict(sorted(congress_counts.items()))
    stats.chamber_counts = dict(sorted(chamber_counts.items()))
    stats.date_min = date_min.isoformat() if date_min else None
    stats.date_max = date_max.isoformat() if date_max else None
    stats.word_count_mismatch_rate = (
        stats.word_count_mismatch_rows / stats.rows_written
        if stats.rows_written
        else 0.0
    )
    return stats
