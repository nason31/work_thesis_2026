"""Build the processed govinfo side of the corpus from Konsti's JSONL.

The raw file (``congress_speeches_2016_present.jsonl``) was produced by
``02_govinfo_dataset.ipynb``, which is not in this repo. Profiling it on
2026-09-27 found four upstream problems; this module repairs the three that the
file still carries enough information to repair, and counts the fourth.
See docs/notes/2026-09-27_govinfo_data_reality.md.

1. **Party lookup ignored chamber.** A surname counted as ambiguous if anyone in
   Congress shared it, so e.g. every Senate speech by Mike Lee (518) came out
   with no party. Fixed: every speaker is re-resolved here against
   congress-legislators, filtered by chamber *and* date.
2. **The state was dropped** from House headers ("Mr. SMITH of Texas."), so
   same-surname House members cannot be told apart. NOT fixable here -- the
   header is not in the file. Those rows stay unresolved and are dropped, and
   the stats count them by chamber and year so the bias is measurable.
3. **Speeches were split only when a member began speaking.** Presiding-officer
   turns, Record narration and inserted material were glued onto the previous
   member's speech. Fixed: each speech is cut at the first of those.
4. **HTML and Record furniture left in the text** -- ``</pre></body></html>``,
   ``[[Page S3236]]``, centered headings, ``{time}`` stamps. Fixed: stripped.

The file's own ``party`` and ``icpsr`` columns are ignored in favour of the
re-resolution; their agreement with it is recorded in the stats. This also
fixes the README's admitted limitation that mid-term party switches were not
reflected: congress-legislators carries day-level ``party_affiliations``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    DELEGATE_STATES,
    EXPECTED_PARTIES,
    GOVINFO_BUILD_STATS_PATH,
    GOVINFO_CAUCUS_PARTY,
    GOVINFO_CORPUS_PATH,
    GOVINFO_END_DATE,
    GOVINFO_EXCLUDED_MEMBERS,
    GOVINFO_JSONL,
    GOVINFO_START_DATE,
    LEGISLATORS_FILES,
    MIN_WORD_COUNT,
)
from src.corpus import CORPUS_SCHEMA, _fingerprint

#: Value of the ``source`` column for every row this loader writes.
SOURCE_TAG = "govinfo"

#: The Stanford corpus columns plus ``icpsr``, which this side has directly and
#: which is the join key for DW-NOMINATE. ``member_id`` is the bioguide id.
GOVINFO_SCHEMA = CORPUS_SCHEMA.append(pa.field("icpsr", pa.int32()))

#: Raw keys every JSONL record must carry.
REQUIRED_KEYS: tuple[str, ...] = ("date", "speaker", "chamber", "speech")

BATCH_SIZE = 50_000

#: congress-legislators party names -> corpus codes.
PARTY_CODES: dict[str, str] = {
    "Democrat": "D",
    "Republican": "R",
    "Independent": "I",
    "Libertarian": "L",
}

_CHAMBER_TYPES: dict[str, str] = {"HOUSE": "rep", "SENATE": "sen"}
_CHAMBER_CODES: dict[str, str] = {"HOUSE": "H", "SENATE": "S"}


# --- text cleaning -----------------------------------------------------

# A presiding-officer turn: an indented paragraph opening with an officer title
# in the Record's capitals, optionally naming the occupant. Titles only -- a
# member's paragraph beginning "The Speaker said..." does not match.
_OFFICER_TURN = re.compile(
    r"^[ \t]+The (?:Acting |ACTING )?"
    r"(?:SPEAKER|PRESIDING OFFICER|CHAIR(?:MAN|WOMAN)?|VICE PRESIDENT|PRESIDENT"
    r"|CHIEF JUSTICE)(?: pro tempore)?(?: \([^)\n]*\))?\.",
    re.MULTILINE,
)

# Record narration: the chamber acting, not a member speaking. Matched only as
# the opening of an indented paragraph, and only these fixed phrases.
_NARRATION_PHRASES: tuple[str, ...] = (
    "The yeas and nays were ordered",
    "The question was taken",
    "The Clerk read",
    "The clerk read",
    "The clerk will call the roll",
    "The legislative clerk",
    "The bill clerk",
    "The assistant legislative clerk",
    "The senior assistant legislative clerk",
    "The result was announced",
    "The material previously referred to",
    "A recorded vote was ordered",
    "The amendment was agreed to",
    "The amendments were agreed to",
    "The motion was agreed to",
    "The motion is agreed to",
    "The motion was rejected",
    "The resolution was agreed to",
    "The preamble was agreed to",
    "The nomination was confirmed",
)
_NARRATION = re.compile(
    r"^[ \t]+(?:" + "|".join(re.escape(p) for p in _NARRATION_PHRASES) + r")\b",
    re.MULTILINE,
)

# A long rule ("____________________") on its own line separates Record items;
# what follows is a different item -- a bill introduction, a cosponsor list, a
# committee letter. Two look-alikes are NOT boundaries and must not cut: the
# short "____" between letters a member inserts within their own turn, and a
# long rule that only frames a "{time} 1430" stamp mid-speech.
_SEPARATOR = re.compile(r"^[ \t]*_{10,}[ \t]*$(?!\s*\{time\})", re.MULTILINE)

_HTML_TAG = re.compile(r"</?(?:pre|body|html)>", re.IGNORECASE)
_PAGE_MARKER = re.compile(r"\[\[Page [^\]\n]*\]\]")
_TIME_STAMP = re.compile(r"\{time\}\s*\d{3,4}")

# Centered lines -- section headings, vote tallies. Speech paragraphs indent 2
# spaces and quoted letters 5-11; headings sit at 12 or more.
_HEADING_LINE = re.compile(r"^ {12,}\S[^\n]*$", re.MULTILINE)

_WHITESPACE = re.compile(r"\s+")


@dataclass
class CleanResult:
    """A cleaned speech and what was done to it."""

    text: str
    cut_at: str | None  # "officer", "narration", "separator", or None
    words_removed: int
    page_markers: int
    html_tags: int
    headings: int


def clean_speech(raw: str) -> CleanResult:
    """Cut a raw govinfo speech down to the member's own words.

    Order matters: the cut comes first, at the earliest officer turn, narration
    paragraph or separator, because everything after it belongs to someone or
    something else. Record furniture is then stripped from what remains, and
    whitespace is normalized exactly as on the Stanford side.
    """
    cuts = [
        (m.start(), kind)
        for kind, pattern in (
            ("officer", _OFFICER_TURN),
            ("narration", _NARRATION),
            ("separator", _SEPARATOR),
        )
        if (m := pattern.search(raw))
    ]
    cut_at = None
    text = raw
    if cuts:
        position, cut_at = min(cuts)
        text = raw[:position]

    page_markers = len(_PAGE_MARKER.findall(text))
    text = _PAGE_MARKER.sub(" ", text)
    html_tags = len(_HTML_TAG.findall(text))
    text = _HTML_TAG.sub(" ", text)
    text = _TIME_STAMP.sub(" ", text)
    headings = len(_HEADING_LINE.findall(text))
    text = _HEADING_LINE.sub(" ", text)

    text = _WHITESPACE.sub(" ", text).strip()
    return CleanResult(
        text=text,
        cut_at=cut_at,
        words_removed=len(raw.split()) - len(text.split()),
        page_markers=page_markers,
        html_tags=html_tags,
        headings=headings,
    )


# --- speaker resolution ------------------------------------------------


def _fold(name: str) -> str:
    """Uppercase, strip accents and straighten apostrophes.

    SÁNCHEZ must equal SANCHEZ, and congress-legislators writes O’Halleran with
    a curly apostrophe where the Record has O'HALLERAN.
    """
    name = name.replace("’", "'").replace("‘", "'")
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).upper()


@dataclass(frozen=True)
class Term:
    """One congress-legislators term, flattened with its member's identity."""

    bioguide: str
    icpsr: int | None
    last: str  # folded
    firsts: frozenset[str]  # folded first name, nickname, official first token
    chamber: str  # "rep" / "sen"
    state: str
    start: dt.date
    end: dt.date
    party: str
    caucus: str | None
    affiliations: tuple[tuple[dt.date, dt.date, str, str | None], ...]

    def party_on(self, day: dt.date) -> tuple[str, str | None]:
        """(party, caucus) on ``day``, honouring mid-term switches."""
        for start, end, party, caucus in self.affiliations:
            if start <= day <= end:
                return party, caucus
        return self.party, self.caucus


def load_terms(paths: Iterable[Path] = LEGISLATORS_FILES) -> list[Term]:
    """Flatten congress-legislators JSON into one Term per served term."""
    terms: list[Term] = []
    for path in paths:
        for person in json.loads(Path(path).read_text(encoding="utf-8")):
            name = person["name"]
            firsts = {_fold(name["first"])}
            if name.get("nickname"):
                firsts.add(_fold(name["nickname"]))
            if name.get("official_full"):
                firsts.add(_fold(name["official_full"].split()[0]))
            for term in person["terms"]:
                affiliations = tuple(
                    (
                        dt.date.fromisoformat(a["start"]),
                        dt.date.fromisoformat(a["end"]),
                        a["party"],
                        a.get("caucus"),
                    )
                    for a in term.get("party_affiliations") or ()
                )
                terms.append(
                    Term(
                        bioguide=person["id"]["bioguide"],
                        icpsr=person["id"].get("icpsr"),
                        last=_fold(name["last"]),
                        firsts=frozenset(firsts),
                        chamber=term["type"],
                        state=term["state"],
                        start=dt.date.fromisoformat(term["start"]),
                        end=dt.date.fromisoformat(term["end"]),
                        party=term.get("party", ""),
                        caucus=term.get("caucus"),
                        affiliations=affiliations,
                    )
                )
    return terms


@dataclass
class Resolution:
    """Outcome of resolving one speaker string."""

    status: str  # "resolved", "no_candidate", "ambiguous"
    term: Term | None = None


class MemberIndex:
    """Resolve a govinfo speaker string to one member, by chamber and date.

    Never guesses: a string that fits more than one member serving in that
    chamber on that date is "ambiguous" and stays unresolved.
    """

    def __init__(self, terms: Iterable[Term]) -> None:
        self._by_last: dict[str, list[Term]] = defaultdict(list)
        for term in terms:
            self._by_last[term.last].append(term)

    def _serving(self, last: str, chamber: str, day: dt.date) -> list[Term]:
        # Terms share their boundary day (one ends Jan 3, the next starts Jan 3),
        # so dedupe by member.
        found = {
            t.bioguide: t
            for t in self._by_last.get(last, ())
            if t.chamber == chamber and t.start <= day <= t.end
        }
        return list(found.values())

    def resolve(self, speaker: str, chamber: str, day: dt.date) -> Resolution:
        """Resolve ``speaker`` as it appears in the govinfo file.

        Handles the forms the file uses: a bare surname ("LEE"), a compound
        surname ("JACKSON LEE", "VAN HOLLEN"), a full name the Record gives to
        disambiguate ("CAROLYN B. MALONEY", "AUSTIN SCOTT"), and the "Manager"
        prefix for House impeachment managers speaking in the Senate.
        """
        name = _fold(speaker).strip()
        chamber_type = _CHAMBER_TYPES[chamber]
        if name.startswith("MANAGER "):
            name, chamber_type = name.removeprefix("MANAGER ").strip(), "rep"

        # Whole string as a surname first, so JACKSON LEE is not read as a
        # first name JACKSON plus surname LEE.
        candidates = self._serving(name, chamber_type, day)
        if not candidates and " " in name:
            tokens = name.replace(".", ". ").split()
            first = tokens[0]
            for split in range(1, len(tokens)):
                last = " ".join(tokens[split:])
                matches = [
                    t
                    for t in self._serving(last, chamber_type, day)
                    if first in t.firsts
                ]
                if matches:
                    candidates = matches
                    break

        if not candidates:
            return Resolution("no_candidate")
        if len(candidates) > 1:
            return Resolution("ambiguous")
        return Resolution("resolved", candidates[0])


# --- helpers -----------------------------------------------------------


def congress_for(day: dt.date) -> int:
    """Congress sitting on ``day``. A new Congress convenes January 3 of odd years."""
    year = day.year
    if year % 2 == 1 and (day.month, day.day) < (1, 3):
        year -= 1
    return (year - 1789) // 2 + 1


def speech_id_for(record: dict[str, object]) -> str:
    """Deterministic id from the record's content.

    The file has no id. A content hash survives re-ordering and partial reruns
    of the upstream notebook, unlike a line number. (date, chamber, speaker,
    speech) was checked unique across all 225,564 rows.
    """
    key = "\x1f".join(str(record[k]) for k in ("date", "chamber", "speaker", "speech"))
    return "gov-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# --- statistics --------------------------------------------------------


@dataclass
class GovinfoBuildStats:
    """Counts and parameters for one govinfo build.

    Serialized to ``results/metrics/govinfo_build_stats.json``. Every repair is
    counted, so the methodology chapter can say how much each one changed.
    """

    source_path: str
    source_bytes: int
    source_sha256: str
    legislators_sha256: dict[str, str]
    output_path: str
    built_at: str
    min_word_count: int
    start_date: str
    end_date: str
    limit: int | None
    source_tag: str = SOURCE_TAG
    rows_read: int = 0
    rows_written: int = 0
    rows_dropped_before_start: int = 0
    rows_dropped_after_end: int = 0
    rows_dropped_unresolved_no_candidate: int = 0
    rows_dropped_unresolved_ambiguous: int = 0
    rows_dropped_delegate: int = 0
    rows_dropped_excluded_member: int = 0
    rows_dropped_short: int = 0
    rows_dropped_duplicate_id: int = 0
    independents_reassigned: int = 0
    # Text repairs, counted over rows in the date window.
    rows_cut: dict[str, int] = field(default_factory=dict)
    words_removed_by_cleaning: int = 0
    words_before_cleaning: int = 0
    page_markers_removed: int = 0
    html_tags_removed: int = 0
    heading_lines_removed: int = 0
    # Speaker resolution, over rows in the date window.
    unresolved_by_chamber: dict[str, int] = field(default_factory=dict)
    unresolved_by_year: dict[str, int] = field(default_factory=dict)
    unresolved_top_speakers: dict[str, int] = field(default_factory=dict)
    file_party_vs_resolved: dict[str, int] = field(default_factory=dict)
    file_icpsr_disagreements: int = 0
    # Distributions of what was written.
    party_counts: dict[str, int] = field(default_factory=dict)
    party_counts_original: dict[str, int] = field(default_factory=dict)
    congress_counts: dict[str, int] = field(default_factory=dict)
    year_counts: dict[str, int] = field(default_factory=dict)
    chamber_counts: dict[str, int] = field(default_factory=dict)
    icpsr_missing: int = 0
    date_min: str | None = None
    date_max: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the stats."""
        return asdict(self)

    def write(self, path: Path = GOVINFO_BUILD_STATS_PATH) -> Path:
        """Write the stats to ``path`` as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


# --- build -------------------------------------------------------------


def _iter_records(src: Path, limit: int | None) -> Iterator[dict[str, object]]:
    """Stream JSONL records, checking each has the keys the build needs."""
    with src.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if limit is not None and line_no > limit:
                return
            record = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in record]
            if missing:
                raise ValueError(
                    f"{src}:{line_no} is missing key(s) {', '.join(missing)}"
                )
            if record["chamber"] not in _CHAMBER_TYPES:
                raise ValueError(
                    f"{src}:{line_no} has unknown chamber {record['chamber']!r}"
                )
            yield record


def _resolve_party(term: Term, day: dt.date) -> tuple[str | None, str, bool]:
    """Return (corpus party or None if excluded, original code, reassigned?).

    Raises on an independent no config rule covers, or one whose configured
    caucus contradicts congress-legislators -- a human decides, the build does
    not guess (same rule as the Stanford side).
    """
    party_name, caucus = term.party_on(day)
    if party_name not in PARTY_CODES:
        raise ValueError(
            f"Unknown party {party_name!r} for {term.bioguide} on {day}; "
            "add it to PARTY_CODES in code/src/govinfo.py."
        )
    original = PARTY_CODES[party_name]
    if original in EXPECTED_PARTIES:
        return original, original, False
    if term.bioguide in GOVINFO_EXCLUDED_MEMBERS:
        return None, original, False
    if term.bioguide in GOVINFO_CAUCUS_PARTY:
        assigned = GOVINFO_CAUCUS_PARTY[term.bioguide]
        if caucus is not None and PARTY_CODES.get(caucus) != assigned:
            raise ValueError(
                f"GOVINFO_CAUCUS_PARTY puts {term.bioguide} ({term.last}) "
                f"with {assigned}, but congress-legislators says they caucus "
                f"with {caucus} on {day}. Fix config.py."
            )
        return assigned, original, True
    raise ValueError(
        f"{term.bioguide} ({term.last}, {term.state}) is {party_name} "
        f"on {day} (caucus: {caucus}) and no rule covers them. Add them to "
        "GOVINFO_CAUCUS_PARTY or GOVINFO_EXCLUDED_MEMBERS in code/src/config.py "
        "and record the choice in docs/decisions.md. The build refuses to guess."
    )


def build_govinfo_corpus(
    src: Path = GOVINFO_JSONL,
    dst: Path = GOVINFO_CORPUS_PATH,
    legislators: tuple[Path, ...] = LEGISLATORS_FILES,
    min_word_count: int = MIN_WORD_COUNT,
    start_date: dt.date = GOVINFO_START_DATE,
    end_date: dt.date = GOVINFO_END_DATE,
    limit: int | None = None,
    overwrite: bool = False,
) -> GovinfoBuildStats:
    """Build the processed govinfo corpus.

    Args:
        src: Konsti's JSONL. Read only; never modified.
        dst: Output parquet, GOVINFO_SCHEMA.
        legislators: congress-legislators JSON files.
        min_word_count: Applied to the *cleaned* text, inclusive.
        start_date, end_date: Inclusive date window; rows outside are dropped.
        limit: Read at most this many raw rows, for smoke runs.
        overwrite: Required to replace an existing ``dst``.

    Raises:
        FileNotFoundError: an input is missing.
        FileExistsError: ``dst`` exists and ``overwrite`` is False.
        ValueError: a malformed record, or an independent no rule covers.
    """
    src, dst = Path(src), Path(dst)
    for path in (src, *legislators):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. The govinfo JSONL comes from the team drive "
                "(data/raw/govinfo/README.md); the legislators JSON from "
                "https://unitedstates.github.io/congress-legislators/."
            )
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"{dst} already exists; pass overwrite=True to replace it."
        )

    stats = GovinfoBuildStats(
        source_path=str(src),
        source_bytes=src.stat().st_size,
        source_sha256=_fingerprint(src),
        legislators_sha256={Path(p).name: _fingerprint(Path(p)) for p in legislators},
        output_path=str(dst),
        built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        min_word_count=min_word_count,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        limit=limit,
    )
    index = MemberIndex(load_terms(legislators))

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    seen_ids: set[str] = set()
    buffer: list[dict[str, object]] = []

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(dst, GOVINFO_SCHEMA, compression="snappy")

    def flush() -> None:
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=GOVINFO_SCHEMA))
            buffer.clear()

    try:
        for record in _iter_records(src, limit):
            stats.rows_read += 1
            day = dt.date.fromisoformat(str(record["date"]))
            if day < start_date:
                stats.rows_dropped_before_start += 1
                continue
            if day > end_date:
                stats.rows_dropped_after_end += 1
                continue

            speaker, chamber = str(record["speaker"]), str(record["chamber"])
            resolution = index.resolve(speaker, chamber, day)
            file_party = str(record.get("party"))
            if resolution.term is None:
                if resolution.status == "ambiguous":
                    stats.rows_dropped_unresolved_ambiguous += 1
                else:
                    stats.rows_dropped_unresolved_no_candidate += 1
                counters["unresolved_chamber"][chamber] += 1
                counters["unresolved_year"][str(day.year)] += 1
                counters["unresolved_speaker"][f"{speaker} ({chamber})"] += 1
                counters["file_vs_resolved"][f"{file_party} -> unresolved"] += 1
                continue
            term = resolution.term

            file_icpsr = record.get("icpsr")
            if file_icpsr is not None and file_icpsr != term.icpsr:
                stats.file_icpsr_disagreements += 1
            # Before party resolution: a delegate's label (Sablan was an
            # Independent) must not trip the independents check.
            if term.state in DELEGATE_STATES:
                stats.rows_dropped_delegate += 1
                counters["file_vs_resolved"][f"{file_party} -> delegate"] += 1
                continue

            party, party_original, reassigned = _resolve_party(term, day)
            counters["file_vs_resolved"][f"{file_party} -> {party_original}"] += 1
            if party is None:
                stats.rows_dropped_excluded_member += 1
                continue

            raw_speech = str(record["speech"])
            cleaned = clean_speech(raw_speech)
            stats.words_before_cleaning += len(raw_speech.split())
            stats.words_removed_by_cleaning += cleaned.words_removed
            counters["cut"][cleaned.cut_at or "none"] += 1
            stats.page_markers_removed += cleaned.page_markers
            stats.html_tags_removed += cleaned.html_tags
            stats.heading_lines_removed += cleaned.headings

            word_count = len(cleaned.text.split())
            if word_count < min_word_count:
                stats.rows_dropped_short += 1
                continue

            speech_id = speech_id_for(record)
            if speech_id in seen_ids:
                stats.rows_dropped_duplicate_id += 1
                continue
            seen_ids.add(speech_id)

            stats.independents_reassigned += reassigned
            congress = congress_for(day)
            buffer.append(
                {
                    "speech_id": speech_id,
                    "date": day,
                    "member_id": term.bioguide,
                    "party": party,
                    "chamber": _CHAMBER_CODES[chamber],
                    "congress_number": congress,
                    "text": cleaned.text,
                    "source": SOURCE_TAG,
                    "last_name": term.last,
                    "state": term.state,
                    "word_count": word_count,
                    "party_original": party_original,
                    "icpsr": term.icpsr,
                }
            )
            stats.rows_written += 1
            counters["party"][party] += 1
            counters["party_original"][party_original] += 1
            counters["congress"][str(congress)] += 1
            counters["year"][str(day.year)] += 1
            counters["chamber"][_CHAMBER_CODES[chamber]] += 1
            stats.icpsr_missing += term.icpsr is None
            iso = day.isoformat()
            stats.date_min = min(stats.date_min or iso, iso)
            stats.date_max = max(stats.date_max or iso, iso)

            if len(buffer) >= BATCH_SIZE:
                flush()
        flush()
    except BaseException:
        # Never leave a half-written corpus behind for a later run to trust.
        writer.close()
        dst.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    stats.rows_cut = dict(sorted(counters["cut"].items()))
    stats.unresolved_by_chamber = dict(sorted(counters["unresolved_chamber"].items()))
    stats.unresolved_by_year = dict(sorted(counters["unresolved_year"].items()))
    stats.unresolved_top_speakers = dict(counters["unresolved_speaker"].most_common(40))
    stats.file_party_vs_resolved = dict(counters["file_vs_resolved"].most_common())
    stats.party_counts = dict(sorted(counters["party"].items()))
    stats.party_counts_original = dict(sorted(counters["party_original"].items()))
    stats.congress_counts = dict(sorted(counters["congress"].items()))
    stats.year_counts = dict(sorted(counters["year"].items()))
    stats.chamber_counts = dict(sorted(counters["chamber"].items()))
    return stats
