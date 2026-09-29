"""Crosswalk from speech-corpus members to Voteview ICPSR ids (DW-NOMINATE).

The DW-NOMINATE validation (docs/decisions.md M5) needs every speech tied to a
Voteview member record, which is keyed on ICPSR. Neither side of the corpus
carries a reliable ICPSR, so this module builds the link once, at the level of
distinct members, and writes it as its own table. The corpus is never modified.

The unit
--------
One row per ``source`` x ``member_id`` x ``congress_number`` x ``chamber`` x
``party_original`` -- the key the crosswalk joins back onto the corpus with.
``party_original`` is in the key because govinfo records party by day: Van Drew
has a Democratic and a Republican unit in the 116th, and Voteview a separate
ICPSR for each.

Stanford side (member_id = Gentzkow's speakerid)
------------------------------------------------
Matched on surname + state + Congress + chamber, never guessing (D21):

1. Candidates: Voteview members of that Congress, chamber and state whose
   normalized surname equals *any* spelling the speakerid carries -- OCR
   variants (HODES/RHODES) differ within one speakerid. If none, members whose
   surname shares a word with it: Stanford keeps only the last word of compound
   surnames (JACKSON LEE -> LEE, VAN HOLLEN -> HOLLEN).
2. One candidate: matched. Several: ``party_original`` breaks the tie (D23).
3. Still several (same surname, state and party -- the Sanchez sisters): the
   speakerid's person part, which is stable across Congresses, carries a match
   made in another Congress over (D22); then, if every candidate but one is
   taken by another member of that Congress, the last one is assigned.
4. Anything left is unmatched, with the reason, never forced.

govinfo side (member_id = bioguide id)
--------------------------------------
Joined on bioguide id + Congress + chamber, with ``party_original`` breaking a
switcher's tie. The corpus's own ``icpsr`` (from congress-legislators) is only
cross-checked: it is the member's original number, so a switcher's later
speeches would join to their old party's record (D21).

What fails loudly
-----------------
A speakerid whose person part points to two people, two members matched to one
ICPSR in the same Congress and chamber, an unknown ``source`` value, a
repeated Voteview key. Each is a broken assumption, not a case to paper over.

Unmatched and unscored members are excluded from the validation only
(``in_validation``); every speech stays in the trend analysis.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    CORPUS_PATH,
    CROSSWALK_PATH,
    CROSSWALK_STATS_PATH,
    CROSSWALK_UNMATCHED_PATH,
    DW_NOMINATE_MEMBERS,
    VOTEVIEW_CONGRESS_RANGE,
    VOTEVIEW_PARTY_CODES,
)
from src.corpus import SOURCE_TAG as STANFORD_TAG
from src.corpus import _fingerprint
from src.govinfo import SOURCE_TAG as GOVINFO_TAG
from src.govinfo import _fold

#: The Voteview columns kept -- exactly those of the hand-made
#: data/processed/dw_nominate_107_119.csv, so both give the same table.
VOTEVIEW_COLUMNS: tuple[str, ...] = (
    "congress",
    "chamber",
    "icpsr",
    "bioguide_id",
    "bioname",
    "state_abbrev",
    "district_code",
    "party_code",
    "nominate_dim1",
    "nominate_dim2",
    "nokken_poole_dim1",
    "nominate_number_of_votes",
)

_VOTEVIEW_CHAMBERS: dict[str, str] = {"House": "H", "Senate": "S"}

#: The crosswalk's key, and the columns it joins onto the corpus with.
UNIT_KEY: tuple[str, ...] = (
    "source",
    "member_id",
    "congress_number",
    "chamber",
    "party_original",
)

CROSSWALK_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("member_id", pa.string()),
        ("congress_number", pa.int16()),
        ("chamber", pa.string()),
        ("party_original", pa.string()),
        ("state", pa.string()),
        ("last_name", pa.string()),  # most common spelling in the corpus
        ("n_speeches", pa.int32()),
        ("icpsr", pa.int32()),  # null when unmatched
        ("bioguide_id", pa.string()),
        ("bioname", pa.string()),
        ("voteview_party_code", pa.int16()),
        ("nominate_dim1", pa.float64()),
        ("nokken_poole_dim1", pa.float64()),
        # How candidates were found: name_exact, name_token, bioguide.
        ("candidate_key", pa.string()),
        # How one was chosen: unique, party, speakerid, elimination.
        ("resolved_by", pa.string()),
        ("unmatched_reason", pa.string()),  # no_candidate, ambiguous
        ("n_candidates", pa.int16()),
        # Voteview has two ICPSRs for this member in this Congress and chamber.
        ("party_switch", pa.bool_()),
        ("has_nominate", pa.bool_()),  # matched and scored (nominate_dim1)
        ("in_validation", pa.bool_()),  # the only rows the validation may use
    ]
)

_CORPUS_COLUMNS: list[str] = [*UNIT_KEY, "state", "last_name", "icpsr"]

_UNMATCHED_FIELDS: tuple[str, ...] = (
    "source",
    "member_id",
    "congress_number",
    "chamber",
    "state",
    "party_original",
    "last_name",
    "name_variants",
    "n_speeches",
    "unmatched_reason",
    "candidates",
)

_NON_LETTER = re.compile(r"[^A-Z ]")

#: How many offending values an error message lists.
_SHOW = 5


# --- names -------------------------------------------------------------


def surname_tokens(name: str) -> tuple[str, ...]:
    """Split a surname into comparable words.

    Uppercase, accents stripped, hyphens split, everything but letters dropped:
    ``Díaz-Balart`` -> ``("DIAZ", "BALART")``, ``O’Halleran`` ->
    ``("OHALLERAN",)``. Dropping stray dots also mends OCR noise such as
    ``SA.NDERS``.
    """
    return tuple(_NON_LETTER.sub("", _fold(name).replace("-", " ")).split())


def _surname_key(name: str) -> str:
    """The surname with its words run together, for exact comparison.

    Joining makes ``DE LAURO`` equal ``DELAURO`` and ``DIAZ-BALART`` equal
    ``DIAZ BALART``.
    """
    return "".join(surname_tokens(name))


# --- inputs ------------------------------------------------------------


def load_voteview(
    path: Path = DW_NOMINATE_MEMBERS,
    congress_range: tuple[int, int] = VOTEVIEW_CONGRESS_RANGE,
) -> pd.DataFrame:
    """Voteview members of the House and Senate in ``congress_range``.

    Reads the raw ``HSall_members.csv`` without modifying it. The result is the
    same table as data/processed/dw_nominate_107_119.csv, up to row order.

    Raises:
        FileNotFoundError: ``path`` is missing.
        ValueError: a needed column is missing, or a (congress, chamber, icpsr)
            key repeats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download HSall_members.csv from "
            f"https://voteview.com/data into {path.parent}."
        )
    frame = pd.read_csv(path)
    missing = [c for c in VOTEVIEW_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}")
    first, last = congress_range
    keep = frame["congress"].between(first, last) & frame["chamber"].isin(
        list(_VOTEVIEW_CHAMBERS)
    )
    frame = frame.loc[keep, list(VOTEVIEW_COLUMNS)].reset_index(drop=True)
    repeated = frame[frame.duplicated(["congress", "chamber", "icpsr"], keep=False)]
    if len(repeated):
        shown = (
            repeated[["congress", "chamber", "icpsr"]].head(_SHOW).to_dict("records")
        )
        raise ValueError(
            f"{path}: {len(repeated)} rows repeat a (congress, chamber, icpsr) key, "
            f"e.g. {shown}"
        )
    return frame


def _most_common(values: pd.Series) -> str:
    """Most frequent value; ties go to the alphabetically first, for stability."""
    counts = values.value_counts()
    return min(counts[counts == counts.max()].index)


def _load_units(corpus: Path) -> pd.DataFrame:
    """Aggregate the corpus to one row per unit, reading only small columns."""
    rows = pq.read_table(corpus, columns=_CORPUS_COLUMNS).to_pandas()
    unknown = sorted(set(rows["source"]) - {STANFORD_TAG, GOVINFO_TAG})
    if unknown:
        raise ValueError(
            f"{corpus}: unknown source value(s) {unknown[:_SHOW]}; the crosswalk "
            f"knows only {STANFORD_TAG!r} and {GOVINFO_TAG!r}"
        )
    units = (
        rows.groupby(list(UNIT_KEY), sort=True)
        .agg(
            n_speeches=("last_name", "size"),
            n_states=("state", "nunique"),
            state=("state", "first"),
            last_name=("last_name", _most_common),
            name_variants=("last_name", lambda s: tuple(sorted(set(s)))),
            corpus_icpsr=("icpsr", "max"),
        )
        .reset_index()
    )
    split = units[units["n_states"] > 1]
    if len(split):
        raise ValueError(
            f"{len(split)} member(s) carry more than one state within a single "
            f"Congress and chamber, e.g. {split['member_id'].head(_SHOW).tolist()}"
        )
    _check_speakerids(units[units["source"] == STANFORD_TAG])
    return units


def _check_speakerids(stanford: pd.DataFrame) -> None:
    """The speakerid layout D22 relies on: congress (3 digits) + person part.

    One unit per speakerid, and its first three digits are its Congress. If a
    replacement dataset breaks this, carrying matches across Congresses would
    be wrong, so the build stops.
    """
    repeated = stanford[stanford["member_id"].duplicated(keep=False)]
    if len(repeated):
        raise ValueError(
            "speakerid(s) appear under more than one chamber, state or party: "
            f"{sorted(set(repeated['member_id']))[:_SHOW]}"
        )
    bad = [
        member_id
        for member_id, congress in zip(
            stanford["member_id"], stanford["congress_number"]
        )
        if not (member_id.isdigit() and member_id[:3] == str(congress))
    ]
    if bad:
        raise ValueError(
            f"speakerid(s) {bad[:_SHOW]} do not start with their Congress; the "
            "speakerid layout the crosswalk relies on (docs/decisions.md D22) "
            "does not hold"
        )


# --- matching ----------------------------------------------------------


@dataclass
class _Match:
    """Working state for one unit."""

    candidates: list[int]  # ICPSRs found by the candidate step
    candidate_key: str | None
    remaining: list[int] = field(default_factory=list)  # after the party step
    icpsr: int | None = None
    resolved_by: str | None = None

    def resolve(self, icpsr: int, how: str) -> None:
        self.icpsr, self.resolved_by = icpsr, how


class _Voteview:
    """Voteview rows indexed the ways the matching needs."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.rows: dict[tuple[int, str, int], dict[str, object]] = {}
        self.by_state: dict[tuple[int, str, str], list[dict[str, object]]] = (
            defaultdict(list)
        )
        self.by_bioguide: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        for record in frame.to_dict("records"):
            congress = int(record["congress"])
            chamber = _VOTEVIEW_CHAMBERS[record["chamber"]]
            icpsr = int(record["icpsr"])
            surname = str(record["bioname"]).split(",")[0]
            record["_tokens"] = frozenset(surname_tokens(surname))
            record["_key"] = _surname_key(surname)
            self.rows[(congress, chamber, icpsr)] = record
            self.by_state[(congress, chamber, record["state_abbrev"])].append(record)
            self.by_bioguide[(congress, chamber, record["bioguide_id"])].append(icpsr)

    def party_code(self, congress: int, chamber: str, icpsr: int) -> float:
        return self.rows[(congress, chamber, icpsr)]["party_code"]

    def bioguide(self, congress: int, chamber: str, icpsr: int) -> str:
        return str(self.rows[(congress, chamber, icpsr)]["bioguide_id"])


def _first_pass(
    match: _Match, congress: int, chamber: str, party: str, vv: _Voteview
) -> None:
    """Resolve a unique candidate, or break a tie on ``party_original``."""
    candidates = sorted(set(match.candidates))
    if len(candidates) == 1:
        match.resolve(candidates[0], "unique")
        return
    code = VOTEVIEW_PARTY_CODES.get(party)
    same_party = [i for i in candidates if vv.party_code(congress, chamber, i) == code]
    if len(same_party) == 1:
        match.resolve(same_party[0], "party")
        return
    match.remaining = same_party or candidates


def _stanford_candidates(unit, vv: _Voteview) -> _Match:
    """Voteview members in the unit's Congress, chamber and state, by surname."""
    block = vv.by_state.get((unit.congress_number, unit.chamber, unit.state), [])
    spellings = {_surname_key(v) for v in unit.name_variants} - {""}
    exact = [int(r["icpsr"]) for r in block if r["_key"] in spellings]
    if exact:
        return _Match(exact, "name_exact")
    words = {w for v in unit.name_variants for w in surname_tokens(v)}
    shared = [int(r["icpsr"]) for r in block if words & r["_tokens"]]
    if shared:
        return _Match(shared, "name_token")
    return _Match([], None)


def _person_bioguides(units: pd.DataFrame, matches: list[_Match], vv) -> dict:
    """speakerid person part -> the one bioguide its matched units point to.

    Raises if a person part points to two people: then the speakerid does not
    identify a person, and no match may be carried across Congresses on it.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for unit, match in zip(units.itertuples(), matches):
        if match.icpsr is not None:
            bioguide = vv.bioguide(unit.congress_number, unit.chamber, match.icpsr)
            seen[unit.member_id[3:]].add(bioguide)
    conflicts = {pid: sorted(b) for pid, b in seen.items() if len(b) > 1}
    if conflicts:
        shown = dict(sorted(conflicts.items())[:_SHOW])
        raise ValueError(
            "speakerid person part(s) matched to more than one member across "
            f"Congresses: {shown}. Carrying matches across Congresses on the "
            "speakerid (docs/decisions.md D22) is unsafe for them."
        )
    return {pid: next(iter(b)) for pid, b in seen.items()}


def _match_stanford(units: pd.DataFrame, vv: _Voteview) -> list[_Match]:
    """Name candidates, party tie-break, then speakerid and elimination."""
    rows = list(units.itertuples())
    matches = [_stanford_candidates(unit, vv) for unit in rows]
    for unit, match in zip(rows, matches):
        if match.candidates:
            _first_pass(
                match, unit.congress_number, unit.chamber, unit.party_original, vv
            )

    # Iterate: a match made by elimination can let the same person be carried
    # into another Congress, and vice versa.
    changed = True
    while changed:
        changed = False
        persons = _person_bioguides(units, matches, vv)
        for unit, match in zip(rows, matches):
            if match.icpsr is None and match.remaining:
                bioguide = persons.get(unit.member_id[3:])
                same = [
                    i
                    for i in match.remaining
                    if vv.bioguide(unit.congress_number, unit.chamber, i) == bioguide
                ]
                if bioguide is not None and len(same) == 1:
                    match.resolve(same[0], "speakerid")
                    changed = True

        taken: dict[tuple[int, str], set[int]] = defaultdict(set)
        for unit, match in zip(rows, matches):
            if match.icpsr is not None:
                taken[(unit.congress_number, unit.chamber)].add(match.icpsr)
        for unit, match in zip(rows, matches):
            if match.icpsr is None and match.remaining:
                seat = (unit.congress_number, unit.chamber)
                free = [i for i in match.remaining if i not in taken[seat]]
                if len(free) == 1:
                    match.resolve(free[0], "elimination")
                    taken[seat].add(free[0])
                    changed = True

    _person_bioguides(units, matches, vv)  # re-check after elimination
    return matches


def _match_govinfo(units: pd.DataFrame, vv: _Voteview) -> list[_Match]:
    """bioguide + Congress + chamber, party tie-break for switchers."""
    matches = []
    for unit in units.itertuples():
        key = (unit.congress_number, unit.chamber, unit.member_id)
        match = _Match(list(vv.by_bioguide.get(key, [])), "bioguide")
        if match.candidates:
            _first_pass(
                match, unit.congress_number, unit.chamber, unit.party_original, vv
            )
        matches.append(match)
    return matches


def _check_one_to_one(units: pd.DataFrame, matches: list[_Match]) -> None:
    """No ICPSR may be claimed by two members of one Congress and chamber."""
    claims: dict[tuple[str, int, str, int], set[str]] = defaultdict(set)
    for unit, match in zip(units.itertuples(), matches):
        if match.icpsr is not None:
            seat = (unit.source, unit.congress_number, unit.chamber, match.icpsr)
            claims[seat].add(unit.member_id)
    clashes = {k: sorted(v) for k, v in claims.items() if len(v) > 1}
    if clashes:
        shown = [
            f"ICPSR {icpsr} ({source}, {congress}th, {chamber}): {', '.join(ids)}"
            for (source, congress, chamber, icpsr), ids in sorted(clashes.items())
        ][:_SHOW]
        raise ValueError(
            "ICPSR(s) matched to more than one member of the same Congress and "
            f"chamber -- {'; '.join(shown)}"
        )


# --- statistics --------------------------------------------------------


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


@dataclass
class CrosswalkBuildStats:
    """Match rates and the members the validation cannot use.

    Serialized to ``results/metrics/crosswalk_build_stats.json``; the unmatched
    members also go to ``crosswalk_unmatched.csv``, for reading.
    """

    output_path: str
    built_at: str
    inputs: dict[str, dict[str, object]] = field(default_factory=dict)
    # source -> units / speeches, matched / in validation, and rates
    summary: dict[str, dict[str, object]] = field(default_factory=dict)
    # source -> congress -> chamber -> counts
    by_congress_chamber: dict[str, dict[str, dict[str, dict[str, int]]]] = field(
        default_factory=dict
    )
    # source -> resolved_by (or "unmatched: <reason>") -> units / speeches
    methods: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    party_switch: list[dict[str, object]] = field(default_factory=list)
    no_nominate: list[dict[str, object]] = field(default_factory=list)
    icpsr_disagreements: list[dict[str, object]] = field(default_factory=list)
    unmatched: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of the stats."""
        return asdict(self)

    def write(self, path: Path = CROSSWALK_STATS_PATH) -> Path:
        """Write the stats to ``path`` as indented JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    def write_unmatched(self, path: Path = CROSSWALK_UNMATCHED_PATH) -> Path:
        """Write the unmatched members, most speeches first, as CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_UNMATCHED_FIELDS)
            writer.writeheader()
            writer.writerows(self.unmatched)
        return path


def _collect_stats(stats: CrosswalkBuildStats, rows: list[dict]) -> None:
    """Fill rates and lists from the finished crosswalk rows."""
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        source, congress, chamber = (
            row["source"],
            str(row["congress_number"]),
            row["chamber"],
        )
        matched = row["icpsr"] is not None
        speeches = row["n_speeches"]

        total = stats.summary.setdefault(
            source,
            dict.fromkeys(
                (
                    "units",
                    "units_matched",
                    "units_in_validation",
                    "speeches",
                    "speeches_matched",
                    "speeches_in_validation",
                ),
                0,
            ),
        )
        cell = cells[source][congress].setdefault(
            chamber,
            dict.fromkeys(
                (
                    "units",
                    "units_matched",
                    "speeches",
                    "speeches_matched",
                    "speeches_in_validation",
                ),
                0,
            ),
        )
        for counts in (total, cell):
            counts["units"] += 1
            counts["units_matched"] += matched
            counts["speeches"] += speeches
            counts["speeches_matched"] += speeches * matched
            counts["speeches_in_validation"] += speeches * row["in_validation"]
        total["units_in_validation"] += row["in_validation"]

        method = row["resolved_by"] or f"unmatched: {row['unmatched_reason']}"
        tally = stats.methods.setdefault(source, {}).setdefault(
            method, {"units": 0, "speeches": 0}
        )
        tally["units"] += 1
        tally["speeches"] += speeches

        brief = {k: row[k] for k in (*UNIT_KEY, "icpsr", "bioname", "n_speeches")}
        if row["party_switch"]:
            stats.party_switch.append(brief)
        if matched and not row["has_nominate"]:
            stats.no_nominate.append(brief)

    for total in stats.summary.values():
        total["unit_match_rate"] = _rate(total["units_matched"], total["units"])
        total["speech_match_rate"] = _rate(total["speeches_matched"], total["speeches"])
        total["speech_validation_rate"] = _rate(
            total["speeches_in_validation"], total["speeches"]
        )
    stats.by_congress_chamber = {
        source: {
            congress: dict(sorted(chambers.items()))
            for congress, chambers in sorted(
                by_congress.items(), key=lambda i: int(i[0])
            )
        }
        for source, by_congress in sorted(cells.items())
    }


# --- build -------------------------------------------------------------


def build_crosswalk(
    corpus: Path = CORPUS_PATH,
    voteview: Path = DW_NOMINATE_MEMBERS,
    dst: Path = CROSSWALK_PATH,
    *,
    congress_range: tuple[int, int] = VOTEVIEW_CONGRESS_RANGE,
    overwrite: bool = False,
) -> CrosswalkBuildStats:
    """Match every corpus member to a Voteview ICPSR and write the crosswalk.

    Args:
        corpus: The merged corpus (``make corpus``). Only its small columns are
            read.
        voteview: Voteview's ``HSall_members.csv``. Read only.
        dst: Output parquet, ``CROSSWALK_SCHEMA``.
        congress_range: Voteview Congresses to match against.
        overwrite: Required to replace an existing ``dst``.

    Returns:
        Match rates and lists for the caller to persist; see
        ``CrosswalkBuildStats.write`` and ``write_unmatched``.

    Raises:
        FileNotFoundError: an input is missing.
        FileExistsError: ``dst`` exists and ``overwrite`` is False.
        ValueError: an unknown source, a speakerid that breaks the layout or
            points to two people, two members on one ICPSR, a bad Voteview file.
    """
    corpus, voteview, dst = Path(corpus), Path(voteview), Path(dst)
    if not corpus.exists():
        raise FileNotFoundError(
            f"The merged corpus is not at {corpus}. Build it first: make corpus"
        )
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"{dst} already exists; pass overwrite=True to replace it."
        )

    units = _load_units(corpus)
    members = load_voteview(voteview, congress_range)
    vv = _Voteview(members)

    stanford = units[units["source"] == STANFORD_TAG].reset_index(drop=True)
    govinfo = units[units["source"] == GOVINFO_TAG].reset_index(drop=True)
    units = pd.concat([stanford, govinfo], ignore_index=True)
    matches = _match_stanford(stanford, vv) + _match_govinfo(govinfo, vv)
    _check_one_to_one(units, matches)

    rows: list[dict[str, object]] = []
    stats = CrosswalkBuildStats(
        output_path=str(dst),
        built_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
    for unit, match in zip(units.itertuples(), matches):
        congress, chamber = int(unit.congress_number), unit.chamber
        record = (
            vv.rows[(congress, chamber, match.icpsr)] if match.icpsr is not None else {}
        )
        bioguide = record.get("bioguide_id")
        dim1 = record.get("nominate_dim1")
        has_nominate = match.icpsr is not None and pd.notna(dim1)
        n_candidates = len(set(match.candidates))
        row = {
            "source": unit.source,
            "member_id": unit.member_id,
            "congress_number": congress,
            "chamber": chamber,
            "party_original": unit.party_original,
            "state": unit.state,
            "last_name": unit.last_name,
            "n_speeches": int(unit.n_speeches),
            "icpsr": match.icpsr,
            "bioguide_id": bioguide,
            "bioname": record.get("bioname"),
            "voteview_party_code": (
                int(record["party_code"]) if match.icpsr is not None else None
            ),
            "nominate_dim1": float(dim1) if has_nominate else None,
            "nokken_poole_dim1": (
                float(record["nokken_poole_dim1"])
                if match.icpsr is not None and pd.notna(record["nokken_poole_dim1"])
                else None
            ),
            "candidate_key": match.candidate_key,
            "resolved_by": match.resolved_by,
            "unmatched_reason": (
                None
                if match.icpsr is not None
                else "no_candidate" if n_candidates == 0 else "ambiguous"
            ),
            "n_candidates": n_candidates,
            "party_switch": match.icpsr is not None
            and len(vv.by_bioguide[(congress, chamber, bioguide)]) > 1,
            "has_nominate": bool(has_nominate),
            "in_validation": bool(has_nominate),
        }
        rows.append(row)

        if match.icpsr is None:
            stats.unmatched.append(
                {
                    **{k: row[k] for k in _UNMATCHED_FIELDS if k in row},
                    "name_variants": " | ".join(unit.name_variants),
                    "candidates": " | ".join(
                        f"{vv.rows[(congress, chamber, i)]['bioname']} ({i})"
                        for i in sorted(set(match.candidates))
                    ),
                }
            )
        corpus_icpsr = unit.corpus_icpsr
        if (
            match.icpsr is not None
            and pd.notna(corpus_icpsr)
            and int(corpus_icpsr) != match.icpsr
        ):
            stats.icpsr_disagreements.append(
                {
                    **{k: row[k] for k in UNIT_KEY},
                    "corpus_icpsr": int(corpus_icpsr),
                    "icpsr": match.icpsr,
                    "bioname": row["bioname"],
                    "n_speeches": row["n_speeches"],
                }
            )

    rows.sort(key=lambda r: tuple(r[k] for k in UNIT_KEY))
    stats.unmatched.sort(key=lambda r: (-r["n_speeches"], r["source"], r["member_id"]))
    _collect_stats(stats, rows)
    stats.inputs = {
        "corpus": {
            "path": str(corpus),
            "bytes": corpus.stat().st_size,
            "sha256": _fingerprint(corpus),
            "rows": int(units["n_speeches"].sum()),
        },
        "voteview": {
            "path": str(voteview),
            "bytes": voteview.stat().st_size,
            "sha256": _fingerprint(voteview),
            "rows": len(members),
            "congress_range": list(congress_range),
        },
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=CROSSWALK_SCHEMA), dst)
    return stats
