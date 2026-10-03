"""Tests for the govinfo -> corpus build.

Fixtures are synthetic and built in-process, so the suite runs without the
763 MB JSONL. They reproduce the upstream defects found on 2026-09-27 (see
docs/notes/2026-09-27_govinfo_data_reality.md): officer turns glued onto
speeches, HTML and page-marker residue, and a surname shared across chambers.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.govinfo import (
    GOVINFO_SCHEMA,
    MemberIndex,
    build_govinfo_corpus,
    clean_speech,
    congress_for,
    load_terms,
    state_code,
)

WORDS = " ".join(["word"] * 60)
SEP = "                          ____________________"

# --- fixture helpers ---------------------------------------------------


def _person(
    bioguide: str,
    last: str,
    first: str,
    terms: list[dict[str, object]],
    icpsr: int | None = None,
) -> dict[str, object]:
    return {
        "id": {"bioguide": bioguide, "icpsr": icpsr},
        "name": {"first": first, "last": last},
        "terms": terms,
    }


def _term(
    kind: str,
    state: str = "NY",
    party: str = "Democrat",
    start: str = "2017-01-03",
    end: str = "2019-01-03",
    **extra: object,
) -> dict[str, object]:
    return {
        "type": kind,
        "state": state,
        "party": party,
        "start": start,
        "end": end,
        **extra,
    }


LEGISLATORS = [
    # One senator LEE, one representative LEE: the chamber must decide.
    _person("S1", "Lee", "Mike", [_term("sen", "UT", "Republican")], icpsr=1),
    _person("H1", "Lee", "Barbara", [_term("rep", "CA")], icpsr=2),
    # Two representatives SMITH: genuinely ambiguous in the House.
    _person("H2", "Smith", "Adam", [_term("rep", "WA")], icpsr=3),
    _person("H3", "Smith", "Jason", [_term("rep", "MO", "Republican")], icpsr=4),
    _person("H4", "Jackson Lee", "Sheila", [_term("rep", "TX")], icpsr=5),
    _person("H5", "Maloney", "Carolyn", [_term("rep", "NY")], icpsr=6),
    _person("H6", "Maloney", "Sean", [_term("rep", "NY")], icpsr=7),
    _person("H7", "O’Halleran", "Tom", [_term("rep", "AZ")], icpsr=8),
    _person("H8", "Schiff", "Adam", [_term("rep", "CA")], icpsr=9),
    # Mid-term switch, Van Drew-style: D, then R from 2018-01-01.
    _person(
        "H9",
        "Switcher",
        "Jeff",
        [
            _term(
                "rep",
                "NJ",
                "Republican",
                party_affiliations=[
                    {"start": "2017-01-03", "end": "2017-12-31", "party": "Democrat"},
                    {"start": "2018-01-01", "end": "2019-01-03", "party": "Republican"},
                ],
            )
        ],
        icpsr=10,
    ),
    # Sanders: independent, caucus D, listed in GOVINFO_CAUCUS_PARTY.
    _person(
        "S000033",
        "Sanders",
        "Bernard",
        [_term("sen", "VT", "Independent", caucus="Democrat")],
        icpsr=29147,
    ),
    # Amash: listed in GOVINFO_EXCLUDED_MEMBERS; R until 2018-01-01.
    _person(
        "A000367",
        "Amash",
        "Justin",
        [
            _term(
                "rep",
                "MI",
                "Independent",
                party_affiliations=[
                    {"start": "2017-01-03", "end": "2017-12-31", "party": "Republican"},
                    {
                        "start": "2018-01-01",
                        "end": "2019-01-03",
                        "party": "Independent",
                    },
                ],
            )
        ],
        icpsr=21143,
    ),
    # An independent delegate: must be dropped, never trip the rule check.
    _person("D1", "Sablan", "Gregorio", [_term("rep", "MP", "Independent")]),
    # An independent no rule covers.
    _person("X1", "Maverick", "Pat", [_term("sen", "AK", "Independent")]),
]


def _record(
    speaker: str,
    chamber: str = "HOUSE",
    date: str = "2017-06-05",
    speech: str | None = None,
    **extra: object,
) -> dict[str, object]:
    return {
        "date": date,
        "speaker": speaker,
        "party": None,
        "icpsr": None,
        "state": None,
        "chamber": chamber,
        "speech": speech if speech is not None else f"Mr. Speaker, {WORDS}",
        **extra,
    }


def _write(tmp_path: Path, records: list[dict[str, object]]) -> tuple[Path, Path]:
    src = tmp_path / "speeches.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    leg = tmp_path / "legislators.json"
    leg.write_text(json.dumps(LEGISLATORS))
    return src, leg


def _build(tmp_path: Path, records: list[dict[str, object]], **kwargs: object):
    src, leg = _write(tmp_path, records)
    dst = tmp_path / "out.parquet"
    stats = build_govinfo_corpus(src=src, dst=dst, legislators=(leg,), **kwargs)
    return stats, (pq.read_table(dst) if dst.exists() else None)


@pytest.fixture()
def index(tmp_path: Path) -> MemberIndex:
    _, leg = _write(tmp_path, [])
    return MemberIndex(load_terms((leg,)))


# --- text cleaning -----------------------------------------------------


def test_speech_is_cut_at_the_first_officer_turn() -> None:
    raw = (
        "Mr. President, I ask unanimous consent.\n"
        "  The PRESIDING OFFICER. Without objection, it is so ordered.\n"
    )
    result = clean_speech(raw)
    assert result.text == "Mr. President, I ask unanimous consent."
    assert result.cut_at == "officer"


def test_officer_turn_naming_the_occupant_is_cut() -> None:
    raw = "Mr. Chair, I yield.\n  The Acting CHAIR (Mr. Doe). The gentleman yields."
    assert clean_speech(raw).text == "Mr. Chair, I yield."


def test_member_paragraph_mentioning_the_speaker_is_not_cut() -> None:
    raw = "Mr. Speaker, thank you.\n  The Speaker said last week that we would act."
    result = clean_speech(raw)
    assert result.cut_at is None
    assert "last week" in result.text


def test_speech_is_cut_at_record_narration() -> None:
    raw = (
        "Mr. Speaker, I call up the bill H.R. 1.\n"
        "  The Clerk read the title of the bill.\n"
        "  The text of the bill is as follows: Be it enacted..."
    )
    result = clean_speech(raw)
    assert result.text == "Mr. Speaker, I call up the bill H.R. 1."
    assert result.cut_at == "narration"


def test_member_paragraph_starting_with_the_is_not_narration() -> None:
    raw = "Mr. Speaker, first.\n  The amendment I offer today fixes this."
    assert clean_speech(raw).cut_at is None


def test_speech_is_cut_at_a_long_separator() -> None:
    raw = f"Mr. Speaker, I yield back.\n\n{SEP}\n\n  By Mr. DURBIN: S. 1. A bill."
    result = clean_speech(raw)
    assert result.text == "Mr. Speaker, I yield back."
    assert result.cut_at == "separator"


def test_short_separator_between_inserted_letters_does_not_cut() -> None:
    raw = (
        "Mr. Speaker, I include these letters.\n     Sincerely, A.\n"
        "                                  ____\n\n     Dear Chairman,\n"
        "  Mr. Speaker, I urge support."
    )
    result = clean_speech(raw)
    assert result.cut_at is None
    assert result.text.endswith("I urge support.")


def test_separator_framing_a_time_stamp_does_not_cut() -> None:
    raw = (
        f"Mr. Speaker, first part.\n\n{SEP}\n\n                              "
        "{time}  1430\n\n  I again thank my colleague."
    )
    result = clean_speech(raw)
    assert result.cut_at is None
    assert result.text == "Mr. Speaker, first part. I again thank my colleague."


def test_record_furniture_is_stripped() -> None:
    raw = (
        "Mr. President, before\n\n[[Page S3236]]\n\nafter.\n\n"
        "                             Infrastructure\n\n"
        "  More words.\n</pre></body>\n</html>"
    )
    result = clean_speech(raw)
    assert result.text == "Mr. President, before after. More words."
    assert (result.page_markers, result.headings) == (1, 1)
    assert result.html_tags == 3


def test_words_removed_is_counted() -> None:
    raw = "one two three\n  The PRESIDING OFFICER. four five."
    assert clean_speech(raw).words_removed == 5


# --- speaker resolution ------------------------------------------------

DAY = dt.date(2017, 6, 5)


def test_surname_unique_in_senate_resolves_despite_a_house_namesake(
    index: MemberIndex,
) -> None:
    """The upstream bug: LEE was unresolved in the Senate because of House Lees."""
    resolution = index.resolve("LEE", "SENATE", DAY)
    assert resolution.status == "resolved"
    assert resolution.term is not None and resolution.term.bioguide == "S1"


def test_shared_surname_in_the_same_chamber_stays_unresolved(
    index: MemberIndex,
) -> None:
    assert index.resolve("SMITH", "HOUSE", DAY).status == "ambiguous"


def test_compound_surname_is_not_split_into_first_and_last(
    index: MemberIndex,
) -> None:
    resolution = index.resolve("JACKSON LEE", "HOUSE", DAY)
    assert resolution.term is not None and resolution.term.bioguide == "H4"


def test_full_name_disambiguates(index: MemberIndex) -> None:
    resolution = index.resolve("CAROLYN B. MALONEY", "HOUSE", DAY)
    assert resolution.term is not None and resolution.term.bioguide == "H5"
    assert index.resolve("MALONEY", "HOUSE", DAY).status == "ambiguous"


def test_curly_apostrophe_matches_straight(index: MemberIndex) -> None:
    resolution = index.resolve("O'HALLERAN", "HOUSE", DAY)
    assert resolution.term is not None and resolution.term.bioguide == "H7"


def test_impeachment_manager_resolves_against_the_house(index: MemberIndex) -> None:
    resolution = index.resolve("Manager SCHIFF", "SENATE", DAY)
    assert resolution.term is not None and resolution.term.bioguide == "H8"


def test_non_member_in_senate_does_not_borrow_a_house_member(
    index: MemberIndex,
) -> None:
    """Trial video of a House member played in the Senate is not a senator."""
    assert index.resolve("SCHIFF", "SENATE", DAY).status == "no_candidate"


def test_member_not_serving_on_the_date_is_not_matched(index: MemberIndex) -> None:
    assert index.resolve("LEE", "SENATE", dt.date(2020, 1, 1)).status == (
        "no_candidate"
    )


def test_state_resolves_a_shared_surname(index: MemberIndex) -> None:
    """Mr. SMITH of Missouri: the case O7 lost 23,075 House speeches to."""
    resolution = index.resolve("SMITH", "HOUSE", DAY, state="MO")
    assert resolution.status == "resolved"
    assert resolution.term is not None and resolution.term.bioguide == "H3"
    assert resolution.by_state


def test_state_is_not_credited_when_the_surname_was_already_unique(
    index: MemberIndex,
) -> None:
    resolution = index.resolve("LEE", "SENATE", DAY, state="UT")
    assert resolution.term is not None and resolution.term.bioguide == "S1"
    assert not resolution.by_state


def test_state_contradicting_the_only_candidate_is_unresolved(
    index: MemberIndex,
) -> None:
    """Ms. LEE of California, in the Senate section, is not Mike Lee (UT).

    Seen 2021-02-12: the joint-session transcript played at the impeachment
    trial. Surname and chamber alone attribute it to the senator.
    """
    resolution = index.resolve("LEE", "SENATE", DAY, state="CA")
    assert resolution.status == "state_mismatch"
    assert resolution.term is None


def test_state_matching_no_namesake_is_unresolved(index: MemberIndex) -> None:
    resolution = index.resolve("SMITH", "HOUSE", DAY, state="TX")
    assert resolution.status == "state_mismatch"


def test_namesakes_from_the_same_state_stay_ambiguous(index: MemberIndex) -> None:
    """Carolyn and Sean Maloney both sat for New York."""
    assert index.resolve("MALONEY", "HOUSE", DAY, state="NY").status == "ambiguous"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("Texas", "TX"),
        ("TEXAS", "TX"),
        ("New York", "NY"),
        ("West Virginia", "WV"),
        ("Northern Mariana Islands", "MP"),
        (None, None),
        # Observed in the 2026-10-02 file: misspelling, regex overrun, non-state.
        ("Virgina", None),
        ("Massachusetts Mr", None),
        ("Japan", None),
    ],
)
def test_state_code(raw: str | None, code: str | None) -> None:
    assert state_code(raw) == code


def test_mid_term_party_switch_is_honoured(index: MemberIndex) -> None:
    term = index.resolve("SWITCHER", "HOUSE", DAY).term
    assert term is not None
    assert term.party_on(dt.date(2017, 6, 5))[0] == "Democrat"
    assert term.party_on(dt.date(2018, 6, 5))[0] == "Republican"


@pytest.mark.parametrize(
    ("day", "congress"),
    [
        (dt.date(2016, 9, 12), 114),
        (dt.date(2017, 1, 2), 114),
        (dt.date(2017, 1, 3), 115),
        (dt.date(2018, 12, 31), 115),
        (dt.date(2025, 1, 3), 119),
    ],
)
def test_congress_from_date(day: dt.date, congress: int) -> None:
    assert congress_for(day) == congress


# --- build -------------------------------------------------------------


def test_output_schema_and_source_tag(tmp_path: Path) -> None:
    _, table = _build(tmp_path, [_record("LEE", "SENATE")])
    assert table is not None
    assert table.schema.equals(GOVINFO_SCHEMA)
    row = table.to_pylist()[0]
    assert row["source"] == "govinfo"
    assert (row["member_id"], row["party"], row["chamber"]) == ("S1", "R", "S")
    assert (row["state"], row["congress_number"], row["icpsr"]) == ("UT", 115, 1)


def test_impeachment_manager_is_written_with_the_house_as_chamber(
    tmp_path: Path,
) -> None:
    """A House manager speaking at a Senate trial is still a House member.

    ``chamber`` is the member's chamber, as on the Stanford side, so the row
    joins to their House record in DW-NOMINATE. docs/decisions.md D20.
    """
    stats, table = _build(tmp_path, [_record("Manager SCHIFF", "SENATE")])
    assert table is not None
    assert table.column("chamber").to_pylist() == ["H"]
    assert stats.chamber_counts == {"H": 1}
    assert stats.rows_chamber_reassigned == 1


def test_file_party_is_ignored_in_favour_of_the_lookup(tmp_path: Path) -> None:
    """The file labels mid-term switchers with their end-of-term party."""
    record = _record("SWITCHER", party="Republican")
    _, table = _build(tmp_path, [record])
    assert table is not None and table.column("party").to_pylist() == ["D"]


def test_date_window_drops_and_counts(tmp_path: Path) -> None:
    stats, table = _build(
        tmp_path,
        [
            _record("LEE", "SENATE", date="2016-09-09"),
            _record("LEE", "SENATE", date="2017-06-05"),
            _record("LEE", "SENATE", date="2026-01-05"),
        ],
        end_date=dt.date(2025, 12, 31),
    )
    assert (stats.rows_dropped_before_start, stats.rows_dropped_after_end) == (1, 1)
    assert table is not None and table.num_rows == 1


def test_word_filter_applies_to_cleaned_text(tmp_path: Path) -> None:
    """Officer text must not carry a short procedural speech over the bar."""
    speech = "Mr. President, I ask consent.\n  The PRESIDING OFFICER. " + WORDS
    stats, table = _build(tmp_path, [_record("LEE", "SENATE", speech=speech)])
    assert stats.rows_dropped_short == 1
    assert table is not None and table.num_rows == 0


def test_caucus_rule_reassigns_listed_independent(tmp_path: Path) -> None:
    stats, table = _build(tmp_path, [_record("SANDERS", "SENATE")])
    assert table is not None
    row = table.to_pylist()[0]
    assert (row["party"], row["party_original"]) == ("D", "I")
    assert stats.independents_reassigned == 1


def test_excluded_member_is_dropped_only_while_independent(tmp_path: Path) -> None:
    stats, table = _build(
        tmp_path,
        [_record("AMASH", date="2017-06-05"), _record("AMASH", date="2018-06-05")],
    )
    assert stats.rows_dropped_excluded_member == 1
    assert table is not None and table.column("party").to_pylist() == ["R"]


def test_independent_delegate_is_dropped_without_raising(tmp_path: Path) -> None:
    stats, table = _build(tmp_path, [_record("SABLAN")])
    assert stats.rows_dropped_delegate == 1
    assert table is not None and table.num_rows == 0


def test_unlisted_independent_raises_and_leaves_no_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="X1"):
        _build(tmp_path, [_record("LEE", "SENATE"), _record("MAVERICK", "SENATE")])
    assert not (tmp_path / "out.parquet").exists()


def test_unresolved_rows_are_counted_by_chamber_and_reason(tmp_path: Path) -> None:
    stats, _ = _build(
        tmp_path, [_record("SMITH"), _record("NOBODY", "SENATE"), _record("LEE")]
    )
    assert stats.rows_dropped_unresolved_ambiguous == 1
    assert stats.rows_dropped_unresolved_no_candidate == 1
    assert stats.unresolved_by_chamber == {"HOUSE": 1, "SENATE": 1}


def test_state_in_the_record_resolves_a_shared_surname(tmp_path: Path) -> None:
    stats, table = _build(tmp_path, [_record("SMITH", state="Missouri")])
    assert table is not None
    assert table.column("member_id").to_pylist() == ["H3"]
    assert stats.rows_resolved_by_state == 1
    assert stats.rows_dropped_unresolved_ambiguous == 0


def test_state_contradicting_the_only_candidate_drops_the_row(
    tmp_path: Path,
) -> None:
    stats, table = _build(tmp_path, [_record("LEE", "SENATE", state="California")])
    assert table is not None and table.num_rows == 0
    assert stats.rows_dropped_unresolved_state_mismatch == 1
    assert stats.unresolved_by_chamber == {"SENATE": 1}


def test_unrecognized_state_is_ignored_and_counted(tmp_path: Path) -> None:
    """A garbled state falls back to surname-only matching, never to a guess."""
    stats, table = _build(tmp_path, [_record("LEE", "SENATE", state="Japan")])
    assert table is not None
    assert table.column("member_id").to_pylist() == ["S1"]
    assert stats.state_unrecognized == {"Japan": 1}


def test_record_without_a_state_key_is_rejected(tmp_path: Path) -> None:
    """The pre-2026-10-02 file has no state and the Senate fetch cap (O8)."""
    record = _record("LEE", "SENATE")
    del record["state"]
    with pytest.raises(ValueError, match="state"):
        _build(tmp_path, [record])


def test_stats_arithmetic_closes(tmp_path: Path) -> None:
    stats, _ = _build(
        tmp_path,
        [
            _record("LEE", "SENATE"),
            _record("LEE", "SENATE"),  # identical -> duplicate id
            _record("SMITH"),
            _record("SABLAN"),
            _record("AMASH", date="2018-06-05"),
            _record("LEE", "SENATE", speech="Mr. President, short."),
            _record("LEE", "SENATE", date="2030-01-01"),
            _record("LEE", "SENATE", state="California"),
        ],
    )
    dropped = (
        stats.rows_dropped_before_start
        + stats.rows_dropped_after_end
        + stats.rows_dropped_unresolved_no_candidate
        + stats.rows_dropped_unresolved_ambiguous
        + stats.rows_dropped_unresolved_state_mismatch
        + stats.rows_dropped_delegate
        + stats.rows_dropped_excluded_member
        + stats.rows_dropped_short
        + stats.rows_dropped_duplicate_id
    )
    assert stats.rows_read == 8
    assert stats.rows_written == 1
    assert dropped + stats.rows_written == stats.rows_read


def test_existing_output_is_not_clobbered_without_overwrite(tmp_path: Path) -> None:
    _build(tmp_path, [_record("LEE", "SENATE")])
    with pytest.raises(FileExistsError):
        _build(tmp_path, [_record("LEE", "SENATE")])
