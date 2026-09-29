"""Tests for the corpus member -> DW-NOMINATE (Voteview ICPSR) crosswalk.

Fixtures are synthetic and built in-process. Each reproduces a case found when
probing the real data on 2026-09-29 (docs/decisions.md D21-D24): compound
surnames the Stanford side cuts to their last word (JACKSON LEE -> LEE), OCR
spelling variants, same-surname members told apart only by party, same-surname
same-party members (the Sanchez sisters), party switchers Voteview gives two
ICPSR numbers, and the Davis pair (VA) that nothing separates.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.crosswalk import (
    CROSSWALK_SCHEMA,
    build_crosswalk,
    load_voteview,
    surname_tokens,
)
from src.merge import MERGED_SCHEMA

# Every column of Voteview's HSall_members.csv, in its order, so the fixture
# mirrors the real file rather than only the columns the code reads.
HSALL_COLUMNS = [
    "congress",
    "chamber",
    "icpsr",
    "state_icpsr",
    "district_code",
    "state_abbrev",
    "party_code",
    "occupancy",
    "last_means",
    "bioname",
    "bioguide_id",
    "born",
    "died",
    "nominate_dim1",
    "nominate_dim2",
    "nominate_log_likelihood",
    "nominate_geo_mean_probability",
    "nominate_number_of_votes",
    "nominate_number_of_errors",
    "conditional",
    "nokken_poole_dim1",
    "nokken_poole_dim2",
]


# --- fixture helpers ---------------------------------------------------


def _vv(
    congress: int,
    chamber: str,
    icpsr: int,
    bioguide: str,
    bioname: str,
    state: str,
    party: int,
    dim1: float | None = 0.1,
) -> dict[str, object]:
    """One Voteview member row, every HSall column present."""
    row: dict[str, object] = dict.fromkeys(HSALL_COLUMNS)
    row.update(
        congress=congress,
        chamber={"H": "House", "S": "Senate", "P": "President"}[chamber],
        icpsr=icpsr,
        state_abbrev=state,
        party_code=party,
        bioname=bioname,
        bioguide_id=bioguide,
        nominate_dim1=dim1,
        nokken_poole_dim1=dim1,
    )
    return row


def _speeches(
    source: str,
    member_id: str,
    congress: int,
    chamber: str,
    state: str,
    last_name: str,
    party_original: str,
    n: int = 1,
    icpsr: int | None = None,
) -> list[dict[str, object]]:
    """``n`` corpus rows for one member, in the merged corpus schema."""
    party = party_original if party_original in ("D", "R") else "D"
    day = dt.date(2001 + 2 * (congress - 107), 6, 1)
    return [
        {
            "speech_id": f"{source}-{member_id}-{congress}-{party_original}-{i}",
            "date": day,
            "member_id": member_id,
            "party": party,
            "chamber": chamber,
            "congress_number": congress,
            "text": "x",
            "source": source,
            "last_name": last_name,
            "state": state,
            "word_count": 60,
            "party_original": party_original,
            "icpsr": icpsr,
        }
        for i in range(n)
    ]


def _stanford(member_id: str, *args: object, **kwargs: object):
    """Stanford rows; the congress is read off the speakerid, as in the data."""
    return _speeches("stanford", member_id, int(member_id[:3]), *args, **kwargs)


def _govinfo(*args: object, **kwargs: object):
    return _speeches("govinfo", *args, **kwargs)


def _write_inputs(
    tmp_path: Path, speeches: list[dict[str, object]], voteview: list[dict]
) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(speeches, schema=MERGED_SCHEMA), corpus)
    members = tmp_path / "HSall_members.csv"
    pd.DataFrame(voteview, columns=HSALL_COLUMNS).to_csv(members, index=False)
    return corpus, members


def _build(tmp_path: Path, speeches: list[dict], voteview: list[dict], **kwargs):
    corpus, members = _write_inputs(tmp_path, speeches, voteview)
    dst = tmp_path / "crosswalk.parquet"
    stats = build_crosswalk(corpus=corpus, voteview=members, dst=dst, **kwargs)
    return stats, pq.read_table(dst).to_pandas()


def _unit(table: pd.DataFrame, member_id: str, party_original: str | None = None):
    """The one crosswalk row for ``member_id`` (and party, when it splits)."""
    rows = table[table.member_id == member_id]
    if party_original is not None:
        rows = rows[rows.party_original == party_original]
    assert len(rows) == 1, rows
    return rows.iloc[0]


# --- Voteview ----------------------------------------------------------


def test_voteview_keeps_house_and_senate_in_the_congress_range(
    tmp_path: Path,
) -> None:
    _, members = _write_inputs(
        tmp_path,
        [],
        [
            _vv(107, "H", 1, "A1", "ONE, A", "AK", 100),
            _vv(119, "S", 2, "A2", "TWO, B", "AK", 200),
            _vv(107, "P", 3, "A3", "BUSH, George", "USA", 200),
            _vv(106, "H", 4, "A4", "OLD, C", "AK", 100),
            _vv(120, "H", 5, "A5", "NEW, D", "AK", 100),
        ],
    )
    voteview = load_voteview(members, congress_range=(107, 119))
    assert sorted(voteview.icpsr) == [1, 2]
    assert sorted(voteview.chamber) == ["House", "Senate"]


def test_voteview_repeated_member_key_raises(tmp_path: Path) -> None:
    row = _vv(107, "H", 1, "A1", "ONE, A", "AK", 100)
    _, members = _write_inputs(tmp_path, [], [row, row])
    with pytest.raises(ValueError, match="congress, chamber, icpsr"):
        load_voteview(members, congress_range=(107, 119))


# --- surname normalization ----------------------------------------------


@pytest.mark.parametrize(
    ("name", "tokens"),
    [
        ("SÁNCHEZ", ("SANCHEZ",)),
        ("Diaz-Balart", ("DIAZ", "BALART")),
        ("O’Halleran", ("OHALLERAN",)),
        ("JACKSON LEE", ("JACKSON", "LEE")),
        ("McMORRIS RODGERS", ("MCMORRIS", "RODGERS")),
    ],
)
def test_surname_tokens(name: str, tokens: tuple[str, ...]) -> None:
    assert surname_tokens(name) == tokens


# --- Stanford side: names ----------------------------------------------


def test_stanford_unique_surname_in_state_matches(tmp_path: Path) -> None:
    _, table = _build(
        tmp_path,
        _stanford("107100010", "H", "AK", "YOUNG", "R", n=3),
        [
            _vv(107, "H", 14066, "Y000033", "YOUNG, Donald Edwin", "AK", 200),
            _vv(107, "H", 29999, "Y000099", "YOUNG, Other", "FL", 200),
        ],
    )
    unit = _unit(table, "107100010")
    assert unit.icpsr == 14066
    assert (unit.candidate_key, unit.resolved_by) == ("name_exact", "unique")
    assert unit.bioguide_id == "Y000033" and unit.n_speeches == 3
    assert unit.in_validation


def test_stanford_minority_ocr_variant_still_matches(tmp_path: Path) -> None:
    """HODES is spelled RHODES on most of his rows; any variant may match."""
    speeches = _stanford("111115660", "H", "NH", "RHODES", "D", n=3)
    speeches += _stanford("111115660", "H", "NH", "HODES", "D", n=1)
    speeches = [dict(s, speech_id=f"s{i}") for i, s in enumerate(speeches)]
    _, table = _build(
        tmp_path, speeches, [_vv(111, "H", 20700, "H001040", "HODES, Paul", "NH", 100)]
    )
    assert _unit(table, "111115660").icpsr == 20700


def test_stanford_compound_surname_matches_on_its_last_word(tmp_path: Path) -> None:
    _, table = _build(
        tmp_path,
        _stanford("112119920", "H", "TX", "LEE", "D"),
        [_vv(112, "H", 29573, "J000032", "JACKSON LEE, Sheila", "TX", 100)],
    )
    unit = _unit(table, "112119920")
    assert unit.icpsr == 29573 and unit.candidate_key == "name_token"


def test_stanford_exact_surname_wins_over_a_compound_one(tmp_path: Path) -> None:
    _, table = _build(
        tmp_path,
        _stanford("112100020", "H", "CA", "LEE", "D"),
        [
            _vv(112, "H", 29778, "L000551", "LEE, Barbara", "CA", 100),
            _vv(112, "H", 20001, "X000001", "JACKSON LEE, Someone", "CA", 100),
        ],
    )
    unit = _unit(table, "112100020")
    assert (unit.icpsr, unit.candidate_key) == (29778, "name_exact")


def test_stanford_namesake_in_another_state_or_chamber_is_no_candidate(
    tmp_path: Path,
) -> None:
    _, table = _build(
        tmp_path,
        _stanford("110100030", "H", "OR", "SMITH", "R"),
        [
            _vv(110, "S", 29386, "S000583", "SMITH, Gordon", "OR", 200),
            _vv(110, "H", 29768, "S000510", "SMITH, Adam", "WA", 100),
        ],
    )
    unit = _unit(table, "110100030")
    assert pd.isna(unit.icpsr) and pd.isna(unit.resolved_by)
    assert unit.unmatched_reason == "no_candidate" and not unit.in_validation


# --- Stanford side: ties ------------------------------------------------


def test_stanford_party_breaks_a_same_surname_tie(tmp_path: Path) -> None:
    voteview = [
        _vv(110, "H", 29905, "M001139", "MILLER, Gary G.", "CA", 200),
        _vv(110, "H", 14256, "M000725", "MILLER, George", "CA", 100),
    ]
    speeches = _stanford("110117730", "H", "CA", "MILLER", "R")
    speeches += _stanford("110117740", "H", "CA", "MILLER", "D")
    _, table = _build(tmp_path, speeches, voteview)
    assert _unit(table, "110117730").icpsr == 29905
    assert _unit(table, "110117740").icpsr == 14256
    assert _unit(table, "110117730").resolved_by == "party"


def test_stanford_switcher_tie_is_broken_on_party_original(tmp_path: Path) -> None:
    """Jeffords: party D after the caucus rule, I in the source, 328 in Voteview."""
    voteview = [
        _vv(107, "S", 14240, "J000072", "JEFFORDS, James Merrill", "VT", 200, 0.014),
        _vv(107, "S", 94240, "J000072", "JEFFORDS, James Merrill", "VT", 328, -0.277),
    ]
    _, table = _build(
        tmp_path, _stanford("107113101", "S", "VT", "JEFFORDS", "I"), voteview
    )
    unit = _unit(table, "107113101")
    assert unit.icpsr == 94240 and unit.resolved_by == "party"
    assert unit.party_switch and unit.nominate_dim1 == pytest.approx(-0.277)


def test_stanford_same_party_namesakes_resolve_via_speakerid_and_elimination(
    tmp_path: Path,
) -> None:
    """The Sanchez sisters: Loretta alone in the 107th, both from the 108th."""
    loretta = [
        _vv(c, "H", 29709, "S000030", "SANCHEZ, Loretta", "CA", 100) for c in (107, 108)
    ]
    linda = [_vv(108, "H", 20310, "S001156", "SÁNCHEZ, Linda T.", "CA", 100)]
    speeches = _stanford("107121190", "H", "CA", "SANCHEZ", "D")
    speeches += _stanford("108121190", "H", "CA", "SANCHEZ", "D")
    speeches += _stanford("108121490", "H", "CA", "SANCHEZ", "D", n=2)
    _, table = _build(tmp_path, speeches, loretta + linda)

    assert _unit(table, "107121190").resolved_by == "unique"
    assert (_unit(table, "108121190").icpsr, _unit(table, "108121190").resolved_by) == (
        29709,
        "speakerid",
    )
    assert (_unit(table, "108121490").icpsr, _unit(table, "108121490").resolved_by) == (
        20310,
        "elimination",
    )
    assert _unit(table, "108121490").n_candidates == 2


def test_stanford_namesakes_nothing_separates_stay_unmatched(tmp_path: Path) -> None:
    """Tom and Jo Ann Davis (VA): same surname, state, party and Congresses."""
    voteview = [
        _vv(c, "H", icpsr, bioguide, name, "VA", 200)
        for c in (107, 108)
        for icpsr, bioguide, name in (
            (29576, "D000136", "DAVIS, Thomas M., III"),
            (20141, "D000590", "DAVIS, Jo Ann"),
        )
    ]
    speeches = [
        row
        for sid in ("107114690", "107114700", "108114690", "108114700")
        for row in _stanford(sid, "H", "VA", "DAVIS", "R")
    ]
    stats, table = _build(tmp_path, speeches, voteview)
    assert table.icpsr.isna().all()
    assert set(table.unmatched_reason) == {"ambiguous"}
    assert stats.summary["stanford"]["units_matched"] == 0


def test_speakerid_pointing_to_two_people_raises(tmp_path: Path) -> None:
    """The speakerid person part is trusted only while it names one person."""
    voteview = [
        _vv(107, "H", 1, "P000001", "PARKER, One", "MS", 100),
        _vv(108, "H", 2, "P000002", "PRICE, Two", "MS", 100),
    ]
    speeches = _stanford("107155550", "H", "MS", "PARKER", "D")
    speeches += _stanford("108155550", "H", "MS", "PRICE", "D")
    with pytest.raises(ValueError, match="155550"):
        _build(tmp_path, speeches, voteview)


def test_two_members_matched_to_one_icpsr_raises(tmp_path: Path) -> None:
    speeches = _stanford("107100040", "H", "AK", "SMITH", "R")
    speeches += _stanford("107100050", "H", "AK", "SMITH", "R")
    with pytest.raises(ValueError, match="107100040.*107100050"):
        _build(tmp_path, speeches, [_vv(107, "H", 7, "S1", "SMITH, A", "AK", 200)])


# --- govinfo side --------------------------------------------------------


def test_govinfo_joins_on_bioguide_not_the_stored_icpsr(tmp_path: Path) -> None:
    """The corpus icpsr is congress-legislators', the pre-switch number."""
    stats, table = _build(
        tmp_path,
        _govinfo("V000133", 117, "H", "NJ", "VAN DREW", "R", icpsr=21980),
        [_vv(117, "H", 91980, "V000133", "VAN DREW, Jefferson", "NJ", 200)],
    )
    unit = _unit(table, "V000133")
    assert unit.icpsr == 91980 and unit.candidate_key == "bioguide"
    assert [
        (d["member_id"], d["corpus_icpsr"], d["icpsr"])
        for d in stats.icpsr_disagreements
    ] == [("V000133", 21980, 91980)]


def test_govinfo_switcher_units_split_by_party_original(tmp_path: Path) -> None:
    voteview = [
        _vv(116, "H", 21980, "V000133", "VAN DREW, Jefferson", "NJ", 100),
        _vv(116, "H", 91980, "V000133", "VAN DREW, Jefferson", "NJ", 200),
    ]
    speeches = _govinfo("V000133", 116, "H", "NJ", "VAN DREW", "D", n=2)
    speeches += _govinfo("V000133", 116, "H", "NJ", "VAN DREW", "R", n=1)
    _, table = _build(tmp_path, speeches, voteview)
    assert _unit(table, "V000133", "D").icpsr == 21980
    assert _unit(table, "V000133", "R").icpsr == 91980
    assert table.party_switch.all() and set(table.resolved_by) == {"party"}


def test_govinfo_member_absent_from_voteview_is_unmatched(tmp_path: Path) -> None:
    _, table = _build(
        tmp_path,
        _govinfo("N000001", 119, "H", "CA", "NEWBIE", "D"),
        [_vv(119, "H", 1, "O000001", "OTHER, A", "CA", 100)],
    )
    assert _unit(table, "N000001").unmatched_reason == "no_candidate"


# --- scores and validation membership ------------------------------------


def test_matched_member_without_a_score_is_kept_out_of_validation(
    tmp_path: Path,
) -> None:
    """Kwanza Hall: served a month, cast too few votes to be scored."""
    stats, table = _build(
        tmp_path,
        _govinfo("H001085", 116, "H", "GA", "HALL", "D"),
        [_vv(116, "H", 21986, "H001085", "HALL, Kwanza", "GA", 100, dim1=None)],
    )
    unit = _unit(table, "H001085")
    assert unit.icpsr == 21986 and not unit.has_nominate and not unit.in_validation
    assert [d["member_id"] for d in stats.no_nominate] == ["H001085"]
    assert stats.unmatched == []


# --- build, stats and outputs ---------------------------------------------


def _mixed_fixture() -> tuple[list[dict], list[dict]]:
    """Stanford: 5 + 3 matched, 2 unmatched. govinfo: 4 matched."""
    speeches = _stanford("107100010", "H", "AK", "YOUNG", "R", n=5)
    speeches += _stanford("107100060", "S", "AK", "STEVENS", "R", n=3)
    speeches += _stanford("107100070", "H", "AK", "GHOST", "D", n=2)
    speeches += _govinfo("M001153", 115, "S", "AK", "MURKOWSKI", "R", n=4)
    voteview = [
        _vv(107, "H", 14066, "Y000033", "YOUNG, Donald Edwin", "AK", 200),
        _vv(107, "S", 12109, "S000888", "STEVENS, Theodore Fulton", "AK", 200),
        _vv(115, "S", 40300, "M001153", "MURKOWSKI, Lisa", "AK", 200),
    ]
    return speeches, voteview


def test_crosswalk_has_one_row_per_member_congress_chamber_party(
    tmp_path: Path,
) -> None:
    speeches, voteview = _mixed_fixture()
    _, table = _build(tmp_path, speeches, voteview)
    written = pq.read_schema(tmp_path / "crosswalk.parquet")
    assert written.remove_metadata().equals(CROSSWALK_SCHEMA)
    assert len(table) == 4
    key = ["source", "member_id", "congress_number", "chamber", "party_original"]
    assert not table.duplicated(key).any()


def test_stats_match_rates_close_on_speeches(tmp_path: Path) -> None:
    speeches, voteview = _mixed_fixture()
    stats, _ = _build(tmp_path, speeches, voteview)
    stanford = stats.summary["stanford"]
    assert (stanford["speeches"], stanford["speeches_matched"]) == (10, 8)
    assert stanford["speech_match_rate"] == pytest.approx(0.8)
    assert (stanford["units"], stanford["units_matched"]) == (3, 2)
    assert stats.summary["govinfo"]["speech_match_rate"] == pytest.approx(1.0)
    assert stats.by_congress_chamber["stanford"]["107"]["H"] == {
        "units": 2,
        "units_matched": 1,
        "speeches": 7,
        "speeches_matched": 5,
        "speeches_in_validation": 5,
    }


def test_unmatched_list_is_written_with_speech_counts(tmp_path: Path) -> None:
    speeches, voteview = _mixed_fixture()
    stats, _ = _build(tmp_path, speeches, voteview)
    path = stats.write_unmatched(tmp_path / "unmatched.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["member_id"], r["n_speeches"], r["unmatched_reason"]) for r in rows] == [
        ("107100070", "2", "no_candidate")
    ]


def test_existing_output_is_not_clobbered_without_overwrite(tmp_path: Path) -> None:
    speeches, voteview = _mixed_fixture()
    _build(tmp_path, speeches, voteview)
    with pytest.raises(FileExistsError):
        _build(tmp_path, speeches, voteview)


def test_missing_corpus_names_the_make_target(tmp_path: Path) -> None:
    _, members = _write_inputs(tmp_path, [], [])
    with pytest.raises(FileNotFoundError, match="make corpus"):
        build_crosswalk(
            corpus=tmp_path / "absent.parquet",
            voteview=members,
            dst=tmp_path / "crosswalk.parquet",
        )


def test_unknown_source_tag_raises(tmp_path: Path) -> None:
    speeches = _speeches("hansard", "X1", 110, "H", "AK", "YOUNG", "R")
    with pytest.raises(ValueError, match="hansard"):
        _build(tmp_path, speeches, [])
