"""Tests for the Stanford -> corpus build.

Fixtures are synthetic and built in-process, so the suite runs without the
681 MB raw parquet on disk. They mirror the real file's quirks deliberately:
``word_count`` and ``date`` are strings there, and the dirty ``speaker`` /
``state`` / ``first_name`` columns are present but must be ignored in favour of
``last_name`` / ``state_map`` / ``chamber_map``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.corpus import build_stanford_corpus

# Superset of what the build needs, plus the dirty columns the real file
# carries so the tests prove they are ignored.
FIXTURE_COLUMNS = (
    "speech_id",
    "speech",
    "date",
    "speaker",
    "first_name",
    "last_name",
    "state",
    "state_map",
    "chamber",
    "chamber_map",
    "word_count",
    "speakerid",
    "party",
    "congress",
)

# --- fixture helpers ---------------------------------------------------


def _row(
    speech_id: str,
    *,
    speech: str = "word " * 80,
    date: int | str = "20010103",
    last_name: str = "SMITH",
    state_map: str = "NY",
    chamber_map: str = "H",
    word_count: int | str = "80",
    speakerid: str = "107118000",
    party: str = "D",
    congress: int = 107,
) -> dict[str, object]:
    """One raw-shaped row, with defaults that survive every filter.

    The dirty columns are filled with the kind of damage the real file has, so
    any test that passes proves the build read the clean columns instead.
    """
    return {
        "speech_id": speech_id,
        "speech": speech,
        "date": date,
        "speaker": f"Mr.. {last_name}",
        "first_name": "Unknown",
        "last_name": last_name,
        "state": "Unknown",
        "state_map": state_map,
        "chamber": None,
        "chamber_map": chamber_map,
        "word_count": word_count,
        "speakerid": speakerid,
        "party": party,
        "congress": congress,
    }


def _write_raw(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    name: str = "raw.parquet",
    row_group_size: int | None = None,
    drop_columns: tuple[str, ...] = (),
) -> Path:
    """Write synthetic rows in the raw Stanford layout to a parquet file."""
    columns = [c for c in FIXTURE_COLUMNS if c not in drop_columns]
    table = pa.table({col: [r[col] for r in rows] for col in columns})
    dst = tmp_path / name
    pq.write_table(table, dst, row_group_size=row_group_size or len(rows))
    return dst


def _build(tmp_path: Path, rows: list[dict[str, object]], **kwargs: object):
    """Build a corpus from synthetic rows; return (table, stats)."""
    src = _write_raw(tmp_path, rows, **kwargs)
    dst = tmp_path / f"corpus_{src.stem}.parquet"
    stats = build_stanford_corpus(src=src, dst=dst)
    return pq.read_table(dst), stats


# --- column mapping and provenance -------------------------------------


def test_columns_are_mapped_onto_the_corpus_schema(tmp_path: Path) -> None:
    table, _ = _build(
        tmp_path,
        [
            _row(
                "s1",
                speech="alpha beta " * 40,
                speakerid="107118030",
                congress=112,
                state_map="TX",
                chamber_map="S",
            )
        ],
    )
    record = table.to_pylist()[0]

    assert record["text"].startswith("alpha beta")
    assert record["member_id"] == "107118030"
    assert record["congress_number"] == 112
    assert record["state"] == "TX"
    assert record["chamber"] == "S"
    assert record["last_name"] == "SMITH"


def test_dirty_speaker_columns_are_ignored(tmp_path: Path) -> None:
    """speaker/state/first_name are damaged in the real file; never read them."""
    table, stats = _build(
        tmp_path,
        [
            _row(
                "s1",
                party="I",
                last_name="SANDERS",
                state_map="VT",
                speakerid="107118220",
            )
        ],
    )
    record = table.to_pylist()[0]

    # Resolved despite speaker being "Mr.. SANDERS" and state being "Unknown".
    assert record["party"] == "D"
    assert record["last_name"] == "SANDERS"
    assert record["state"] == "VT"
    assert stats.independents_reassigned == 1
    assert "speaker" not in table.column_names
    assert "first_name" not in table.column_names


def test_every_row_carries_the_stanford_source_tag(tmp_path: Path) -> None:
    table, _ = _build(tmp_path, [_row("s1"), _row("s2"), _row("s3")])

    assert table.column("source").to_pylist() == ["stanford"] * 3


def test_output_schema_is_exactly_the_documented_columns(tmp_path: Path) -> None:
    table, _ = _build(tmp_path, [_row("s1")])

    assert set(table.column_names) == {
        "speech_id",
        "date",
        "member_id",
        "party",
        "chamber",
        "congress_number",
        "text",
        "source",
        "last_name",
        "state",
        "word_count",
        "party_original",
    }


# --- raw dtypes --------------------------------------------------------


@pytest.mark.parametrize("raw_date", [20010103, "20010103"])
def test_yyyymmdd_parses_from_both_int_and_string(
    tmp_path: Path, raw_date: int | str
) -> None:
    table, _ = _build(tmp_path, [_row("s1", date=raw_date)])

    assert table.schema.field("date").type == pa.date32()
    assert table.column("date").to_pylist() == [dt.date(2001, 1, 3)]


@pytest.mark.parametrize("raw_wc", [80, "80"])
def test_word_count_parses_from_both_int_and_string(
    tmp_path: Path, raw_wc: int | str
) -> None:
    """The real file stores word_count as a string."""
    table, _ = _build(tmp_path, [_row("s1", word_count=raw_wc)])

    assert table.schema.field("word_count").type == pa.int32()
    assert table.column("word_count").to_pylist() == [80]


# --- word-count filter -------------------------------------------------


def test_word_filter_boundary_is_inclusive(tmp_path: Path) -> None:
    table, stats = _build(
        tmp_path,
        [
            _row("keep", word_count="50", speech="word " * 50),
            _row("drop", word_count="49", speech="word " * 49),
        ],
    )

    assert table.column("speech_id").to_pylist() == ["keep"]
    assert stats.rows_dropped_short == 1


# --- party rules -------------------------------------------------------


@pytest.mark.parametrize(
    ("speakerid", "last_name", "state_map"),
    [
        ("114118221", "SANDERS", "VT"),
        ("113122291", "KING", "ME"),
        ("107113101", "JEFFORDS", "VT"),
        ("112116471", "LIEBERMAN", "CT"),
    ],
)
def test_roster_independents_are_mapped_to_their_caucus_party(
    tmp_path: Path, speakerid: str, last_name: str, state_map: str
) -> None:
    table, stats = _build(
        tmp_path,
        [
            _row(
                "s1",
                party="I",
                last_name=last_name,
                state_map=state_map,
                speakerid=speakerid,
            )
        ],
    )
    record = table.to_pylist()[0]

    assert record["party"] == "D"
    assert record["party_original"] == "I"
    assert stats.independents_reassigned == 1
    assert stats.party_corrections_applied == 0


def test_party_correction_is_applied_and_counted_separately(tmp_path: Path) -> None:
    """Crenshaw (FL) is coded I in the 107th but was a Republican throughout."""
    table, stats = _build(
        tmp_path,
        [
            _row(
                "s1",
                party="I",
                last_name="CRENSHAW",
                state_map="FL",
                speakerid="107119090",
            )
        ],
    )
    record = table.to_pylist()[0]

    assert record["party"] == "R"
    assert record["party_original"] == "I"
    assert stats.party_corrections_applied == 1
    assert stats.independents_reassigned == 0


def test_unhandled_independent_raises_and_names_the_member(tmp_path: Path) -> None:
    src = _write_raw(
        tmp_path,
        [
            _row("s1"),
            _row(
                "s2",
                party="I",
                last_name="MCMYSTERY",
                state_map="OR",
                congress=110,
                speakerid="110999999",
            ),
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        build_stanford_corpus(src=src, dst=tmp_path / "corpus.parquet")

    message = str(excinfo.value)
    assert "MCMYSTERY" in message
    assert "OR" in message
    assert "110999999" in message
    assert "CAUCUS_PARTY" in message


def test_unhandled_independent_fails_before_writing_output(tmp_path: Path) -> None:
    src = _write_raw(
        tmp_path,
        [
            _row(
                "s1",
                party="I",
                last_name="MCMYSTERY",
                state_map="OR",
                speakerid="110999999",
            )
        ],
    )
    dst = tmp_path / "corpus.parquet"

    with pytest.raises(ValueError):
        build_stanford_corpus(src=src, dst=dst)

    assert not dst.exists()


def test_unexpected_party_code_raises(tmp_path: Path) -> None:
    """Party codes beyond D/R/I must not slip through silently."""
    src = _write_raw(
        tmp_path, [_row("s1"), _row("s2", party="P", last_name="ODD", state_map="OR")]
    )

    with pytest.raises(ValueError, match="party codes outside"):
        build_stanford_corpus(src=src, dst=tmp_path / "corpus.parquet")


def test_declared_party_rows_pass_through_untouched(tmp_path: Path) -> None:
    """A Republican must pass through even if a roster member shares a name."""
    table, stats = _build(
        tmp_path,
        [
            _row("d", party="D"),
            _row("r", party="R"),
            _row("king", party="R", last_name="KING", state_map="IA"),
        ],
    )
    by_id = {r["speech_id"]: r for r in table.to_pylist()}

    assert by_id["d"]["party"] == "D"
    assert by_id["r"]["party"] == "R"
    assert by_id["king"]["party"] == "R"
    assert stats.independents_reassigned == 0


# --- membership exclusions ---------------------------------------------


@pytest.mark.parametrize("state_map", ["DC", "PR", "VI", "GU", "AS", "MP"])
def test_non_voting_delegates_are_dropped(tmp_path: Path, state_map: str) -> None:
    table, stats = _build(
        tmp_path, [_row("keep"), _row("delegate", state_map=state_map)]
    )

    assert table.column("speech_id").to_pylist() == ["keep"]
    assert stats.rows_dropped_delegate == 1


def test_excluded_member_is_dropped(tmp_path: Path) -> None:
    """Barkley caucused with neither party, so the caucus rule has no answer."""
    table, stats = _build(
        tmp_path,
        [
            _row("keep"),
            _row(
                "barkley",
                party="I",
                last_name="BARKLEY",
                state_map="MN",
                speakerid="107113431",
            ),
        ],
    )

    assert table.column("speech_id").to_pylist() == ["keep"]
    assert stats.rows_dropped_excluded_member == 1
    assert stats.rows_dropped_delegate == 0


def test_delegate_rows_do_not_trigger_the_independent_check(tmp_path: Path) -> None:
    """Sablan (MP) is an unrostered independent, but delegates drop out first."""
    table, stats = _build(
        tmp_path,
        [_row("keep"), _row("sablan", party="I", last_name="SABLAN", state_map="MP")],
    )

    assert table.column("speech_id").to_pylist() == ["keep"]
    assert stats.rows_dropped_delegate == 1


# --- integrity and cleaning -------------------------------------------


def test_missing_raw_column_raises_naming_it(tmp_path: Path) -> None:
    src = _write_raw(tmp_path, [_row("s1")], drop_columns=("state_map",))

    with pytest.raises(ValueError, match="state_map"):
        build_stanford_corpus(src=src, dst=tmp_path / "corpus.parquet")


def test_duplicate_speech_ids_keep_the_first_and_are_counted(tmp_path: Path) -> None:
    table, stats = _build(
        tmp_path,
        [
            _row("dup", speech="first version " * 30),
            _row("dup", speech="second version " * 30),
            _row("unique"),
        ],
    )

    assert table.column("speech_id").to_pylist() == ["dup", "unique"]
    assert table.to_pylist()[0]["text"].startswith("first version")
    assert stats.rows_dropped_duplicate_id == 1


def test_empty_and_whitespace_only_text_is_dropped(tmp_path: Path) -> None:
    table, stats = _build(
        tmp_path,
        [_row("ok"), _row("blank", speech="   \n\t  "), _row("null", speech=None)],
    )

    assert table.column("speech_id").to_pylist() == ["ok"]
    assert stats.rows_dropped_empty_text == 2


def test_whitespace_is_normalized_but_content_is_not_stripped(tmp_path: Path) -> None:
    messy = "  Mr. Speaker,\n\n  I  rise   today\tin support. " + "word " * 60
    table, _ = _build(tmp_path, [_row("s1", speech=messy)])
    text = table.to_pylist()[0]["text"]

    assert text.startswith("Mr. Speaker, I rise today in support.")
    assert "  " not in text
    assert "\n" not in text


# --- streaming ---------------------------------------------------------


def test_row_group_size_does_not_change_the_result(tmp_path: Path) -> None:
    rows = [
        _row(f"s{i}", word_count=str(40 + i), speech="word " * (40 + i))
        for i in range(40)
    ]

    single, single_stats = _build(tmp_path, rows, name="single.parquet")
    chunked, chunked_stats = _build(
        tmp_path, rows, name="chunked.parquet", row_group_size=7
    )

    assert single.equals(chunked)
    assert single_stats.rows_written == chunked_stats.rows_written


# --- statistics --------------------------------------------------------


def test_stats_arithmetic_closes(tmp_path: Path) -> None:
    _, stats = _build(
        tmp_path,
        [
            _row("ok1"),
            _row("ok2"),
            _row("short", word_count="10", speech="word " * 10),
            _row("blank", speech="  "),
            _row("dup"),
            _row("dup"),
            _row("delegate", state_map="GU"),
            _row(
                "barkley",
                party="I",
                last_name="BARKLEY",
                state_map="MN",
                speakerid="107113431",
            ),
        ],
    )

    assert stats.rows_read == 8
    assert stats.rows_written == 3
    assert (
        stats.rows_read
        == stats.rows_written
        + stats.rows_dropped_delegate
        + stats.rows_dropped_excluded_member
        + stats.rows_dropped_short
        + stats.rows_dropped_empty_text
        + stats.rows_dropped_duplicate_id
    )


def test_stats_record_the_run_parameters_and_distributions(tmp_path: Path) -> None:
    _, stats = _build(
        tmp_path,
        [
            _row("s1", party="D", congress=107, date="20010103"),
            _row("s2", party="R", congress=114, date="20170103", chamber_map="S"),
            _row(
                "s3",
                party="I",
                last_name="SANDERS",
                state_map="VT",
                congress=110,
                speakerid="110118221",
            ),
        ],
    )
    payload = stats.to_dict()

    assert payload["min_word_count"] == 50
    assert payload["source_tag"] == "stanford"
    assert payload["party_counts"] == {"D": 2, "R": 1}
    assert payload["party_counts_original"] == {"D": 1, "I": 1, "R": 1}
    assert payload["congress_counts"] == {"107": 1, "110": 1, "114": 1}
    assert payload["chamber_counts"] == {"H": 2, "S": 1}
    assert payload["date_min"] == "2001-01-03"
    assert payload["date_max"] == "2017-01-03"
    assert payload["raw_schema"]["speech_id"]


# --- overwrite guard and limit -----------------------------------------


def test_existing_output_is_not_clobbered_without_overwrite(tmp_path: Path) -> None:
    src = _write_raw(tmp_path, [_row("s1")])
    dst = tmp_path / "corpus.parquet"
    dst.write_bytes(b"do not lose me")

    with pytest.raises(FileExistsError):
        build_stanford_corpus(src=src, dst=dst)
    assert dst.read_bytes() == b"do not lose me"

    build_stanford_corpus(src=src, dst=dst, overwrite=True)
    assert pq.read_table(dst).num_rows == 1


def test_limit_caps_rows_read(tmp_path: Path) -> None:
    rows = [_row(f"s{i}") for i in range(30)]

    _, stats = _build(tmp_path, rows, row_group_size=4)
    assert stats.rows_read == 30

    src = _write_raw(tmp_path, rows, name="limited.parquet", row_group_size=4)
    stats = build_stanford_corpus(
        src=src, dst=tmp_path / "limited_corpus.parquet", limit=10
    )
    assert stats.rows_read == 10
    assert stats.rows_written == 10
