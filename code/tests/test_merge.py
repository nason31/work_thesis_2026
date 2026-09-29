"""Tests for the Stanford + govinfo merge.

Inputs are small processed-corpus files written in-process with the real
schemas, each next to the stats JSON its build would have left, so the suite
runs without the real data. The default fixture mirrors the real seam:
Stanford ends inside the 114th Congress on 2016-09-09, and govinfo picks up in
the same Congress on 2016-09-12.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import merge
from src.corpus import CORPUS_SCHEMA
from src.govinfo import GOVINFO_SCHEMA
from src.merge import MERGED_SCHEMA, merge_corpora

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_merged_corpus import main as build_main

# --- fixture helpers ---------------------------------------------------


def _stanford_row(speech_id: str, **overrides: object) -> dict[str, object]:
    """One processed Stanford row; the defaults pass every check."""
    row: dict[str, object] = {
        "speech_id": speech_id,
        "date": dt.date(2013, 1, 10),
        "member_id": "113118221",
        "party": "D",
        "chamber": "S",
        "congress_number": 113,
        "text": "word " * 60,
        "source": "stanford",
        "last_name": "SANDERS",
        "state": "VT",
        "word_count": 60,
        "party_original": "I",
    }
    row.update(overrides)
    return row


def _govinfo_row(speech_id: str, **overrides: object) -> dict[str, object]:
    """One processed govinfo row; the defaults pass every check."""
    row: dict[str, object] = {
        "speech_id": speech_id,
        "date": dt.date(2017, 6, 5),
        "member_id": "M000355",
        "party": "R",
        "chamber": "S",
        "congress_number": 115,
        "text": "word " * 70,
        "source": "govinfo",
        "last_name": "MCCONNELL",
        "state": "KY",
        "word_count": 70,
        "party_original": "R",
        "icpsr": 14921,
    }
    row.update(overrides)
    return row


def _stanford_rows() -> list[dict[str, object]]:
    """113th and 114th Congress; the last row is Stanford's real last day."""
    return [
        _stanford_row("1130000001"),
        _stanford_row("1130000002", party="R", chamber="H", member_id="113120350"),
        _stanford_row("1140000003", date=dt.date(2016, 9, 9), congress_number=114),
    ]


def _govinfo_rows() -> list[dict[str, object]]:
    """115th, then govinfo's real first day in the 114th, with no ICPSR."""
    return [
        _govinfo_row("gov-0000000000000001"),
        _govinfo_row(
            "gov-0000000000000002",
            date=dt.date(2016, 9, 12),
            congress_number=114,
            party="D",
            chamber="H",
            icpsr=None,
        ),
    ]


def _write_input(
    tmp_path: Path,
    name: str,
    rows: list[dict[str, object]],
    schema: pa.Schema,
    stats: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write a processed corpus plus the stats JSON a full build would leave."""
    path = tmp_path / f"corpus_{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    if stats is None:
        stats = {
            "limit": None,
            "rows_written": len(rows),
            "source_sha256": f"raw-{name}",
        }
    stats_path = tmp_path / f"{name}_build_stats.json"
    stats_path.write_text(json.dumps(stats))
    return path, stats_path


def _merge(
    tmp_path: Path,
    *,
    stanford_rows: list[dict[str, object]] | None = None,
    govinfo_rows: list[dict[str, object]] | None = None,
    stanford_schema: pa.Schema = CORPUS_SCHEMA,
    govinfo_schema: pa.Schema = GOVINFO_SCHEMA,
    stanford_stats: dict[str, object] | None = None,
    govinfo_stats: dict[str, object] | None = None,
    dst: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, merge.MergedBuildStats]:
    """Write both inputs and merge them; return (output path, stats)."""
    s_path, s_stats = _write_input(
        tmp_path,
        "stanford",
        _stanford_rows() if stanford_rows is None else stanford_rows,
        stanford_schema,
        stanford_stats,
    )
    g_path, g_stats = _write_input(
        tmp_path,
        "govinfo",
        _govinfo_rows() if govinfo_rows is None else govinfo_rows,
        govinfo_schema,
        govinfo_stats,
    )
    dst = dst or tmp_path / "corpus.parquet"
    stats = merge_corpora(
        stanford=s_path,
        govinfo=g_path,
        dst=dst,
        stanford_stats=s_stats,
        govinfo_stats=g_stats,
        overwrite=overwrite,
    )
    return dst, stats


# --- what gets written -------------------------------------------------


def test_every_row_of_both_inputs_is_written(tmp_path: Path) -> None:
    dst, stats = _merge(tmp_path)

    assert pq.read_table(dst).num_rows == 5
    assert stats.rows_written == 5


def test_rows_pass_through_unchanged_stanford_first(tmp_path: Path) -> None:
    dst, _ = _merge(tmp_path)

    # Stanford rows have no ICPSR yet; the merge adds the column as null.
    expected = [{**row, "icpsr": None} for row in _stanford_rows()] + _govinfo_rows()
    assert pq.read_table(dst).to_pylist() == expected


def test_output_has_the_merged_schema(tmp_path: Path) -> None:
    dst, _ = _merge(tmp_path)

    assert MERGED_SCHEMA.names == CORPUS_SCHEMA.names + ["icpsr"]
    assert pq.ParquetFile(dst).schema_arrow.equals(MERGED_SCHEMA)


# --- output safety -----------------------------------------------------


def test_existing_output_is_kept_without_overwrite(tmp_path: Path) -> None:
    dst = tmp_path / "corpus.parquet"
    dst.write_text("previous corpus")

    with pytest.raises(FileExistsError, match="overwrite"):
        _merge(tmp_path, dst=dst)
    assert dst.read_text() == "previous corpus"


def test_overwrite_replaces_existing_output(tmp_path: Path) -> None:
    dst = tmp_path / "corpus.parquet"
    dst.write_text("previous corpus")

    _merge(tmp_path, dst=dst, overwrite=True)
    assert pq.read_table(dst).num_rows == 5


def test_output_that_is_an_input_is_refused(tmp_path: Path) -> None:
    # Writing over an input while still reading it would destroy it.
    dst = tmp_path / "corpus_stanford.parquet"

    with pytest.raises(ValueError, match="own input"):
        _merge(tmp_path, dst=dst, overwrite=True)


def test_failed_write_keeps_the_previous_output_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dst = tmp_path / "corpus.parquet"
    dst.write_text("previous corpus")

    def fail(spec: merge.MergeInput, writer: pq.ParquetWriter) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(merge, "_stream", fail)
    with pytest.raises(OSError, match="disk full"):
        _merge(tmp_path, dst=dst, overwrite=True)
    assert dst.read_text() == "previous corpus"
    assert not (tmp_path / "corpus.parquet.tmp").exists()


# --- input checks ------------------------------------------------------


@pytest.mark.parametrize(
    "missing, target", [("stanford", "make stanford"), ("govinfo", "make govinfo")]
)
def test_missing_input_names_the_build_to_run(
    tmp_path: Path, missing: str, target: str
) -> None:
    s_path, s_stats = _write_input(
        tmp_path, "stanford", _stanford_rows(), CORPUS_SCHEMA
    )
    g_path, g_stats = _write_input(tmp_path, "govinfo", _govinfo_rows(), GOVINFO_SCHEMA)
    {"stanford": s_path, "govinfo": g_path}[missing].unlink()
    # The old layout: corpus.parquet still holds a previous build. A failed
    # merge must leave it alone even with overwrite.
    dst = tmp_path / "corpus.parquet"
    dst.write_text("previous corpus")

    with pytest.raises(FileNotFoundError, match=target):
        merge_corpora(
            stanford=s_path,
            govinfo=g_path,
            dst=dst,
            stanford_stats=s_stats,
            govinfo_stats=g_stats,
            overwrite=True,
        )
    assert dst.read_text() == "previous corpus"


def test_missing_column_is_named(tmp_path: Path) -> None:
    schema = pa.schema([f for f in CORPUS_SCHEMA if f.name != "last_name"])

    with pytest.raises(ValueError, match="missing last_name"):
        _merge(tmp_path, stanford_schema=schema)


def test_wrong_column_type_is_named(tmp_path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("icpsr", pa.int64()) if f.name == "icpsr" else f
            for f in GOVINFO_SCHEMA
        ]
    )

    with pytest.raises(ValueError, match="icpsr is int64, expected int32"):
        _merge(tmp_path, govinfo_schema=schema)


def test_input_without_build_stats_is_refused(tmp_path: Path) -> None:
    s_path, s_stats = _write_input(
        tmp_path, "stanford", _stanford_rows(), CORPUS_SCHEMA
    )
    g_path, g_stats = _write_input(tmp_path, "govinfo", _govinfo_rows(), GOVINFO_SCHEMA)
    g_stats.unlink()

    with pytest.raises(FileNotFoundError, match="build stats"):
        merge_corpora(
            stanford=s_path,
            govinfo=g_path,
            dst=tmp_path / "corpus.parquet",
            stanford_stats=s_stats,
            govinfo_stats=g_stats,
        )


@pytest.mark.parametrize(
    "stats, message",
    [
        ({"limit": 50000, "rows_written": 3}, "--limit"),
        ({"limit": None, "rows_written": 99}, "different builds"),
        ({"rows_written": 3}, "cannot vouch"),
    ],
)
def test_input_that_is_not_its_full_build_is_refused(
    tmp_path: Path, stats: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _merge(tmp_path, stanford_stats=stats)


def test_wrong_source_tag_is_refused(tmp_path: Path) -> None:
    rows = _govinfo_rows()
    rows[0]["source"] = "stanford"

    with pytest.raises(ValueError, match="unexpected source"):
        _merge(tmp_path, govinfo_rows=rows)


@pytest.mark.parametrize(
    "side, day",
    [
        ("stanford", dt.date(2016, 9, 10)),
        ("govinfo", dt.date(2016, 9, 9)),
        ("govinfo", dt.date(2026, 1, 5)),
    ],
)
def test_dates_across_the_seam_are_refused(
    tmp_path: Path, side: str, day: dt.date
) -> None:
    stanford, govinfo = _stanford_rows(), _govinfo_rows()
    (stanford if side == "stanford" else govinfo)[-1]["date"] = day

    with pytest.raises(ValueError, match=day.isoformat()):
        _merge(tmp_path, stanford_rows=stanford, govinfo_rows=govinfo)


def test_speech_id_in_both_inputs_is_named(tmp_path: Path) -> None:
    govinfo = _govinfo_rows()
    govinfo[0]["speech_id"] = "1130000001"

    with pytest.raises(ValueError, match="1130000001"):
        _merge(tmp_path, govinfo_rows=govinfo)


@pytest.mark.parametrize("column, value", [("party", "I"), ("chamber", "J")])
def test_unexpected_code_is_named(tmp_path: Path, column: str, value: str) -> None:
    rows = _stanford_rows()
    rows[0][column] = value

    with pytest.raises(ValueError, match=f"unexpected {column}.*{value}"):
        _merge(tmp_path, stanford_rows=rows)


def test_null_in_a_key_column_is_refused(tmp_path: Path) -> None:
    rows = _govinfo_rows()
    rows[0]["party"] = None

    with pytest.raises(ValueError, match="null value.*'party'"):
        _merge(tmp_path, govinfo_rows=rows)


def test_congress_gap_is_named(tmp_path: Path) -> None:
    govinfo = [_govinfo_row("gov-0000000000000001", congress_number=116)]

    with pytest.raises(ValueError, match=r"\[115\]"):
        _merge(tmp_path, govinfo_rows=govinfo)


def test_empty_input_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        _merge(tmp_path, govinfo_rows=[])


# --- statistics --------------------------------------------------------


def test_cell_counts_split_by_congress_source_chamber_party(tmp_path: Path) -> None:
    _, stats = _merge(tmp_path)

    assert stats.cell_counts == {
        "113": {"stanford": {"H": {"R": 1}, "S": {"D": 1}}},
        "114": {"govinfo": {"H": {"D": 1}}, "stanford": {"S": {"D": 1}}},
        "115": {"govinfo": {"S": {"R": 1}}},
    }


def test_year_counts_split_by_source_and_chamber(tmp_path: Path) -> None:
    _, stats = _merge(tmp_path)

    assert stats.year_counts == {
        "2013": {"stanford": {"H": 1, "S": 1}},
        "2016": {"govinfo": {"H": 1}, "stanford": {"S": 1}},
        "2017": {"govinfo": {"S": 1}},
    }


def test_seam_and_date_range_are_recorded(tmp_path: Path) -> None:
    _, stats = _merge(tmp_path)

    assert stats.seam == {"stanford_last": "2016-09-09", "govinfo_first": "2016-09-12"}
    assert (stats.date_min, stats.date_max) == ("2013-01-10", "2017-06-05")


def test_rows_and_missing_icpsr_are_counted_by_source(tmp_path: Path) -> None:
    _, stats = _merge(tmp_path)

    assert stats.rows_by_source == {"stanford": 3, "govinfo": 2}
    assert stats.icpsr_missing_by_source == {"stanford": 3, "govinfo": 1}


def test_inputs_trace_back_to_processed_and_raw_files(tmp_path: Path) -> None:
    _, stats = _merge(tmp_path)
    processed = tmp_path / "corpus_stanford.parquet"

    stanford = stats.inputs["stanford"]
    assert stanford["sha256"] == hashlib.sha256(processed.read_bytes()).hexdigest()
    assert stanford["rows"] == 3
    assert stanford["raw_source_sha256"] == "raw-stanford"
    assert stats.inputs["govinfo"]["raw_source_sha256"] == "raw-govinfo"


def test_stats_write_as_json(tmp_path: Path) -> None:
    _, stats = _merge(tmp_path)

    path = stats.write(tmp_path / "merged_build_stats.json")
    assert json.loads(path.read_text())["rows_written"] == 5


# --- command line ------------------------------------------------------


def _cli_args(tmp_path: Path) -> tuple[list[str], Path, Path]:
    """Write both inputs; return (argv, stanford input, stats output)."""
    s_path, s_stats = _write_input(
        tmp_path, "stanford", _stanford_rows(), CORPUS_SCHEMA
    )
    g_path, g_stats = _write_input(tmp_path, "govinfo", _govinfo_rows(), GOVINFO_SCHEMA)
    stats_out = tmp_path / "merged_build_stats.json"
    argv = [
        "--stanford", str(s_path),
        "--govinfo", str(g_path),
        "--stanford-stats", str(s_stats),
        "--govinfo-stats", str(g_stats),
        "--output", str(tmp_path / "corpus.parquet"),
        "--stats-path", str(stats_out),
    ]  # fmt: skip
    return argv, s_path, stats_out


def test_cli_writes_the_corpus_and_its_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, _, stats_out = _cli_args(tmp_path)

    assert build_main(argv) == 0
    assert pq.read_table(tmp_path / "corpus.parquet").num_rows == 5
    assert json.loads(stats_out.read_text())["rows_written"] == 5
    assert "written" in capsys.readouterr().out


def test_cli_reports_a_failed_check_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, s_path, stats_out = _cli_args(tmp_path)
    s_path.unlink()

    assert build_main(argv) == 1
    assert "make stanford" in capsys.readouterr().err
    assert not stats_out.exists()
