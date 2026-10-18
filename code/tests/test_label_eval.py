"""Tests for the label check: agreement statistics, the sheet, the S8 checks.

Expected values are worked out by hand in the comments, not with the code
under test. The two scripts run end to end on tiny synthetic files.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_labels
import make_label_check_sheet

from src.analysis import attach_labels
from src.config import LABEL_MODEL, SOURCE_BREAK_CONGRESS
from src.label_eval import (
    SHEET_COLUMNS,
    accuracy,
    agreement,
    build_check_sheet,
    coder_labels,
    cohen_kappa,
    compare,
    confusion_matrix,
    label_shares,
    macro_f1,
    procedural_crosstab,
)

# --- agreement ----------------------------------------------------------------------


def test_perfect_agreement_is_one_on_every_statistic() -> None:
    a = ["x", "y", "x", "z"]

    assert agreement(a, list(a)) == {
        "n": 4,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "kappa": 1.0,
    }


def test_agreement_at_chance_has_zero_kappa() -> None:
    # p_o = 2/4; each rater uses x and y half the time, p_e = 0.5 -> kappa 0.
    a, b = ["x", "x", "y", "y"], ["x", "y", "x", "y"]

    assert accuracy(a, b) == 0.5
    assert cohen_kappa(a, b) == pytest.approx(0.0)


def _two_by_two() -> tuple[list[str], list[str]]:
    """50 items, rater a in rows: [[20, 5], [10, 15]] over labels A, B."""
    pairs = [("A", "A")] * 20 + [("A", "B")] * 5 + [("B", "A")] * 10 + [("B", "B")] * 15
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_kappa_on_a_known_table() -> None:
    # p_o = 35/50 = 0.7. Row sums 25, 25; column sums 30, 20:
    # p_e = (25*30 + 25*20) / 50^2 = 0.5. kappa = (0.7 - 0.5) / 0.5 = 0.4.
    a, b = _two_by_two()

    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert accuracy(a, b) == pytest.approx(0.7)


def test_macro_f1_on_a_known_table() -> None:
    # A: TP 20, FP 10, FN 5 -> 40/55. B: TP 15, FP 5, FN 10 -> 30/45.
    # Mean = (0.727272... + 0.666666...) / 2 = 0.696969...
    a, b = _two_by_two()

    assert macro_f1(a, b) == pytest.approx((40 / 55 + 30 / 45) / 2)
    # Symmetric: swapping the raters swaps FP and FN.
    assert macro_f1(b, a) == pytest.approx(macro_f1(a, b))


def test_kappa_is_undefined_when_both_raters_use_one_label() -> None:
    assert math.isnan(cohen_kappa(["x", "x"], ["x", "x"]))


def test_the_confusion_matrix_follows_the_codebook_order() -> None:
    matrix = confusion_matrix(
        ["policy_debate", "procedural", "procedural"],
        ["procedural", "procedural", "ceremonial"],
        order=("procedural", "ceremonial", "policy_debate"),
    )

    assert list(matrix.index) == ["procedural", "ceremonial", "policy_debate"]
    assert matrix.loc["procedural", "procedural"] == 1
    assert matrix.loc["procedural", "ceremonial"] == 1
    assert matrix.loc["policy_debate", "procedural"] == 1
    assert matrix.to_numpy().sum() == 3


def _rater(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["speech_id", "speech_type", "topic", "topic_block"]
    ).set_index("speech_id")


def test_topics_are_compared_only_where_both_raters_saw_policy_debate() -> None:
    first = _rater(
        [
            ("s1", "policy_debate", "health", "economic"),
            ("s2", "policy_debate", "immigration", "social_cultural"),
            ("s3", "ceremonial", None, None),
        ]
    )
    second = _rater(
        [
            ("s1", "policy_debate", "welfare_labor", "economic"),
            ("s2", "procedural", None, None),
            ("s3", "ceremonial", None, None),
        ]
    )

    types_a, _ = compare(first, second, "speech_type")
    topics_a, topics_b = compare(first, second, "topic")
    blocks_a, blocks_b = compare(first, second, "topic_block")

    assert list(types_a.index) == ["s1", "s2", "s3"]
    assert list(topics_a) == ["health"] and list(topics_b) == ["welfare_labor"]
    assert list(blocks_a) == list(blocks_b) == ["economic"]


# --- the check sheet -------------------------------------------------------------------


def _candidates(n_per_cell: int = 20) -> pd.DataFrame:
    rows = []
    for chamber in ("H", "S"):
        for source, term in (("stanford", "Bush"), ("stanford", "Obama"),
                             ("govinfo", "Trump")):  # fmt: skip
            for n in range(n_per_cell):
                rows.append(
                    {
                        "speech_id": f"{chamber}-{source}-{term}-{n:02d}",
                        "chamber": chamber,
                        "source": source,
                        "term": term,
                        "congress_number": 110 if source == "stanford" else 116,
                        "text": "text",
                    }
                )
    return pd.DataFrame(rows)


STRATA = ("chamber", "source", "term")


def test_the_sheet_is_blind_with_the_double_coded_rows_first() -> None:
    sheet = build_check_sheet(_candidates(), 60, 12, STRATA, seed=42)

    assert tuple(sheet.columns) == SHEET_COLUMNS
    assert not {"speech_type", "topic", "topic_block", "reason"} & set(sheet.columns)
    assert len(sheet) == 60
    assert sheet["overlap"].tolist() == [1] * 12 + [0] * 48
    assert (sheet[["coder1_type", "coder2_topic"]] == "").all().all()


def test_the_sample_and_its_double_coded_part_are_stratified() -> None:
    candidates = _candidates()

    sheet = build_check_sheet(candidates, 60, 12, STRATA, seed=42)

    cells = candidates.set_index("speech_id").loc[sheet["speech_id"]]
    assert cells.groupby(list(STRATA)).size().tolist() == [10] * 6
    double = cells[sheet.set_index("speech_id")["overlap"] == 1]
    assert double.groupby(list(STRATA)).size().tolist() == [2] * 6


def test_the_sheet_is_reproducible() -> None:
    first = build_check_sheet(_candidates(), 60, 12, STRATA, seed=42)

    assert first.equals(build_check_sheet(_candidates(), 60, 12, STRATA, seed=42))


def _coded(rows: list[tuple]) -> pd.DataFrame:
    sheet = pd.DataFrame(
        rows, columns=["speech_id", "coder1_type", "coder1_topic"]
    ).assign(coder2_type=None, coder2_topic=None, overlap=0)
    return sheet


def test_coder_labels_take_valid_codes_and_skip_uncoded_rows() -> None:
    sheet = _coded(
        [
            ("s1", "policy_debate", " health "),
            ("s2", "procedural", None),
            ("s3", None, None),
        ]
    )

    labels = coder_labels(sheet, "coder1")

    assert list(labels.index) == ["s1", "s2"]
    assert labels.loc["s1", "topic"] == "health"
    assert labels.loc["s1", "topic_block"] == "economic"
    assert pd.isna(labels.loc["s2", "topic"])


def test_coder_typos_are_listed_not_dropped() -> None:
    sheet = _coded(
        [
            ("s1", "Policy_Debate", "health"),
            ("s2", "policy_debate", "healthcare"),
            ("s3", "ceremonial", "health"),
            ("s4", None, "health"),
        ]
    )

    with pytest.raises(ValueError) as error:
        coder_labels(sheet, "coder1")

    message = str(error.value)
    assert "4 code(s)" in message
    assert all(s in message for s in ("s1", "s2", "s3", "s4"))


def _labels_dir(tmp_path: Path, rows: list[dict], run: str = "L1") -> Path:
    labels = tmp_path / "labels"
    labels.mkdir(exist_ok=True)
    (labels / f"label_manifest_s8_{run}.json").write_text(
        json.dumps(
            {
                "model": LABEL_MODEL,
                "stages": [{"selection": {"from_run": "S8RUN"}}],
            }
        )
    )
    (labels / f"label_s8_{LABEL_MODEL}_{run}.jsonl").write_text(
        "".join(json.dumps({"error": None, **r}) + "\n" for r in rows)
    )
    return labels


def test_the_sheet_script_never_overwrites_a_sheet(tmp_path: Path, capsys) -> None:
    rows = [
        {"speech_id": f"s{n}", "speech_type": "procedural", "topic": None}
        for n in range(6)
    ]
    labels = _labels_dir(tmp_path, rows)
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"speech_id": f"s{n}", "party": "D", "chamber": "HS"[n % 2],
                 "source": "stanford", "congress_number": 110, "text": f"text {n}"}
                for n in range(6)
            ]
        ),
        corpus,
    )  # fmt: skip
    out = tmp_path / "check"
    argv = [
        "--run", "L1", "--size", "4", "--overlap", "2", "--labels-dir", str(labels),
        "--corpus", str(corpus), "--output-dir", str(out),
    ]  # fmt: skip

    assert make_label_check_sheet.main(argv) == 0
    sheet_file = out / "label_check_s8_L1.csv"
    sheet = pd.read_csv(sheet_file, dtype={"speech_id": str})
    assert len(sheet) == 4 and sheet["overlap"].sum() == 2
    assert "speech_type" not in sheet.columns
    sheet_file.write_text(sheet_file.read_text() + "# someone's codes\n")
    coded = sheet_file.read_bytes()

    assert make_label_check_sheet.main(argv) == 2
    assert "exists" in capsys.readouterr().err
    assert sheet_file.read_bytes() == coded


# --- the analysis side --------------------------------------------------------------------


def test_attach_labels_adds_the_columns_and_keeps_every_speech(tmp_path: Path) -> None:
    labels = _labels_dir(
        tmp_path,
        [
            {"speech_id": "s1", "speech_type": "policy_debate", "topic": "law_crime"},
            {"speech_id": "s2", "speech_type": None, "topic": None, "error": "boom"},
        ],
    )
    frame = pd.DataFrame({"speech_id": ["s1", "s2", "s3"], "ideology": [0.1, 0, 0.3]})

    out = attach_labels(frame, "L1", labels)

    assert out["speech_id"].tolist() == ["s1", "s2", "s3"]
    assert out.loc[0, "speech_type"] == "policy_debate"
    assert out.loc[0, "topic_block"] == "security_foreign"
    assert out.loc[1:, ["speech_type", "topic", "topic_block"]].isna().all().all()


def _scored_frame() -> pd.DataFrame:
    """Six scored, labelled speeches across the 114th and 115th."""
    return pd.DataFrame(
        {
            "speech_id": [f"s{n}" for n in range(6)],
            "chamber": ["H"] * 6,
            "source": ["stanford", "stanford", "govinfo", "govinfo", "govinfo", "govinfo"],
            "congress_number": [114, 114, 114, 115, 115, 115],
            "speech_type": ["procedural", "policy_debate", "procedural",
                            "procedural", "policy_debate", "ceremonial"],
            "topic_block": [None, "economic", None, None, "governance", None],
            "procedural": [True, False, False, True, False, False],
            "ideology": [0.0, -0.6, 0.2, 0.0, 0.5, 0.1],
            "tone": [0.0, 0.4, 0.1, 0.0, 0.7, 0.0],
        }
    )  # fmt: skip


def test_label_shares_sum_to_one_and_mark_the_mixed_congress() -> None:
    shares = label_shares(_scored_frame(), ["congress_number", "chamber"])

    assert shares.groupby("congress_number")["share"].sum().tolist() == [1.0, 1.0]
    # 114th: 2 of 3 procedural.
    row = shares[
        (shares["congress_number"] == 114) & (shares["speech_type"] == "procedural")
    ]
    assert row["share"].iloc[0] == pytest.approx(2 / 3)
    assert set(shares.loc[shares["mixed_source"], "congress_number"]) == {
        SOURCE_BREAK_CONGRESS
    }


def test_the_procedural_crosstab_counts_label_against_flag() -> None:
    table = procedural_crosstab(_scored_frame())

    # procedural label: s0 and s3 flagged, s2 scored.
    assert table.loc["procedural", "all models 0.0"] == 2
    assert table.loc["procedural", "scored"] == 1
    assert table.loc["All", "All"] == 6


def test_the_evaluation_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """Scores are faked; the sheet, label runs, agreement and outputs are real."""
    rows = [
        {"speech_id": "s0", "speech_type": "procedural", "topic": None},
        {"speech_id": "s1", "speech_type": "policy_debate", "topic": "health"},
        {"speech_id": "s2", "speech_type": "procedural", "topic": None},
        {"speech_id": "s3", "speech_type": "procedural", "topic": None},
        {"speech_id": "s4", "speech_type": "policy_debate", "topic": "government_process"},
        {"speech_id": "s5", "speech_type": "ceremonial", "topic": None},
    ]  # fmt: skip
    labels = _labels_dir(tmp_path, rows)
    sheet = pd.DataFrame(
        {
            "speech_id": [r["speech_id"] for r in rows],
            "overlap": [1, 1, 1, 0, 0, 0],
            "chamber": "H", "source": "govinfo", "congress_number": 115, "text": "t",
            "coder1_type": ["procedural", "policy_debate", "ceremonial",
                            "procedural", "policy_debate", "ceremonial"],
            "coder1_topic": [None, "health", None, None, "government_process", None],
            "coder2_type": ["procedural", "policy_debate", "ceremonial", None, None, None],
            "coder2_topic": [None, "welfare_labor", None, None, None, None],
        }
    )  # fmt: skip
    sheet_file = tmp_path / "sheet.csv"
    sheet.to_csv(sheet_file, index=False)
    seen = {}

    def fake_scores(run, *args):
        seen["run"] = run
        return _scored_frame().drop(columns=["speech_type", "topic_block"])

    monkeypatch.setattr(evaluate_labels, "load_scored_run", fake_scores)
    metrics, plots = tmp_path / "metrics", tmp_path / "plots"

    code = evaluate_labels.main(
        ["--sheet", str(sheet_file), "--run", "L1", "--labels-dir", str(labels),
         "--metrics-dir", str(metrics), "--plots-dir", str(plots)]
    )  # fmt: skip

    assert code == 0
    assert seen["run"] == "S8RUN"  # the label run's own source
    stats = pd.read_csv(metrics / "label_agreement_L1.csv")
    llm = stats[(stats["rater_a"] == LABEL_MODEL) & (stats["level"] == "speech_type")]
    # 5 of 6 agree with coder 1 (s2: procedural vs ceremonial).
    assert llm["n"].iloc[0] == 6
    assert llm["accuracy"].iloc[0] == pytest.approx(5 / 6)
    coders = stats[(stats["rater_a"] == "coder1") & (stats["level"] == "topic")]
    # Only s1 is policy debate for both coders, and they differ on its topic.
    assert coders["n"].iloc[0] == 1 and coders["accuracy"].iloc[0] == 0.0
    checks = json.loads((metrics / "label_consistency_L1.json").read_text())
    assert checks["procedural_label_vs_zero_flag"] == {
        "labelled_procedural": 3, "flagged_all_zero": 2, "both": 2,
        "label_only": 1, "flag_only": 0,
    }  # fmt: skip
    assert (plots / "label_type_shares_L1.png").exists()
    assert (metrics / "label_confusion_L1.csv").exists()
    assert (metrics / "label_eval_L1.json").exists()


def test_the_label_share_figure_keeps_chambers_apart_and_marks_the_break() -> None:
    from src.plots import plot_label_shares

    shares = label_shares(_scored_frame().assign(chamber=["H", "S"] * 3),
                          ["congress_number", "chamber"])  # fmt: skip

    fig = plot_label_shares(shares, "t")

    house, senate = fig.axes[:2]
    for ax in (house, senate):
        assert any(
            list(line.get_xdata()) == [SOURCE_BREAK_CONGRESS, SOURCE_BREAK_CONGRESS]
            for line in ax.get_lines()
        )
    assert house.get_title(loc="left") == "House"
    assert senate.get_title(loc="left") == "Senate"


def test_a_congress_without_a_speech_type_is_drawn_at_zero() -> None:
    """Not skipped: a skipped point would draw a line across it."""
    from src.plots import plot_label_shares

    shares = pd.DataFrame(
        {
            "congress_number": [113, 114, 115, 113, 115],
            "chamber": ["H"] * 5,
            "speech_type": ["policy_debate"] * 3 + ["ceremonial"] * 2,
            "share": [0.8, 1.0, 0.7, 0.2, 0.3],
        }
    )

    fig = plot_label_shares(shares, "t", order=("ceremonial", "policy_debate"))

    (ceremonial,) = [
        ln for ln in fig.axes[0].get_lines() if ln.get_label() == "ceremonial"
    ]
    assert list(ceremonial.get_xdata()) == [113, 114, 115]
    assert list(ceremonial.get_ydata()) == [0.2, 0.0, 0.3]
