"""The hand check of the speech labels, and how far raters agree.

Three pieces (docs/decisions.md L4):

- **The check sheet**: a seeded, stratified sample from a label run, with empty
  columns for two coders. The model's labels are deliberately left out, so the
  coding is blind. The first rows are coded by both coders.
- **Agreement**: accuracy, macro-F1 and Cohen's kappa for any two raters --
  the model against a coder, coder against coder, model against model --
  with confusion matrices. Implemented with numpy and pandas; scikit-learn is
  not a dependency.
- **Consistency with the S8 scores**, which costs nothing: does a speech
  labelled procedural get the 0.0 the scoring prompt asks for procedural
  speech, and how do scores and label shares vary by speech type, source and
  Congress.

Speech type is compared on every speech both raters labelled. Topic and block
are compared only where both raters called the speech a policy debate -- a
topic exists only there, so a disagreement on type is not counted twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    POLICY_SPEECH_TYPE,
    SOURCE_BREAK_CONGRESS,
    SPEECH_TYPES,
    TOPIC_BLOCKS,
    TOPICS,
)
from src.labeling import BLOCK_OF_TOPIC, stratified_subset

#: Codebook order for each level that agreement is reported on.
LEVELS: dict[str, tuple[str, ...]] = {
    "speech_type": SPEECH_TYPES,
    "topic": TOPICS,
    "topic_block": tuple(TOPIC_BLOCKS),
}

#: The coders' columns, per coder.
CODER_COLUMNS: dict[str, tuple[str, str]] = {
    "coder1": ("coder1_type", "coder1_topic"),
    "coder2": ("coder2_type", "coder2_topic"),
}

#: The sheet's columns, in order. No model labels: the coding is blind.
SHEET_COLUMNS: tuple[str, ...] = (
    "speech_id",
    "overlap",
    "chamber",
    "source",
    "congress_number",
    "text",
    *CODER_COLUMNS["coder1"],
    *CODER_COLUMNS["coder2"],
)

#: Longest text an Excel cell holds; longer ones are cut off when opened there.
EXCEL_CELL_LIMIT = 32_767


# --- agreement -------------------------------------------------------------------


def _labels(a: Sequence[str], b: Sequence[str], order: Sequence[str] = ()) -> list[str]:
    """Every label either rater used: codebook order first, then any others."""
    used = set(a) | set(b)
    return [x for x in order if x in used] + sorted(used - set(order))


def confusion_matrix(
    a: Sequence[str], b: Sequence[str], order: Sequence[str] = ()
) -> pd.DataFrame:
    """Counts, rater ``a`` in rows and rater ``b`` in columns."""
    if len(a) != len(b):
        raise ValueError(f"raters label {len(a)} and {len(b)} items")
    labels = _labels(a, b, order)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in zip(a, b):
        matrix[index[x], index[y]] += 1
    return pd.DataFrame(matrix, index=labels, columns=labels)


def accuracy(a: Sequence[str], b: Sequence[str]) -> float:
    """Share of items both raters labelled the same."""
    if not len(a):
        return float("nan")
    return float(np.mean([x == y for x, y in zip(a, b)]))


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa: agreement beyond what the raters' label shares predict.

    kappa = (p_o - p_e) / (1 - p_e), with p_o the observed agreement and p_e
    the agreement expected if each rater kept their own label shares but
    labelled independently. Undefined (nan) when p_e = 1, i.e. both raters
    used one and the same label throughout.
    """
    matrix = confusion_matrix(a, b).to_numpy(dtype=float)
    n = matrix.sum()
    if n == 0:
        return float("nan")
    observed = np.trace(matrix) / n
    expected = float(matrix.sum(axis=1) @ matrix.sum(axis=0)) / n**2
    if expected == 1:
        return float("nan")
    return float((observed - expected) / (1 - expected))


def macro_f1(a: Sequence[str], b: Sequence[str]) -> float:
    """Unweighted mean over labels of F1 = 2TP / (2TP + FP + FN).

    Over every label either rater used. Each label's F1 is the same whichever
    rater is taken as the reference (FP and FN swap), so this is symmetric.
    """
    matrix = confusion_matrix(a, b).to_numpy()
    if matrix.sum() == 0:
        return float("nan")
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    return float(np.mean(2 * tp / (2 * tp + fp + fn)))


def agreement(a: Sequence[str], b: Sequence[str]) -> dict[str, Any]:
    """n, accuracy, macro-F1 and Cohen's kappa of two raters on the same items."""
    return {
        "n": len(a),
        "accuracy": accuracy(a, b),
        "macro_f1": macro_f1(a, b),
        "kappa": cohen_kappa(a, b),
    }


def compare(
    first: pd.DataFrame, second: pd.DataFrame, level: str
) -> tuple[pd.Series, pd.Series]:
    """Two raters' labels at ``level``, on the speeches where both apply.

    Each frame is indexed by speech_id with columns speech_type, topic and
    topic_block, null where the rater gave no label. Speech type: every speech
    both labelled. Topic and block: only speeches both called policy debate.
    """
    both = first.join(second, how="inner", lsuffix="_a", rsuffix="_b")
    both = both.dropna(subset=["speech_type_a", "speech_type_b"])
    if level != "speech_type":
        both = both[
            (both["speech_type_a"] == POLICY_SPEECH_TYPE)
            & (both["speech_type_b"] == POLICY_SPEECH_TYPE)
        ].dropna(subset=[f"{level}_a", f"{level}_b"])
    both = both.sort_index()
    return both[f"{level}_a"], both[f"{level}_b"]


def compare_raters(
    raters: dict[str, pd.DataFrame], pairs: Sequence[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agreement per pair and level, and the confusion matrices in long form.

    Returns (agreement table, confusion table). The confusion table has one
    row per pair, level and cell: ``label_a`` (first rater), ``label_b``, ``n``.
    """
    stats, cells = [], []
    for a, b in pairs:
        for level, order in LEVELS.items():
            first, second = compare(raters[a], raters[b], level)
            stats.append(
                {"rater_a": a, "rater_b": b, "level": level}
                | agreement(list(first), list(second))
            )
            matrix = confusion_matrix(list(first), list(second), order)
            for label_a, row in matrix.iterrows():
                for label_b, n in row.items():
                    cells.append(
                        {
                            "rater_a": a,
                            "rater_b": b,
                            "level": level,
                            "label_a": label_a,
                            "label_b": label_b,
                            "n": int(n),
                        }
                    )
    return pd.DataFrame(stats), pd.DataFrame(cells)


# --- the check sheet ---------------------------------------------------------------


def build_check_sheet(
    frame: pd.DataFrame,
    size: int,
    overlap: int,
    strata: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """A blind coding sheet: ``size`` speeches, the first ``overlap`` double-coded.

    ``frame`` holds one row per candidate speech with the ``strata`` columns and
    the sheet's own (speech_id, chamber, source, congress_number, text). The
    sample is stratified, and so is its double-coded part, which is put first.
    Rows are shuffled within each part, so the order gives nothing away.
    """
    if overlap > size:
        raise ValueError(f"overlap {overlap} is larger than the sheet ({size})")
    chosen = set(stratified_subset(frame, size, strata, seed))
    sample = frame[frame["speech_id"].isin(chosen)]
    double = set(stratified_subset(sample, overlap, strata, seed))
    sample = sample.assign(overlap=sample["speech_id"].isin(double).astype(int))
    sample = sample.sample(frac=1, random_state=seed).sort_values(
        "overlap", ascending=False, kind="stable"
    )
    for column in (*CODER_COLUMNS["coder1"], *CODER_COLUMNS["coder2"]):
        sample[column] = ""
    return sample[list(SHEET_COLUMNS)].reset_index(drop=True)


def _blank(value: Any) -> bool:
    """An empty cell: missing, or only whitespace."""
    return bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())


def coder_labels(sheet: pd.DataFrame, coder: str) -> pd.DataFrame:
    """One coder's labels from a filled sheet, indexed by speech_id.

    Uncoded rows (no type) are left out. Codes are trimmed of surrounding
    whitespace and otherwise taken as typed.

    Raises:
        ValueError: listing every row whose codes are not in the codebook, or
            whose topic does not fit its type -- fix the sheet, nothing is
            dropped silently.
    """
    type_column, topic_column = CODER_COLUMNS[coder]
    problems, records = [], []
    for row in sheet.itertuples(index=False):
        speech_type = getattr(row, type_column)
        topic = getattr(row, topic_column)
        if _blank(speech_type):
            if not _blank(topic):
                problems.append(f"{row.speech_id}: topic {topic!r} without a type")
            continue
        speech_type = str(speech_type).strip()
        topic = None if _blank(topic) else str(topic).strip()
        if speech_type not in SPEECH_TYPES:
            problems.append(f"{row.speech_id}: type {speech_type!r} not in codebook")
        elif speech_type == POLICY_SPEECH_TYPE and topic not in TOPICS:
            problems.append(f"{row.speech_id}: topic {topic!r} not in codebook")
        elif speech_type != POLICY_SPEECH_TYPE and topic is not None:
            problems.append(f"{row.speech_id}: topic {topic!r} on a {speech_type}")
        else:
            records.append(
                {
                    "speech_id": str(row.speech_id),
                    "speech_type": speech_type,
                    "topic": topic,
                    "topic_block": BLOCK_OF_TOPIC.get(topic) if topic else None,
                }
            )
    if problems:
        raise ValueError(
            f"{coder}: {len(problems)} code(s) to fix in the sheet:\n  "
            + "\n  ".join(problems)
        )
    return pd.DataFrame(
        records, columns=["speech_id", "speech_type", "topic", "topic_block"]
    ).set_index("speech_id")


# --- consistency with the S8 scores ---------------------------------------------------


def procedural_crosstab(frame: pd.DataFrame) -> pd.DataFrame:
    """Speech type against the scores' procedural flag (every model at 0.0).

    ``frame`` is a scored run with labels attached; an unlabelled speech is
    its own row, not dropped.
    """
    return pd.crosstab(
        frame["speech_type"].fillna("(no label)"),
        frame["procedural"].map({True: "all models 0.0", False: "scored"}),
        margins=True,
    )


def scores_by_type(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean |ideology|, mean ideology and mean tone per chamber and speech type."""
    data = frame.dropna(subset=["speech_type"]).assign(
        abs_ideology=lambda f: f["ideology"].abs()
    )
    return (
        data.groupby(["chamber", "speech_type"])
        .agg(
            n=("speech_id", "size"),
            mean_abs_ideology=("abs_ideology", "mean"),
            mean_ideology=("ideology", "mean"),
            mean_tone=("tone", "mean"),
            procedural_share=("procedural", "mean"),
        )
        .reset_index()
    )


def label_shares(
    frame: pd.DataFrame, by: Sequence[str], column: str = "speech_type"
) -> pd.DataFrame:
    """Share of each label of ``column`` within each ``by`` group.

    Over labelled speeches only. Topic and block shares are therefore shares
    of policy debate. In a stratified sample these are sample shares, not
    corpus shares (S11). ``mixed_source`` marks the 114th Congress, which
    holds both sources.
    """
    data = frame.dropna(subset=[column])
    counts = data.groupby([*by, column]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby(list(by))["n"].transform("sum")
    if "congress_number" in by:
        counts["mixed_source"] = counts["congress_number"] == SOURCE_BREAK_CONGRESS
    return counts
