"""Analysis of a scoring run: scores joined to speeches, members and DW-NOMINATE.

Feeds ``code/notebooks/04_score_analysis.ipynb`` (preliminary findings for
Meeting 2). Three rules hold throughout:

- **Chambers are never pooled.** The govinfo Senate is thinner than Stanford's
  (O8), and the sample gives every Congress x party x chamber cell the same
  quota, so a pooled mean is not a corpus mean (S11).
- **Every interval is clustered by member.** Many speeches come from the same
  member; treating them as independent would overstate the evidence. A member
  is their career-long ICPSR where the crosswalk has one, else the
  source-native id.
- **Nothing here is final.** The figures are for discussion, not citation.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from src.config import CORPUS_PATH, CROSSWALK_PATH, PRESIDENTS_BY_CONGRESS, SCORES_DIR
from src.crosswalk import UNIT_KEY
from src.runs import (
    effective_rows,
    ensemble_path,
    find_label,
    is_ok,
    model_path,
    read_rows,
    run_models,
)
from src.validation import CROSSWALK_COLUMNS

#: Two-sided 95% normal quantile; intervals here are normal approximations.
Z95 = 1.959964

#: The default cell: one value per Congress, chamber and party.
CELL = ("congress_number", "chamber", "party")

_ENSEMBLE_COLUMNS = {
    "ideology_score_mean": "ideology",
    "ideology_score_std": "ideology_std",
    "tone_score_mean": "tone",
    "tone_score_std": "tone_std",
}

#: Corpus columns carried onto each speech.
_SPEECH_COLUMNS = ("date", "source", "member_id", "last_name", "state", "word_count")


# --- loading -------------------------------------------------------------


def _model_frame(scores_dir: Path, label: str, model: str, run: str) -> pd.DataFrame:
    """One model's answer per speech: the successful attempt where there is one."""
    rows = effective_rows(read_rows(model_path(scores_dir, label, model, run)))
    return pd.DataFrame(
        [
            {
                "speech_id": speech_id,
                f"ideology_{model}": row.get("ideology_score") if is_ok(row) else None,
                f"tone_{model}": row.get("tone_score") if is_ok(row) else None,
                f"reasoning_{model}": row.get("reasoning"),
            }
            for speech_id, row in rows.items()
        ],
        columns=[
            "speech_id",
            f"ideology_{model}",
            f"tone_{model}",
            f"reasoning_{model}",
        ],
    )


def load_scored_run(
    run: str,
    scores_dir: Path = SCORES_DIR,
    corpus: Path = CORPUS_PATH,
    crosswalk: Path = CROSSWALK_PATH,
) -> pd.DataFrame:
    """One row per scored speech, with everything the analysis needs.

    The ensemble scores (``ideology``, ``tone`` and their cross-model spread),
    each model's score, tone and justification, the speech's date, source and
    member from the corpus, the member's Voteview scores through the crosswalk,
    ``member_key`` for clustering, ``procedural`` (every model at exactly 0.0,
    as the prompt instructs for procedural speech) and the presidency.

    Raises:
        ValueError: a scored speech is not in the corpus -- the run belongs to
            another build of it.
    """
    label = find_label(scores_dir, run)
    models = run_models(scores_dir, run)

    frame = pd.DataFrame(read_rows(ensemble_path(scores_dir, label, run)))
    frame["speech_id"] = frame["speech_id"].astype(str)
    frame = frame.rename(columns=_ENSEMBLE_COLUMNS)
    for model in models:
        frame = frame.merge(
            _model_frame(scores_dir, label, model, run),
            on="speech_id",
            how="left",
            validate="one_to_one",
        )
    scored = [f"ideology_{model}" for model in models]
    frame["procedural"] = frame[scored].notna().any(axis=1) & frame[scored].fillna(
        0.0
    ).eq(0.0).all(axis=1)

    ids = frame["speech_id"].tolist()
    speeches = pq.read_table(
        corpus,
        columns=["speech_id", *dict.fromkeys((*_SPEECH_COLUMNS, *UNIT_KEY))],
        filters=[("speech_id", "in", ids)],
    ).to_pandas()
    missing = sorted(set(ids) - set(speeches["speech_id"]))
    if missing:
        raise ValueError(f"{len(missing)} scored speech(es) are not in {corpus}")
    members = pd.read_parquet(crosswalk, columns=[*UNIT_KEY, *CROSSWALK_COLUMNS])
    speeches = speeches.merge(members, on=list(UNIT_KEY), how="left", validate="m:1")
    speeches["in_validation"] = speeches["in_validation"].fillna(False).astype(bool)
    speeches["member_key"] = np.where(
        speeches["icpsr"].notna(),
        "icpsr:" + speeches["icpsr"].astype("Int64").astype(str),
        speeches["source"] + ":" + speeches["member_id"],
    )
    speeches = speeches.drop(columns=["congress_number", "chamber"])

    frame = frame.merge(speeches, on="speech_id", how="left", validate="one_to_one")
    return add_presidency(frame)


def _term_labels() -> dict[int, str]:
    """Congress -> its presidential term, e.g. "Trump 2017-2020".

    Consecutive Congresses under one president form a term, so Trump's two
    terms stay apart. Years run from the first Congress's start to the last
    one's end.
    """
    labels: dict[int, str] = {}
    ordered = sorted(PRESIDENTS_BY_CONGRESS.items())
    for name, group in groupby(ordered, key=lambda item: item[1][0]):
        congresses = [congress for congress, _ in group]
        first = 1789 + 2 * (congresses[0] - 1)
        last = 1789 + 2 * (congresses[-1] - 1) + 1
        for congress in congresses:
            labels[congress] = f"{name} {first}-{last}"
    return labels


def add_presidency(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``president``, ``president_party``, ``term`` and ``in_opposition``.

    Raises:
        KeyError: a Congress with no entry in PRESIDENTS_BY_CONGRESS.
    """
    unknown = sorted(set(frame["congress_number"]) - set(PRESIDENTS_BY_CONGRESS))
    if unknown:
        raise KeyError(f"no president recorded for Congress {unknown}")
    out = frame.copy()
    out["president"] = out["congress_number"].map(
        lambda c: PRESIDENTS_BY_CONGRESS[c][0]
    )
    out["president_party"] = out["congress_number"].map(
        lambda c: PRESIDENTS_BY_CONGRESS[c][1]
    )
    out["term"] = out["congress_number"].map(_term_labels())
    out["in_opposition"] = out["party"] != out["president_party"]
    return out


# --- clustered estimates ---------------------------------------------------


def _clustered_mean(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Mean and its cluster-robust standard error (CR1, as statsmodels does).

    SE = sqrt(G / (G - 1) * sum_g e_g^2) / N, where e_g is the sum of the
    residuals in cluster g. With one cluster there is no error to estimate.
    """
    mean = float(values.mean())
    sums = pd.Series(values - mean).groupby(clusters).sum()
    g = len(sums)
    if g < 2:
        return mean, float("nan")
    return mean, float(np.sqrt(g / (g - 1) * (sums**2).sum()) / len(values))


def cell_means(
    frame: pd.DataFrame,
    value: str = "ideology",
    by: Sequence[str] = CELL,
    cluster: str = "member_key",
) -> pd.DataFrame:
    """Mean of ``value`` per group, with a member-clustered SE and 95% interval."""
    rows = []
    for key, group in frame.dropna(subset=[value]).groupby(list(by)):
        mean, se = _clustered_mean(
            group[value].to_numpy(dtype=float), group[cluster].to_numpy()
        )
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                "mean": mean,
                "se": se,
                "ci_low": mean - Z95 * se,
                "ci_high": mean + Z95 * se,
                "n": len(group),
                "n_members": group[cluster].nunique(),
            }
        )
    return pd.DataFrame(rows)


def party_gap(cells: pd.DataFrame) -> pd.DataFrame:
    """R - D per Congress and chamber, from `cell_means` output.

    The parties' speeches come from different members, so their errors are
    independent and add in quadrature.
    """
    wide = cells.pivot_table(
        index=["congress_number", "chamber"], columns="party", values=["mean", "se"]
    )
    gap = pd.DataFrame(
        {
            "gap": wide[("mean", "R")] - wide[("mean", "D")],
            "se": np.sqrt(wide[("se", "R")] ** 2 + wide[("se", "D")] ** 2),
        }
    ).reset_index()
    gap["ci_low"] = gap["gap"] - Z95 * gap["se"]
    gap["ci_high"] = gap["gap"] + Z95 * gap["se"]
    return gap


def trend(
    frame: pd.DataFrame,
    value: str = "ideology",
    congress_range: tuple[int, int] | None = None,
    by: Sequence[str] = ("chamber", "party"),
    cluster: str = "member_key",
) -> pd.DataFrame:
    """Linear trend of ``value`` per Congress, per group, clustered by member.

    OLS of the speech-level score on the Congress number. ``change_over_span``
    is the slope times the Congresses spanned -- the fitted change from first
    to last. A straight line is a summary, not a claim that change was linear.
    """
    data = frame.dropna(subset=[value])
    if congress_range is not None:
        low, high = congress_range
        data = data[data["congress_number"].between(low, high)]
    rows = []
    for key, group in data.groupby(list(by)):
        x = sm.add_constant(group["congress_number"].astype(float))
        fit = sm.OLS(group[value].astype(float), x).fit(
            cov_type="cluster",
            cov_kwds={"groups": pd.factorize(group[cluster])[0]},
        )
        first, last = group["congress_number"].min(), group["congress_number"].max()
        slope, se = fit.params["congress_number"], fit.bse["congress_number"]
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                "span": f"{first}-{last}",
                "slope": slope,
                "se": se,
                "ci_low": slope - Z95 * se,
                "ci_high": slope + Z95 * se,
                "p": fit.pvalues["congress_number"],
                "change_over_span": slope * (last - first),
                "n": len(group),
                "n_members": group[cluster].nunique(),
            }
        )
    return pd.DataFrame(rows)


# --- members and examples --------------------------------------------------


def member_means(
    frame: pd.DataFrame, value: str = "ideology", min_speeches: int = 5
) -> pd.DataFrame:
    """Each validated member's average score next to their Voteview scores.

    Only ``in_validation`` speeches, and only members with at least
    ``min_speeches`` of them: one speech is a noisy reading of a member (P5).
    ``nominate_dim1`` is constant over a career; ``nokken_poole_dim1`` varies by
    Congress, so it is averaged over the member's speeches. A party switcher
    appears once per party. Exploratory -- the unit of the validation is still
    open (O9).
    """
    data = frame[frame["in_validation"]]
    members = (
        data.groupby(["icpsr", "party"])
        .agg(
            bioname=("bioname", "first"),
            n_speeches=(value, "size"),
            **{value: (value, "mean")},
            nominate_dim1=("nominate_dim1", "first"),
            nokken_poole_dim1=("nokken_poole_dim1", "mean"),
        )
        .reset_index()
    )
    return members[members["n_speeches"] >= min_speeches].reset_index(drop=True)


def disagreements(
    frame: pd.DataFrame, models: Sequence[str], top: int = 10
) -> pd.DataFrame:
    """The speeches two models score furthest apart; ``gap`` = first - second."""
    first, second = models
    out = frame.assign(gap=frame[f"ideology_{first}"] - frame[f"ideology_{second}"])
    order = out["gap"].abs().sort_values(ascending=False).index
    return out.loc[order].head(top).reset_index(drop=True)


def speech_texts(
    speech_ids: Sequence[str], corpus: Path = CORPUS_PATH
) -> dict[str, str]:
    """The text of the given speeches, read without loading the whole corpus."""
    table = pq.read_table(
        corpus,
        columns=["speech_id", "text"],
        filters=[("speech_id", "in", list(speech_ids))],
    )
    return dict(
        zip(table.column("speech_id").to_pylist(), table.column("text").to_pylist())
    )
