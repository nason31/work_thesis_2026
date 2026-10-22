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
from itertools import groupby, pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from src.config import (
    AGE_BINS,
    CORPUS_PATH,
    CROSSWALK_PATH,
    DW_NOMINATE_MEMBERS,
    EXCLUDED_SPEECH_TYPES,
    GENERATIONS,
    LABELS_DIR,
    PRESIDENTS_BY_CONGRESS,
    SCORES_DIR,
    TENURE_BINS,
)
from src.crosswalk import UNIT_KEY
from src.labeling import LABEL_COLUMNS, load_labels
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


# --- speech labels -----------------------------------------------------------


def attach_labels(
    frame: pd.DataFrame, run: str, labels_dir: Path = LABELS_DIR
) -> pd.DataFrame:
    """Add ``speech_type``, ``topic`` and ``topic_block`` from label run ``run``.

    A left join on speech_id: a speech the run did not label, or never labelled
    validly, gets nulls rather than being dropped. ``topic`` and
    ``topic_block`` are null for every type but policy debate (L2).

    Raises:
        ValueError: ``frame`` repeats a speech_id.
    """
    labels = load_labels(run, labels_dir)[["speech_id", *LABEL_COLUMNS]]
    return frame.merge(labels, on="speech_id", how="left", validate="one_to_one")


def substantive(
    frame: pd.DataFrame, exclude: Sequence[str] = EXCLUDED_SPEECH_TYPES
) -> pd.DataFrame:
    """Speeches whose label is not in ``exclude`` (default procedural, ceremonial).

    Unlabelled speeches are left out too: whether they are substantive is
    unknown. Needs `attach_labels` first.
    """
    keep = frame["speech_type"].notna() & ~frame["speech_type"].isin(list(exclude))
    return frame[keep]


# --- member background: age, generation, tenure ----------------------------


def _bin_labels(bins: Sequence[int]) -> list[str]:
    """Labels for left-closed bins: (45, 55) -> 'under 45', '45–54', '55+'."""
    labels = [f"under {bins[0]}"]
    labels += [f"{low}–{high - 1}" for low, high in pairwise(bins)]
    return labels + [f"{bins[-1]}+"]


#: Group labels in display order.
AGE_GROUPS: tuple[str, ...] = tuple(_bin_labels(AGE_BINS))
GENERATION_ORDER: tuple[str, ...] = tuple(GENERATIONS)
TENURE_GROUPS: tuple[str, ...] = (
    "1st Congress",
    *(f"{low}–{high - 1}" for low, high in pairwise(TENURE_BINS)),
    f"{TENURE_BINS[-1]}+",
)


def _cut(values: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    """Left-closed binning into plain strings (missing stays missing)."""
    edges = [-np.inf, *bins, np.inf]
    return pd.cut(values, edges, right=False, labels=list(labels)).astype("string")


def member_background(path: Path = DW_NOMINATE_MEMBERS) -> pd.DataFrame:
    """Birth year, first Congress and tenure per member and Congress.

    From the raw Voteview member file, over every Congress since 1789, so
    service before 2001 counts. One row per ICPSR and Congress (House and
    Senate); ``tenure`` counts the distinct Congresses served up to and
    including this one, in either chamber. Voteview gives a party switcher a
    new ICPSR, so their tenure restarts (D23).
    """
    voteview = pd.read_csv(path, usecols=["congress", "chamber", "icpsr", "born"])
    voteview = voteview[voteview["chamber"].isin(["House", "Senate"])]
    per = (
        voteview.groupby(["icpsr", "congress"], as_index=False)["born"]
        .first()
        .sort_values(["icpsr", "congress"])
    )
    per["tenure"] = per.groupby("icpsr").cumcount() + 1
    per["first_congress"] = per.groupby("icpsr")["congress"].transform("min")
    per["icpsr"] = per["icpsr"].astype("Int64")
    return per.rename(columns={"congress": "congress_number"})


def add_background(frame: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Add each speaker's ``born``, ``age``, ``tenure`` and their groups.

    ``age`` is the speech's year minus the birth year (a year only, so +-1);
    ``age_group``, ``generation`` and ``tenure_group`` bin them as config.py
    defines. Speakers the crosswalk did not match (no ICPSR) get nulls.
    """
    out = frame.assign(icpsr=frame["icpsr"].astype("Int64")).merge(
        background, on=["icpsr", "congress_number"], how="left", validate="m:1"
    )
    out["age"] = pd.to_datetime(out["date"]).dt.year - out["born"]
    out["age_group"] = _cut(out["age"], AGE_BINS, AGE_GROUPS)
    edges = [high + 0.5 for _, high in list(GENERATIONS.values())[:-1]]
    out["generation"] = _cut(out["born"], edges, GENERATION_ORDER)
    out["tenure_group"] = _cut(out["tenure"], TENURE_BINS, TENURE_GROUPS)
    return out


def toward_party(frame: pd.DataFrame, value: str = "ideology") -> pd.Series:
    """The score in the party's own direction: R as is, D sign-flipped.

    Higher means further from the centre toward the party's side -- the
    member's extremity, comparable across the two parties.
    """
    return frame[value] * np.where(frame["party"] == "R", 1.0, -1.0)


def relative_to_cell(
    frame: pd.DataFrame, value: str, cell: Sequence[str] = CELL
) -> pd.Series:
    """``value`` minus its mean in the speech's cell (default Congress, chamber,
    party): how a speech compares with its own party, chamber and Congress."""
    return frame[value] - frame.groupby(list(cell))[value].transform("mean")


def within_cell_ols(
    frame: pd.DataFrame,
    value: str,
    terms: Sequence[str],
    cell: Sequence[str] = CELL,
    by: Sequence[str] = ("chamber",),
    cluster: str = "member_key",
) -> pd.DataFrame:
    """OLS of ``value`` on ``terms`` within cells, per group of ``by``.

    Every variable is demeaned within its cell (Congress, chamber and party by
    default), so each coefficient compares speeches of the same party, chamber
    and Congress -- cell fixed effects. Standard errors are clustered by
    member; they ignore the degrees of freedom the cell means use up, which is
    negligible with thousands of speeches per group.
    """
    data = frame.dropna(subset=[value, *terms])
    rows = []
    for key, group in data.groupby(list(by)):
        means = group.groupby(list(cell))[[value, *terms]].transform("mean")
        y = group[value] - means[value]
        x = group[list(terms)] - means[list(terms)]
        fit = sm.OLS(y.astype(float), x.astype(float)).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(group[cluster])[0]}
        )
        for term in terms:
            coef, se = fit.params[term], fit.bse[term]
            rows.append(
                {
                    **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                    "term": term,
                    "coef": coef,
                    "se": se,
                    "ci_low": coef - Z95 * se,
                    "ci_high": coef + Z95 * se,
                    "p": fit.pvalues[term],
                    "n": len(group),
                    "n_members": group[cluster].nunique(),
                }
            )
    return pd.DataFrame(rows)


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


def party_gap(
    cells: pd.DataFrame, by: Sequence[str] = ("congress_number", "chamber")
) -> pd.DataFrame:
    """R - D per group of ``by`` (default: Congress and chamber), from `cell_means`.

    The parties' speeches come from different members, so their errors are
    independent and add in quadrature.
    """
    wide = cells.pivot_table(index=list(by), columns="party", values=["mean", "se"])
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


def gap_trend(
    frame: pd.DataFrame,
    value: str = "ideology",
    by: Sequence[str] = ("chamber",),
    congress_range: tuple[int, int] | None = None,
    cluster: str = "member_key",
) -> pd.DataFrame:
    """How fast the party gap changes per Congress, per group, clustered by member.

    OLS of the speech-level score on party, Congress and their interaction:
    ``value ~ R + congress + R x congress``. The interaction is the change in
    the R - D gap per Congress; ``change_over_span`` is it times the
    Congresses spanned. ``gap_first`` is the fitted gap in the first Congress.
    A straight line is a summary, not a claim that change was linear.
    """
    data = frame.dropna(subset=[value])
    if congress_range is not None:
        low, high = congress_range
        data = data[data["congress_number"].between(low, high)]
    rows = []
    for key, group in data.groupby(list(by)):
        if group["party"].nunique() < 2:
            continue
        first, last = group["congress_number"].min(), group["congress_number"].max()
        republican = (group["party"] == "R").astype(float)
        congress = group["congress_number"].astype(float) - first
        x = sm.add_constant(
            pd.DataFrame(
                {
                    "R": republican,
                    "congress": congress,
                    "R_x_congress": republican * congress,
                }
            )
        )
        fit = sm.OLS(group[value].astype(float), x).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(group[cluster])[0]}
        )
        slope, se = fit.params["R_x_congress"], fit.bse["R_x_congress"]
        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                "span": f"{first}-{last}",
                "gap_first": fit.params["R"],
                "gap_slope": slope,
                "se": se,
                "ci_low": slope - Z95 * se,
                "ci_high": slope + Z95 * se,
                "p": fit.pvalues["R_x_congress"],
                "change_over_span": slope * (last - first),
                "n": len(group),
                "n_members": group[cluster].nunique(),
            }
        )
    return pd.DataFrame(rows)


# --- members and examples --------------------------------------------------

#: One member's record in a Congress, beyond the score.
_MEMBER_FIRST = ("bioname", "state", "chamber")


def member_trajectories(
    frame: pd.DataFrame, value: str = "ideology", min_speeches: int = 10
) -> pd.DataFrame:
    """Each member's mean per Congress, beside their party's mean that Congress.

    One row per member (career-long ICPSR), party and Congress with at least
    ``min_speeches`` speeches. ``se`` treats the member's speeches in that
    Congress as independent. ``party_mean`` is the mean of all speeches by the
    member's party in their chamber that Congress, and ``relative`` the
    member's distance from it. ``nokken_poole_dim1`` is their per-Congress
    DW-NOMINATE. Only members the crosswalk matched (``icpsr``) are included.
    """
    data = frame.dropna(subset=[value, "icpsr"])
    party = (
        data.groupby(["congress_number", "chamber", "party"])[value]
        .mean()
        .rename("party_mean")
    )
    grouped = data.groupby(["icpsr", "party", "congress_number"])
    out = grouped.agg(
        **{name: (name, "first") for name in _MEMBER_FIRST},
        n=(value, "size"),
        mean=(value, "mean"),
        sd=(value, "std"),
        nokken_poole_dim1=("nokken_poole_dim1", "mean"),
    ).reset_index()
    out = out[out["n"] >= min_speeches].copy()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["ci_low"] = out["mean"] - Z95 * out["se"]
    out["ci_high"] = out["mean"] + Z95 * out["se"]
    out = out.join(party, on=["congress_number", "chamber", "party"])
    out["relative"] = out["mean"] - out["party_mean"]
    out["icpsr"] = out["icpsr"].astype("Int64")
    return out.drop(columns="sd").reset_index(drop=True)


def member_trends(
    frame: pd.DataFrame,
    value: str = "ideology",
    min_speeches: int = 10,
    min_congresses: int = 5,
) -> pd.DataFrame:
    """Which members moved: each member's linear trend over the Congresses.

    Uses the member-Congresses `member_trajectories` keeps (at least
    ``min_speeches`` speeches) and members with at least ``min_congresses``
    of them. ``slope`` is the OLS slope of the member's speech-level score on
    the Congress number (HC1 standard errors); ``slope_relative`` the same for
    the distance from their party's mean, so a member who moved only as their
    party did has ~0; ``slope_nokken_poole`` the slope of their per-Congress
    DW-NOMINATE, the votes-based counterpart.
    """
    kept = member_trajectories(frame, value, min_speeches)
    counts = kept.groupby(["icpsr", "party"])["congress_number"].nunique()
    members = counts[counts >= min_congresses].index
    data = frame.dropna(subset=[value, "icpsr"]).assign(
        icpsr=lambda f: f["icpsr"].astype("Int64")
    )
    data = data.merge(
        kept[["icpsr", "party", "congress_number", "party_mean"]],
        on=["icpsr", "party", "congress_number"],
    )
    rows = []
    for (icpsr, party), group in data.groupby(["icpsr", "party"]):
        if (icpsr, party) not in members:
            continue
        x = sm.add_constant(group["congress_number"].astype(float))
        fit = sm.OLS(group[value].astype(float), x).fit(cov_type="HC1")
        relative = sm.OLS((group[value] - group["party_mean"]).astype(float), x).fit(
            cov_type="HC1"
        )
        per_congress = kept[(kept["icpsr"] == icpsr) & (kept["party"] == party)]
        np_scores = per_congress.dropna(subset=["nokken_poole_dim1"])
        np_slope = (
            np.polyfit(np_scores["congress_number"], np_scores["nokken_poole_dim1"], 1)[
                0
            ]
            if len(np_scores) >= 2
            else np.nan
        )
        slope, se = fit.params["congress_number"], fit.bse["congress_number"]
        first, last = group["congress_number"].min(), group["congress_number"].max()
        rows.append(
            {
                "icpsr": icpsr,
                "party": party,
                "bioname": group["bioname"].iloc[0],
                "state": group["state"].iloc[0],
                "chambers": "".join(sorted(group["chamber"].unique())),
                "span": f"{first}-{last}",
                "congresses": len(per_congress),
                "n": len(group),
                "mean": group[value].mean(),
                "slope": slope,
                "se": se,
                "ci_low": slope - Z95 * se,
                "ci_high": slope + Z95 * se,
                "p": fit.pvalues["congress_number"],
                "change_over_span": slope * (last - first),
                "slope_relative": relative.params["congress_number"],
                "p_relative": relative.pvalues["congress_number"],
                "slope_nokken_poole": np_slope,
            }
        )
    return pd.DataFrame(rows)


def member_means(
    frame: pd.DataFrame, value: str = "ideology", min_speeches: int = 5
) -> pd.DataFrame:
    """Each validated member's average score next to their Voteview scores.

    Only ``in_validation`` speeches, and only members with at least
    ``min_speeches`` of them: one speech is a noisy reading of a member (P5).
    ``nominate_dim1`` is constant over a career; ``nokken_poole_dim1`` varies by
    Congress, so it is averaged over the member's speeches. A party switcher
    appears once per party. Exploratory, a career-level view: the validation
    the thesis reports is member x Congress (A5, `validation_design`).
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
