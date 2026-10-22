"""Which DW-NOMINATE validation the thesis reports (docs/decisions.md O9, A5).

`validation.py` correlates each speech with its speaker's Voteview score. This
module compares the designs O9 left open, on a run already on disk (no API
calls):

- **Unit.** A speech; a member in one Congress (``member_congress``); or a
  member over their career (``career``). A member unit is the mean of its
  speeches, kept only with at least ``min_speeches`` of them.
- **Statistic.** Within-party Pearson and Spearman; the pooled r *within party
  x chamber x Congress cells*, which neither the party gap nor a party's swing
  in and out of opposition (P8) can produce; the benchmark regressed on party
  and the score; the overall r, for completeness only (P5).
- **Uncertainty.** Every interval is a percentile bootstrap over members,
  stratified by party: a member's speeches and Congresses are drawn together,
  so a frequent speaker is not counted as many independent readings (A1).
- **Reliability.** How well a unit's mean reproduces itself across random
  halves of its speeches. It does not use DW-NOMINATE, so it can explain the
  gap between speech- and member-level r without being tuned on it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

from src.analysis import CELL, relative_to_cell, substantive
from src.config import (
    RANDOM_SEED,
    SOURCE_BREAK_CONGRESS,
    VALIDATION_BOOTSTRAP,
    VALIDATION_SPLITS,
)

#: What one observation is, by the columns that identify it. A member is their
#: career-long ICPSR (``member_key``); Voteview gives a party switcher a second
#: ICPSR, and ``party`` keeps the rare remaining mix apart (D23).
UNITS: dict[str, tuple[str, ...]] = {
    "speech": ("speech_id",),
    "member_congress": ("member_key", "party", "chamber", "congress_number"),
    "career": ("member_key", "party"),
}

#: What a unit is compared within for the within-cell statistics. A speech or a
#: member-Congress sits in one party x chamber x Congress cell; a career spans
#: several, so its speeches are first taken relative to their cells and the
#: career is then compared within its party.
COMPARED_WITHIN: dict[str, tuple[str, ...]] = {
    "speech": CELL,
    "member_congress": CELL,
    "career": ("party",),
}

#: Which speeches count as procedural and are left out of the validation.
PROCEDURAL_RULES: tuple[str, ...] = ("none", "zero_score", "labels")

#: Fewer units than this, and a correlation is not computed.
MIN_UNITS = 3

#: Statistics of one design, in report order.
STATISTICS: tuple[str, ...] = (
    "within_D_pearson",
    "within_D_spearman",
    "within_R_pearson",
    "within_R_spearman",
    "within_cell_pearson",
    "within_cell_spearman",
    "within_cell_D",
    "within_cell_R",
    "within_cell_H",
    "within_cell_S",
    "coef_score",
    "partial_r2",
    "overall_pearson",
    "party_only_r",
)


# --- which speeches, which units ------------------------------------------------


def validation_speeches(
    frame: pd.DataFrame, rule: str, score: str = "ideology"
) -> pd.DataFrame:
    """The speeches a validation design may use.

    Only ``in_validation`` speeches with a score (D24). ``rule`` then leaves
    out procedural speech: ``none`` keeps everything; ``zero_score`` drops
    speeches every model of the run scored exactly 0.0, as the prompt asks for
    procedural speech; ``labels`` drops what the trend analysis drops (A3:
    labelled procedural or ceremonial, or unlabelled). This filters the
    validation only -- the trend analysis never loses a speech here.

    Raises:
        ValueError: an unknown rule, or a party other than D and R.
    """
    if rule not in PROCEDURAL_RULES:
        raise ValueError(f"unknown procedural rule {rule!r}; use {PROCEDURAL_RULES}")
    data = frame[frame["in_validation"] & frame[score].notna()]
    other = sorted(set(data["party"]) - {"D", "R"})
    if other:
        raise ValueError(f"validation expects parties D and R only, found {other}")
    if rule == "zero_score":
        data = data[~data["procedural"]]
    elif rule == "labels":
        data = substantive(data)
    return data


def _design_speeches(
    speeches: pd.DataFrame,
    unit: str,
    benchmark: str,
    score: str,
    min_speeches: int,
) -> pd.DataFrame:
    """The speeches of every unit with at least ``min_speeches`` of them.

    Adds ``_unit`` (a unit id, 0..), ``_score`` and ``_bench``, and both taken
    relative to the speech's party x chamber x Congress cell (``_score_rel``,
    ``_bench_rel``) over the speeches kept.
    """
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; use {tuple(UNITS)}")
    keys = list(UNITS[unit])
    data = speeches.dropna(subset=[score, benchmark])
    size = data.groupby(keys)[score].transform("size")
    data = data[size >= min_speeches].copy()
    data["_unit"] = data.groupby(keys, sort=True).ngroup()
    data["_score"] = data[score].astype(float)
    data["_bench"] = data[benchmark].astype(float)
    data["_score_rel"] = relative_to_cell(data, "_score")
    data["_bench_rel"] = relative_to_cell(data, "_bench")
    return data


def unit_table(
    speeches: pd.DataFrame,
    unit: str,
    benchmark: str,
    score: str = "ideology",
    min_speeches: int = 1,
) -> pd.DataFrame:
    """One row per unit with at least ``min_speeches`` speeches.

    ``score`` and ``bench`` are the means over the unit's speeches (a
    member-Congress has one Nokken-Poole value; a career averages it over its
    speeches); ``score_rel`` and ``bench_rel`` the same after each speech is
    taken relative to its party x chamber x Congress cell, which for a career
    removes when and where the member served. ``cluster`` is what the
    bootstrap draws (member and party); ``group`` the units a unit is compared
    within (`COMPARED_WITHIN`). Speeches without a benchmark value are left out.

    Raises:
        ValueError: an unknown unit.
    """
    data = _design_speeches(speeches, unit, benchmark, score, min_speeches)
    agg: dict[str, tuple[str, str]] = {
        "member_key": ("member_key", "first"),
        "icpsr": ("icpsr", "first"),
        "party": ("party", "first"),
        "n": ("_score", "size"),
        "score": ("_score", "mean"),
        "bench": ("_bench", "mean"),
        "score_rel": ("_score_rel", "mean"),
        "bench_rel": ("_bench_rel", "mean"),
    }
    if unit == "career":
        agg["congresses"] = ("congress_number", "nunique")
    else:
        agg["chamber"] = ("chamber", "first")
        agg["congress_number"] = ("congress_number", "first")
    units = data.groupby("_unit").agg(**agg).reset_index(drop=True)
    units["cluster"] = units["member_key"] + "|" + units["party"]
    units["group"] = units.groupby(list(COMPARED_WITHIN[unit])).ngroup()
    return units


# --- statistics -----------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r; NaN below MIN_UNITS or without variation."""
    if len(x) < MIN_UNITS:
        return float("nan")
    dx, dy = x - x.mean(), y - y.mean()
    denominator = np.sqrt((dx**2).sum() * (dy**2).sum())
    return float((dx * dy).sum() / denominator) if denominator > 0 else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho: Pearson r of the ranks, ties averaged."""
    if len(x) < MIN_UNITS:
        return float("nan")
    return _pearson(rankdata(x), rankdata(y))


def _demean(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``values`` minus their group's mean, and each value's group size."""
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=values, minlength=len(counts))
    means = sums / np.maximum(counts, 1)
    return values - means[groups], counts[groups]


def _sse(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Residual sum of squares of the OLS fit of ``y`` on ``x``, and its coefficients."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(((y - x @ beta) ** 2).sum()), beta


def _arrays(units: pd.DataFrame) -> dict[str, np.ndarray]:
    """The unit columns the statistics need, as arrays. ``is_s`` only where a
    unit has one chamber (not a career)."""
    arrays = {
        "is_r": (units["party"] == "R").to_numpy(),
        "group": units["group"].to_numpy(dtype=np.int64),
        "score": units["score"].to_numpy(dtype=float),
        "bench": units["bench"].to_numpy(dtype=float),
        "score_rel": units["score_rel"].to_numpy(dtype=float),
        "bench_rel": units["bench_rel"].to_numpy(dtype=float),
    }
    if "chamber" in units:
        arrays["is_s"] = (units["chamber"] == "S").to_numpy()
    return arrays


def _statistics(a: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Every statistic of `STATISTICS` on one set of units (see `design_statistics`)."""
    is_r, score, bench = a["is_r"], a["score"], a["bench"]
    out: dict[str, float] = {}
    for party, mask in (("D", ~is_r), ("R", is_r)):
        out[f"within_{party}_pearson"] = _pearson(score[mask], bench[mask])
        out[f"within_{party}_spearman"] = _spearman(score[mask], bench[mask])

    score_w, size = _demean(a["score_rel"], a["group"])
    bench_w, _ = _demean(a["bench_rel"], a["group"])
    kept = size >= 2  # a unit alone in its cell carries no within-cell contrast
    out["within_cell_pearson"] = _pearson(score_w[kept], bench_w[kept])
    out["within_cell_spearman"] = _spearman(score_w[kept], bench_w[kept])
    for party, mask in (("D", kept & ~is_r), ("R", kept & is_r)):
        out[f"within_cell_{party}"] = _pearson(score_w[mask], bench_w[mask])
    is_s = a.get("is_s")
    for chamber, mask in (("H", ~is_s), ("S", is_s)) if is_s is not None else ():
        out[f"within_cell_{chamber}"] = _pearson(
            score_w[kept & mask], bench_w[kept & mask]
        )
    for chamber in ("H", "S"):
        out.setdefault(f"within_cell_{chamber}", float("nan"))

    if len(score) >= MIN_UNITS and is_r.any() and (~is_r).any():
        ones = np.ones(len(score))
        sse_party, _ = _sse(np.column_stack([ones, is_r]), bench)
        sse_full, beta = _sse(np.column_stack([ones, is_r, score]), bench)
        out["coef_score"] = float(beta[2])
        out["partial_r2"] = 1.0 - sse_full / sse_party if sse_party > 0 else np.nan
    else:
        out["coef_score"] = out["partial_r2"] = float("nan")
    out["overall_pearson"] = _pearson(score, bench)
    out["party_only_r"] = _pearson(is_r.astype(float), bench)
    return out


def design_statistics(units: pd.DataFrame) -> dict[str, float]:
    """The statistics of one design, from `unit_table`.

    - ``within_{D,R}_{pearson,spearman}``: score against benchmark within a
      party, over all its units -- so over Congresses too, where a party's
      shared swing in and out of opposition can enter.
    - ``within_cell_{pearson,spearman}``: both parties pooled after each unit's
      relative score and benchmark are taken relative to the units it is
      compared within (``group``); ``within_cell_{D,R}`` the same per party.
      Only how members are ordered *within* their party, chamber and Congress
      is left. A unit alone in its cell is left out. ``within_cell_{H,S}``
      per chamber, where a unit has one: the sample's equal quota per chamber
      gives a senator about four times a representative's speeches, so a
      threshold keeps senators first.
    - ``coef_score`` and ``partial_r2``: OLS of the benchmark on a Republican
      dummy and the score; the score's coefficient, and the share of the
      variance party leaves that the score explains.
    - ``overall_pearson``: both parties together, for completeness (P5);
      ``party_only_r``: what party alone gives, for comparison.
    """
    return _statistics(_arrays(units))


def benchmark_regression(units: pd.DataFrame) -> dict[str, float]:
    """``bench ~ R + score``: the score's coefficient with a member-clustered SE."""
    x = sm.add_constant(
        pd.DataFrame(
            {"R": (units["party"] == "R").astype(float), "score": units["score"]}
        )
    )
    fit = sm.OLS(units["bench"].astype(float), x).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(units["cluster"])[0]}
    )
    return {
        "coef_score": float(fit.params["score"]),
        "coef_score_se": float(fit.bse["score"]),
        "coef_score_p": float(fit.pvalues["score"]),
    }


# --- the bootstrap over members -------------------------------------------------


def cluster_draws(
    cluster: Sequence[object],
    stratum: Sequence[object],
    n_draws: int,
    seed: int = RANDOM_SEED,
) -> Iterator[np.ndarray]:
    """Row indices of ``n_draws`` bootstrap samples over clusters.

    In every stratum (party) as many clusters as it has are drawn with
    replacement, and each drawn cluster brings all its rows, so a member's
    speeches or Congresses are never split up. A cluster takes the stratum of
    its first row.
    """
    codes, _ = pd.factorize(pd.Series(cluster))
    order = np.argsort(codes, kind="stable")
    rows_of = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    stratum = np.asarray(stratum)
    stratum_of = stratum[[rows[0] for rows in rows_of]]
    strata = [np.flatnonzero(stratum_of == s) for s in pd.unique(stratum_of)]
    rng = np.random.default_rng(seed)
    for _ in range(n_draws):
        picked = np.concatenate([rng.choice(ids, size=len(ids)) for ids in strata])
        yield np.concatenate([rows_of[i] for i in picked])


def bootstrap(
    arrays: Mapping[str, np.ndarray],
    statistic: Callable[[Mapping[str, np.ndarray]], dict[str, float]],
    cluster: Sequence[object],
    stratum: Sequence[object],
    n_draws: int = VALIDATION_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """``statistic`` on each draw of `cluster_draws`; one row per draw."""
    rows = []
    for idx in cluster_draws(cluster, stratum, n_draws, seed):
        rows.append(statistic({name: values[idx] for name, values in arrays.items()}))
    return pd.DataFrame(rows)


def with_intervals(
    point: Mapping[str, float], draws: pd.DataFrame, level: float = 0.95
) -> dict[str, float]:
    """Each statistic with its percentile interval: ``name``, ``name_ci_low``, ``name_ci_high``."""
    tail = (1 - level) / 2 * 100
    out: dict[str, float] = {}
    for name, value in point.items():
        out[name] = value
        column = draws[name].dropna() if name in draws else pd.Series(dtype=float)
        low, high = (
            np.percentile(column, [tail, 100 - tail])
            if len(column)
            else (np.nan, np.nan)
        )
        out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(low), float(high)
    return out


# --- reliability ----------------------------------------------------------------


def spearman_brown(r: float) -> float:
    """Reliability of the whole from the correlation of its two halves."""
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")


def split_half_reliability(
    speeches: pd.DataFrame,
    unit: str,
    benchmark: str,
    score: str = "ideology",
    min_speeches: int = 2,
    n_splits: int = VALIDATION_SPLITS,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """How well a member unit's mean score reproduces itself.

    Each unit's speeches are split at random into two halves (sizes n // 2 and
    the rest), the halves' means are correlated across units, and the
    correlation, averaged over ``n_splits`` splits, is stepped up to the full
    unit with Spearman-Brown. ``D`` and ``R``: the unit means within each
    party; ``cell``: both parties, within the units' cells, as
    ``within_cell_pearson`` reads them. Uses the same units as `unit_table`
    (``benchmark`` only decides which speeches have a value); the benchmark
    itself is never looked at.

    Raises:
        ValueError: ``unit`` is ``speech`` -- see `speech_icc`.
    """
    if unit == "speech":
        raise ValueError("a single speech has no halves; use speech_icc")
    data = _design_speeches(speeches, unit, benchmark, score, max(min_speeches, 2))
    unit_id = data["_unit"].to_numpy()
    n_per_unit = np.bincount(unit_id)
    n_units = len(n_per_unit)
    first_rows = data.groupby("_unit").head(1).sort_values("_unit")
    is_r = (first_rows["party"] == "R").to_numpy()
    groups = first_rows.groupby(list(COMPARED_WITHIN[unit])).ngroup().to_numpy()
    values = data["_score"].to_numpy(dtype=float)
    relative = data["_score_rel"].to_numpy(dtype=float)
    starts = np.cumsum(n_per_unit) - n_per_unit

    rng = np.random.default_rng(seed)
    found: dict[str, list[float]] = {"D": [], "R": [], "cell": []}
    for _ in range(n_splits):
        order = np.lexsort((rng.random(len(unit_id)), unit_id))
        position = np.empty(len(unit_id), dtype=np.int64)
        position[order] = np.arange(len(unit_id)) - starts[unit_id[order]]
        half = 2 * unit_id + (position < n_per_unit[unit_id] // 2)
        counts = np.bincount(half, minlength=2 * n_units)
        a, b = _halves(half, values, counts, n_units)
        found["D"].append(_pearson(a[~is_r], b[~is_r]))
        found["R"].append(_pearson(a[is_r], b[is_r]))
        ar, br = _halves(half, relative, counts, n_units)
        ar_w, size = _demean(ar, groups)
        br_w, _ = _demean(br, groups)
        found["cell"].append(_pearson(ar_w[size >= 2], br_w[size >= 2]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN: too few units
        return {key: spearman_brown(float(np.nanmean(rs))) for key, rs in found.items()}


def _halves(
    half: np.ndarray, values: np.ndarray, counts: np.ndarray, n_units: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each unit's mean in its two halves."""
    means = np.bincount(half, weights=values, minlength=2 * n_units) / counts
    return means[0::2], means[1::2]


def icc_oneway(values: np.ndarray, groups: Sequence[object]) -> float:
    """ICC(1): the share of a single reading's variance that lies between groups.

    One-way random effects, for unequal group sizes (n0 in place of n).
    """
    codes, _ = pd.factorize(pd.Series(groups))
    values = np.asarray(values, dtype=float)
    n_groups, n_total = codes.max() + 1, len(values)
    sizes = np.bincount(codes)
    means = np.bincount(codes, weights=values) / sizes
    between = (sizes * (means - values.mean()) ** 2).sum() / (n_groups - 1)
    within = ((values - means[codes]) ** 2).sum() / (n_total - n_groups)
    n0 = (n_total - (sizes**2).sum() / n_total) / (n_groups - 1)
    return float((between - within) / (between + (n0 - 1) * within))


def speech_icc(
    speeches: pd.DataFrame, benchmark: str, score: str = "ideology"
) -> dict[str, float]:
    """How reliably one speech reads its speaker's position in that Congress.

    ICC(1) of speeches grouped by member x Congress: ``D`` and ``R`` on the
    score within each party, ``cell`` on the score relative to its cell. A
    speech-level r is about r_true x sqrt(ICC); a member unit's mean has the
    Spearman-Brown reliability of its n speeches. Member-Congresses with one
    speech carry no within-member information and are left out.
    """
    data = _design_speeches(speeches, "member_congress", benchmark, score, 2)
    out = {}
    for party in ("D", "R"):
        rows = data[data["party"] == party]
        out[party] = icc_oneway(rows["_score"].to_numpy(), rows["_unit"].to_numpy())
    out["cell"] = icc_oneway(data["_score_rel"].to_numpy(), data["_unit"].to_numpy())
    return out


# --- one design, the grid -------------------------------------------------------


def design_row(
    speeches: pd.DataFrame,
    unit: str,
    benchmark: str,
    min_speeches: int = 1,
    score: str = "ideology",
    n_draws: int = VALIDATION_BOOTSTRAP,
    n_splits: int = VALIDATION_SPLITS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Counts, statistics with member-bootstrap intervals, and reliability.

    ``rel_{D,R,cell}`` is the split-half reliability of a member unit, or the
    ICC of a single speech; ``disatt_{D,R,cell}`` the matching within-party or
    within-cell r divided by its square root -- corrected for the LLM side
    only, as if the benchmark were measured without error.
    """
    units = unit_table(speeches, unit, benchmark, score, min_speeches)
    row: dict[str, object] = {
        "unit": unit,
        "min_speeches": min_speeches,
        "benchmark": benchmark,
        "score": score,
        "n_speeches": int(units["n"].sum()),
    }
    for party in ("D", "R"):
        rows = units[units["party"] == party]
        row[f"n_units_{party}"] = len(rows)
        row[f"n_members_{party}"] = int(rows["cluster"].nunique())
    arrays = _arrays(units)
    draws = bootstrap(
        arrays, _statistics, units["cluster"], units["party"], n_draws, seed
    )
    row.update(with_intervals(_statistics(arrays), draws))
    reg = benchmark_regression(units)
    row["coef_score_se"], row["coef_score_p"] = (
        reg["coef_score_se"],
        reg["coef_score_p"],
    )

    if unit == "speech":
        reliability = speech_icc(speeches, benchmark, score)
    else:
        reliability = split_half_reliability(
            speeches, unit, benchmark, score, min_speeches, n_splits, seed
        )
    observed = {
        "D": row["within_D_pearson"],
        "R": row["within_R_pearson"],
        "cell": row["within_cell_pearson"],
    }
    for key, rel in reliability.items():
        row[f"rel_{key}"] = rel
        row[f"disatt_{key}"] = observed[key] / np.sqrt(rel) if rel > 0 else np.nan
    return row


def validation_grid(
    frame: pd.DataFrame,
    benchmarks: Sequence[str],
    thresholds: Sequence[int],
    rules: Sequence[str] = PROCEDURAL_RULES,
    score: str = "ideology",
    n_draws: int = VALIDATION_BOOTSTRAP,
    n_splits: int = VALIDATION_SPLITS,
    seed: int = RANDOM_SEED,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """`design_row` for every procedural rule, benchmark, unit and threshold.

    Speech level once per rule and benchmark; each member unit at every
    threshold. ``progress`` is called with each finished row.
    """
    rows = []
    for rule in rules:
        speeches = validation_speeches(frame, rule, score)
        for benchmark in benchmarks:
            designs = [("speech", 1)] + [
                (unit, k) for unit in ("member_congress", "career") for k in thresholds
            ]
            for unit, k in designs:
                row = {
                    "procedural": rule,
                    **design_row(
                        speeches, unit, benchmark, k, score, n_draws, n_splits, seed
                    ),
                }
                rows.append(row)
                if progress is not None:
                    progress(row)
    return pd.DataFrame(rows)


# --- how the procedural definitions differ --------------------------------------


def procedural_overlap(frame: pd.DataFrame) -> dict[str, object]:
    """How the two procedural rules differ on the validation speeches.

    Counts dropped by each rule and by both, and the speech types the
    0.0-score rule drops (and keeps).
    """
    data = frame[frame["in_validation"]]
    zero = data["procedural"].to_numpy()
    kept_by_labels = data.index.isin(substantive(data).index)
    by_labels = ~kept_by_labels
    types = data["speech_type"].fillna("unlabelled")
    return {
        "speeches": len(data),
        "dropped_by_zero_score": int(zero.sum()),
        "dropped_by_labels": int(by_labels.sum()),
        "dropped_by_both": int((zero & by_labels).sum()),
        "dropped_by_zero_score_only": int((zero & ~by_labels).sum()),
        "dropped_by_labels_only": int((~zero & by_labels).sum()),
        "zero_score_by_type": types[zero].value_counts().to_dict(),
        "type_counts": types.value_counts().to_dict(),
    }


# --- who the frequent speakers are ----------------------------------------------

#: The eras `_era` names, in order.
ERAS: tuple[str, ...] = ("stanford", "mixed_114", "govinfo")


def _era(congress: pd.Series) -> pd.Series:
    """Stanford before the 114th, mixed in it, govinfo after (D15)."""
    return pd.Series(
        np.select(
            [congress < SOURCE_BREAK_CONGRESS, congress == SOURCE_BREAK_CONGRESS],
            ["stanford", "mixed_114"],
            "govinfo",
        ),
        index=congress.index,
    )


def _toward_party(bench: pd.Series, party: pd.Series) -> pd.Series:
    """A benchmark in the party's own direction: R as is, D sign-flipped."""
    return bench * np.where(party == "R", 1.0, -1.0)


def _voteview_members(population: pd.DataFrame) -> pd.DataFrame:
    """Voteview's D and R members with the corpus's codes: party D/R, chamber
    H/S, ``congress_number``. Other party codes (independents) are left out."""
    return population[population["party_code"].isin([100, 200])].assign(
        party=lambda f: np.where(f["party_code"] == 100, "D", "R"),
        chamber=lambda f: np.where(f["chamber"] == "Senate", "S", "H"),
        congress_number=lambda f: f["congress"],
        icpsr=lambda f: f["icpsr"].astype("Int64"),
    )


def selection_by_congress(
    speeches: pd.DataFrame,
    thresholds: Sequence[int],
    population: pd.DataFrame,
    benchmark: str = "nokken_poole_dim1",
    score: str = "ideology",
) -> pd.DataFrame:
    """Member x Congress units kept at each threshold, per Congress, against Voteview.

    One row per threshold (1 = every sampled member-Congress), Congress,
    chamber and party: ``units`` kept, ``members`` Voteview scored in that
    cell, ``coverage`` (units / members), and the benchmark's mean toward the
    party's side for the kept units and for all members. Cells with no kept
    unit get ``units`` 0. Where `selection_profile` pools Congresses into
    eras, this shows whether a threshold empties particular Congresses.
    """
    members = _voteview_members(population).dropna(subset=[benchmark])
    members = members.assign(toward=_toward_party(members[benchmark], members["party"]))
    cell = ["congress_number", "chamber", "party"]
    everyone = (
        members.groupby(cell)
        .agg(members=("icpsr", "nunique"), toward_all=("toward", "mean"))
        .reset_index()
    )
    rows = []
    for k in (1, *thresholds):
        units = unit_table(speeches, "member_congress", benchmark, score, k)
        units = units.assign(toward=_toward_party(units["bench"], units["party"]))
        kept = (
            units.groupby(cell)
            .agg(units=("cluster", "size"), toward_kept=("toward", "mean"))
            .reset_index()
        )
        table = everyone.merge(kept, on=cell, how="left").assign(threshold=k)
        table["units"] = table["units"].fillna(0).astype(int)
        table["coverage"] = table["units"] / table["members"]
        rows.append(table)
    columns = ["threshold", *cell, "units", "members", "coverage"]
    return pd.concat(rows, ignore_index=True)[[*columns, "toward_kept", "toward_all"]]


def selection_profile(
    speeches: pd.DataFrame,
    thresholds: Sequence[int],
    background: pd.DataFrame,
    population: pd.DataFrame,
    score: str = "ideology",
) -> pd.DataFrame:
    """Who the member units are at each threshold, against everyone.

    One row per unit (``member_congress`` against Nokken-Poole, ``career``
    against ``nominate_dim1``), threshold and party; threshold 1 is every
    sampled member unit, and ``voteview`` every member Voteview scored in the
    Congresses ``population`` covers (from `crosswalk.load_voteview`). Columns:
    units, members, the share of the party's speeches they hold, the Senate's
    share of units, the eras' shares (member x Congress), mean tenure and the
    first-Congress share (from `analysis.member_background`), and the
    benchmark's mean toward the party's side and its SD -- a narrower SD
    lowers any r (range restriction).
    """
    population = _voteview_members(population)
    tenure = background[["icpsr", "congress_number", "tenure"]].assign(
        icpsr=lambda f: f["icpsr"].astype("Int64")
    )
    rows = []
    for unit, benchmark in (
        ("member_congress", "nokken_poole_dim1"),
        ("career", "nominate_dim1"),
    ):
        total = speeches.dropna(subset=[score, benchmark]).groupby("party").size()
        for k in (1, *thresholds):
            units = unit_table(speeches, unit, benchmark, score, k)
            if unit == "member_congress":
                units = units.assign(icpsr=units["icpsr"].astype("Int64")).merge(
                    tenure, on=["icpsr", "congress_number"], how="left"
                )
            for party, group in units.groupby("party"):
                row = {
                    "unit": unit,
                    "threshold": k,
                    "party": party,
                    "units": len(group),
                    "members": group["cluster"].nunique(),
                    "speech_share": group["n"].sum() / total[party],
                    "bench_toward_party": _toward_party(
                        group["bench"], group["party"]
                    ).mean(),
                    "bench_sd": group["bench"].std(),
                }
                if unit == "member_congress":
                    era = _era(group["congress_number"]).value_counts(normalize=True)
                    row |= {
                        "senate_share": (group["chamber"] == "S").mean(),
                        **{f"era_{e}": era.get(e, 0.0) for e in ERAS},
                        "tenure_mean": group["tenure"].mean(),
                        "first_congress_share": (group["tenure"] == 1).mean(),
                    }
                rows.append(row)

        if unit == "member_congress":
            people = population.merge(
                tenure, on=["icpsr", "congress_number"], how="left", validate="m:1"
            )
        else:
            people = population.drop_duplicates(["icpsr", "party"])
        for party, group in people.dropna(subset=[benchmark]).groupby("party"):
            row = {
                "unit": unit,
                "threshold": "voteview",
                "party": party,
                "units": len(group),
                "members": group["icpsr"].nunique(),
                "bench_toward_party": _toward_party(
                    group[benchmark], group["party"]
                ).mean(),
                "bench_sd": group[benchmark].std(),
            }
            if unit == "member_congress":
                era = _era(group["congress_number"]).value_counts(normalize=True)
                row |= {
                    "senate_share": (group["chamber"] == "S").mean(),
                    **{f"era_{e}": era.get(e, 0.0) for e in ERAS},
                    "tenure_mean": group["tenure"].mean(),
                    "first_congress_share": (group["tenure"] == 1).mean(),
                }
            rows.append(row)
    return pd.DataFrame(rows)


# --- does a member's movement track their votes' (RQ2 diagnostic) ---------------


def within_cell(units: pd.DataFrame) -> pd.DataFrame:
    """Add ``score_within`` and ``bench_within``: ``score_rel`` and ``bench_rel``
    minus their mean over the units in the same ``group``."""
    groups = units["group"].to_numpy(dtype=np.int64)
    score, _ = _demean(units["score_rel"].to_numpy(dtype=float), groups)
    bench, _ = _demean(units["bench_rel"].to_numpy(dtype=float), groups)
    return units.assign(score_within=score, bench_within=bench)


def congress_changes(units: pd.DataFrame) -> pd.DataFrame:
    """Member x Congress units -> one row per member, chamber and pair of
    consecutive Congresses in which both units were kept.

    ``d_score`` and ``d_bench``: the change in the mean score and in
    Nokken-Poole; ``d_score_within`` and ``d_bench_within``: the change in the
    member's position relative to their party, chamber and Congress, so a
    party's swing in and out of opposition drops out.
    """
    data = within_cell(units).sort_values(["cluster", "chamber", "congress_number"])
    columns = ["congress_number", "score", "bench", "score_within", "bench_within"]
    previous = data.groupby(["cluster", "chamber"])[columns].shift(1)
    consecutive = data["congress_number"] - previous["congress_number"] == 1
    data, previous = data[consecutive], previous[consecutive]
    return pd.DataFrame(
        {
            "cluster": data["cluster"],
            "party": data["party"],
            "chamber": data["chamber"],
            "congress_number": data["congress_number"],
            "d_score": data["score"] - previous["score"],
            "d_bench": data["bench"] - previous["bench"],
            "d_score_within": data["score_within"] - previous["score_within"],
            "d_bench_within": data["bench_within"] - previous["bench_within"],
        }
    ).reset_index(drop=True)


def _change_statistics(a: Mapping[str, np.ndarray]) -> dict[str, float]:
    is_r = a["is_r"]
    out = {}
    for party, mask in (("D", ~is_r), ("R", is_r)):
        out[f"within_{party}_pearson"] = _pearson(
            a["d_score"][mask], a["d_bench"][mask]
        )
    out["within_cell_pearson"] = _pearson(a["d_score_within"], a["d_bench_within"])
    out["within_cell_spearman"] = _spearman(a["d_score_within"], a["d_bench_within"])
    return out


def change_statistics(
    changes: pd.DataFrame,
    n_draws: int = VALIDATION_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Does the change in a member's score track the change in their votes?

    Pearson r of ``d_score`` with ``d_bench`` within each party, and of the
    within-cell changes pooled, with member-bootstrap intervals. A diagnostic
    for RQ2, not a validation of levels.
    """
    arrays = {
        "is_r": (changes["party"] == "R").to_numpy(),
        **{
            name: changes[name].to_numpy(dtype=float)
            for name in ("d_score", "d_bench", "d_score_within", "d_bench_within")
        },
    }
    draws = bootstrap(
        arrays, _change_statistics, changes["cluster"], changes["party"], n_draws, seed
    )
    out = with_intervals(_change_statistics(arrays), draws)
    out["n_pairs_D"] = int((changes["party"] == "D").sum())
    out["n_pairs_R"] = int((changes["party"] == "R").sum())
    out["n_members"] = int(changes["cluster"].nunique())
    return out
