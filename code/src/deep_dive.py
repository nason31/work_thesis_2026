"""The deep dive on polarization: chamber control, opposition within members,
topic blocks, and who moved the parties.

Feeds ``code/notebooks/08_deep_dive.ipynb``, on the `d100k` run and its labels
(the same speeches as notebook 07, A6). Five analyses, each new beside what
07 already shows:

1. **Chamber control.** Who held the majority in each chamber, resolved by
   the speech's date where the Senate changed hands within a Congress, and
   whether the speaker's party is in the minority of their own chamber.
2. **Opposition within members.** Notebook 07's opposition effect (model A),
   with the chamber minority added (B), and with member fixed effects (C):
   does the same person speak further out, or more harshly, once their party
   loses power?
3. **Topic blocks in one model.** The party gap per block and Congress from
   one saturated regression, so blocks and Congresses can be tested against
   each other with a joint, member-clustered covariance.
4. **Agenda against positions.** How much of the change in the policy-debate
   gap comes from the parties moving within blocks, and how much from the
   debate shifting between blocks (two-factor Shapley decomposition).
5. **Replacement against adaptation.** How much of each party's move comes
   from members who stayed, and how much from who entered and who left.

The rules of `analysis.py` hold: chambers are never pooled, every interval is
clustered by member, and nothing here is final. Labels are not yet
hand-checked (L4).
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from src.analysis import Z95, add_presidency, cell_means, within_cell_ols
from src.config import (
    CAUCUS_PARTY,
    CORPUS_PATH,
    CROSSWALK_PATH,
    DELEGATE_STATES,
    DW_NOMINATE_MEMBERS,
    GOVINFO_CAUCUS_PARTY,
    GOVINFO_END_DATE,
    GOVINFO_EXCLUDED_MEMBERS,
    PRESIDENTS_BY_CONGRESS,
    RANDOM_SEED,
    RAW_DIR,
    SOURCE_BREAK_CONGRESS,
    SOURCE_BREAK_DATE,
    TOPIC_BLOCKS,
    VOTEVIEW_PARTY_CODES,
)
from src.crosswalk import UNIT_KEY, load_voteview
from src.validation_design import cluster_draws

# --- constants -----------------------------------------------------------------

# Majority party of each chamber per Congress, 107th-119th (independents
# counted with the party they caucus with). House: history.house.gov, "Party
# Divisions of the House of Representatives". Senate: the party divisions
# senate.gov publishes ("Party Division", senate.gov/history/partydiv.htm),
# checked on 2026-10-27 against the table that cites it, since senate.gov
# refuses automated requests. SENATE_MAJORITY holds the Congresses with one
# majority throughout; the two that changed hands are in
# SENATE_MAJORITY_CHANGES and are resolved by the speech's date.
HOUSE_MAJORITY: dict[int, str] = {
    107: "R", 108: "R", 109: "R",
    110: "D", 111: "D",
    112: "R", 113: "R", 114: "R", 115: "R",
    116: "D", 117: "D",
    118: "R", 119: "R",
}  # fmt: skip
SENATE_MAJORITY: dict[int, str] = {
    108: "R", 109: "R",
    110: "D", 111: "D", 112: "D", 113: "D",
    114: "R", 115: "R", 116: "R",
    118: "D",
    119: "R",
}  # fmt: skip

# Each period runs from its date to the next one's (or the Congress's end).
# 107th: 50-50, Vice President Gore deciding until 20 January 2001, then
# Vice President Cheney; Jeffords left the Republicans and the Democrats
# organized the Senate on 6 June 2001 (he announced it on 24 May). 117th:
# Republicans until 20 January 2021, when Ossoff, Warnock and Padilla were
# sworn in and Vice President Harris broke the 50-50 tie.
SENATE_MAJORITY_CHANGES: dict[int, tuple[tuple[dt.date, str], ...]] = {
    107: (
        (dt.date(2001, 1, 3), "D"),
        (dt.date(2001, 1, 20), "R"),
        (dt.date(2001, 6, 6), "D"),
    ),
    117: (
        (dt.date(2021, 1, 3), "R"),
        (dt.date(2021, 1, 20), "D"),
    ),
}

# The Stanford data code Arlen Specter (PA, Senate) as a Democrat from the
# 109th Congress, but he left the Republicans only on 28 April 2009 (O13; his
# 111th speeches are flagged as a party switch, D23). His miscoded speeches
# are the ones labelled D before that date. Identified by Voteview's bioname,
# which the crosswalk matched on every one of them; the speakerid changes
# every Congress and Stanford's last_name has OCR variants.
SPECTER_BIONAME_PREFIX: str = "SPECTER,"
SPECTER_SWITCH_DATE: dt.date = dt.date(2009, 4, 28)

# A party x block x Congress x chamber cell with fewer policy debates is
# reported before the topic model is read (deep dive, analysis 3).
TOPIC_MIN_CELL: int = 30

# Bootstrap draws over members for the two decompositions; seeded with
# config.RANDOM_SEED.
DEEP_DIVE_BOOTSTRAP: int = 1000

# Base and comparison of the topic decomposition, as Congresses. A period's
# shares and means are averaged over its Congresses, each weighted equally.
DECOMPOSITION_COMPARISONS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "107→118": ((107,), (118,)),
    "107–114→115–119": (tuple(range(107, 115)), tuple(range(115, 120))),
}

# Base and comparison Congress of the per-block test of the topic model: the
# 118th is the last complete Congress.
TOPIC_TEST_CONGRESSES: tuple[int, int] = (107, 118)

# Windows of the cumulative replacement decomposition, first and last Congress.
TURNOVER_WINDOWS: dict[str, tuple[int, int]] = {
    "2001–2016 (107th–114th)": (107, 114),
    "2001–2025 (107th–119th)": (107, 119),
}

#: The opposition models, in order: terms, and the cell their fixed effects
#: sit in. A is notebook 07's (a party-chamber intercept), C compares each
#: member with themselves.
OPPOSITION_MODELS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "A": (("congress_number", "opposition"), ("chamber", "party")),
    "B": (("congress_number", "opposition", "chamber_minority"), ("chamber", "party")),
    "C": (("congress_number", "opposition", "chamber_minority"), ("member_key",)),
}

#: The four constellations of power, in display order.
CONSTELLATIONS: tuple[str, ...] = (
    "White House, chamber majority",
    "White House, chamber minority",
    "opposition, chamber majority",
    "opposition, chamber minority",
)

_PARTIES = ("D", "R")


# --- 1. chamber control -----------------------------------------------------------


def _as_date(value: object) -> dt.date:
    return pd.Timestamp(value).date()


def chamber_majority(chamber: str, congress: int, date: object) -> str:
    """The party holding the majority of ``chamber`` on ``date``.

    Raises:
        KeyError: a Congress or chamber with no entry.
    """
    if chamber == "H":
        return HOUSE_MAJORITY[congress]
    if chamber != "S":
        raise KeyError(f"unknown chamber {chamber!r}")
    if congress in SENATE_MAJORITY_CHANGES:
        day = _as_date(date)
        periods = SENATE_MAJORITY_CHANGES[congress]
        held = [party for start, party in periods if start <= day]
        return held[-1] if held else periods[0][1]
    return SENATE_MAJORITY[congress]


def add_chamber_control(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``chamber_majority`` and ``chamber_minority``.

    ``chamber_majority`` is the majority party of the speaker's chamber on the
    speech's date; ``chamber_minority`` is 1 when the speaker's party is not
    it.

    Raises:
        KeyError: a Congress with no majority recorded.
    """
    unknown = sorted(set(frame["congress_number"]) - set(HOUSE_MAJORITY))
    if unknown:
        raise KeyError(f"no majority recorded for Congress {unknown}")
    out = frame.copy()
    keys = out[["chamber", "congress_number", "date"]].drop_duplicates()
    majority = {
        (c, n, d): chamber_majority(c, n, d) for c, n, d in keys.itertuples(index=False)
    }
    rows = zip(out["chamber"], out["congress_number"], out["date"])
    out["chamber_majority"] = [majority[key] for key in rows]
    out["chamber_minority"] = (out["party"] != out["chamber_majority"]).astype(int)
    return out


def majority_table() -> pd.DataFrame:
    """One row per Congress and majority period: who held each chamber.

    The Senate's mid-Congress changes appear as one row per period, with the
    date it began.
    """
    rows = []
    for congress in sorted(HOUSE_MAJORITY):
        if congress in SENATE_MAJORITY_CHANGES:
            periods = SENATE_MAJORITY_CHANGES[congress]
        else:
            periods = ((None, SENATE_MAJORITY[congress]),)
        for start, senate in periods:
            rows.append(
                {
                    "congress_number": congress,
                    "senate_from": start,
                    "house_majority": HOUSE_MAJORITY[congress],
                    "senate_majority": senate,
                }
            )
    return pd.DataFrame(rows)


def constellation(frame: pd.DataFrame) -> pd.Series:
    """Each speech's constellation of power, one of `CONSTELLATIONS`."""
    house = np.where(frame["in_opposition"].astype(bool), "opposition", "White House")
    chamber = np.where(
        frame["chamber_minority"].astype(bool), "chamber minority", "chamber majority"
    )
    return pd.Series(
        [f"{h}, {c}" for h, c in zip(house, chamber)], index=frame.index, dtype=object
    )


def control_cells(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean of ``value`` per chamber, party and constellation, clustered by member.

    Needs `add_chamber_control` and ``in_opposition``. Columns as
    `analysis.cell_means`, plus ``constellation``.
    """
    data = frame.assign(constellation=constellation(frame))
    return cell_means(data, value, by=("chamber", "party", "constellation"))


# --- 2. opposition within members ----------------------------------------------------


def opposition_models(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Models A, B and C of `OPPOSITION_MODELS` for ``value``, per chamber and party.

    `analysis.within_cell_ols`, clustered by member, one row per model, group
    and term. Needs ``opposition`` (0/1, out of the White House),
    ``chamber_minority`` and ``member_key``.
    """
    fits = []
    for model, (terms, cell) in OPPOSITION_MODELS.items():
        fit = within_cell_ols(
            frame, value, list(terms), cell=cell, by=("chamber", "party")
        )
        fits.append(fit.assign(model=model, value=value))
    return pd.concat(fits, ignore_index=True)


def identifying_members(
    frame: pd.DataFrame,
    terms: Sequence[str] = ("opposition", "chamber_minority"),
    by: Sequence[str] = ("chamber", "party"),
) -> pd.DataFrame:
    """How many members identify each term under member fixed effects.

    Within a member, only a term that changes contributes: a member whose
    party never left the White House, say, adds nothing to the opposition
    coefficient of model C. One row per group and term: members, members whose
    term takes both values, and their speeches.
    """
    rows = []
    for key, group in frame.groupby(list(by)):
        for term in terms:
            varies = group.groupby("member_key")[term].nunique() > 1
            moving = varies[varies].index
            rows.append(
                {
                    **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                    "term": term,
                    "n_members": group["member_key"].nunique(),
                    "n_identifying": len(moving),
                    "n_speeches_identifying": int(
                        group["member_key"].isin(moving).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def check_reproduces(
    fit: pd.DataFrame, committed: pd.DataFrame, rtol: float = 1e-8
) -> None:
    """Raise unless ``fit`` reproduces a committed `within_cell_ols` table.

    Compares ``coef`` and ``se`` per chamber, party and term.

    Raises:
        ValueError: a row is missing on either side, or a value differs.
    """
    key = ["chamber", "party", "term"]
    merged = fit.merge(
        committed, on=key, how="outer", suffixes=("", "_committed"), indicator=True
    )
    missing = merged[merged["_merge"] != "both"]
    if len(missing):
        raise ValueError(f"rows on one side only: {missing[key].to_dict('records')}")
    for column in ("coef", "se"):
        off = ~np.isclose(
            merged[column], merged[f"{column}_committed"], rtol=rtol, atol=0
        )
        if off.any():
            raise ValueError(
                f"{column} differs from the committed table: "
                f"{merged.loc[off, [*key, column, f'{column}_committed']].to_dict('records')}"
            )


def specter_miscoded(frame: pd.DataFrame) -> pd.Series:
    """Speeches by Specter labelled D before his switch (O13, D23)."""
    names = frame["bioname"].astype("string").fillna("")
    dates = pd.to_datetime(frame["date"]).dt.date
    return (
        names.str.upper().str.startswith(SPECTER_BIONAME_PREFIX)
        & (frame["chamber"] == "S")
        & (frame["party"] == "D")
        & (dates < SPECTER_SWITCH_DATE)
    )


def robustness_samples(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The speech sets of the robustness check: all, and two left out (D23, O13)."""
    switch = frame["party_switch"].astype("boolean").fillna(False).astype(bool)
    return {
        "all speeches": frame,
        "without party switchers": frame[~switch],
        "without Specter 2005–2009": frame[~specter_miscoded(frame)],
    }


# --- 3. topic blocks in one model -------------------------------------------------------


def topic_cell_sizes(
    policy: pd.DataFrame, minimum: int = TOPIC_MIN_CELL
) -> pd.DataFrame:
    """Policy debates per Congress, chamber, topic block and party.

    Every combination appears, empty ones with n = 0; ``below_min`` marks a
    cell under ``minimum``.
    """
    index = pd.MultiIndex.from_product(
        [
            sorted(policy["congress_number"].unique()),
            sorted(policy["chamber"].unique()),
            list(TOPIC_BLOCKS),
            list(_PARTIES),
        ],
        names=["congress_number", "chamber", "topic_block", "party"],
    )
    sizes = (
        policy.groupby(list(index.names))
        .agg(n=("speech_id", "size"), n_members=("member_key", "nunique"))
        .reindex(index, fill_value=0)
        .reset_index()
    )
    sizes["below_min"] = sizes["n"] < minimum
    return sizes


def _cell_name(party: str, block: str, congress: int) -> str:
    return f"{party}:{block}:{congress}"


def fit_topic_model(policy: pd.DataFrame, chamber: str, value: str = "ideology"):
    """Saturated OLS of ``value`` on party x topic block x Congress, one chamber.

    One indicator per cell and no intercept, so each coefficient is a cell's
    mean, named ``party:block:congress``; the covariance is clustered by
    member (statsmodels' CR1 with its small-sample factor), so contrasts across
    cells are tested with the members they share. Congress is a category, not
    a trend. Returns the statsmodels result.
    """
    data = policy[policy["chamber"] == chamber].dropna(subset=[value, "topic_block"])
    names = [
        _cell_name(p, b, int(c))
        for p, b, c in zip(data["party"], data["topic_block"], data["congress_number"])
    ]
    design = pd.get_dummies(pd.Series(names, index=data.index), dtype=float)
    return sm.OLS(data[value].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["member_key"])[0]}
    )


def _contrast(fit, weights: Mapping[str, float]) -> np.ndarray:
    row = np.zeros(len(fit.params))
    names = list(fit.params.index)
    for name, weight in weights.items():
        row[names.index(name)] = weight
    return row


def _gap_weights(block: str, congress: int) -> dict[str, float]:
    return {
        _cell_name("R", block, congress): 1.0,
        _cell_name("D", block, congress): -1.0,
    }


def topic_gaps(fit, chamber: str) -> pd.DataFrame:
    """R - D per topic block and Congress from `fit_topic_model`, with 95% intervals.

    A cell either party lacks gets no gap.
    """
    cells = {tuple(name.split(":")) for name in fit.params.index}
    rows = []
    for party, block, congress in sorted(cells):
        if party != "R" or ("D", block, congress) not in cells:
            continue
        test = fit.t_test(_contrast(fit, _gap_weights(block, int(congress))))
        gap, se = float(np.squeeze(test.effect)), float(np.squeeze(test.sd))
        rows.append(
            {
                "congress_number": int(congress),
                "chamber": chamber,
                "topic_block": block,
                "gap": gap,
                "se": se,
                "ci_low": gap - Z95 * se,
                "ci_high": gap + Z95 * se,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["topic_block", "congress_number"], ignore_index=True
    )


def _change_weights(block: str, base: int, compare: int) -> dict[str, float]:
    weights = _gap_weights(block, compare)
    for name, weight in _gap_weights(block, base).items():
        weights[name] = weights.get(name, 0.0) - weight
    return weights


def topic_gap_tests(
    fit, chamber: str, congresses: tuple[int, int] = TOPIC_TEST_CONGRESSES
) -> pd.DataFrame:
    """Did each block's gap grow from the first Congress to the second?

    One row per block: the change in R - D, its member-clustered SE, interval
    and two-sided p. A last row, ``topic_block = "all (joint)"``, holds the
    Wald test that every block's change is the same (chi-squared with one
    degree of freedom fewer than blocks).

    Raises:
        ValueError: a block has no speeches for a party in either Congress --
            leaving it out would silently change the joint test.
    """
    base, compare = congresses
    blocks = list(TOPIC_BLOCKS)
    missing = [
        _cell_name(p, b, c)
        for b in blocks
        for p in _PARTIES
        for c in (base, compare)
        if _cell_name(p, b, c) not in fit.params.index
    ]
    if missing:
        raise ValueError(f"{chamber}: no speeches in cell(s) {missing}")
    rows = []
    for block in blocks:
        test = fit.t_test(_contrast(fit, _change_weights(block, base, compare)))
        change, se = float(np.squeeze(test.effect)), float(np.squeeze(test.sd))
        rows.append(
            {
                "chamber": chamber,
                "topic_block": block,
                "span": f"{base}-{compare}",
                "change": change,
                "se": se,
                "ci_low": change - Z95 * se,
                "ci_high": change + Z95 * se,
                "p": float(np.squeeze(test.pvalue)),
            }
        )
    if len(blocks) > 1:
        first = _contrast(fit, _change_weights(blocks[0], base, compare))
        matrix = np.array(
            [
                _contrast(fit, _change_weights(b, base, compare)) - first
                for b in blocks[1:]
            ]
        )
        wald = fit.wald_test(matrix, scalar=True, use_f=False)
        rows.append(
            {
                "chamber": chamber,
                "topic_block": "all (joint)",
                "span": f"{base}-{compare}",
                "chi2": float(wald.statistic),
                "df": len(blocks) - 1,
                "p": float(wald.pvalue),
            }
        )
    return pd.DataFrame(rows)


# --- 4. agenda against positions ---------------------------------------------------------


def block_cells(policy: pd.DataFrame, value: str = "ideology") -> pd.DataFrame:
    """Share and mean of each topic block per chamber, party and Congress.

    ``share`` is the block's share of the party's policy debates in that
    chamber and Congress -- a sample share within the cell, which estimates the
    cell's corpus share because every cell is drawn at random with the same
    quota (S11).
    """
    data = policy.dropna(subset=[value, "topic_block"])
    grouped = data.groupby(["chamber", "party", "congress_number", "topic_block"])[
        value
    ]
    cells = grouped.agg(n="size", mean="mean").reset_index()
    total = cells.groupby(["chamber", "party", "congress_number"])["n"].transform("sum")
    cells["share"] = cells["n"] / total
    return cells


def _period(cells: pd.DataFrame, congresses: Sequence[int]) -> pd.DataFrame:
    """Shares and means per chamber, party and block, averaged over ``congresses``.

    Each Congress weighs the same; a block absent from a Congress has share 0
    there and no mean.
    """
    blocks = list(TOPIC_BLOCKS)
    part = cells[cells["congress_number"].isin(congresses)]
    index = pd.MultiIndex.from_product(
        [
            sorted(part["chamber"].unique()),
            list(_PARTIES),
            list(congresses),
            blocks,
        ],
        names=["chamber", "party", "congress_number", "topic_block"],
    )
    full = part.set_index(list(index.names)).reindex(index)
    full["share"] = full["share"].fillna(0.0)
    return full.groupby(["chamber", "party", "topic_block"])[["share", "mean"]].mean()


def decompose_gap(base: pd.DataFrame, compare: pd.DataFrame) -> pd.DataFrame:
    """Split the change in the policy-debate gap into positions and agenda.

    ``base`` and ``compare`` hold ``share`` and ``mean`` per chamber, party and
    block (as `_period` returns). Per party, its overall mean is
    G = sum_k s_k mu_k, and the gap is G_R - G_D. Its change splits exactly,
    with no residual, into

    - positions: sum_k s̄_k (mu'_k - mu_k), the parties moving within blocks;
    - agenda:    sum_k (mū_k - c) (s'_k - s_k), the debate shifting between them;

    the two-factor Shapley values (bars are the averages of base and
    comparison). Shares sum to one, so the agenda total does not depend on the
    centre c; a single block's does, and c is the party's own mean (the
    average of its G in the two periods): a block adds to the gap when it gains
    share and sits further out than the party's average. A block missing in a
    period is taken at that centre.

    Returns one row per chamber, component (``total``, ``positions``,
    ``agenda``) and block (``all`` for the sums), with ``value``, plus
    ``gap_base`` and ``gap_compare`` on the ``total`` rows.
    """
    joined = base.join(compare, lsuffix="_0", rsuffix="_1").sort_index()
    rows = []
    for chamber in joined.index.get_level_values("chamber").unique():
        parts: dict[str, dict[str, float]] = {}
        gaps = [0.0, 0.0]
        for party, sign in (("R", 1.0), ("D", -1.0)):
            p = joined.loc[(chamber, party)]
            g0 = float((p["share_0"] * p["mean_0"].fillna(0.0)).sum())
            g1 = float((p["share_1"] * p["mean_1"].fillna(0.0)).sum())
            centre = (g0 + g1) / 2
            mean_0 = p["mean_0"].fillna(centre)
            mean_1 = p["mean_1"].fillna(centre)
            positions = (p["share_0"] + p["share_1"]) / 2 * (mean_1 - mean_0)
            agenda = ((mean_0 + mean_1) / 2 - centre) * (p["share_1"] - p["share_0"])
            for block in p.index:
                entry = parts.setdefault(block, {"positions": 0.0, "agenda": 0.0})
                entry["positions"] += sign * float(positions[block])
                entry["agenda"] += sign * float(agenda[block])
            gaps[0] += sign * g0
            gaps[1] += sign * g1
        positions = sum(v["positions"] for v in parts.values())
        agenda = sum(v["agenda"] for v in parts.values())
        rows.append(
            {
                "chamber": chamber,
                "component": "total",
                "topic_block": "all",
                "value": gaps[1] - gaps[0],
                "gap_base": gaps[0],
                "gap_compare": gaps[1],
            }
        )
        rows += [
            {
                "chamber": chamber,
                "component": "positions",
                "topic_block": "all",
                "value": positions,
            },
            {
                "chamber": chamber,
                "component": "agenda",
                "topic_block": "all",
                "value": agenda,
            },
        ]
        for block in TOPIC_BLOCKS:
            for component in ("positions", "agenda"):
                rows.append(
                    {
                        "chamber": chamber,
                        "component": component,
                        "topic_block": block,
                        "value": parts.get(block, {}).get(component, 0.0),
                    }
                )
    return pd.DataFrame(rows)


def gap_decomposition(
    policy: pd.DataFrame,
    comparisons: Mapping[
        str, tuple[Sequence[int], Sequence[int]]
    ] = DECOMPOSITION_COMPARISONS,
    value: str = "ideology",
) -> pd.DataFrame:
    """`decompose_gap` for each comparison, one row per comparison, chamber,
    component and block."""
    cells = block_cells(policy, value)
    return pd.concat(
        [
            decompose_gap(_period(cells, base), _period(cells, compare)).assign(
                comparison=name
            )
            for name, (base, compare) in comparisons.items()
        ],
        ignore_index=True,
    )


def _percentile_intervals(
    point: pd.DataFrame, draws: list[pd.DataFrame], key: Sequence[str], column: str
) -> pd.DataFrame:
    """95% percentile intervals of ``column`` over the draws, joined onto ``point``."""
    stacked = pd.concat(draws, ignore_index=True)
    bounds = (
        stacked.groupby(list(key))[column]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
        .reset_index()
    )
    return point.merge(bounds, on=list(key), how="left", validate="one_to_one")


def _bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], pd.DataFrame],
    key: Sequence[str],
    column: str,
    n_draws: int,
    seed: int,
) -> pd.DataFrame:
    """``statistic`` with percentile intervals from a member bootstrap, per chamber.

    Members are drawn with replacement within party (`validation_design.
    cluster_draws`), each bringing all their speeches, separately in each
    chamber, so a chamber's interval never borrows the other's members.
    """
    point = statistic(frame)
    draws = []
    for chamber, part in frame.groupby("chamber"):
        part = part.reset_index(drop=True)
        for idx in cluster_draws(part["member_key"], part["party"], n_draws, seed):
            draws.append(statistic(part.iloc[idx]))
    return _percentile_intervals(point, draws, key, column)


def bootstrap_gap_decomposition(
    policy: pd.DataFrame,
    comparisons: Mapping[
        str, tuple[Sequence[int], Sequence[int]]
    ] = DECOMPOSITION_COMPARISONS,
    value: str = "ideology",
    n_draws: int = DEEP_DIVE_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """`gap_decomposition` with 95% percentile intervals over members."""
    return _bootstrap(
        policy.dropna(subset=[value, "topic_block"]),
        lambda f: gap_decomposition(f, comparisons, value),
        ("comparison", "chamber", "component", "topic_block"),
        "value",
        n_draws,
        seed,
    )


# --- 5. replacement against adaptation --------------------------------------------------


def chamber_membership(path: Path = DW_NOMINATE_MEMBERS) -> pd.DataFrame:
    """Who sat in which chamber in each Congress, 107th-119th, from Voteview.

    One row per ICPSR, Congress and chamber (``H``/``S``), with ``party``
    from Voteview's party code (D 100, R 200; others, such as independents,
    null). Membership is actual service, not presence in the sample.
    Non-voting delegates are left out, as they are from the corpus (D10).
    """
    voteview = load_voteview(path)
    voteview = voteview[~voteview["state_abbrev"].isin(DELEGATE_STATES)]
    codes = {
        code: party for party, code in VOTEVIEW_PARTY_CODES.items() if party in _PARTIES
    }
    return pd.DataFrame(
        {
            "icpsr": voteview["icpsr"].astype("Int64"),
            "congress_number": voteview["congress"].astype(int),
            "chamber": voteview["chamber"].map({"House": "H", "Senate": "S"}),
            "party": voteview["party_code"].map(codes),
        }
    )


def _served(membership: pd.DataFrame) -> set[tuple[int, int, str]]:
    return set(
        zip(
            membership["icpsr"].astype(int),
            membership["congress_number"],
            membership["chamber"],
        )
    )


def classify_turnover(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Each speech's role in the step into and out of its Congress.

    ``as_earlier`` (the speech's Congress as t of the step t -> t+1):
    ``stayer`` if the speaker sits in the same chamber in t+1, else ``exit``.
    ``as_later`` (as t+1 of the step t -> t+1): ``stayer`` if they sat in the
    same chamber in t, else ``entrant``. A move from House to Senate is an exit
    from one and an entry into the other; a party switcher's new ICPSR enters
    and the old one leaves. Speeches without an ICPSR are ``unmatched``.
    """
    served = _served(membership)
    icpsr = frame["icpsr"].astype("Int64")
    earlier, later = [], []
    for member, congress, chamber in zip(
        icpsr, frame["congress_number"], frame["chamber"]
    ):
        if pd.isna(member):
            earlier.append("unmatched")
            later.append("unmatched")
            continue
        member = int(member)
        earlier.append(
            "stayer" if (member, congress + 1, chamber) in served else "exit"
        )
        later.append(
            "stayer" if (member, congress - 1, chamber) in served else "entrant"
        )
    return frame.assign(as_earlier=earlier, as_later=later)


def _group_sums(frame: pd.DataFrame, role: str, value: str) -> pd.DataFrame:
    return (
        frame.groupby(["chamber", "party", "congress_number", role])[value]
        .agg(["sum", "size"])
        .unstack(role)
    )


def _turnover_steps(classified: pd.DataFrame, value: str) -> pd.DataFrame:
    """The decomposition of every step t -> t+1, per chamber and party.

    With speech-weighted means m and speech shares w (s stayers, n entrants,
    x exits, u unmatched; primes for t+1):

        dM = (m_s' - m_s) + w_n (m_n - m_s') - w_x (m_x - m_s)
             + [w_u' (m_u' - m_s') - w_u (m_u - m_s)]

    an identity, since M_t = m_s + w_x (m_x - m_s) + w_u (m_u - m_s) and
    likewise for t+1. Terms: adaptation, entry, exit, unmatched.
    """
    early = _group_sums(classified, "as_earlier", value)
    late = _group_sums(classified, "as_later", value)
    rows = []
    for (chamber, party, congress), _ in early.iterrows():
        nxt = (chamber, party, congress + 1)
        if nxt not in late.index:
            continue
        e, lt = early.loc[(chamber, party, congress)], late.loc[nxt]

        def part(table, role):
            n = table.get(("size", role), 0.0)
            n = 0.0 if pd.isna(n) else float(n)
            s = table.get(("sum", role), 0.0)
            s = 0.0 if pd.isna(s) else float(s)
            return n, s

        n_s, s_s = part(e, "stayer")
        n_x, s_x = part(e, "exit")
        n_u, s_u = part(e, "unmatched")
        n_s1, s_s1 = part(lt, "stayer")
        n_n, s_n = part(lt, "entrant")
        n_u1, s_u1 = part(lt, "unmatched")
        if n_s == 0 or n_s1 == 0:
            raise ValueError(
                f"no stayer speeches in {chamber} {party} {congress}->{congress + 1}"
            )
        m_s, m_s1 = s_s / n_s, s_s1 / n_s1
        total_t, total_t1 = n_s + n_x + n_u, n_s1 + n_n + n_u1

        def term(n, s, total, reference):
            return 0.0 if n == 0 else n / total * (s / n - reference)

        rows.append(
            {
                "chamber": chamber,
                "party": party,
                "step": f"{congress}-{congress + 1}",
                "congress_number": congress + 1,
                "mean_before": (s_s + s_x + s_u) / total_t,
                "mean_after": (s_s1 + s_n + s_u1) / total_t1,
                "adaptation": m_s1 - m_s,
                "entry": term(n_n, s_n, total_t1, m_s1),
                "exit": -term(n_x, s_x, total_t, m_s),
                "unmatched": term(n_u1, s_u1, total_t1, m_s1)
                - term(n_u, s_u, total_t, m_s),
                "w_entrants": n_n / total_t1,
                "w_exits": n_x / total_t,
                "n_before": int(total_t),
                "n_after": int(total_t1),
            }
        )
    steps = pd.DataFrame(rows)
    steps["delta"] = steps["mean_after"] - steps["mean_before"]
    steps["replacement"] = steps["entry"] + steps["exit"]
    return steps


def turnover_decomposition(
    frame: pd.DataFrame, membership: pd.DataFrame, value: str = "ideology"
) -> pd.DataFrame:
    """Replacement against adaptation for every pair of consecutive Congresses.

    One row per chamber, party and step, from speech-weighted means: each
    sampled speech counts once, since the run draws speeches, not members.
    ``adaptation + entry + exit + unmatched == delta`` exactly;
    ``replacement`` is entry + exit.
    """
    data = frame.dropna(subset=[value])
    return _turnover_steps(classify_turnover(data, membership), value)


def stayers_without_speeches(
    frame: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Stayers of each step who have no sampled speech in one of its Congresses.

    Per chamber, party and step: members Voteview places in the chamber in
    both Congresses, under that party in the earlier one, and how many of them
    have no speech in the sample in the earlier or the later Congress. Their
    position enters the adaptation term only through the Congress where they do
    speak. Independents (Voteview code 328, e.g. Sanders, King) carry no party
    in `chamber_membership` and are not counted here; the decomposition itself
    classifies their speeches like anyone else's.
    """
    spoke = set(
        zip(
            frame["icpsr"].dropna().astype(int),
            frame.loc[frame["icpsr"].notna(), "congress_number"],
            frame.loc[frame["icpsr"].notna(), "chamber"],
        )
    )
    members = membership.dropna(subset=["party"])
    present = set(
        zip(
            members["icpsr"].astype(int), members["congress_number"], members["chamber"]
        )
    )
    rows = []
    for (chamber, party), group in members.groupby(["chamber", "party"]):
        congresses = sorted(group["congress_number"].unique())
        for congress in congresses[:-1]:
            now = group[group["congress_number"] == congress]["icpsr"].astype(int)
            stayers = [m for m in now if (m, congress + 1, chamber) in present]
            rows.append(
                {
                    "chamber": chamber,
                    "party": party,
                    "step": f"{congress}-{congress + 1}",
                    "n_stayers": len(stayers),
                    "no_speech_before": sum(
                        (m, congress, chamber) not in spoke for m in stayers
                    ),
                    "no_speech_after": sum(
                        (m, congress + 1, chamber) not in spoke for m in stayers
                    ),
                }
            )
    return pd.DataFrame(rows)


def cumulative_turnover(
    steps: pd.DataFrame, windows: Mapping[str, tuple[int, int]] = TURNOVER_WINDOWS
) -> pd.DataFrame:
    """The steps summed over each window: each party's move from its first to
    its last Congress, split into adaptation, entry, exit and unmatched.

    Long format: one row per window, chamber, party and component (``delta``,
    ``adaptation``, ``replacement``, ``entry``, ``exit``, ``unmatched``).
    """
    components = ["delta", "adaptation", "replacement", "entry", "exit", "unmatched"]
    rows = []
    for name, (first, last) in windows.items():
        inside = steps[steps["congress_number"].between(first + 1, last)]
        sums = inside.groupby(["chamber", "party"])[components].sum()
        for (chamber, party), row in sums.iterrows():
            for component in components:
                rows.append(
                    {
                        "window": name,
                        "chamber": chamber,
                        "party": party,
                        "component": component,
                        "value": float(row[component]),
                    }
                )
    return pd.DataFrame(rows)


def turnover_robustness(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    value: str = "ideology",
    windows: Mapping[str, tuple[int, int]] = TURNOVER_WINDOWS,
) -> pd.DataFrame:
    """`cumulative_turnover` on each speech set of `robustness_samples`.

    A member the data relabel without a new ICPSR -- Specter, coded D from the
    109th on the Stanford side (O13), and the other Stanford switchers (D23) --
    stays a "stayer" across the relabelling, so their move counts as
    adaptation of both parties rather than as exit and entry. Point estimates
    only; the main decomposition keeps every speech, to match notebook 07.
    """
    return pd.concat(
        [
            cumulative_turnover(
                turnover_decomposition(sample, membership, value), windows
            ).assign(sample=name, n_speeches=len(sample))
            for name, sample in robustness_samples(frame).items()
        ],
        ignore_index=True,
    )


def bootstrap_turnover(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    value: str = "ideology",
    windows: Mapping[str, tuple[int, int]] = TURNOVER_WINDOWS,
    n_draws: int = DEEP_DIVE_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """`cumulative_turnover` with 95% percentile intervals over members.

    Who stays, enters or leaves is fixed by actual membership; the bootstrap
    redraws only whose speeches are seen.
    """
    classified = classify_turnover(frame.dropna(subset=[value]), membership)
    return _bootstrap(
        classified,
        lambda f: cumulative_turnover(_turnover_steps(f, value), windows),
        ("window", "chamber", "party", "component"),
        "value",
        n_draws,
        seed,
    )


# =================================================================================
# Notebook 09: extensions of the deep dive
# =================================================================================
#
# Feeds ``code/notebooks/09_deep_dive_extensions.ipynb``, on the same `d100k`
# run, labels and substantive speeches as notebook 08. Nothing above is
# changed; the functions below are new and only call it. Nothing here fixes the
# last Congress or year: they are read from the data, so a later scoring run
# that reaches further needs only its run id.

# --- constants ---------------------------------------------------------------------

#: External data in data/raw/external (see its README); read, never written.
EXTERNAL_DIR: Path = RAW_DIR / "external"
COUNTYPRES_PATH: Path = EXTERNAL_DIR / "countypres_2000-2024.csv"
PUBLIC_LAWS_PATH: Path = EXTERNAL_DIR / "public_laws_by_congress.csv"

#: A Congress (or a year) counts as complete once the data reach 1 December of
#: its last year: Congress rarely sits in the last weeks of December.
COMPLETE_FROM: tuple[int, int] = (12, 1)

#: Fixed list of events, set before any result was seen (A8): date and the
#: topic codes (L2) the event is about.
EVENTS: dict[str, tuple[dt.date, tuple[str, ...]]] = {
    "9/11": (dt.date(2001, 9, 11), ("defense_foreign",)),
    "Iraq War": (dt.date(2003, 3, 20), ("defense_foreign",)),
    "Lehman collapse": (dt.date(2008, 9, 15), ("economy_tax_budget",)),
    "ACA signed": (dt.date(2010, 3, 23), ("health",)),
    "Covid emergency": (dt.date(2020, 3, 13), ("health", "economy_tax_budget")),
    "Dobbs decision": (dt.date(2022, 6, 24), ("social_cultural",)),
}

#: Half-width of an event window in days (six months), and the shortest window
#: the notebook accepts without stopping.
EVENT_WINDOW_DAYS: int = 182
EVENT_MIN_DAYS: int = 91

#: Width of a bin in the descriptive quarterly view around an event, and how
#: many bins on each side.
EVENT_QUARTER_DAYS: int = 91
EVENT_QUARTERS: int = 4

#: Minimum substantive speeches for a member x Congress unit (A3, A5).
MEMBER_MIN_SPEECHES: int = 10

#: Robustness definition of a moderate: a member mean this close to 0.
CENTRIST_BAND: float = 0.2

#: Keyword lists (A8), fixed before any result was seen. Whole words or phrases,
#: case ignored; inflections are written out. ``other_party`` is per speaker
#: party, in OTHER_PARTY_TERMS. The hand check (HALT 2, 50 hits a list) failed
#: two first lists: ``presidential_power`` (40% true hits: "unconstitutional"
#: is mostly about laws and courts, "power grab" and "abuse of power" often
#: about Congress) is dropped; ``hostility`` (60%: "radical" and "extremist"
#: often non-political or about terrorism) is replaced by targeted phrases and
#: checked again (``deepdive_keyword_check_hostility_v2.csv``).
KEYWORD_LISTS: dict[str, tuple[str, ...]] = {
    "cooperation": (
        "bipartisan", "bipartisanship", "compromise", "compromises",
        "common ground", "across the aisle", "both sides of the aisle",
        "work together", "working together", "worked together", "both parties",
    ),
    "hostility": (
        "socialist", "socialists", "socialism", "radical left", "radical right",
        "far left", "far-left", "far right", "far-right", "radical agenda",
        "radical policies", "radical policy", "extremist agenda", "extreme agenda",
        "environmental extremists", "environmental radicals",
        "radical environmentalists", "radical environmentalism",
    ),
    "deficit": (
        "deficit", "deficits", "national debt", "debt ceiling", "debt limit",
        "fiscal responsibility", "balanced budget",
    ),
    "undo": (
        "repeal", "repeals", "repealed", "repealing", "overturn", "overturned",
        "overturning", "undo", "undoing", "undid", "roll back", "rolls back",
        "rolled back", "rolling back", "rollback", "rollbacks",
    ),
}  # fmt: skip

#: Naming the other party, by the speaker's party. Bare "Republican" and
#: "Democratic" never count (republican form of government, democratic
#: institutions), and neither does the speaker's own party.
_OTHER_SIDE = (
    "the other side of the aisle", "my friends on the other side",
    "my colleagues on the other side", "those on the other side",
    "members on the other side",
)  # fmt: skip
OTHER_PARTY_TERMS: dict[str, tuple[str, ...]] = {
    "D": (
        "Republicans", "Republican Party", "GOP", "my Republican colleagues",
        "Republican leadership", "Republican majority", "Republican leader",
        "Republican leaders", *_OTHER_SIDE,
    ),
    "R": (
        "Democrats", "Democratic Party", "Democrat Party", "my Democratic colleagues",
        "Democratic leadership", "Democrat leadership", "Democratic majority",
        "Democratic leader", "Democratic leaders", "Democrat leaders", *_OTHER_SIDE,
    ),
}  # fmt: skip

#: Phrases taken out of the text before a list is matched (regular expressions,
#: case ignored): uses of a listed word that are not what the list measures.
KEYWORD_EXCLUDE: dict[str, tuple[str, ...]] = {
    "cooperation": (r"compromis(?:e|es|ed|ing)\s+(?:our|the|national|homeland)\s+security",),
    "hostility": (
        r"soviet\s+socialist",
        r"national\s+socialists?",
        r"socialist\s+republics?",
    ),
    "deficit": (
        r"trade\s+deficits?",
        r"attention\s+deficit",
        r"deficits?\s+of\s+(?:trust|confidence)",
    ),
    "other_party": (),
    "undo": (),
}  # fmt: skip

#: What a true hit is, written into each check sheet (HALT 2, A8).
KEYWORD_CHECK_DEFINITIONS: dict[str, str] = {
    "cooperation": "the passage speaks about cooperation between the parties",
    "hostility": "the word describes political opponents or their policies",
    "other_party": "the words mean the other party",
    "deficit": "the passage is about public finances",
    "undo": "the passage is about taking back laws or decisions",
}
KEYWORD_CHECK_SIZE: int = 50
KEYWORD_CONTEXT: int = 150

#: Tercile labels of a state's presidential margin, closest first.
MARGIN_TERCILES: tuple[str, ...] = ("close", "middle", "safe")

#: The two outcomes of the member-level analyses.
VALUES: tuple[str, ...] = ("extremity", "tone")


# --- shared helpers ----------------------------------------------------------------


def congress_days(congress: int) -> tuple[dt.date, dt.date]:
    """First day of a Congress and the first day of the next (3 January)."""
    start = 1789 + 2 * (congress - 1)
    return dt.date(start, 1, 3), dt.date(start + 2, 1, 3)


def congress_of(date: object) -> int:
    """The Congress sitting on ``date``."""
    day = _as_date(date)
    congress = (day.year - 1789) // 2 + 1
    if day < congress_days(congress)[0]:
        congress -= 1
    return congress


def last_complete_congress(last_date: object) -> int:
    """The last Congress the data cover to 1 December of its second year."""
    day = _as_date(last_date)
    congress = congress_of(day)
    end_year = congress_days(congress)[1].year - 1
    if day < dt.date(end_year, *COMPLETE_FROM):
        congress -= 1
    return congress


def incomplete_year(last_date: object) -> int | None:
    """The data's last year if they stop before 1 December of it, else None."""
    day = _as_date(last_date)
    return day.year if day < dt.date(day.year, *COMPLETE_FROM) else None


def load_corpus_keyword_hits(
    corpus: Path = CORPUS_PATH,
    crosswalk: Path = CROSSWALK_PATH,
    batch_size: int = 20_000,
) -> pd.DataFrame:
    """Every corpus speech with its keyword hits (`keyword_hits`), but not its text.

    The corpus is read in batches and each batch's text is dropped once it is
    matched, so the whole text never sits in memory. Adds ``member_key``
    (as `analysis.load_scored_run`: the career-long ICPSR from the crosswalk,
    else the source-native id; the corpus's own ``icpsr`` is not the Voteview
    join key, D21) and the presidency.
    """
    columns = [
        "speech_id", "date", "member_id", "party", "chamber", "congress_number",
        "text", "source", "word_count", "party_original",
    ]  # fmt: skip
    parts = []
    for batch in pq.ParquetFile(corpus).iter_batches(
        batch_size=batch_size, columns=columns
    ):
        frame = batch.to_pandas()
        parts.append(frame.drop(columns="text").join(keyword_hits(frame)))
    speeches = pd.concat(parts, ignore_index=True)
    members = pd.read_parquet(crosswalk, columns=[*UNIT_KEY, "icpsr"])
    speeches = speeches.merge(members, on=list(UNIT_KEY), how="left", validate="m:1")
    speeches["member_key"] = np.where(
        speeches["icpsr"].notna(),
        "icpsr:" + speeches["icpsr"].astype("Int64").astype(str),
        speeches["source"] + ":" + speeches["member_id"],
    )
    return add_presidency(speeches)


def fit_cell_model(frame: pd.DataFrame, cells: Sequence[str], value: str):
    """Saturated OLS of ``value``: one indicator per cell, no intercept.

    Each coefficient is a cell's mean, named by its ``cells`` values joined
    with ``:``; the covariance is clustered by member (CR1), so contrasts
    across cells are tested with the members they share. The general form of
    `fit_topic_model`.
    """
    data = frame.dropna(subset=[value, *cells])
    names = data[list(cells)].astype(str).agg(":".join, axis=1)
    design = pd.get_dummies(names, dtype=float)
    return sm.OLS(data[value].astype(float), design).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["member_key"])[0]}
    )


def cell_contrast(fit, weights: Mapping[str, float]) -> dict[str, float]:
    """A contrast of cell means: estimate, SE, 95% interval and two-sided p.

    Raises:
        ValueError: a weighted cell is not in the model (no speeches).
    """
    missing = [name for name in weights if name not in fit.params.index]
    if missing:
        raise ValueError(f"no speeches in cell(s) {missing}")
    test = fit.t_test(_contrast(fit, weights))
    estimate, se = float(np.squeeze(test.effect)), float(np.squeeze(test.sd))
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - Z95 * se,
        "ci_high": estimate + Z95 * se,
        "p": float(np.squeeze(test.pvalue)),
    }


# --- A. events ----------------------------------------------------------------------


def event_windows(
    events: Mapping[str, tuple[dt.date, tuple[str, ...]]] = EVENTS,
    half_width: int = EVENT_WINDOW_DAYS,
    minimum: int = EVENT_MIN_DAYS,
) -> pd.DataFrame:
    """Each event's window: before = [d - w, d), after = [d, d + w).

    ``w`` is ``half_width`` unless the event's Congress begins or ends sooner;
    then both sides are cut to the nearer boundary, so no window crosses a
    Congress and the two sides stay equally long. ``truncated`` marks a cut
    window, ``too_short`` one under ``minimum`` days a side.
    """
    rows = []
    for name, (day, topics) in events.items():
        congress = congress_of(day)
        start, end = congress_days(congress)
        days = min(half_width, (day - start).days, (end - day).days)
        rows.append(
            {
                "event": name,
                "date": day,
                "topics": ", ".join(topics),
                "congress_number": congress,
                "days": days,
                "before_from": day - dt.timedelta(days=days),
                "after_to": day + dt.timedelta(days=days),
                "truncated": days < half_width,
                "too_short": days < minimum,
            }
        )
    return pd.DataFrame(rows)


def event_speeches(policy: pd.DataFrame, window: Mapping[str, object]) -> pd.DataFrame:
    """The policy debates inside one event window, with ``period`` (before /
    after) and ``group`` (event: the event's topics; other: every other topic)."""
    dates = pd.to_datetime(policy["date"])
    day = pd.Timestamp(window["date"])
    days = pd.Timedelta(days=int(window["days"]))
    inside = (dates >= day - days) & (dates < day + days)
    out = policy[inside].copy()
    out["period"] = np.where(dates[inside] < day, "before", "after")
    topics = str(window["topics"]).split(", ")
    out["group"] = np.where(out["topic"].isin(topics), "event", "other")
    return out


def event_cell_sizes(
    policy: pd.DataFrame, windows: pd.DataFrame, minimum: int = TOPIC_MIN_CELL
) -> pd.DataFrame:
    """Speeches and members per event, chamber, party, period and group.

    Every combination appears, empty ones with n = 0; ``below_min`` marks a
    cell under ``minimum``.
    """
    parts = []
    for window in windows.to_dict("records"):
        data = event_speeches(policy, window)
        index = pd.MultiIndex.from_product(
            [("H", "S"), _PARTIES, ("before", "after"), ("event", "other")],
            names=["chamber", "party", "period", "group"],
        )
        sizes = (
            data.groupby(list(index.names))
            .agg(n=("speech_id", "size"), n_members=("member_key", "nunique"))
            .reindex(index, fill_value=0)
            .reset_index()
        )
        parts.append(sizes.assign(event=window["event"], days=window["days"]))
    out = pd.concat(parts, ignore_index=True)
    out["below_min"] = out["n"] < minimum
    return out


def _event_cell(party: str, period: str, group: str) -> str:
    return f"{party}:{period}:{group}"


def event_effects(policy: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Change from before to after each event, in its topics and in the rest.

    Per event and chamber, one saturated model (`fit_cell_model`) on party x
    period x group, clustered by member, for ideology and for tone. Rows:

    - ``measure = "gap"``, ``party = "R−D"``: the change in R - D;
    - ``measure = "tone"``, ``party`` D or R: the change in the party's tone;

    each for ``group`` event, other, and ``event − other`` (the difference of
    the two changes, i.e. how much the event's topics moved beyond the rest).
    """
    rows = []
    for window in windows.to_dict("records"):
        data = event_speeches(policy, window)
        for chamber, part in data.groupby("chamber"):
            base = {
                "event": window["event"],
                "chamber": chamber,
                "days": window["days"],
            }
            fits = {
                "gap": fit_cell_model(part, ("party", "period", "group"), "ideology"),
                "tone": fit_cell_model(part, ("party", "period", "group"), "tone"),
            }
            for measure, parties in (("gap", ("R−D",)), ("tone", _PARTIES)):
                for party in parties:
                    change = {}
                    for group in ("event", "other"):
                        change[group] = _event_change(measure, party, group)
                        rows.append(
                            {**base, "measure": measure, "party": party,
                             "group": group,
                             **cell_contrast(fits[measure], change[group])}
                        )  # fmt: skip
                    both = dict(change["event"])
                    for name, weight in change["other"].items():
                        both[name] = both.get(name, 0.0) - weight
                    rows.append(
                        {**base, "measure": measure, "party": party,
                         "group": "event − other", **cell_contrast(fits[measure], both)}
                    )  # fmt: skip
    return pd.DataFrame(rows)


def _event_change(measure: str, party: str, group: str) -> dict[str, float]:
    """Weights of after - before for one group: of R - D, or of one party."""
    signs = {"R": 1.0, "D": -1.0} if measure == "gap" else {party: 1.0}
    weights: dict[str, float] = {}
    for p, sign in signs.items():
        weights[_event_cell(p, "after", group)] = sign
        weights[_event_cell(p, "before", group)] = -sign
    return weights


def flag_small_event_cells(effects: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """Mark each `event_effects` row that rests on a cell under the minimum.

    A row's contrast uses its party (both for the gap), both periods and its
    group (both for ``event − other``). ``small_cell`` is True when any of
    those cells is ``below_min`` in ``sizes`` (`event_cell_sizes`);
    ``small_cells`` names them with their speeches and members.
    """
    small = sizes[sizes["below_min"]]
    flags, notes = [], []
    for row in effects.to_dict("records"):
        parties = list(_PARTIES) if row["party"] == "R−D" else [row["party"]]
        groups = (
            ["event", "other"] if row["group"] == "event − other" else [row["group"]]
        )
        hit = small[
            (small["event"] == row["event"])
            & (small["chamber"] == row["chamber"])
            & small["party"].isin(parties)
            & small["group"].isin(groups)
        ]
        flags.append(len(hit) > 0)
        notes.append(
            "; ".join(
                f"{r['party']} {r['period']} {r['group']}: n={r['n']}, {r['n_members']} members"
                for r in hit.to_dict("records")
            )
        )
    return effects.assign(small_cell=flags, small_cells=notes)


def event_quarterly(
    policy: pd.DataFrame,
    events: Mapping[str, tuple[dt.date, tuple[str, ...]]] = EVENTS,
    width: int = EVENT_QUARTER_DAYS,
    quarters: int = EVENT_QUARTERS,
) -> pd.DataFrame:
    """Descriptive: each party's mean ideology and tone in the event's topics,
    in bins of ``width`` days, ``quarters`` on each side of the event.

    Bin k holds [d + k*width, d + (k+1)*width); bin 0 starts on the event day.
    Unlike the tested windows, the bins may cross a Congress boundary;
    ``congress_boundary`` gives where (in bins) the nearest one falls.
    """
    parts = []
    for name, (day, topics) in events.items():
        data = policy[policy["topic"].isin(topics)]
        offset = (pd.to_datetime(data["date"]) - pd.Timestamp(day)).dt.days
        data = data.assign(quarter=np.floor(offset / width).astype(int))
        data = data[data["quarter"].between(-quarters, quarters - 1)]
        start, end = congress_days(congress_of(day))
        nearest = min(((start - day).days, (end - day).days), key=abs)
        for value in ("ideology", "tone"):
            cells = cell_means(data, value, by=("chamber", "party", "quarter"))
            parts.append(
                cells.assign(event=name, value=value, congress_boundary=nearest / width)
            )
    return pd.concat(parts, ignore_index=True)


# --- B. entry cohorts ---------------------------------------------------------------


def prior_service_switchers(path: Path = DW_NOMINATE_MEMBERS) -> pd.DataFrame:
    """ICPSRs whose person served earlier under another ICPSR.

    Voteview gives a party switcher a new ICPSR, so `analysis.member_background`
    restarts their tenure and they look like newcomers. A person is their
    bioguide id; any ICPSR whose first Congress is later than the person's
    first is listed, with both first Congresses.
    """
    voteview = pd.read_csv(
        path, usecols=["congress", "chamber", "icpsr", "bioguide_id", "bioname"]
    )
    voteview = voteview[voteview["chamber"].isin(["House", "Senate"])]
    voteview = voteview.dropna(subset=["bioguide_id"])
    per = voteview.groupby(["bioguide_id", "icpsr"], as_index=False).agg(
        bioname=("bioname", "first"), first_congress=("congress", "min")
    )
    per["person_first_congress"] = per.groupby("bioguide_id")[
        "first_congress"
    ].transform("min")
    out = per[per["first_congress"] > per["person_first_congress"]].copy()
    out["icpsr"] = out["icpsr"].astype("Int64")
    return out.sort_values(["first_congress", "bioname"], ignore_index=True)


def cohort_extremity(
    members: pd.DataFrame,
    exclude_icpsr: Sequence[int] = (),
    value: str = "extremity_vs_cell",
    first: int = 107,
) -> pd.DataFrame:
    """Each entry cohort's ``value`` in its first Congress.

    ``members`` is `analysis.add_background` output with ``value`` (extremity
    relative to the speaker's party, chamber and Congress, taken over all
    speeches). Kept: speeches in the speaker's first Congress
    (``first_congress``, any chamber, A4), from ``first`` on, without the ICPSRs
    in ``exclude_icpsr``. Means per chamber, party and cohort, clustered by
    member, as `analysis.cell_means` (``cohort`` = the first Congress).
    """
    icpsr = members["icpsr"].astype("Int64")
    keep = (
        members["first_congress"].notna()
        & (members["congress_number"] == members["first_congress"])
        & (members["first_congress"] >= first)
        & ~icpsr.isin(list(exclude_icpsr)).fillna(False)
    )
    data = members[keep].assign(cohort=members.loc[keep, "first_congress"].astype(int))
    return cell_means(data, value, by=("chamber", "party", "cohort"))


# --- 9. dissenters ------------------------------------------------------------------


def moderates(units: pd.DataFrame, band: float = CENTRIST_BAND) -> pd.DataFrame:
    """Flag moderate member x Congress units.

    ``units`` is `analysis.member_trajectories` (one row per member and
    Congress, ``mean`` its mean ideology). Per chamber and Congress, a unit is
    ``moderate`` when its mean lies closer to the median unit of the other
    party than to that of its own (ties count as not moderate), and
    ``centrist`` when |mean| < ``band`` (the robustness definition).
    """
    out = units.copy()
    medians = out.groupby(["chamber", "congress_number", "party"])["mean"].median()
    own = [
        medians.get((c, n, p))
        for c, n, p in zip(out["chamber"], out["congress_number"], out["party"])
    ]
    other = [
        medians.get((c, n, "R" if p == "D" else "D"))
        for c, n, p in zip(out["chamber"], out["congress_number"], out["party"])
    ]
    own, other = np.array(own, dtype=float), np.array(other, dtype=float)
    out["median_own"], out["median_other"] = own, other
    out["moderate"] = (np.abs(out["mean"] - other) < np.abs(out["mean"] - own)).astype(
        int
    )
    out["centrist"] = (out["mean"].abs() < band).astype(int)
    return out


def _dissent_stats(units: pd.DataFrame) -> pd.DataFrame:
    """SD of member means and shares of moderates per chamber, party, Congress."""
    flagged = moderates(units)
    return (
        flagged.groupby(["chamber", "party", "congress_number"])
        .agg(
            sd=("mean", "std"),
            share_moderate=("moderate", "mean"),
            share_centrist=("centrist", "mean"),
            n_members=("mean", "size"),
        )
        .reset_index()
    )


def dissent_table(
    units: pd.DataFrame, n_draws: int = DEEP_DIVE_BOOTSTRAP, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """`_dissent_stats` with 95% percentile intervals over members.

    Within each chamber and Congress, members are drawn with replacement
    within their party, and the medians that define a moderate are drawn
    with them. Long format: one row per chamber, party, Congress and
    statistic (``sd``, ``share_moderate``, ``share_centrist``).
    """
    point = _dissent_stats(units).melt(
        id_vars=["chamber", "party", "congress_number", "n_members"],
        var_name="statistic",
        value_name="value",
    )
    rng = np.random.default_rng(seed)
    draws = []
    groups = [
        part.reset_index(drop=True)
        for _, part in units.groupby(["chamber", "congress_number"])
    ]
    for _ in range(n_draws):
        sample = []
        for part in groups:
            for _, rows in part.groupby("party"):
                sample.append(rows.iloc[rng.integers(0, len(rows), len(rows))])
        stats = _dissent_stats(pd.concat(sample, ignore_index=True))
        draws.append(
            stats.drop(columns="n_members").melt(
                id_vars=["chamber", "party", "congress_number"],
                var_name="statistic",
                value_name="value",
            )
        )
    return _percentile_intervals(
        point, draws, ("chamber", "party", "congress_number", "statistic"), "value"
    )


def moderate_exit(
    units: pd.DataFrame, membership: pd.DataFrame, last_congress: int
) -> pd.DataFrame:
    """`moderates` plus ``left``: 1 when the member does not sit in the same
    chamber in the next Congress (actual service, `chamber_membership`).

    Units in ``last_congress`` are dropped: who leaves after it is not known
    yet. ``member_key`` is added for clustering.
    """
    served = _served(membership)
    out = moderates(units[units["congress_number"] < last_congress])
    out["left"] = [
        int((int(i), int(c) + 1, ch) not in served)
        for i, c, ch in zip(out["icpsr"], out["congress_number"], out["chamber"])
    ]
    out["member_key"] = "icpsr:" + out["icpsr"].astype("Int64").astype(str)
    return out


# --- 4. the last Congress -----------------------------------------------------------


def person_service(path: Path = DW_NOMINATE_MEMBERS) -> pd.DataFrame:
    """Who served in which Congress and chamber, 107th-119th, as a person.

    One row per ICPSR, Congress and chamber, with ``person``: the bioguide id,
    or the ICPSR where Voteview has none. Delegates are left out (D10).
    """
    voteview = load_voteview(path)
    voteview = voteview[~voteview["state_abbrev"].isin(DELEGATE_STATES)]
    person = voteview["bioguide_id"].where(
        voteview["bioguide_id"].notna(), "icpsr:" + voteview["icpsr"].astype(str)
    )
    return pd.DataFrame(
        {
            "icpsr": voteview["icpsr"].astype("Int64"),
            "person": person,
            "congress_number": voteview["congress"].astype(int),
            "chamber": voteview["chamber"].map({"House": "H", "Senate": "S"}),
        }
    )


def add_last_congress(
    frame: pd.DataFrame, service: pd.DataFrame, last_congress: int
) -> pd.DataFrame:
    """Add ``last_congress``: 1 in a member's last Congress before leaving.

    A speech's Congress is the speaker's last when the person (bioguide id,
    so a party switcher's new ICPSR does not count as leaving) sits in neither
    chamber in the next Congress. A move from House to Senate is not leaving.
    Speeches in ``last_congress`` (who leaves after it is unknown) and
    speeches without an ICPSR get a null.
    """
    person_of = service.drop_duplicates("icpsr").set_index("icpsr")["person"]
    sitting = set(zip(service["person"], service["congress_number"]))
    icpsr = frame["icpsr"].astype("Int64")
    flags = []
    for member, congress in zip(icpsr, frame["congress_number"]):
        if (
            pd.isna(member)
            or congress >= last_congress
            or member not in person_of.index
        ):
            flags.append(np.nan)
            continue
        flags.append(float((person_of[member], int(congress) + 1) not in sitting))
    return frame.assign(last_congress=flags)


#: Model C of notebook 08 with the last Congress added.
LAST_CONGRESS_TERMS: tuple[str, ...] = (*OPPOSITION_MODELS["C"][0], "last_congress")


def last_congress_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Model C (member fixed effects) plus ``last_congress``, for extremity and
    tone, per chamber and party, clustered by member."""
    fits = []
    for value in VALUES:
        fit = within_cell_ols(
            frame, value, list(LAST_CONGRESS_TERMS), cell=("member_key",),
            by=("chamber", "party"),
        )  # fmt: skip
        fits.append(fit.assign(value=value))
    return pd.concat(fits, ignore_index=True)


# --- keyword lists (2 + 7, 6, 3) -------------------------------------------------------


def compile_keywords(terms: Sequence[str]) -> re.Pattern:
    """One pattern matching any of ``terms`` as whole words, case ignored.

    Longer terms come first, so a phrase wins over a word inside it; a space
    in a term matches any run of whitespace.
    """
    ordered = sorted(set(terms), key=len, reverse=True)
    body = "|".join(r"\s+".join(map(re.escape, t.split())) for t in ordered)
    return re.compile(rf"\b(?:{body})\b", re.IGNORECASE)


def compile_exclusions(patterns: Sequence[str]) -> re.Pattern | None:
    """The exclusion phrases as one pattern (whole words, case ignored)."""
    if not patterns:
        return None
    return re.compile(r"\b(?:" + "|".join(patterns) + r")\b", re.IGNORECASE)


def blank(text: str, exclusion: re.Pattern | None) -> str:
    """``text`` with every exclusion phrase replaced by as many spaces, so
    positions in the text stay where they were."""
    if exclusion is None:
        return text
    return exclusion.sub(lambda m: " " * len(m.group()), text)


def _patterns(
    name: str, party: str | None = None
) -> tuple[re.Pattern, re.Pattern | None]:
    terms = OTHER_PARTY_TERMS[party] if name == "other_party" else KEYWORD_LISTS[name]
    return compile_keywords(terms), compile_exclusions(KEYWORD_EXCLUDE[name])


def _count(
    texts: pd.Series, pattern: re.Pattern, exclusion: re.Pattern | None
) -> pd.Series:
    """Matches of ``pattern`` in each text, after the exclusions are blanked."""
    return texts.map(lambda t: len(pattern.findall(blank(t, exclusion))))


def keyword_hits(frame: pd.DataFrame, text: str = "text") -> pd.DataFrame:
    """Hits per speech for every list: ``<list>_hits`` (count) and ``<list>``
    (0/1). ``other_party`` matches the terms for the speaker's party."""
    out = pd.DataFrame(index=frame.index)
    texts = frame[text].fillna("")
    for name in [*KEYWORD_LISTS, "other_party"]:
        if name == "other_party":
            counts = pd.Series(0, index=frame.index)
            for party in _PARTIES:
                rows = frame["party"] == party
                pattern, exclusion = _patterns(name, party)
                counts[rows] = _count(texts[rows], pattern, exclusion)
        else:
            counts = _count(texts, *_patterns(name))
        out[f"{name}_hits"] = counts.astype(int)
        out[name] = (counts > 0).astype(int)
    return out


def keyword_sample(
    frame: pd.DataFrame,
    name: str,
    n: int = KEYWORD_CHECK_SIZE,
    seed: int = RANDOM_SEED,
    context: int = KEYWORD_CONTEXT,
    text: str = "text",
) -> pd.DataFrame:
    """``n`` random hits of list ``name`` for a hand check of precision.

    First ``n`` speeches with a hit are drawn, then one hit within each, so
    every speech weighs the same. Each row shows ``context`` characters on
    either side of the hit, and an empty ``correct`` column for the coder.
    """
    rng = np.random.default_rng(seed)
    hits = frame[frame[f"{name}_hits"] > 0]
    picked = hits.iloc[
        np.sort(rng.choice(len(hits), size=min(n, len(hits)), replace=False))
    ]
    rows = []
    for row in picked.to_dict("records"):
        pattern, exclusion = _patterns(
            name, row["party"] if name == "other_party" else None
        )
        matches = list(pattern.finditer(blank(row[text], exclusion)))
        match = matches[rng.integers(0, len(matches))]
        low, high = max(0, match.start() - context), match.end() + context
        rows.append(
            {
                "list": name,
                "speech_id": row["speech_id"],
                "date": row["date"],
                "party": row["party"],
                "chamber": row["chamber"],
                "speaker": row.get("bioname") or row.get("last_name"),
                "match": match.group(),
                "snippet": row[text][low:high].replace("\n", " "),
                "correct": "",
            }
        )
    return pd.DataFrame(rows)


def write_keyword_check(sample: pd.DataFrame, path: Path, definition: str) -> Path:
    """Write a check sheet: a first line ``# correct = <definition>``, then the
    CSV. Refuses to overwrite a sheet, which may already hold a coder's work.

    Raises:
        FileExistsError: ``path`` exists.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; a coder may have filled it in")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# correct = {definition} (1 = yes, 0 = no)\n")
        sample.to_csv(handle, index=False)
    return path


def read_keyword_check(path: Path) -> pd.DataFrame:
    """A filled-in check sheet, without its definition line."""
    return pd.read_csv(path, skiprows=1)


def keyword_precision(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of true hits per list from filled-in sheets (``correct`` 1/0).

    Raises:
        ValueError: a sheet has a ``correct`` value that is not 0 or 1.
    """
    rows = []
    for name, sheet in sheets.items():
        correct = pd.to_numeric(sheet["correct"], errors="coerce")
        if not correct.isin([0, 1]).all():
            raise ValueError(f"{name}: 'correct' must be 0 or 1 in every row")
        rows.append({"list": name, "n_checked": len(correct), "n_correct": int(correct.sum()),
                     "precision": float(correct.mean())})  # fmt: skip
    return pd.DataFrame(rows)


def keyword_shares(
    frame: pd.DataFrame,
    flags: pd.DataFrame,
    lists: Sequence[str],
    by: Sequence[str],
) -> pd.DataFrame:
    """Share of speeches with a hit per list and group, clustered by member.

    Columns as `analysis.cell_means` (``mean`` is the share), plus ``list``
    and ``per_10k_words``: hits per 10,000 words in the group (a ratio of
    sums, no interval), which does not depend on how long speeches are.
    """
    data = frame.join(flags)
    parts = []
    for name in lists:
        shares = cell_means(data, name, by=by)
        sums = data.groupby(list(by))[[f"{name}_hits", "word_count"]].sum()
        rate = (1e4 * sums[f"{name}_hits"] / sums["word_count"]).rename("per_10k_words")
        parts.append(
            shares.merge(rate.reset_index(), on=list(by), how="left").assign(list=name)
        )
    return pd.concat(parts, ignore_index=True)


def keyword_counts_by_year(
    flags: pd.DataFrame, years: pd.Series, lists: Sequence[str]
) -> pd.DataFrame:
    """Speeches with a hit per list and year (the control that no list is empty)."""
    return flags[list(lists)].groupby(years.rename("year")).sum().reset_index()


def keyword_opposition_effects(
    frame: pd.DataFrame, flags: pd.DataFrame, lists: Sequence[str]
) -> pd.DataFrame:
    """Out of the White House against in, net of the trend, per list.

    Notebook 07's model A (A6) on the 0/1 hit flag: per chamber and party, OLS
    on the Congress number and ``opposition``, member-clustered. The
    ``opposition`` coefficient is the difference in the share of speeches
    with a hit.
    """
    data = frame.join(flags).assign(opposition=frame["in_opposition"].astype(float))
    fits = []
    for name in lists:
        fit = within_cell_ols(
            data, name, ["congress_number", "opposition"], cell=("chamber", "party"),
            by=("chamber", "party"),
        )  # fmt: skip
        fits.append(fit.assign(list=name))
    return pd.concat(fits, ignore_index=True)


# --- 13. swing states --------------------------------------------------------------------


def state_two_party_share(path: Path = COUNTYPRES_PATH) -> pd.DataFrame:
    """Democratic share of the two-party presidential vote per state and year.

    Some counties report votes by mode (election day, absentee, ...) beside or
    instead of a TOTAL row: a county's TOTAL row is used where it has one,
    and its modes are summed otherwise (data/raw/external/README.md).
    """
    data = pd.read_csv(path, dtype={"county_fips": str})
    data["mode"] = data["mode"].fillna("")
    county = data["county_fips"].fillna("name:" + data["county_name"])
    key = [data["year"], data["state_po"], county]
    has_total = data["mode"].eq("TOTAL").groupby(key).transform("any")
    kept = data[~has_total | data["mode"].eq("TOTAL")]
    votes = (
        kept[kept["party"].isin(["DEMOCRAT", "REPUBLICAN"])]
        .groupby(["year", "state_po", "party"])["candidatevotes"]
        .sum()
        .unstack("party")
    )
    out = pd.DataFrame(
        {
            "dem_votes": votes["DEMOCRAT"],
            "rep_votes": votes["REPUBLICAN"],
            "d_share": votes["DEMOCRAT"] / (votes["DEMOCRAT"] + votes["REPUBLICAN"]),
        }
    ).reset_index()
    return out.rename(columns={"state_po": "state"})


def state_margins(shares: pd.DataFrame, congresses: Sequence[int]) -> pd.DataFrame:
    """Each state's presidential margin before each Congress.

    ``margin`` = mean |D two-party share - 0.5| over the last two presidential
    elections held before the Congress's first year (one where the data hold
    only one, ``n_elections``). Small = competitive. ``tercile`` splits the
    states of each Congress into close, middle and safe (DC left out: no
    senators).
    """
    shares = shares[shares["state"] != "DC"]
    rows = []
    for congress in congresses:
        start = congress_days(congress)[0].year
        years = sorted(y for y in shares["year"].unique() if y < start)[-2:]
        part = shares[shares["year"].isin(years)]
        margin = (part["d_share"] - 0.5).abs().groupby(part["state"]).mean()
        for state, value in margin.items():
            rows.append({"state": state, "congress_number": congress, "margin": value,
                         "n_elections": len(years), "elections": "+".join(map(str, years))})  # fmt: skip
    out = pd.DataFrame(rows)
    out["tercile"] = out.groupby("congress_number")["margin"].transform(
        lambda m: pd.qcut(m, 3, labels=list(MARGIN_TERCILES)).astype(str)
    )
    return out


def swing_state_effects(senate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Senators by their state's margin: tercile means and a linear term.

    ``senate`` holds Senate speeches with ``margin``, ``tercile``,
    ``extremity_vs_cell`` and ``tone_vs_cell`` (each relative to the party,
    chamber and Congress). Returns the tercile means per party (clustered by
    member) and, per party, the OLS slope of extremity and tone on the margin
    in units of 10 points (0.1), within party x Congress cells.
    """
    means, slopes = [], []
    data = senate.assign(margin_10=senate["margin"] * 10)
    for value in VALUES:
        means.append(
            cell_means(
                data, f"{value}_vs_cell", by=("chamber", "party", "tercile")
            ).assign(value=value)
        )
        slopes.append(
            within_cell_ols(data, value, ["margin_10"], by=("chamber", "party")).assign(
                value=value
            )
        )
    return pd.concat(means, ignore_index=True), pd.concat(slopes, ignore_index=True)


# --- D. the gap's jumps at each change of the White House -----------------------------


def white_house_changes(
    congresses: Sequence[int],
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Consecutive Congresses in ``congresses`` between which the president's
    party changed, as `gap_decomposition` comparisons ("110→111": ...)."""
    shown = sorted(c for c in congresses if c in PRESIDENTS_BY_CONGRESS)
    return {
        f"{a}→{b}": ((a,), (b,))
        for a, b in pairwise(shown)
        if b == a + 1 and PRESIDENTS_BY_CONGRESS[a][1] != PRESIDENTS_BY_CONGRESS[b][1]
    }


# --- 1. common ground -------------------------------------------------------------------


def _common_ground_cell(d: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """Crossing shares and overlap of two score samples.

    Strict inequalities: a speech at the other party's median is not across
    it. The overlap coefficient sum_v min(p_D(v), p_R(v)) is taken over the
    observed values, so it is exact for scores on a grid (0 none, 1 same).
    """
    if not len(d) or not len(r):
        return {
            "d_right_of_r_median": np.nan,
            "r_left_of_d_median": np.nan,
            "overlap": np.nan,
        }
    values = np.round(np.concatenate([d, r]), 6)
    grid, codes = np.unique(values, return_inverse=True)
    p_d = np.bincount(codes[: len(d)], minlength=len(grid)) / len(d)
    p_r = np.bincount(codes[len(d) :], minlength=len(grid)) / len(r)
    return {
        "d_right_of_r_median": float(np.mean(d > np.median(r))),
        "r_left_of_d_median": float(np.mean(r < np.median(d))),
        "overlap": float(np.minimum(p_d, p_r).sum()),
    }


def common_ground(
    frame: pd.DataFrame,
    by: Sequence[str] = ("congress_number", "chamber"),
    value: str = "ideology",
) -> pd.DataFrame:
    """Per group: share of D speeches right of the R median, of R speeches left
    of the D median, and the overlap of the two distributions. Long format,
    one row per group and ``measure``, with ``n_D`` and ``n_R``."""
    rows = []
    data = frame.dropna(subset=[value])
    for key, group in data.groupby(list(by)):
        d = group.loc[group["party"] == "D", value].to_numpy(dtype=float)
        r = group.loc[group["party"] == "R", value].to_numpy(dtype=float)
        labels = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for measure, number in _common_ground_cell(d, r).items():
            rows.append(
                {
                    **labels,
                    "measure": measure,
                    "value": number,
                    "n_D": len(d),
                    "n_R": len(r),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_common_ground(
    frame: pd.DataFrame,
    by: Sequence[str] = ("congress_number", "chamber"),
    value: str = "ideology",
    n_draws: int = DEEP_DIVE_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """`common_ground` with 95% percentile intervals from a member bootstrap
    (`_bootstrap`: members drawn within party, per chamber)."""
    return _bootstrap(
        frame.dropna(subset=[value]),
        lambda f: common_ground(f, by, value),
        (*by, "measure"),
        "value",
        n_draws,
        seed,
    )


# --- G. tone per topic block ------------------------------------------------------------


def topic_party_values(fit, chamber: str) -> pd.DataFrame:
    """Each cell of `fit_topic_model` as a party's mean per block and Congress,
    with its member-clustered 95% interval (columns as `analysis.cell_means`,
    without n)."""
    rows = []
    for name in fit.params.index:
        party, block, congress = name.split(":")
        mean, se = float(fit.params[name]), float(fit.bse[name])
        rows.append({"congress_number": int(congress), "chamber": chamber, "topic_block": block,
                     "party": party, "mean": mean, "se": se,
                     "ci_low": mean - Z95 * se, "ci_high": mean + Z95 * se})  # fmt: skip
    return pd.DataFrame(rows).sort_values(
        ["topic_block", "party", "congress_number"], ignore_index=True
    )


def topic_party_change_tests(
    fit, chamber: str, congresses: tuple[int, int]
) -> pd.DataFrame:
    """Did each party's value in each block change between two Congresses?

    One row per party and block (change, member-clustered SE, interval, p),
    and per party a row ``topic_block = "all (joint)"`` with the Wald test that
    every block changed by the same amount.

    Raises:
        ValueError: a block has no speeches for a party in either Congress.
    """
    base, compare = congresses
    blocks = list(TOPIC_BLOCKS)
    rows = []
    for party in _PARTIES:
        weights = {
            b: {_cell_name(party, b, compare): 1.0, _cell_name(party, b, base): -1.0}
            for b in blocks
        }
        for block in blocks:
            result = cell_contrast(fit, weights[block])
            rows.append({"chamber": chamber, "party": party, "topic_block": block,
                         "span": f"{base}-{compare}", "change": result["estimate"],
                         **{k: result[k] for k in ("se", "ci_low", "ci_high", "p")}})  # fmt: skip
        first = _contrast(fit, weights[blocks[0]])
        matrix = np.array([_contrast(fit, weights[b]) - first for b in blocks[1:]])
        wald = fit.wald_test(matrix, scalar=True, use_f=False)
        rows.append({"chamber": chamber, "party": party, "topic_block": "all (joint)",
                     "span": f"{base}-{compare}", "chi2": float(wald.statistic),
                     "df": len(blocks) - 1, "p": float(wald.pvalue)})  # fmt: skip
    return pd.DataFrame(rows)


# --- 3. naming the other party: tone -----------------------------------------------------


def naming_tone(
    frame: pd.DataFrame, flags: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tone of speeches that name the other party against those that do not.

    Returns the raw means per chamber, party and ``names_other`` (clustered by
    member), and the difference within party x chamber x Congress cells
    (`within_cell_ols`, clustered by member).
    """
    data = frame.assign(names_other=flags["other_party"])
    raw = cell_means(data, "tone", by=("chamber", "party", "names_other"))
    within = within_cell_ols(data, "tone", ["names_other"], by=("chamber", "party"))
    return raw, within


# --- 11. laws and the gap ----------------------------------------------------------------


def load_public_laws(path: Path = PUBLIC_LAWS_PATH) -> pd.DataFrame:
    """Public laws enacted per Congress (data/raw/external), with ``complete``."""
    laws = pd.read_csv(path)
    laws["complete"] = laws["complete"].astype(str).str.lower().eq("true")
    return laws[["congress_number", "public_laws", "complete"]]


def fisher_interval(r: float, n: int, controls: int = 0) -> tuple[float, float]:
    """95% interval of a Pearson (or partial, with ``controls``) correlation by
    Fisher's z, SE = 1 / sqrt(n - 3 - controls)."""
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3 - controls)
    return float(np.tanh(z - Z95 * se)), float(np.tanh(z + Z95 * se))


def laws_gap_correlation(table: pd.DataFrame) -> pd.DataFrame:
    """Correlation of the party gap with the laws enacted, per chamber.

    ``table`` holds ``congress_number``, ``chamber``, ``gap``, ``public_laws``
    and ``complete``; only complete Congresses enter. ``raw``: Pearson r;
    ``detrended``: r of the two series' residuals from a linear trend in the
    Congress number (one control in the Fisher interval). Descriptive.
    """
    rows = []
    for chamber, part in table[table["complete"]].groupby("chamber"):
        x = part["congress_number"].to_numpy(dtype=float)
        gap = part["gap"].to_numpy(dtype=float)
        laws = part["public_laws"].to_numpy(dtype=float)
        n = len(part)
        series = {
            "raw": (gap, laws, 0),
            "detrended": (_residual(gap, x), _residual(laws, x), 1),
        }
        for kind, (a, b, controls) in series.items():
            r = float(np.corrcoef(a, b)[0, 1])
            low, high = fisher_interval(r, n, controls)
            rows.append({"chamber": chamber, "kind": kind, "r": r, "ci_low": low,
                         "ci_high": high, "n_congresses": n,
                         "congresses": f"{int(x.min())}-{int(x.max())}"})  # fmt: skip
    return pd.DataFrame(rows)


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(x, y, 1)
    return y - (intercept + slope * x)


# =================================================================================
# Notebook 10: designs, topics over time, Voteview
# =================================================================================

VOTEVIEW_DIR: Path = DW_NOMINATE_MEMBERS.parent
ROLLCALLS_PATH: Path = VOTEVIEW_DIR / "HSall_rollcalls.csv"
VOTES_PATH: Path = VOTEVIEW_DIR / "HSall_votes.csv"

#: The four policy blocks. "other" is a remainder, not a policy area, so it is
#: left out of the tests whether the blocks differ.
POLICY_BLOCKS: tuple[str, ...] = tuple(b for b in TOPIC_BLOCKS if b != "other")

#: Window of the White House event study, in Congresses from the new
#: president's first (k = 0), and the reference Congress.
EVENT_STUDY_WINDOW: tuple[int, int] = (-2, 1)
EVENT_STUDY_REFERENCE: int = -1

#: Terciles of a senator's own-party margin, low (state leans to the other
#: party) to high (stronghold of their own).
OWN_MARGIN_TERCILES: tuple[str, ...] = ("low", "middle", "high")

#: Bands of |member mean| that count as near the centre (analysis 9); 0.2 is
#: notebook 09's (CENTRIST_BAND).
CENTRIST_BANDS: tuple[float, ...] = (0.15, CENTRIST_BAND, 0.25)

# Voteview codes independents 328. As in the corpus (D2, D18), each counts with
# the party they caucus with: the speech-side rules (CAUCUS_PARTY via the
# crosswalk, GOVINFO_CAUCUS_PARTY) plus two members who never appear as "I" in
# the corpus but do in Voteview, keyed on bioguide id. Excluded, as in the
# corpus: no caucus (D8, D18). A 328 member no rule covers stops the build (D4).
VOTEVIEW_EXTRA_CAUCUS: dict[str, str] = {
    "G000280": "R",  # Virgil Goode (VA, House, 107th): Independent, caucused R
    "K000401": "R",  # Kevin Kiley (CA, House, 119th): caucuses R (D18)
}
VOTEVIEW_EXCLUDED: frozenset[str] = frozenset(
    {"B001237"}  # Dean Barkley (MN, Senate, 107th): no caucus (D8)
)

#: Voteview cast codes: 1-3 yea (yea, paired yea, announced yea), 4-6 nay;
#: 7-9 present or not voting, 0 not a member.
CAST_YEA: tuple[int, ...] = (1, 2, 3)
CAST_NAY: tuple[int, ...] = (4, 5, 6)

#: A roll call is near-unanimous when the yea share of yea + nay is at least
#: this, or at most one minus it.
NEAR_UNANIMOUS: float = 0.9

#: Roll calls that aim to undo something: whole words in vote_desc,
#: vote_question or dtl_desc, case ignored. "rescission(s)" deliberately not.
ROLLCALL_UNDO_TERMS: tuple[str, ...] = (
    "repeal", "repeals", "repealed", "repealing",
    "disapproval", "disapproving",
    "overturn", "overturns", "overturned", "overturning",
    "nullify", "nullifies", "nullified",
)  # fmt: skip
ROLLCALL_TEXT_COLUMNS: tuple[str, ...] = ("vote_desc", "vote_question", "dtl_desc")
#: Blanked before matching (decided at HALT 2, 2026-11-03, from the hand
#: check): disapprovals of arms sales and exports, of a presidential report,
#: and resolutions that only express disapproval. Each phrase is anchored on
#: the disapproval it qualifies, so blanking it removes that hit.
ROLLCALL_UNDO_EXCLUDE: tuple[str, ...] = (
    r"disapprov\w*\s+(?:of\s+)?(?:the\s+|a\s+)?proposed\s+(?:foreign\s+military\s+sales?|exports?|sales?)",
    r"disapprov\w*\s+(?:of\s+)?(?:the\s+|a\s+)?presidential\s+report",
    r"expressing\s+(?:its\s+|the\s+)?disapproval",
)
#: Congressional Review Act resolutions as the brief names them, by the title
#: the Act prescribes (kept to show what the wider rule adds).
CRA_TITLE_PATTERN: re.Pattern = re.compile(
    r"providing\s+for\s+congressional\s+disapproval\s+under\s+chapter\s+8\s+of\s+title\s+5",
    re.IGNORECASE,
)
#: The wider rule (HALT 2): a joint resolution whose text, after the
#: exclusions, names chapter 8 of title 5, or a disapproval with "rule" within
#: 40 characters after it; not a disapproval of an act of the DC Council.
CRA_RULE_PATTERN: re.Pattern = re.compile(
    r"chapter\s+8\s+of\s+title\s+5|(?:congressional\s+disapproval|disapproving).{0,40}?\brule\b",
    re.IGNORECASE | re.DOTALL,
)
DC_COUNCIL_PATTERN: re.Pattern = re.compile(
    r"district\s+of\s+columbia\s+council|council\s+of\s+the\s+district\s+of\s+columbia",
    re.IGNORECASE,
)
JOINT_RESOLUTION_PATTERN: re.Pattern = re.compile(r"^(?:HJRES|SJRES)\d+$")
ROLLCALL_UNDO_DEFINITION: str = (
    "the vote aims to repeal or overturn an existing law, regulation or rule"
)


# --- 1. do the four policy blocks differ? ------------------------------------------------


def _party_change_weights(
    party: str, block: str, base: int, compare: int
) -> dict[str, float]:
    return {
        _cell_name(party, block, compare): 1.0,
        _cell_name(party, block, base): -1.0,
    }


def policy_block_tests(
    fit,
    chamber: str,
    congresses: tuple[int, int],
    party: str | None = None,
    blocks: Sequence[str] = POLICY_BLOCKS,
) -> pd.DataFrame:
    """Did the policy blocks change by different amounts between two Congresses?

    ``fit`` is `fit_topic_model`. Without ``party`` the change is in R - D
    (weights as `topic_gap_tests`); with it, in that party's value (as
    `topic_party_change_tests`). Rows by ``kind``:

    - ``block``: each block's change, member-clustered SE, interval, p;
    - ``joint``: the Wald test that all ``blocks`` changed alike
      (chi-squared, one degree of freedom fewer than blocks);
    - ``pair``: each pair's difference in change, its p, and ``p_holm``,
      Holm-adjusted over the pairs.

    Raises:
        ValueError: a cell has no speeches (`cell_contrast`).
    """
    base, compare = congresses
    blocks = list(blocks)
    weights = {
        b: (
            _change_weights(b, base, compare)
            if party is None
            else _party_change_weights(party, b, base, compare)
        )
        for b in blocks
    }
    label = {"chamber": chamber, "party": party or "R−D", "span": f"{base}-{compare}"}
    keep = ("se", "ci_low", "ci_high", "p")
    rows = []
    for block in blocks:
        result = cell_contrast(fit, weights[block])
        rows.append({**label, "kind": "block", "topic_block": block,
                     "change": result["estimate"], **{k: result[k] for k in keep}})  # fmt: skip
    first = _contrast(fit, weights[blocks[0]])
    matrix = np.array([_contrast(fit, weights[b]) - first for b in blocks[1:]])
    wald = fit.wald_test(matrix, scalar=True, use_f=False)
    rows.append({**label, "kind": "joint", "topic_block": "policy blocks (joint)",
                 "chi2": float(wald.statistic), "df": len(blocks) - 1,
                 "p": float(wald.pvalue)})  # fmt: skip
    pairs = []
    for a, b in combinations(blocks, 2):
        difference = dict(weights[a])
        for name, weight in weights[b].items():
            difference[name] = difference.get(name, 0.0) - weight
        result = cell_contrast(fit, difference)
        pairs.append({**label, "kind": "pair", "topic_block": f"{a} − {b}",
                      "change": result["estimate"], **{k: result[k] for k in keep}})  # fmt: skip
    adjusted = multipletests([p["p"] for p in pairs], method="holm")[1]
    for pair, p_holm in zip(pairs, adjusted):
        pair["p_holm"] = float(p_holm)
    return pd.DataFrame(rows + pairs)


# --- 2. which topic carried the growth, and when? ---------------------------------------

#: Key of a step-decomposition row.
STEP_KEY: tuple[str, ...] = ("comparison", "chamber", "component", "topic_block")


def consecutive_steps(
    congresses: Sequence[int],
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Every step from one Congress to the next, as `gap_decomposition`
    comparisons ("107→108": ((107,), (108,)), ...)."""
    shown = sorted(set(congresses))
    return {f"{a}→{b}": ((a,), (b,)) for a, b in pairwise(shown) if b == a + 1}


def step_periods(
    steps: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> dict[str, str]:
    """Each step's period: the presidential term it lies within
    (`analysis.add_presidency`'s ``term``, e.g. "Bush 2001-2008"), or, where
    the president's party changes, "switch 110→111"."""
    congresses = sorted({c for a, b in steps.values() for c in (*a, *b)})
    terms = add_presidency(
        pd.DataFrame({"congress_number": congresses, "party": "D"})
    ).set_index("congress_number")["term"]
    out = {}
    for name, ((a,), (b,)) in steps.items():
        if PRESIDENTS_BY_CONGRESS[a][1] != PRESIDENTS_BY_CONGRESS[b][1]:
            out[name] = f"switch {name}"
        elif terms[a] == terms[b]:
            out[name] = terms[a]
        else:
            out[name] = f"{terms[a]} → {terms[b]}"
    return out


def step_decomposition(
    policy: pd.DataFrame,
    steps: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
    value: str = "ideology",
) -> pd.DataFrame:
    """`gap_decomposition` for each step, plus two components per block:

    - ``contribution``: positions + agenda, the block's part of the change in
      the policy-debate gap (weighted by its share of the debates);
    - ``within``: the change in R - D within the block itself, unweighted
      (how much the block polarized, whatever its size).

    Rows as `gap_decomposition` (``comparison``, ``chamber``, ``component``,
    ``topic_block``, ``value``).
    """
    decomposition = gap_decomposition(policy, steps, value)
    per_block = decomposition[decomposition["topic_block"] != "all"]
    contribution = (
        per_block.groupby(["comparison", "chamber", "topic_block"], as_index=False)[
            "value"
        ]
        .sum()
        .assign(component="contribution")
    )
    cells = block_cells(policy, value)
    means = cells.pivot_table(
        index=["chamber", "congress_number", "topic_block"],
        columns="party",
        values="mean",
    ).reindex(columns=list(_PARTIES))
    gap = (means["R"] - means["D"]).to_dict()
    rows = []
    for name, (base, compare) in steps.items():
        (a,), (b,) = base, compare
        for chamber in sorted(cells["chamber"].unique()):
            for block in TOPIC_BLOCKS:
                change = gap.get((chamber, b, block), np.nan) - gap.get(
                    (chamber, a, block), np.nan
                )
                rows.append({"comparison": name, "chamber": chamber, "component": "within",
                             "topic_block": block, "value": change})  # fmt: skip
    return pd.concat(
        [decomposition, contribution, pd.DataFrame(rows)], ignore_index=True
    )


def bootstrap_step_decomposition(
    policy: pd.DataFrame,
    steps: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
    value: str = "ideology",
    n_draws: int = DEEP_DIVE_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`step_decomposition` with 95% percentile intervals, and its draws.

    The member bootstrap of `_bootstrap` (members drawn within party, per
    chamber, `cluster_draws`): each draw is taken once and serves every step,
    and the draws do not depend on the steps, so a step's interval is the one
    `bootstrap_gap_decomposition` gives for it. The draws are returned (with
    ``draw``) because sums over steps need them.
    """
    frame = policy.dropna(subset=[value, "topic_block"])
    point = step_decomposition(frame, steps, value)
    draws = []
    for _, part in frame.groupby("chamber"):
        part = part.reset_index(drop=True)
        for i, idx in enumerate(
            cluster_draws(part["member_key"], part["party"], n_draws, seed)
        ):
            draws.append(
                step_decomposition(part.iloc[idx], steps, value).assign(draw=i)
            )
    stacked = pd.concat(draws, ignore_index=True)
    return _percentile_intervals(point, [stacked], STEP_KEY, "value"), stacked


def _sum_by(frame: pd.DataFrame, key: Sequence[str]) -> pd.DataFrame:
    """``value`` summed per ``key``; null where any summand is null."""
    groups = [frame[c] for c in key]
    total = frame["value"].groupby(groups).sum()
    missing = frame["value"].isna().groupby(groups).any()
    return total.where(~missing).rename("value").reset_index()


def _bootstrap_wald(point: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    """Wald test that all entries of ``point`` are equal, with the covariance
    of their differences from the first taken over ``draws`` (rows)."""
    difference = point[1:] - point[0]
    spread = draws[:, 1:] - draws[:, [0]]
    covariance = np.atleast_2d(np.cov(spread, rowvar=False))
    statistic = float(difference @ np.linalg.solve(covariance, difference))
    return statistic, float(chi2.sf(statistic, len(difference)))


def period_sums(
    point: pd.DataFrame,
    draws: pd.DataFrame,
    periods: Mapping[str, str],
    blocks: Sequence[str] = POLICY_BLOCKS,
) -> pd.DataFrame:
    """Steps summed within periods (`step_periods`).

    Rows ``kind = "sum"``: per period, chamber, component and block, the
    summed ``value`` with a 95% percentile interval over the summed draws,
    ``n_steps``, ``share_of_period`` (of the period's total change; for
    positions, agenda and contribution) and ``share_of_growth`` (of the total
    change over all steps). Rows ``kind = "joint"``: per period, chamber and
    component (``contribution``, ``within``), a Wald test that the ``blocks``
    are equal, with the bootstrap covariance (chi-squared, blocks - 1 df).
    Steps not in ``periods`` are left out.
    """
    key = ["period", "chamber", "component", "topic_block"]

    def summed(frame: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
        data = frame.assign(period=frame["comparison"].map(periods))
        return _sum_by(data[data["period"].notna()], [*extra, *key])

    sums = summed(point)
    drawn = summed(draws, ("draw",))
    bounds = (
        drawn.groupby(key)["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
        .reset_index()
    )
    sums = sums.merge(bounds, on=key, how="left", validate="one_to_one")
    counts = pd.Series(list(periods.values())).value_counts()
    sums["n_steps"] = sums["period"].map(counts)
    totals = sums[sums["component"] == "total"].set_index(["period", "chamber"])[
        "value"
    ]
    period_total = [
        totals.get((p, c), np.nan) for p, c in zip(sums["period"], sums["chamber"])
    ]
    sums["share_of_period"] = np.where(
        sums["component"].isin(["positions", "agenda", "contribution"]),
        sums["value"] / np.asarray(period_total, dtype=float),
        np.nan,
    )
    grand = point[point["component"] == "total"].groupby("chamber")["value"].sum()
    sums["share_of_growth"] = sums["value"] / sums["chamber"].map(grand)
    sums["kind"] = "sum"
    joint = []
    for (period, chamber), _ in sums.groupby(["period", "chamber"]):
        for component in ("contribution", "within"):
            mine = sums[(sums["period"] == period) & (sums["chamber"] == chamber)
                        & (sums["component"] == component)]  # fmt: skip
            values = mine.set_index("topic_block")["value"].reindex(list(blocks))
            wide = drawn[(drawn["period"] == period) & (drawn["chamber"] == chamber)
                         & (drawn["component"] == component)]  # fmt: skip
            wide = wide.pivot(index="draw", columns="topic_block", values="value")
            wide = wide.reindex(columns=list(blocks)).dropna()
            statistic, p = _bootstrap_wald(values.to_numpy(float), wide.to_numpy(float))
            joint.append({"period": period, "chamber": chamber, "component": component,
                          "topic_block": "policy blocks (joint)", "kind": "joint",
                          "chi2": statistic, "df": len(blocks) - 1, "p": p,
                          "n_draws": len(wide)})  # fmt: skip
    return pd.concat([sums, pd.DataFrame(joint)], ignore_index=True)


# --- 3. a majority lost in one chamber only -----------------------------------------------


def chamber_flip_cases(congresses: Sequence[int]) -> pd.DataFrame:
    """Steps t -> t+1 in which exactly one chamber changes hands while the
    White House stays with one party.

    The majority on the last day of t is compared with that on the first day
    of t+1 (`chamber_majority`, so the Senate's changes within a Congress are
    read by date). ``status`` is "robustness" where the flipping chamber also
    changed hands within t or t+1 (the 107th Senate), else "main"; ``note``
    flags a control chamber that changed hands within a Congress.
    """
    rows = []
    for t, u in pairwise(sorted(set(congresses))):
        if u != t + 1 or PRESIDENTS_BY_CONGRESS[t][1] != PRESIDENTS_BY_CONGRESS[u][1]:
            continue
        flips = {}
        for chamber in ("H", "S"):
            before = chamber_majority(
                chamber, t, congress_days(t)[1] - dt.timedelta(days=1)
            )
            after = chamber_majority(chamber, u, congress_days(u)[0])
            if before != after:
                flips[chamber] = (before, after)
        if len(flips) != 1:
            continue
        ((chamber, (loses, gains)),) = flips.items()
        other = "S" if chamber == "H" else "H"
        changed = [c for c in (t, u) if c in SENATE_MAJORITY_CHANGES]
        impure = chamber == "S" and bool(changed)
        note = (
            f"control chamber (Senate) changed hands within the {changed[0]}th"
            if other == "S" and changed
            else ""
        )
        rows.append({"case": f"{t}→{u}", "before": t, "after": u, "chamber": chamber,
                     "loses": loses, "gains": gains,
                     "president_party": PRESIDENTS_BY_CONGRESS[t][1],
                     "status": "robustness" if impure else "main", "note": note})  # fmt: skip
    return pd.DataFrame(rows)


def chamber_flip_data(
    frame: pd.DataFrame,
    cases: pd.DataFrame,
    served: set[tuple[int, int, str]],
    role: str = "loses",
    same_chamber: bool = True,
    stanford_only: Sequence[int] = (SOURCE_BREAK_CONGRESS,),
    placebo: bool = False,
) -> pd.DataFrame:
    """Speeches for the chamber-flip difference-in-differences, stacked by case.

    Per case: the speeches of the party in ``role`` ("loses" or "gains" the
    flipping chamber) in t and t+1 (``placebo``: t-1 and t, the same groups a
    Congress earlier; a case with no t-1 in ``frame`` is skipped), in both
    chambers. ``treated`` = the flipping chamber, ``post`` = the later
    Congress. ``same_chamber`` keeps members who sat in the same chamber in
    both Congresses (``served``: `_served` of `chamber_membership`), so the
    comparison is within the same people. In the Congresses of
    ``stanford_only`` only Stanford speeches are kept, so no case reaches
    across the 2016-09-10 source break.
    """
    parts = []
    available = set(frame["congress_number"])
    for case in cases.to_dict("records"):
        t, u = case["before"], case["after"]
        if placebo:
            t, u = t - 1, t
        if t not in available or u not in available:
            continue
        data = frame[
            (frame["party"] == case[role]) & frame["congress_number"].isin((t, u))
        ]
        data = data[
            ~(data["congress_number"].isin(list(stanford_only)) & (data["source"] == "govinfo"))
        ]  # fmt: skip
        if same_chamber:
            icpsr = data["icpsr"].astype("Int64")
            both = [
                not pd.isna(i) and (int(i), t, c) in served and (int(i), u, c) in served
                for i, c in zip(icpsr, data["chamber"])
            ]
            data = data[np.asarray(both, dtype=bool)]
        data = data.assign(
            case=case["case"],
            post=(data["congress_number"] == u).astype(float),
            treated=(data["chamber"] == case["chamber"]).astype(float),
        )
        data["post_treated"] = data["post"] * data["treated"]
        data["member_case"] = data["member_key"] + "|" + case["case"]
        parts.append(data)
    if not parts:
        columns = ["case", "post", "treated", "post_treated", "member_case"]
        return frame.iloc[0:0].assign(**{c: pd.Series(dtype=object) for c in columns})
    return pd.concat(parts, ignore_index=True)


def chamber_flip_did(data: pd.DataFrame, value: str) -> pd.DataFrame:
    """Difference-in-differences of ``value`` on `chamber_flip_data`.

    Rows by ``kind``, all clustered by member:

    - ``case``: per case, member fixed effects + post + post x treated; the
      ``post_treated`` coefficient is the change in the flipping chamber
      beyond the change of the same party in the other chamber;
    - ``own change``: per case and chamber, the change from t to t+1 within
      members (the two changes the case compares);
    - ``pooled``: all cases stacked, member x case and case x post fixed
      effects, one ``post_treated`` coefficient.
    """
    by_case = within_cell_ols(
        data, value, ["post", "post_treated"], cell=("member_key",), by=("case",)
    )
    by_case = by_case[by_case["term"] == "post_treated"].assign(kind="case")
    own = within_cell_ols(
        data, value, ["post"], cell=("member_key",), by=("case", "chamber")
    ).assign(kind="own change")
    posts = {
        f"post_{name}": ((data["case"] == name) & (data["post"] == 1)).astype(float)
        for name in sorted(data["case"].unique())
    }
    stacked = data.assign(**posts, pooled="pooled")
    pooled = within_cell_ols(
        stacked, value, [*posts, "post_treated"], cell=("member_case",), by=("pooled",)
    )
    pooled = pooled[pooled["term"] == "post_treated"].drop(columns="pooled")
    pooled = pooled.assign(kind="pooled", case="pooled")
    return pd.concat([by_case, own, pooled], ignore_index=True).assign(value=value)


# --- 4. event study around the changes of the White House --------------------------------


def wh_event_study_data(
    frame: pd.DataFrame, window: tuple[int, int] = EVENT_STUDY_WINDOW
) -> pd.DataFrame:
    """Speeches around each change of the White House, stacked by change.

    For each change in `white_house_changes`, ``k`` is the Congress minus the
    new president's first, kept from ``window[0]`` to ``window[1]`` as far as
    the data reach; a speech appears once for each change whose window holds
    its Congress. ``treated`` = the party that loses the White House.
    """
    changes = white_house_changes(sorted(frame["congress_number"].unique()))
    parts = []
    for name, (_, (first,)) in changes.items():
        losing = PRESIDENTS_BY_CONGRESS[first - 1][1]
        k = frame["congress_number"] - first
        data = frame[k.between(*window)].copy()
        data["k"] = (data["congress_number"] - first).astype(int)
        data["switch"] = name
        data["treated"] = (data["party"] == losing).astype(float)
        data["member_switch"] = data["member_key"] + "|" + name
        parts.append(data)
    return pd.concat(parts, ignore_index=True)


def _k_term(prefix: str, k: int) -> str:
    return f"{prefix}_k{k:+d}"


def wh_event_study(
    data: pd.DataFrame, value: str, reference: int = EVENT_STUDY_REFERENCE
) -> pd.DataFrame:
    """Event study of ``value`` on `wh_event_study_data`, per chamber.

    Within member x change (fixed effects), with change x k fixed effects and
    ``chamber_minority`` (as model C), the effects ``treated x k`` for every
    k but ``reference``: how much more the losing party's members changed than
    the winning party's, relative to Congress k = ``reference``. The k = -2
    effect is the pre-trend test. Clustered by member. Rows: the effects
    (with ``k``) and ``chamber_minority``.
    """
    extra, effects = {}, {}
    for (switch, k), _ in data.groupby(["switch", "k"]):
        if k != reference:
            extra[f"fe_{switch}_{k:+d}"] = (
                (data["switch"] == switch) & (data["k"] == k)
            ).astype(float)
    for k in sorted(data["k"].unique()):
        if k != reference:
            effects[_k_term("treated", int(k))] = data["treated"] * (data["k"] == k)
    stacked = data.assign(**extra, **effects)
    fit = within_cell_ols(
        stacked, value, [*extra, *effects, "chamber_minority"],
        cell=("member_switch",), by=("chamber",),
    )  # fmt: skip
    out = fit[fit["term"].isin([*effects, "chamber_minority"])].copy()
    out["k"] = [
        int(t.split("_k")[1]) if t.startswith("treated_k") else np.nan
        for t in out["term"]
    ]
    return out.assign(value=value)


def party_own_paths(
    data: pd.DataFrame, value: str, reference: int = EVENT_STUDY_REFERENCE
) -> pd.DataFrame:
    """Descriptive: each side's own path around the changes, within members.

    Per chamber and side (``role``: the party losing or gaining the White
    House), member x change fixed effects and one indicator per k but
    ``reference``, pooled over the changes; each coefficient is the change
    from k = ``reference``. Clustered by member.
    """
    terms = {
        _k_term("k", int(k)): (data["k"] == k).astype(float)
        for k in sorted(data["k"].unique())
        if k != reference
    }
    stacked = data.assign(
        **terms,
        role=np.where(data["treated"] == 1, "loses White House", "gains White House"),
    )
    fit = within_cell_ols(
        stacked, value, list(terms), cell=("member_switch",), by=("chamber", "role")
    )
    fit["k"] = [int(t.split("_k")[1]) for t in fit["term"]]
    return fit.assign(value=value)


# --- 5. naming the other party: chamber minority and opposition --------------------------


def names_power_effects(frame: pd.DataFrame, flag: str = "other_party") -> pd.DataFrame:
    """Model C of notebook 08 (`OPPOSITION_MODELS`: member fixed effects,
    linear trend, ``opposition``, ``chamber_minority``) on the 0/1 ``flag``
    (a linear probability model), per chamber and party, clustered by member.
    Each coefficient is a change in the share of speeches that name the other
    party."""
    terms, cell = OPPOSITION_MODELS["C"]
    data = frame.assign(**{flag: frame[flag].astype(float)})
    return within_cell_ols(data, flag, list(terms), cell=cell, by=("chamber", "party"))


# --- 6. swing states: the own party's margin, and within the same senator -----------------


def state_leans(shares: pd.DataFrame, congresses: Sequence[int]) -> pd.DataFrame:
    """Each state's Democratic lean before each Congress: the mean of
    (D two-party share - 0.5) over the same elections as `state_margins`
    (the last two before the Congress's first year), signed. DC left out."""
    shares = shares[shares["state"] != "DC"]
    rows = []
    for congress in congresses:
        start = congress_days(congress)[0].year
        years = sorted(y for y in shares["year"].unique() if y < start)[-2:]
        part = shares[shares["year"].isin(years)]
        lean = (part["d_share"] - 0.5).groupby(part["state"]).mean()
        for state, value in lean.items():
            rows.append({"state": state, "congress_number": congress, "d_lean": value,
                         "n_elections": len(years), "elections": "+".join(map(str, years))})  # fmt: skip
    return pd.DataFrame(rows)


def add_own_margin(senate: pd.DataFrame, leans: pd.DataFrame) -> pd.DataFrame:
    """Add ``own_margin``: the state's lean toward the senator's own party
    (D: the Democratic lean; R: minus it), so positive is a stronghold of the
    own party and negative a state leaning to the other; ``own_margin_10`` in
    units of 10 points; and ``own_tercile``, terciles over the party's
    senators in each Congress (each senator once, ties broken by order)."""
    out = senate.merge(
        leans[["state", "congress_number", "d_lean"]],
        on=["state", "congress_number"], how="left", validate="m:1",
    )  # fmt: skip
    out["own_margin"] = np.where(out["party"] == "D", out["d_lean"], -out["d_lean"])
    out["own_margin_10"] = out["own_margin"] * 10
    key = ["member_key", "congress_number", "party"]
    units = out.dropna(subset=["own_margin"]).drop_duplicates(key)[[*key, "own_margin"]]
    rank = units.groupby(["party", "congress_number"])["own_margin"].rank(
        method="first"
    )
    size = units.groupby(["party", "congress_number"])["own_margin"].transform("size")
    third = np.ceil(3 * rank / size).astype(int) - 1
    units["own_tercile"] = np.asarray(OWN_MARGIN_TERCILES)[third]
    return out.merge(
        units.drop(columns="own_margin"), on=key, how="left", validate="m:1"
    )


def own_margin_effects(senate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Senators by their own party's margin: tercile means and a linear term.

    ``senate`` is `add_own_margin` output with ``extremity_vs_cell`` and
    ``tone_vs_cell``. Returns the means per party and ``own_tercile``
    (clustered by member) and, per party, the slope of extremity and tone per
    10 points of own margin within party x Congress cells.
    """
    means, slopes = [], []
    for value in VALUES:
        means.append(
            cell_means(
                senate, f"{value}_vs_cell", by=("chamber", "party", "own_tercile")
            ).assign(value=value)
        )
        slopes.append(
            within_cell_ols(senate, value, ["own_margin_10"], by=("chamber", "party"))
            .assign(value=value)
        )  # fmt: skip
    return pd.concat(means, ignore_index=True), pd.concat(slopes, ignore_index=True)


def own_margin_within(senate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The own-margin slope within the same senator.

    Per party: senator fixed effects plus one indicator per Congress (the
    party's Congress effects), slope per 10 points, clustered by member; only
    senators whose own margin changes identify it. Returns the slopes and,
    per party, how many senators have variation and how large it is (the
    range of their own margin across Congresses, in points).
    """
    congresses = sorted(senate["congress_number"].unique())[1:]
    dummies = {
        f"congress_{c}": (senate["congress_number"] == c).astype(float)
        for c in congresses
    }
    data = senate.assign(**dummies)
    slopes = []
    for value in VALUES:
        fit = within_cell_ols(
            data, value, ["own_margin_10", *dummies], cell=("member_key",),
            by=("chamber", "party"),
        )  # fmt: skip
        slopes.append(fit[fit["term"] == "own_margin_10"].assign(value=value))
    ranges = (
        senate.dropna(subset=["own_margin"])
        .groupby(["party", "member_key"])["own_margin"]
        .agg(lambda m: 100 * (m.max() - m.min()))
    )
    varying = ranges[ranges > 1e-9]
    variation = pd.DataFrame(
        {
            "n_senators": ranges.groupby("party").size(),
            "n_with_variation": varying.groupby("party").size(),
            "median_range_points": varying.groupby("party").median(),
            "mean_range_points": varying.groupby("party").mean(),
            "max_range_points": varying.groupby("party").max(),
        }
    ).reset_index()
    return pd.concat(slopes, ignore_index=True), variation


# --- 7. topic blocks around the events ------------------------------------------------------


def event_block_cell_sizes(
    policy: pd.DataFrame, windows: pd.DataFrame, minimum: int = TOPIC_MIN_CELL
) -> pd.DataFrame:
    """Speeches and members per event, chamber, party, period and topic block.

    Every combination appears, empty ones with n = 0; ``below_min`` marks a
    cell under ``minimum``.
    """
    parts = []
    for window in windows.to_dict("records"):
        data = event_speeches(policy, window)
        index = pd.MultiIndex.from_product(
            [("H", "S"), _PARTIES, ("before", "after"), list(TOPIC_BLOCKS)],
            names=["chamber", "party", "period", "topic_block"],
        )
        sizes = (
            data.groupby(list(index.names))
            .agg(n=("speech_id", "size"), n_members=("member_key", "nunique"))
            .reindex(index, fill_value=0)
            .reset_index()
        )
        parts.append(sizes.assign(event=window["event"], days=window["days"]))
    out = pd.concat(parts, ignore_index=True)
    out["below_min"] = out["n"] < minimum
    return out


def _contrast_or_missing(fit, weights: Mapping[str, float]) -> dict[str, object]:
    """`cell_contrast`, or nulls with a note when a cell has no speeches."""
    try:
        return {**cell_contrast(fit, weights), "note": ""}
    except ValueError:
        empty = {k: np.nan for k in ("estimate", "se", "ci_low", "ci_high", "p")}
        return {**empty, "note": "no speeches in a cell"}


def event_block_effects(policy: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Change from before to after each event in every topic block.

    Per event and chamber, one saturated model (`fit_cell_model`) on party x
    period x block, clustered by member, for ideology and for tone; and per
    block a linear probability model (party x period) of a policy debate
    being in the block. Rows by ``measure``:

    - ``gap`` (``party`` "R−D"): the change in R - D within the block;
    - ``tone`` (D, R): the change in the party's tone within the block;
    - ``agenda`` (D, R): the change in the block's share of the party's
      policy debates.

    A contrast with an empty cell is null, with a ``note``.
    """
    rows = []
    for window in windows.to_dict("records"):
        data = event_speeches(policy, window)
        for chamber, part in data.groupby("chamber"):
            base = {
                "event": window["event"],
                "chamber": chamber,
                "days": window["days"],
            }
            cells = ("party", "period", "topic_block")
            fits = {
                "gap": fit_cell_model(part, cells, "ideology"),
                "tone": fit_cell_model(part, cells, "tone"),
            }
            for block in TOPIC_BLOCKS:
                gap = {}
                for party, sign in (("R", 1.0), ("D", -1.0)):
                    gap[f"{party}:after:{block}"] = sign
                    gap[f"{party}:before:{block}"] = -sign
                rows.append({**base, "measure": "gap", "party": "R−D", "topic_block": block,
                             **_contrast_or_missing(fits["gap"], gap)})  # fmt: skip
                in_block = part.assign(
                    in_block=(part["topic_block"] == block).astype(float)
                )
                share = fit_cell_model(in_block, ("party", "period"), "in_block")
                for party in _PARTIES:
                    tone = {
                        f"{party}:after:{block}": 1.0,
                        f"{party}:before:{block}": -1.0,
                    }
                    rows.append({**base, "measure": "tone", "party": party,
                                 "topic_block": block,
                                 **_contrast_or_missing(fits["tone"], tone)})  # fmt: skip
                    agenda = {f"{party}:after": 1.0, f"{party}:before": -1.0}
                    rows.append({**base, "measure": "agenda", "party": party,
                                 "topic_block": block,
                                 **_contrast_or_missing(share, agenda)})  # fmt: skip
    return pd.DataFrame(rows)


def flag_small_block_cells(
    effects: pd.DataFrame, sizes: pd.DataFrame, minimum: int = TOPIC_MIN_CELL
) -> pd.DataFrame:
    """Mark each `event_block_effects` row that rests on a cell under ``minimum``.

    ``gap`` uses both parties' block cells, ``tone`` its party's, both
    periods; ``agenda`` uses the party's policy debates in each period (all
    blocks). ``small_cell`` flags the row, ``small_cells`` names the cells.
    """
    totals = (
        sizes.groupby(["event", "chamber", "party", "period"], as_index=False)
        .agg(n=("n", "sum"), n_members=("n_members", "sum"))
        .assign(topic_block="all blocks")
    )
    flags, notes = [], []
    for row in effects.to_dict("records"):
        parties = list(_PARTIES) if row["party"] == "R−D" else [row["party"]]
        if row["measure"] == "agenda":
            used = totals[(totals["event"] == row["event"]) & (totals["chamber"] == row["chamber"])
                          & totals["party"].isin(parties)]  # fmt: skip
        else:
            used = sizes[(sizes["event"] == row["event"]) & (sizes["chamber"] == row["chamber"])
                         & sizes["party"].isin(parties)
                         & (sizes["topic_block"] == row["topic_block"])]  # fmt: skip
        small = used[used["n"] < minimum]
        flags.append(len(small) > 0)
        notes.append("; ".join(
            f"{r['party']} {r['period']}: n={r['n']}" + (
                f", {r['n_members']} members" if row["measure"] != "agenda" else "")
            for r in small.to_dict("records")))  # fmt: skip
    return effects.assign(small_cell=flags, small_cells=notes)


# --- 8. new members against continuing ones -------------------------------------------------


def cohort_vs_continuing(
    members: pd.DataFrame,
    exclude_icpsr: Sequence[int] = (),
    value: str = "extremity",
    first: int = 107,
) -> pd.DataFrame:
    """New members against the continuing members of their party, chamber and
    Congress.

    ``members`` is `analysis.add_background` output. A speech is ``new`` when
    its Congress is the speaker's first in either chamber (A4), else
    continuing; from ``first`` on; the ICPSRs in ``exclude_icpsr`` are left
    out entirely (as `cohort_extremity`). The difference new - continuing in
    ``value`` comes from `within_cell_ols` within party x chamber x Congress,
    clustered by member, at three ``level``s: per cohort (the Congress,
    ``cohort``), pooled per party x chamber, and per ``presidency``
    (`analysis.add_presidency`'s ``term``).
    ``n_new_members`` counts the new members behind each estimate. Cells with
    no new or no continuing member are left out.
    """
    icpsr = members["icpsr"].astype("Int64")
    keep = (
        members["first_congress"].notna()
        & (members["congress_number"] >= first)
        & ~icpsr.isin(list(exclude_icpsr)).fillna(False)
    )
    data = members[keep].copy()
    data["new"] = (data["congress_number"] == data["first_congress"]).astype(float)
    data["presidency"] = data["term"]
    cell = ("chamber", "party", "congress_number")
    data = data[data.groupby(list(cell))["new"].transform("nunique") > 1]
    levels = {
        "cohort": cell,
        "pooled": ("chamber", "party"),
        "presidency": ("chamber", "party", "presidency"),
    }
    parts = []
    for level, by in levels.items():
        fit = within_cell_ols(data, value, ["new"], cell=cell, by=by)
        new = (
            data[data["new"] == 1]
            .groupby(list(by))["member_key"]
            .nunique()
            .rename("n_new_members")
            .reset_index()
        )
        parts.append(fit.merge(new, on=list(by), how="left").assign(level=level))
    out = pd.concat(parts, ignore_index=True)
    return out.rename(columns={"congress_number": "cohort"})


# --- 9. the centre disappears ----------------------------------------------------------------


def _centrist_shares(units: pd.DataFrame, bands: Sequence[float]) -> pd.DataFrame:
    rows = []
    for band in bands:
        shares = (
            (units["mean"].abs() < band)
            .groupby([units["chamber"], units["party"], units["congress_number"]])
            .agg(["mean", "size"])
            .rename(columns={"mean": "value", "size": "n_members"})
            .reset_index()
        )
        rows.append(shares.assign(band=band))
    return pd.concat(rows, ignore_index=True)


def centrist_table(
    units: pd.DataFrame,
    bands: Sequence[float] = CENTRIST_BANDS,
    n_draws: int = DEEP_DIVE_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Share of member x Congress units with |mean| under each band, per
    chamber, party and Congress, with 95% percentile intervals.

    The bootstrap is `dissent_table`'s, draw for draw (members drawn with
    replacement within party, per chamber and Congress, the same random
    sequence), so the 0.2 band reproduces its ``share_centrist`` with its
    interval.
    """
    point = _centrist_shares(units, bands)
    rng = np.random.default_rng(seed)
    groups = [
        part.reset_index(drop=True)
        for _, part in units.groupby(["chamber", "congress_number"])
    ]
    draws = []
    for _ in range(n_draws):
        sample = []
        for part in groups:
            for _, rows in part.groupby("party"):
                sample.append(rows.iloc[rng.integers(0, len(rows), len(rows))])
        drawn = _centrist_shares(pd.concat(sample, ignore_index=True), bands)
        draws.append(drawn.drop(columns="n_members"))
    return _percentile_intervals(
        point, draws, ("chamber", "party", "congress_number", "band"), "value"
    )


# --- 10. Voteview: voting across party lines, and votes to undo -------------------------------


def load_rollcalls(
    path: Path = ROLLCALLS_PATH, first: int = 107, last_date: object = GOVINFO_END_DATE
) -> pd.DataFrame:
    """House and Senate roll calls from Congress ``first`` to ``last_date``
    (as D15), with the columns the analyses need and ``chamber`` as H/S."""
    columns = ["congress", "chamber", "rollnumber", "date", "bill_number",
               "vote_result", "vote_desc", "vote_question", "dtl_desc"]  # fmt: skip
    text = {c: str for c in ("bill_number", "vote_result", *ROLLCALL_TEXT_COLUMNS)}
    calls = pd.read_csv(path, usecols=columns, dtype=text)
    calls = calls[
        (calls["congress"] >= first)
        & calls["chamber"].isin(["House", "Senate"])
        & (pd.to_datetime(calls["date"]) <= pd.Timestamp(last_date))
    ].copy()
    calls["chamber"] = calls["chamber"].map({"House": "H", "Senate": "S"})
    return calls.rename(columns={"congress": "congress_number"}).reset_index(drop=True)


def voteview_caucus_party(
    members: Path = DW_NOMINATE_MEMBERS, crosswalk: Path = CROSSWALK_PATH
) -> pd.DataFrame:
    """Party of every Voteview member record, 107th on, independents assigned.

    One row per Congress, chamber (H/S) and ICPSR: ``party`` D or R, ``rule``
    (``party code``, ``caucus`` for an independent assigned as in D2/D18, or
    ``excluded``, party null) and ``independent`` (Voteview code 328).
    Delegates are left out (D10). Caucus rules by bioguide id: the corpus's
    Stanford rules (CAUCUS_PARTY, reached through the crosswalk), its govinfo
    rules (GOVINFO_CAUCUS_PARTY) and VOTEVIEW_EXTRA_CAUCUS.

    Raises:
        ValueError: a member with another party code, or an independent no
            rule covers.
    """
    voteview = load_voteview(members)
    voteview = voteview[~voteview["state_abbrev"].isin(DELEGATE_STATES)]
    links = pd.read_parquet(crosswalk, columns=["source", "member_id", "bioguide_id"])
    stanford = links[
        (links["source"] == "stanford") & links["member_id"].isin(CAUCUS_PARTY)
    ]
    caucus = {
        b: CAUCUS_PARTY[m]
        for m, b in zip(stanford["member_id"], stanford["bioguide_id"])
    }
    caucus |= GOVINFO_CAUCUS_PARTY | VOTEVIEW_EXTRA_CAUCUS
    excluded = set(GOVINFO_EXCLUDED_MEMBERS) | set(VOTEVIEW_EXCLUDED)
    codes = {VOTEVIEW_PARTY_CODES["D"]: "D", VOTEVIEW_PARTY_CODES["R"]: "R"}
    independent = voteview["party_code"] == VOTEVIEW_PARTY_CODES["I"]
    unknown = voteview[~voteview["party_code"].isin(codes) & ~independent]
    if len(unknown):
        raise ValueError(
            f"party codes with no rule: {unknown[['bioname', 'party_code']].values.tolist()}"
        )
    uncovered = voteview[
        independent & ~voteview["bioguide_id"].isin(set(caucus) | excluded)
    ]
    if len(uncovered):
        raise ValueError(
            f"independents no caucus rule covers: {sorted(set(uncovered['bioname']))}"
        )
    party = voteview["party_code"].map(codes)
    rule = np.where(independent, "caucus", "party code")
    assigned = voteview["bioguide_id"].map(caucus)
    party = party.where(~independent, assigned)
    out_rule = np.where(
        independent & voteview["bioguide_id"].isin(excluded), "excluded", rule
    )
    party = party.where(out_rule != "excluded")
    return pd.DataFrame(
        {
            "congress_number": voteview["congress"].astype(int),
            "chamber": voteview["chamber"].map({"House": "H", "Senate": "S"}),
            "icpsr": voteview["icpsr"].astype(int),
            "bioname": voteview["bioname"],
            "party": party,
            "rule": out_rule,
            "independent": independent.to_numpy(),
        }
    ).reset_index(drop=True)


def load_rollcall_votes(
    rollcalls: pd.DataFrame,
    parties: pd.DataFrame,
    path: Path = VOTES_PATH,
    members: Path = DW_NOMINATE_MEMBERS,
    chunksize: int = 2_000_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Every member's vote on the roll calls in ``rollcalls``, with their party.

    ``HSall_votes.csv`` is read in chunks of ``chunksize`` rows and only five
    columns; rows are kept from the first Congress of ``rollcalls`` and only
    for its roll calls (so votes after its last date drop out). ``yea`` and
    ``nay`` follow CAST_YEA and CAST_NAY. Party from ``parties``
    (`voteview_caucus_party`) on Congress, chamber and ICPSR. Returns the
    votes and counts of what was left out: rows of the president (Voteview
    records his positions), rows whose ICPSR is not a member that Congress
    (delegates among them), and rows of excluded independents.
    """
    first = int(rollcalls["congress_number"].min())
    keys = rollcalls[["congress_number", "chamber", "rollnumber"]]
    presidents = set(
        pd.read_csv(members, usecols=["chamber", "icpsr"]).query(
            "chamber == 'President'"
        )["icpsr"]
    )
    columns = ["congress", "chamber", "rollnumber", "icpsr", "cast_code"]
    kinds = {"congress": "int16", "chamber": str, "rollnumber": "int32",
             "icpsr": "int32", "cast_code": "int8"}  # fmt: skip
    counts = {"rows_read": 0, "rows_after_last_date": 0, "president": 0,
              "not_a_member": 0, "excluded_independent": 0}  # fmt: skip
    parts = []
    lookup = parties[
        ["congress_number", "chamber", "icpsr", "party", "rule", "independent"]
    ]
    for chunk in pd.read_csv(path, usecols=columns, dtype=kinds, chunksize=chunksize):
        chunk = chunk[
            (chunk["congress"] >= first) & chunk["chamber"].isin(["House", "Senate"])
        ]
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={"congress": "congress_number"})
        chunk["chamber"] = chunk["chamber"].map({"House": "H", "Senate": "S"})
        counts["rows_read"] += len(chunk)
        before = len(chunk)
        chunk = chunk.merge(
            keys, on=["congress_number", "chamber", "rollnumber"], how="inner"
        )
        counts["rows_after_last_date"] += before - len(chunk)
        is_president = chunk["icpsr"].isin(presidents)
        counts["president"] += int(is_president.sum())
        chunk = chunk[~is_president].merge(
            lookup,
            on=["congress_number", "chamber", "icpsr"],
            how="left",
            validate="m:1",
        )
        counts["not_a_member"] += int(chunk["rule"].isna().sum())
        counts["excluded_independent"] += int((chunk["rule"] == "excluded").sum())
        chunk = chunk[chunk["party"].notna()].copy()
        chunk["independent"] = chunk["independent"].astype(bool)
        chunk["yea"] = chunk["cast_code"].isin(CAST_YEA)
        chunk["nay"] = chunk["cast_code"].isin(CAST_NAY)
        parts.append(chunk.drop(columns="rule"))
    votes = pd.concat(parts, ignore_index=True)
    return votes, counts


def rollcall_party_votes(
    votes: pd.DataFrame, include_independents: bool = True
) -> pd.DataFrame:
    """Per roll call: each party's yeas and nays, majorities and type.

    ``d_yes_share`` / ``r_yes_share``: yeas over yeas + nays. A party's
    majority is ``yea``, ``nay`` or ``none`` (a tie, or nobody voting).
    ``party_unity``: a majority of D against a majority of R (the
    Congressional Quarterly definition); ``bipartisan``: both majorities on
    the same side; ``no_majority``: at least one party tied or absent.
    ``near_unanimous``: yeas over all yeas + nays at least NEAR_UNANIMOUS or
    at most 1 - NEAR_UNANIMOUS. ``include_independents=False`` drops every
    vote of a Voteview independent (the robustness check).
    """
    data = votes if include_independents else votes[~votes["independent"]]
    key = ["congress_number", "chamber", "rollnumber"]
    tallies = data.groupby([*key, "party"])[["yea", "nay"]].sum().unstack("party")
    tallies = tallies.reindex(
        columns=pd.MultiIndex.from_product([["yea", "nay"], list(_PARTIES)]),
        fill_value=0,
    ).fillna(0)
    out = pd.DataFrame(index=tallies.index)
    for party in _PARTIES:
        yea, nay = tallies[("yea", party)], tallies[("nay", party)]
        lower = party.lower()
        out[f"{lower}_yea"], out[f"{lower}_nay"] = yea.astype(int), nay.astype(int)
        out[f"{lower}_yes_share"] = yea / (yea + nay).where(yea + nay > 0)
        out[f"{lower}_majority"] = np.select(
            [yea > nay, nay > yea], ["yea", "nay"], "none"
        )
    out = out.reset_index()
    yeas = out["d_yea"] + out["r_yea"]
    cast = yeas + out["d_nay"] + out["r_nay"]
    out["yes_share"] = yeas / cast.where(cast > 0)
    sides = out["d_majority"].ne("none") & out["r_majority"].ne("none")
    out["party_unity"] = sides & out["d_majority"].ne(out["r_majority"])
    out["bipartisan"] = sides & out["d_majority"].eq(out["r_majority"])
    out["no_majority"] = ~sides
    out["near_unanimous"] = (out["yes_share"] >= NEAR_UNANIMOUS) | (
        out["yes_share"] <= 1 - NEAR_UNANIMOUS
    )
    return out


def rollcall_summary(
    party_votes: pd.DataFrame, by: Sequence[str] = ("congress_number", "chamber")
) -> pd.DataFrame:
    """Per group: roll calls, and the shares that are party-unity and
    bipartisan votes (of all roll calls, ties included), with the ties."""
    return (
        party_votes.groupby(list(by))
        .agg(
            n_votes=("rollnumber", "size"),
            share_party_unity=("party_unity", "mean"),
            share_bipartisan=("bipartisan", "mean"),
            n_no_majority=("no_majority", "sum"),
        )
        .reset_index()
    )


def measure_gap_correlation(
    table: pd.DataFrame, column: str, gap: str = "gap"
) -> pd.DataFrame:
    """`laws_gap_correlation` for any per-Congress ``column``: Pearson r of
    ``gap`` with it per chamber over the ``complete`` Congresses, raw and
    after a linear trend in the Congress number is removed from both (one
    control in the Fisher interval). Descriptive."""
    rows = []
    for chamber, part in table[table["complete"]].groupby("chamber"):
        x = part["congress_number"].to_numpy(dtype=float)
        a = part[gap].to_numpy(dtype=float)
        b = part[column].to_numpy(dtype=float)
        n = len(part)
        for kind, (u, v, controls) in {
            "raw": (a, b, 0),
            "detrended": (_residual(a, x), _residual(b, x), 1),
        }.items():
            r = float(np.corrcoef(u, v)[0, 1])
            low, high = fisher_interval(r, n, controls)
            rows.append({"chamber": chamber, "measure": column, "kind": kind, "r": r,
                         "ci_low": low, "ci_high": high, "n_congresses": n,
                         "congresses": f"{int(x.min())}-{int(x.max())}"})  # fmt: skip
    return pd.DataFrame(rows)


def rollcall_text(rollcalls: pd.DataFrame) -> pd.Series:
    """vote_desc, vote_question and dtl_desc joined with " | " (nulls skipped)."""
    parts = rollcalls[list(ROLLCALL_TEXT_COLUMNS)]
    return parts.apply(lambda r: " | ".join(str(v) for v in r if pd.notna(v)), axis=1)


def flag_rollcall_undo(
    rollcalls: pd.DataFrame, passed: Sequence[str] = (), exclude: bool = True
) -> pd.DataFrame:
    """Add the flags for votes to undo:

    - ``undo``: ROLLCALL_UNDO_TERMS in the roll call's text (whole words, case
      ignored), after ROLLCALL_UNDO_EXCLUDE is blanked (``exclude=False``:
      without the exclusions, as the hand-check sample was drawn);
    - ``cra``: a Congressional Review Act resolution -- a joint resolution
      (JOINT_RESOLUTION_PATTERN) matching CRA_RULE_PATTERN after the
      exclusions, and not about the DC Council (DC_COUNCIL_PATTERN);
    - ``cra_title``: the brief's narrower rule, CRA_TITLE_PATTERN alone;
    - ``cra_passed``: ``cra`` with a vote_result in ``passed``.
    """
    text = rollcall_text(rollcalls)
    exclusion = compile_exclusions(ROLLCALL_UNDO_EXCLUDE) if exclude else None
    cleaned = text.map(lambda t: blank(t, exclusion))
    pattern = compile_keywords(ROLLCALL_UNDO_TERMS)
    joint = (
        rollcalls["bill_number"]
        .fillna("")
        .str.upper()
        .map(lambda m: bool(JOINT_RESOLUTION_PATTERN.match(m)))
    )
    rule = cleaned.map(lambda t: bool(CRA_RULE_PATTERN.search(t)))
    council = text.map(lambda t: bool(DC_COUNCIL_PATTERN.search(t)))
    out = rollcalls.assign(
        undo=cleaned.map(lambda t: bool(pattern.search(t))).to_numpy(),
        cra=(joint & rule & ~council).to_numpy(),
        cra_title=text.map(lambda t: bool(CRA_TITLE_PATTERN.search(t))).to_numpy(),
    )
    out["cra_passed"] = out["cra"] & out["vote_result"].isin(list(passed))
    return out


def rollcall_keyword_sample(
    flagged: pd.DataFrame,
    n: int = KEYWORD_CHECK_SIZE,
    seed: int = RANDOM_SEED,
    context: int = KEYWORD_CONTEXT,
) -> pd.DataFrame:
    """``n`` random roll calls flagged ``undo``, for a hand check of precision,
    in the format of `keyword_sample`: first the roll calls are drawn, then
    one hit within each; ``context`` characters on either side; an empty
    ``correct`` column for the coder."""
    rng = np.random.default_rng(seed)
    hits = flagged[flagged["undo"]].reset_index(drop=True)
    picked = hits.iloc[
        np.sort(rng.choice(len(hits), size=min(n, len(hits)), replace=False))
    ]
    pattern = compile_keywords(ROLLCALL_UNDO_TERMS)
    rows = []
    for row, text in zip(picked.to_dict("records"), rollcall_text(picked)):
        matches = list(pattern.finditer(text))
        match = matches[rng.integers(0, len(matches))]
        low, high = max(0, match.start() - context), match.end() + context
        rows.append(
            {
                "list": "rollcall_undo",
                "rollcall": f"{row['congress_number']}-{row['chamber']}-{row['rollnumber']}",
                "date": row["date"],
                "chamber": row["chamber"],
                "bill_number": row["bill_number"],
                "vote_question": row["vote_question"],
                "vote_result": row["vote_result"],
                "cra": row["cra"],
                "match": match.group(),
                "snippet": text[low:high].replace("\n", " "),
                "correct": "",
            }
        )
    return pd.DataFrame(rows)


def rollcall_undo_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Per Congress and chamber: roll calls, undo votes (number and share),
    CRA votes and passed CRA votes, and whether the Congress is the first
    after a change of the White House (`white_house_changes`). With a
    ``final_passage`` column, also the share of undo votes among the final
    passage votes (``share_undo_final``)."""
    firsts = {
        b[0]
        for _, b in white_house_changes(
            sorted(flagged["congress_number"].unique())
        ).values()
    }
    out = (
        flagged.groupby(["congress_number", "chamber"])
        .agg(
            n_votes=("rollnumber", "size"),
            n_undo=("undo", "sum"),
            n_cra=("cra", "sum"),
            n_cra_passed=("cra_passed", "sum"),
        )
        .reset_index()
    )
    out["share_undo"] = out["n_undo"] / out["n_votes"]
    if "final_passage" in flagged:
        final = flagged[flagged["final_passage"]]
        counts = final.groupby(["congress_number", "chamber"]).agg(
            n_final=("rollnumber", "size"), n_undo_final=("undo", "sum")
        )
        out = out.merge(
            counts.reset_index(), on=["congress_number", "chamber"], how="left"
        )
        out["share_undo_final"] = out["n_undo_final"] / out["n_final"]
    out["first_after_wh_change"] = out["congress_number"].isin(firsts)
    return out


def cra_resolutions(flagged: pd.DataFrame) -> pd.DataFrame:
    """Per Congress: distinct CRA resolutions (bill_number) voted on, and
    those passed by at least one chamber. Passing one chamber is not
    becoming law: the other chamber and the president may still stop it."""
    cra = flagged[flagged["cra"]]
    voted = (
        cra.groupby("congress_number")["bill_number"]
        .nunique()
        .rename("n_resolutions_voted")
    )
    passed = (
        cra[cra["cra_passed"]]
        .groupby("congress_number")["bill_number"]
        .nunique()
        .rename("n_resolutions_passed_a_chamber")
    )
    return pd.concat([voted, passed], axis=1).fillna(0).astype(int).reset_index()


#: Bills and joint resolutions -- the measures that can become law. Simple and
#: concurrent resolutions (HRES, SRES, HCONRES, SCONRES) cannot.
LAW_MEASURE_PATTERN: re.Pattern = re.compile(r"^(?:HR|S|HJRES|SJRES)\d+$")
#: Final-passage questions (case ignored): passage, suspension and passage,
#: concurring in or agreeing to the other chamber's amendments, conference
#: reports, the Senate's vote on a joint resolution, veto overrides. A
#: question that also matches FINAL_PASSAGE_EXCLUDE is a procedural motion.
FINAL_PASSAGE_PATTERN: re.Pattern = re.compile(
    r"^(?:on )?passage\b|sus?pend (?:the )?rules and pass\b|conference report$"
    r"|\bconcur(?:ring)? in (?:the )?senate amendments?$"
    r"|\bagree(?:ing)? to (?:the )?senate amendments?$"
    r"|^on the joint resolution$|overrid|objections of the president",
    re.IGNORECASE,
)
FINAL_PASSAGE_EXCLUDE: re.Pattern = re.compile(
    r"recommit|table|instruct|reconsider|\bcommit\b|\brefer\b|proceed|cloture|previous question"
    r"|consideration",
    re.IGNORECASE,
)


def is_final_passage(rollcalls: pd.DataFrame) -> pd.Series:
    """True for a final vote on a bill or joint resolution: a law measure
    (LAW_MEASURE_PATTERN on ``bill_number``) and a final-passage question
    (FINAL_PASSAGE_PATTERN, not FINAL_PASSAGE_EXCLUDE)."""
    question = rollcalls["vote_question"].fillna("")
    measure = rollcalls["bill_number"].fillna("").str.upper()
    return (
        measure.map(lambda m: bool(LAW_MEASURE_PATTERN.match(m)))
        & question.map(lambda q: bool(FINAL_PASSAGE_PATTERN.search(q)))
        & ~question.map(lambda q: bool(FINAL_PASSAGE_EXCLUDE.search(q)))
    )


def vote_question_counts(rollcalls: pd.DataFrame) -> pd.DataFrame:
    """Roll calls per chamber and vote_question, with how many of them
    `is_final_passage` counts, most frequent first."""
    data = rollcalls.assign(final_passage=is_final_passage(rollcalls))
    return (
        data.groupby(["chamber", "vote_question"])
        .agg(n=("rollnumber", "size"), n_final_passage=("final_passage", "sum"))
        .reset_index()
        .sort_values(["n"], ascending=False, ignore_index=True)
    )


# =================================================================================
# Notebook 10, analyses 4b and 4c: the power ladder, and after elections
# =================================================================================

# --- 4b. the power ladder ------------------------------------------------------------------

#: The power of the speaker's party: the White House and the majority of the
#: speaker's own chamber, both, one or neither. The first is the reference.
POWER_LEVELS: tuple[str, ...] = (
    "full power",
    "White House only",
    "chamber only",
    "nothing",
)
_POWER_TERMS: dict[str, str] = {
    "White House only": "power_white_house_only",
    "chamber only": "power_chamber_only",
    "nothing": "power_nothing",
}
#: A power level x party x chamber cell under either count is marked.
POWER_MIN_SPEECHES: int = 30
POWER_MIN_MEMBERS: int = 10

#: The president's constellation: how many chambers their party holds.
PRESIDENT_CONSTELLATIONS: tuple[str, ...] = ("unified", "one chamber", "none")


def add_power_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``power_level`` (one of `POWER_LEVELS`) and a 0/1 column per level
    but the reference.

    Needs ``in_opposition`` (`analysis.add_presidency`) and
    ``chamber_minority`` (`add_chamber_control`).
    """
    opposition = frame["in_opposition"].astype(bool).to_numpy()
    minority = frame["chamber_minority"].astype(bool).to_numpy()
    level = np.select(
        [~opposition & ~minority, ~opposition & minority, opposition & ~minority],
        list(POWER_LEVELS[:3]),
        POWER_LEVELS[3],
    )
    out = frame.assign(power_level=level)
    for name, term in _POWER_TERMS.items():
        out[term] = (out["power_level"] == name).astype(float)
    return out


def power_ladder(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """The power ladder of ``value``, per chamber and party, clustered by member.

    Three models, all with member fixed effects and a linear trend in the
    Congress (as model C of `OPPOSITION_MODELS`), in ``model``:

    - ``additive``: ``opposition`` + ``chamber_minority``, i.e. model C itself
      (the reproduction check);
    - ``ladder``: one indicator per level but "full power", so each
      coefficient is the level against full power;
    - ``interaction``: the additive model plus ``opposition x
      chamber_minority``. Its coefficient is beta(nothing) - beta(White House
      only) - beta(chamber only) of the ladder (the same model,
      reparametrized), so its Wald test (``chi2``, one degree of freedom) is
      whether losing both is more than the sum of losing each.

    Needs `add_power_level`, ``opposition`` and ``member_key``.
    """
    terms, cell = OPPOSITION_MODELS["C"]
    by = ("chamber", "party")
    data = frame.assign(
        opposition_x_minority=frame["opposition"] * frame["chamber_minority"]
    )
    additive = within_cell_ols(data, value, list(terms), cell=cell, by=by)
    ladder = within_cell_ols(
        data, value, ["congress_number", *_POWER_TERMS.values()], cell=cell, by=by
    )
    names = {term: name for name, term in _POWER_TERMS.items()}
    ladder["power_level"] = ladder["term"].map(names)
    interaction = within_cell_ols(
        data, value, [*terms, "opposition_x_minority"], cell=cell, by=by
    )
    interaction = interaction[interaction["term"] == "opposition_x_minority"].copy()
    interaction["chi2"] = (interaction["coef"] / interaction["se"]) ** 2
    interaction["df"] = 1
    return pd.concat(
        [
            additive.assign(model="additive"),
            ladder.assign(model="ladder"),
            interaction.assign(model="interaction"),
        ],
        ignore_index=True,
    ).assign(value=value)


def power_ladder_cells(
    frame: pd.DataFrame,
    min_speeches: int = POWER_MIN_SPEECHES,
    min_members: int = POWER_MIN_MEMBERS,
) -> pd.DataFrame:
    """Speeches, members and member x Congress units per chamber, party and
    power level; every combination appears, empty ones with zeros.

    ``small`` marks a cell under ``min_speeches`` or ``min_members``;
    ``n_members_varying`` counts the members of the chamber and party whose
    level changes (only they identify the ladder under member fixed effects).
    """
    data = frame.assign(
        unit=frame["member_key"] + "|" + frame["congress_number"].astype(str)
    )
    index = pd.MultiIndex.from_product(
        [("H", "S"), _PARTIES, list(POWER_LEVELS)],
        names=["chamber", "party", "power_level"],
    )
    out = (
        data.groupby(list(index.names))
        .agg(
            n=("speech_id", "size"),
            n_members=("member_key", "nunique"),
            n_units=("unit", "nunique"),
            congresses=(
                "congress_number",
                lambda s: ", ".join(str(c) for c in sorted(set(s))),
            ),
        )
        .reindex(index)
        .reset_index()
    )
    for column in ("n", "n_members", "n_units"):
        out[column] = out[column].fillna(0).astype(int)
    out["congresses"] = out["congresses"].fillna("")
    levels = data.groupby(["chamber", "party", "member_key"])["power_level"].nunique()
    varying = (
        (levels > 1).groupby(["chamber", "party"]).sum().rename("n_members_varying")
    )
    out = out.merge(varying.reset_index(), on=["chamber", "party"], how="left")
    out["small"] = (out["n"] < min_speeches) | (out["n_members"] < min_members)
    return out


def power_constellations() -> pd.DataFrame:
    """Per Congress and majority period: the president (`PRESIDENTS_BY_CONGRESS`),
    the House and Senate majorities (`majority_table`), and the president's
    constellation, one of `PRESIDENT_CONSTELLATIONS`.

    The president is the one the constant gives the whole Congress; in the
    first weeks of the 107th and 117th (to 20 January) the outgoing president
    was still in office, which ``note`` says.
    """
    table = majority_table()
    table["president"] = table["congress_number"].map(
        lambda c: PRESIDENTS_BY_CONGRESS[c][0]
    )
    table["president_party"] = table["congress_number"].map(
        lambda c: PRESIDENTS_BY_CONGRESS[c][1]
    )
    held = (table["house_majority"] == table["president_party"]).astype(int) + (
        table["senate_majority"] == table["president_party"]
    ).astype(int)
    table["constellation"] = held.map(dict(zip((2, 1, 0), PRESIDENT_CONSTELLATIONS)))
    first = table["senate_from"].notna() & (
        table["senate_from"]
        == table["congress_number"].map(lambda c: congress_days(c)[0])
    )
    table["note"] = np.where(
        first,
        "president as the constant gives the Congress; the outgoing president served to 20 January",
        "",
    )
    return table


# --- 4c. after elections: how does the losing side react? -----------------------------------

#: The general federal elections in the data: presidential years and midterms.
ELECTION_YEARS: tuple[int, ...] = tuple(range(2002, 2025, 2))
#: An election x chamber x party x period cell with fewer speeches is kept and
#: marked (†).
ELECTION_MIN_CELL: int = 30
#: Fewer members speaking in both windows (on a side) leave no clustered
#: standard error to speak of: with one, the change is fitted exactly and the
#: SE collapses towards 0. Such an estimate is left null, with a note.
ELECTION_MIN_MEMBERS_BOTH: int = 2


def election_day(year: int) -> dt.date:
    """The general election: the Tuesday after the first Monday in November."""
    first = dt.date(year, 11, 1)
    monday = first + dt.timedelta(days=(0 - first.weekday()) % 7)
    return monday + dt.timedelta(days=1)


def election_outcomes(years: Sequence[int] = ELECTION_YEARS) -> pd.DataFrame:
    """Who lost each election, per chamber, from the constants.

    Presidential years (divisible by four): the party that does not win the
    presidency (`PRESIDENTS_BY_CONGRESS` of the next Congress) loses, in both
    chambers; at a re-election that is the challenger's party. Midterms: the
    party that loses the chamber's majority (`chamber_majority` on the last
    day of the Congress against the first day of the next); a chamber whose
    majority stays has ``status`` "no change" and no loser.
    """
    rows = []
    for year in years:
        day = election_day(year)
        congress = congress_of(day)
        end = congress_days(congress)[1]
        presidential = year % 4 == 0
        for chamber in ("H", "S"):
            if presidential:
                winner = PRESIDENTS_BY_CONGRESS[congress + 1][1]
                loser = "R" if winner == "D" else "D"
            else:
                before = chamber_majority(chamber, congress, end - dt.timedelta(days=1))
                after = chamber_majority(chamber, congress + 1, end)
                loser, winner = (before, after) if before != after else (None, None)
            rows.append({
                "election": str(year), "election_day": day, "congress_number": congress,
                "election_type": "presidential" if presidential else "midterm",
                "chamber": chamber, "loser": loser, "winner": winner,
                "status": "no change" if loser is None else "loser",
                "white_house_changed": presidential and (
                    PRESIDENTS_BY_CONGRESS[congress][1] != PRESIDENTS_BY_CONGRESS[congress + 1][1]),
            })  # fmt: skip
    return pd.DataFrame(rows)


def corpus_sitting_days(corpus: Path = CORPUS_PATH) -> pd.DataFrame:
    """Every chamber x day with at least one speech in the corpus, and its source:
    the days the Record shows each chamber sitting (decided at HALT 1 of 4c,
    2026-11-06, over the days of a scored sample)."""
    days = pd.read_parquet(corpus, columns=["date", "chamber", "source"])
    days["day"] = pd.to_datetime(days["date"]).dt.date
    return (
        days[["chamber", "day", "source"]]
        .drop_duplicates()
        .sort_values(["chamber", "day"], ignore_index=True)
    )


def post_election_windows(
    days: pd.DataFrame, outcomes: pd.DataFrame, cut_at_break: bool = True
) -> pd.DataFrame:
    """The before and after window of each election and chamber, in sitting days.

    ``days`` is `corpus_sitting_days`. After: every sitting day from election
    day to the end of the Congress (the lame duck). Before: as many sitting
    days, the last ones before election day, never before the Congress began.
    With ``cut_at_break`` the before window also stays on the election's side
    of the 2016-09-10 source break, so it can be shorter (decided at HALT 1 of
    4c: 2016 in the House); ``short_before`` marks that. ``before_days`` and
    ``after_days`` hold the days.
    """
    rows = []
    for outcome in outcomes.to_dict("records"):
        day, congress = outcome["election_day"], outcome["congress_number"]
        start, end = congress_days(congress)
        if cut_at_break and day >= SOURCE_BREAK_DATE:
            start = max(start, SOURCE_BREAK_DATE)
        sitting = days.loc[days["chamber"] == outcome["chamber"], "day"]
        after = sorted(d for d in sitting if day <= d < end)
        earlier = sorted(d for d in sitting if start <= d < day)
        before = earlier[max(len(earlier) - len(after), 0) :] if after else []
        rows.append({
            "election": outcome["election"], "chamber": outcome["chamber"],
            "n_days_before": len(before), "n_days_after": len(after),
            "before_from": before[0] if before else None, "before_to": before[-1] if before else None,
            "after_from": after[0] if after else None, "after_to": after[-1] if after else None,
            "short_before": len(before) < len(after), "cut_at_break": cut_at_break,
            "before_days": tuple(before), "after_days": tuple(after),
        })  # fmt: skip
    return pd.DataFrame(rows)


def post_election_data(
    frame: pd.DataFrame, windows: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """The speeches in each election's windows, stacked by election.

    Adds ``election``, ``election_type``, ``status``, ``period`` ("before",
    "after"), ``post`` (after = 1), ``loser`` (1 for the losing party; null
    where the chamber had no change), ``post_loser`` and ``member_election``.
    """
    speech_day = pd.to_datetime(frame["date"]).dt.date
    keyed = outcomes.set_index(["election", "chamber"])
    parts = []
    for window in windows.to_dict("records"):
        outcome = keyed.loc[(window["election"], window["chamber"])]
        in_chamber = frame["chamber"] == window["chamber"]
        before = in_chamber & speech_day.isin(set(window["before_days"]))
        after = in_chamber & speech_day.isin(set(window["after_days"]))
        data = frame[before | after].copy()
        data["period"] = np.where(after[before | after], "after", "before")
        data["post"] = (data["period"] == "after").astype(float)
        data["loser"] = (
            (data["party"] == outcome["loser"]).astype(float)
            if outcome["status"] == "loser"
            else np.nan
        )
        data["post_loser"] = data["post"] * data["loser"]
        data["member_election"] = data["member_key"] + "|" + window["election"]
        parts.append(data.assign(election=window["election"],
                                 election_type=outcome["election_type"],
                                 status=outcome["status"]))  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def post_election_cell_sizes(
    data: pd.DataFrame, minimum: int = ELECTION_MIN_CELL
) -> pd.DataFrame:
    """Speeches and members per election, chamber, party and period, and the
    members who speak in both windows (only they identify a change within
    members). ``below_min`` marks a cell under ``minimum`` speeches."""
    keys = ["election", "chamber", "party", "period"]
    sizes = (
        data.groupby(keys)
        .agg(n=("speech_id", "size"), n_members=("member_key", "nunique"))
        .reset_index()
    )
    periods = data.groupby(["election", "chamber", "party", "member_key"])[
        "period"
    ].nunique()
    both = (periods == 2).groupby(["election", "chamber", "party"]).sum()
    sizes = sizes.merge(
        both.rename("n_members_both").reset_index(),
        on=["election", "chamber", "party"],
        how="left",
    )
    sizes["below_min"] = sizes["n"] < minimum
    return sizes


def _members_in_both(part: pd.DataFrame, side: str | None = None) -> int:
    """Members of ``part`` (one election and chamber) who speak in both windows;
    ``side`` "loser" or "winner" keeps one side."""
    if side is not None:
        part = part[part["loser"] == (1.0 if side == "loser" else 0.0)]
    return int((part.groupby("member_key")["period"].nunique() == 2).sum())


def _missing_fit(group: dict[str, object], term: str, note: str) -> dict[str, object]:
    empty = {k: np.nan for k in ("coef", "se", "ci_low", "ci_high", "p")}
    return {**group, "term": term, **empty, "n": 0, "n_members": 0, "note": note}


def post_election_effects(data: pd.DataFrame, value: str) -> pd.DataFrame:
    """After minus before an election, the losing side against the winning
    side, within members; clustered by member. Descriptive only.

    Rows by ``kind``:

    - ``election``: per election and chamber with a loser, member fixed
      effects + post + post x loser; ``post_loser`` is how much more the
      losing party's members changed than the winning party's. Null, with a
      ``note``, where fewer than `ELECTION_MIN_MEMBERS_BOTH` members of a
      side speak in both windows (``n_both_loser``, ``n_both_winner``);
    - ``pooled``: per election type and chamber, stacked, member x election
      and election x post fixed effects, one ``post_loser``;
    - ``trend``: the pooled model plus ``post_loser`` x decades since 2002;
      its coefficient is how much the losers' reaction changed per decade;
    - ``own change``: per election, chamber and party (no-change midterms
      included), member fixed effects + post: each side's own change; null
      under the same minimum (``n_both``).
    """
    rows = []
    losers = data[data["status"] == "loser"]
    for (election, chamber), part in losers.groupby(["election", "chamber"]):
        group = {"election": election, "chamber": chamber,
                 "election_type": part["election_type"].iloc[0],
                 "n_both_loser": _members_in_both(part, "loser"),
                 "n_both_winner": _members_in_both(part, "winner")}  # fmt: skip
        if (
            min(group["n_both_loser"], group["n_both_winner"])
            < ELECTION_MIN_MEMBERS_BOTH
        ):
            note = f"under {ELECTION_MIN_MEMBERS_BOTH} members of a side speak in both windows"
            rows.append({**_missing_fit(group, "post_loser", note), "kind": "election"})
            continue
        fit = within_cell_ols(
            part, value, ["post", "post_loser"], cell=("member_key",), by=("election",)
        )
        fit = fit[fit["term"] == "post_loser"].drop(columns="election")
        rows.extend(
            {**group, **r, "kind": "election", "note": ""}
            for r in fit.to_dict("records")
        )
    for (kind, chamber), part in losers.groupby(["election_type", "chamber"]):
        posts = {
            f"post_{e}": ((part["election"] == e) & (part["post"] == 1)).astype(float)
            for e in sorted(part["election"].unique())
        }
        decades = (part["election"].astype(int) - ELECTION_YEARS[0]) / 10
        stacked = part.assign(**posts, post_loser_trend=part["post_loser"] * decades)
        group = {
            "election": f"pooled {kind}",
            "chamber": chamber,
            "election_type": kind,
        }
        for name, terms, keep in (
            ("pooled", [*posts, "post_loser"], "post_loser"),
            ("trend", [*posts, "post_loser", "post_loser_trend"], "post_loser_trend"),
        ):
            fit = within_cell_ols(stacked, value, terms, cell=("member_election",),
                                  by=("election_type",))  # fmt: skip
            fit = fit[fit["term"] == keep].drop(columns="election_type")
            rows.extend(
                {**group, **r, "kind": name, "note": ""} for r in fit.to_dict("records")
            )
    for (election, chamber, party), part in data.groupby(
        ["election", "chamber", "party"]
    ):
        group = {"election": election, "chamber": chamber, "party": party,
                 "election_type": part["election_type"].iloc[0],
                 "status": part["status"].iloc[0], "n_both": _members_in_both(part)}  # fmt: skip
        if group["n_both"] < ELECTION_MIN_MEMBERS_BOTH:
            note = f"under {ELECTION_MIN_MEMBERS_BOTH} members speak in both windows"
            rows.append({**_missing_fit(group, "post", note), "kind": "own change"})
            continue
        fit = within_cell_ols(
            part, value, ["post"], cell=("member_key",), by=("election",)
        )
        rows.extend({**group, **r, "kind": "own change", "note": ""}
                    for r in fit.drop(columns="election").to_dict("records"))  # fmt: skip
    return pd.DataFrame(rows).assign(value=value)


def flag_small_election_cells(
    effects: pd.DataFrame, sizes: pd.DataFrame, minimum: int = ELECTION_MIN_CELL
) -> pd.DataFrame:
    """Mark each `post_election_effects` row that rests on a cell under
    ``minimum`` speeches: an election row uses both parties' cells of that
    election and chamber, an own change its party's; pooled and trend rows
    are not marked. ``small_cell`` flags the row, ``small_cells`` names the
    cells."""
    flags, notes = [], []
    for row in effects.to_dict("records"):
        if row["kind"] not in ("election", "own change"):
            flags.append(False)
            notes.append("")
            continue
        used = sizes[
            (sizes["election"] == row["election"])
            & (sizes["chamber"] == row["chamber"])
        ]
        if row["kind"] == "own change":
            used = used[used["party"] == row["party"]]
        small = used[used["n"] < minimum]
        flags.append(len(small) > 0)
        notes.append("; ".join(f"{r['party']} {r['period']}: n={r['n']}, {r['n_members']} members"
                               for r in small.to_dict("records")))  # fmt: skip
    return effects.assign(small_cell=flags, small_cells=notes)
