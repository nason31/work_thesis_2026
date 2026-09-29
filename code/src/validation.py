"""Validate LLM ideology scores against DW-NOMINATE (docs/decisions.md M5).

A scoring run writes one JSONL file per model to ``results/scores/``. This
module turns a run into one row per speech, attaches each speaker's Voteview
scores through the member crosswalk (``crosswalk.py``), and correlates the two.

Three things this deliberately does *not* hide:

- **Within-party correlation is reported next to the overall one.** Across
  both parties a high r mostly re-measures the party split, which the pilot's
  party check (P1) already showed. Whether the scores order members *within*
  a party the way their votes do is the stronger test.
- **Only ``in_validation`` speeches are correlated** -- matched and scored
  members. The rest stay in the frame, flagged, so they can be counted.
- **Nothing is chosen that is still open** (O9): both benchmarks, and the
  procedural speeches, are reported as they are; the caller filters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from src.config import CORPUS_PATH, CROSSWALK_PATH, ENSEMBLE_MODELS
from src.crosswalk import UNIT_KEY

#: Crosswalk columns carried onto each speech.
CROSSWALK_COLUMNS: tuple[str, ...] = (
    "icpsr",
    "bioname",
    "nominate_dim1",
    "nokken_poole_dim1",
    "party_switch",
    "unmatched_reason",
    "in_validation",
)

#: Below this many speeches a group gets no coefficient: two points always
#: lie on a line.
MIN_N = 3

#: How many offending values an error message lists.
_SHOW = 5


def score_column(model: str) -> str:
    """Column holding ``model``'s ideology score in a run frame."""
    return f"ideology_{model}"


ENSEMBLE_COLUMN = score_column("ensemble")


# --- a run -------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """JSONL rows, with ``speech_id`` kept a string (it looks numeric)."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run_scores(
    scores_dir: Path, run: str, models: tuple[str, ...] = ENSEMBLE_MODELS
) -> pd.DataFrame:
    """One row per speech: each model's ideology score and their mean.

    Reads the raw per-model files, not the run's ensemble file, so the
    individual scores stay inspectable (CLAUDE.md). The mean is over the models
    that returned a score, as the run's own ensemble is (S4); ``n_models``
    counts them. ``procedural`` marks speeches every scoring model put at
    exactly 0.0, which the prompt reserves for procedural speech.

    Raises:
        FileNotFoundError: a model's file for ``run`` is missing.
    """
    frame: pd.DataFrame | None = None
    for model in models:
        path = Path(scores_dir) / f"pilot_{model.replace('/', '_')}_{run}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No scores for {model} in run {run}: {path}")
        rows = pd.DataFrame(_read_jsonl(path), columns=["speech_id", "ideology_score"])
        rows = rows.rename(columns={"ideology_score": score_column(model)})
        rows[score_column(model)] = pd.to_numeric(rows[score_column(model)])
        frame = (
            rows if frame is None else frame.merge(rows, on="speech_id", how="outer")
        )

    columns = [score_column(m) for m in models]
    frame[ENSEMBLE_COLUMN] = frame[columns].mean(axis=1, skipna=True)
    frame["n_models"] = frame[columns].notna().sum(axis=1)
    frame["procedural"] = (frame["n_models"] > 0) & (
        frame[columns].fillna(0.0).eq(0.0).all(axis=1)
    )
    return frame


# --- DW-NOMINATE ---------------------------------------------------------


def attach_benchmarks(
    scores: pd.DataFrame,
    corpus: Path = CORPUS_PATH,
    crosswalk: Path = CROSSWALK_PATH,
) -> pd.DataFrame:
    """Add each speech's party and its speaker's Voteview scores.

    speech_id -> corpus (party and the crosswalk key) -> crosswalk. Every
    scored speech is kept; unmatched ones carry ``in_validation = False``.

    Raises:
        ValueError: a scored speech is not in the corpus -- the corpus was
            rebuilt under the run, or the run is from another corpus.
    """
    ids = scores["speech_id"].astype(str).tolist()
    speeches = pq.read_table(
        corpus,
        columns=["speech_id", "party", *UNIT_KEY],
        filters=[("speech_id", "in", ids)],
    ).to_pandas()
    missing = sorted(set(ids) - set(speeches["speech_id"]))
    if missing:
        raise ValueError(
            f"{len(missing)} scored speech(es) are not in {corpus}, e.g. "
            f"{missing[:_SHOW]}"
        )
    table = pd.read_parquet(crosswalk, columns=[*UNIT_KEY, *CROSSWALK_COLUMNS])
    speeches = speeches.merge(table, on=list(UNIT_KEY), how="left", validate="m:1")
    speeches["in_validation"] = speeches["in_validation"].fillna(False).astype(bool)
    return scores.merge(speeches, on="speech_id", how="left", validate="1:1")


def _finite(value: float) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def _group_stats(x: pd.Series, y: pd.Series) -> dict[str, object]:
    """n, Pearson r with p and 95% CI, Spearman rho with p."""
    n = len(x)
    result: dict[str, object] = {
        "n": n,
        "pearson_r": None,
        "pearson_p": None,
        "pearson_ci95": None,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if n < MIN_N or x.nunique() < 2 or y.nunique() < 2:
        return result
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    result.update(
        pearson_r=_finite(pearson.statistic),
        pearson_p=_finite(pearson.pvalue),
        spearman_rho=_finite(spearman.statistic),
        spearman_p=_finite(spearman.pvalue),
    )
    if n > MIN_N:
        low, high = pearson.confidence_interval(0.95)
        result["pearson_ci95"] = [_finite(low), _finite(high)]
    return result


def correlate(
    frame: pd.DataFrame, score: str, benchmark: str, by: str = "party"
) -> dict[str, dict[str, object]]:
    """Correlate ``score`` with ``benchmark`` over all speeches and per ``by``.

    Uses only ``in_validation`` rows where both values exist. Returns
    ``{"all": {...}, "<group>": {...}}``; a group below ``MIN_N`` speeches, or
    with no variation, gets ``n`` and no coefficients.
    """
    usable = frame[frame["in_validation"]].dropna(subset=[score, benchmark])
    result = {"all": _group_stats(usable[score], usable[benchmark])}
    for group, rows in sorted(usable.groupby(by), key=lambda item: str(item[0])):
        result[str(group)] = _group_stats(rows[score], rows[benchmark])
    return result
