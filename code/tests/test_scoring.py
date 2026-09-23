"""Tests for the scoring helpers and the pilot's pure logic.

Nothing here touches a network or needs an API key: the parts worth testing are
prompt rendering, response parsing, validation, cost arithmetic, stratified
allocation and ensemble aggregation. The provider calls themselves are thin
wrappers around two SDKs and are exercised by the `--sample-size 6` run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pilot_run import _mean_std, allocate, build_ensemble

from src.scoring import (
    MODEL_SPECS,
    ScoreParseError,
    ScoreResult,
    estimate_cost,
    extract_json,
    load_prompt_template,
    render_prompt,
    specs_for,
    validate_scores,
)

# --- prompt ------------------------------------------------------------


def test_render_substitutes_without_tripping_on_literal_braces() -> None:
    """The real template shows a JSON example, so .format() would raise."""
    template = 'Return {"ideology_score": <float>}\n\nSPEECH:\n{speech_text}'

    rendered = render_prompt(template, "Mr. Speaker, I rise today.")

    assert "Mr. Speaker, I rise today." in rendered
    assert '{"ideology_score": <float>}' in rendered
    assert "{speech_text}" not in rendered
    with pytest.raises((KeyError, IndexError, ValueError)):
        template.format(speech_text="x")  # guards the reason for using replace


def test_real_template_loads_and_renders() -> None:
    template = load_prompt_template()

    rendered = render_prompt(template, "SPEECH BODY HERE")

    assert "SPEECH BODY HERE" in rendered
    assert "{speech_text}" not in rendered


def test_template_without_placeholder_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("no placeholder here")

    with pytest.raises(ValueError, match="speech_text"):
        load_prompt_template(path)


# --- response parsing --------------------------------------------------


def test_extract_json_from_bare_object() -> None:
    assert extract_json('{"ideology_score": 0.5}') == {"ideology_score": 0.5}


def test_extract_json_ignores_reasoning_prose_before_the_object() -> None:
    """deepseek-reasoner emits its reasoning before the answer."""
    raw = 'Let me think about this.\nThe speech argues for lower taxes.\n{"a": 1}'

    assert extract_json(raw) == {"a": 1}


def test_extract_json_handles_markdown_fences() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_handles_anthropic_prefill() -> None:
    """The assistant turn is prefilled with `{`, re-prepended before parsing."""
    assert extract_json('{"ideology_score": -0.4, "tone_score": 0.2}') == {
        "ideology_score": -0.4,
        "tone_score": 0.2,
    }


def test_extract_json_ignores_braces_inside_strings() -> None:
    raw = '{"reasoning": "uses the phrase {big government}", "ideology_score": 0.8}'

    assert extract_json(raw)["ideology_score"] == 0.8


def test_extract_json_takes_only_the_first_object() -> None:
    assert extract_json('{"a": 1} trailing {"b": 2}') == {"a": 1}


@pytest.mark.parametrize("raw", ["no json at all", "", '{"unterminated": 1'])
def test_extract_json_rejects_unusable_responses(raw: str) -> None:
    with pytest.raises(ScoreParseError):
        extract_json(raw)


# --- validation --------------------------------------------------------


def test_validate_accepts_a_well_formed_payload() -> None:
    ideology, tone, reasoning = validate_scores(
        {"ideology_score": -0.7, "tone_score": 0.3, "reasoning": "because"}
    )

    assert (ideology, tone, reasoning) == (-0.7, 0.3, "because")


def test_validate_accepts_integer_scores() -> None:
    ideology, tone, _ = validate_scores({"ideology_score": 1, "tone_score": 0})

    assert (ideology, tone) == (1.0, 0.0)


@pytest.mark.parametrize(
    "payload",
    [
        {"tone_score": 0.5},
        {"ideology_score": 0.5},
        {"ideology_score": 1.5, "tone_score": 0.5},
        {"ideology_score": -2.0, "tone_score": 0.5},
        {"ideology_score": 0.5, "tone_score": 1.5},
        {"ideology_score": 0.5, "tone_score": -0.1},
        {"ideology_score": "left", "tone_score": 0.5},
        {"ideology_score": True, "tone_score": 0.5},
        {"ideology_score": None, "tone_score": 0.5},
    ],
)
def test_validate_rejects_bad_payloads(payload: dict) -> None:
    """Out of range is a failure, not something to clip into range."""
    with pytest.raises(ScoreParseError):
        validate_scores(payload)


def test_missing_reasoning_defaults_to_empty_string() -> None:
    _, _, reasoning = validate_scores({"ideology_score": 0.0, "tone_score": 0.0})

    assert reasoning == ""


# --- model specs -------------------------------------------------------


def test_every_ensemble_model_has_a_spec_and_a_price() -> None:
    from src.config import ENSEMBLE_MODELS, MODEL_PRICING

    specs = specs_for(ENSEMBLE_MODELS)

    assert set(specs) == set(ENSEMBLE_MODELS)
    assert all(model in MODEL_PRICING for model in ENSEMBLE_MODELS)


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(KeyError, match="no ModelSpec"):
        specs_for(("some-model-we-never-configured",))


def test_no_model_claims_a_temperature() -> None:
    """The providers removed the control; the manifest must not imply otherwise."""
    from src.config import ENSEMBLE_MODELS

    for model in ENSEMBLE_MODELS:
        assert MODEL_SPECS[model].effective_temperature == "provider default"


def test_anthropic_uses_a_server_enforced_schema() -> None:
    """Prefill returns 400 on current Claude models, so the schema is required."""
    from src.scoring import RESPONSE_SCHEMA

    claude = MODEL_SPECS["claude-sonnet-4-6"]

    assert claude.supports_json_schema is True
    assert RESPONSE_SCHEMA["schema"]["required"] == [
        "ideology_score",
        "tone_score",
        "reasoning",
    ]
    assert RESPONSE_SCHEMA["schema"]["additionalProperties"] is False


def test_reasoning_model_gets_more_output_headroom() -> None:
    """Reasoning tokens are billed from the same budget as the answer.

    At 1024 the reasoner spent the whole cap reasoning on a 623-word speech and
    returned empty content.
    """
    reasoner = MODEL_SPECS["deepseek-reasoner"]

    assert (
        reasoner.max_output_tokens > MODEL_SPECS["gpt-4o-2024-11-20"].max_output_tokens
    )
    assert reasoner.max_output_tokens >= 8192


def test_truncation_is_recorded_not_raised() -> None:
    """A truncated response must become a null row, not end the run."""
    from src.scoring import TruncatedResponseError

    assert issubclass(TruncatedResponseError, ScoreParseError)


def test_retired_model_is_gone_from_the_ensemble() -> None:
    """claude-3-5-sonnet-20241022 was retired (404) before the first run."""
    from src.config import ENSEMBLE_MODELS

    assert "claude-3-5-sonnet-20241022" not in ENSEMBLE_MODELS
    assert "claude-3-5-sonnet-20241022" not in MODEL_SPECS


# --- cost --------------------------------------------------------------


def test_cost_uses_per_million_pricing() -> None:
    # GPT-4o: $2.50/1M in, $10.00/1M out
    cost = estimate_cost("gpt-4o-2024-11-20", 1_000_000, 1_000_000)

    assert cost == pytest.approx(12.50)


def test_cost_scales_linearly() -> None:
    assert estimate_cost("gpt-4o-2024-11-20", 2_000, 1_000) == pytest.approx(
        2 * estimate_cost("gpt-4o-2024-11-20", 1_000, 500)
    )


def test_cost_for_unpriced_model_is_rejected() -> None:
    with pytest.raises(KeyError, match="no pricing"):
        estimate_cost("mystery-model", 1, 1)


# --- stratified allocation ---------------------------------------------


def test_allocation_sums_to_the_requested_total() -> None:
    strata = [(c, p) for c in range(107, 115) for p in ("D", "R")]
    available = dict.fromkeys(strata, 10_000)

    quota = allocate(strata, available, 200)

    assert sum(quota.values()) == 200


def test_allocation_is_near_equal_across_strata() -> None:
    strata = [(c, p) for c in range(107, 115) for p in ("D", "R")]
    available = dict.fromkeys(strata, 10_000)

    quota = allocate(strata, available, 200)

    assert min(quota.values()) == 12
    assert max(quota.values()) == 13


def test_allocation_redistributes_away_from_a_short_stratum() -> None:
    strata = [("a",), ("b",), ("c",), ("d",)]
    available = {("a",): 1, ("b",): 100, ("c",): 100, ("d",): 100}

    quota = allocate(strata, available, 40)

    assert quota[("a",)] == 1
    assert sum(quota.values()) == 40


def test_allocation_caps_at_what_the_corpus_holds() -> None:
    strata = [("a",), ("b",)]
    available = {("a",): 3, ("b",): 4}

    quota = allocate(strata, available, 100)

    assert quota == {("a",): 3, ("b",): 4}


def test_allocation_is_deterministic() -> None:
    strata = [(c, p) for c in range(107, 115) for p in ("D", "R")]
    available = dict.fromkeys(strata, 10_000)

    assert allocate(strata, available, 200) == allocate(
        list(reversed(strata)), available, 200
    )


# --- ensemble aggregation ----------------------------------------------


def _result(model: str, ideology: float | None, tone: float | None) -> ScoreResult:
    return ScoreResult(
        model=model,
        ideology_score=ideology,
        tone_score=tone,
        reasoning="r",
        error=None if ideology is not None else "boom",
    )


def test_mean_std_of_three_values() -> None:
    mean, std = _mean_std([0.0, 0.5, 1.0])

    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.5)


def test_single_value_has_no_standard_deviation() -> None:
    """One model agreeing with itself is not agreement."""
    mean, std = _mean_std([0.4])

    assert mean == pytest.approx(0.4)
    assert std is None


def test_no_values_gives_nulls() -> None:
    assert _mean_std([]) == (None, None)


def test_ensemble_averages_all_three_models() -> None:
    speeches = [{"speech_id": "s1", "party": "R", "congress_number": 110}]
    by_speech = {
        "s1": [
            _result("m1", 0.2, 0.1),
            _result("m2", 0.4, 0.3),
            _result("m3", 0.6, 0.2),
        ]
    }

    row = build_ensemble(speeches, by_speech)[0]

    assert row["ideology_score_mean"] == pytest.approx(0.4)
    assert row["n_models"] == 3
    assert row["party"] == "R"


def test_ensemble_averages_survivors_and_records_the_count() -> None:
    """A failed model must not silently look like a three-model average."""
    speeches = [{"speech_id": "s1", "party": "D", "congress_number": 111}]
    by_speech = {
        "s1": [
            _result("m1", -0.5, 0.2),
            _result("m2", -0.3, 0.4),
            _result("m3", None, None),
        ]
    }

    row = build_ensemble(speeches, by_speech)[0]

    assert row["ideology_score_mean"] == pytest.approx(-0.4)
    assert row["n_models"] == 2


def test_ensemble_row_survives_total_failure() -> None:
    speeches = [{"speech_id": "s1", "party": "D", "congress_number": 111}]
    by_speech = {"s1": [_result("m1", None, None)]}

    row = build_ensemble(speeches, by_speech)[0]

    assert row["ideology_score_mean"] is None
    assert row["ideology_score_std"] is None
    assert row["n_models"] == 0


def test_ensemble_covers_every_sampled_speech() -> None:
    speeches = [
        {"speech_id": "s1", "party": "D", "congress_number": 107},
        {"speech_id": "s2", "party": "R", "congress_number": 114},
    ]

    rows = build_ensemble(speeches, {"s1": [_result("m1", 0.1, 0.1)]})

    assert [r["speech_id"] for r in rows] == ["s1", "s2"]
    assert rows[1]["n_models"] == 0
