"""Tests for the speech labels: codebook, validation, calls, ids, cost, files.

Nothing here touches a network. Provider calls get fake clients that hand back
real SDK response objects, so the call, parsing, validation and retry logic run
as in production. Expected values are worked out by hand in the comments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from openai.types.chat import ChatCompletion
from tenacity import wait_none

from src import scoring
from src.config import (
    LABEL_CHECK_MODEL,
    LABEL_MODEL,
    LABEL_PROMPT_PATH,
    SPEECH_TYPES,
    TOPIC_BLOCKS,
    TOPICS,
)
from src.labeling import (
    LABEL_MODEL_SPECS,
    LabelParseError,
    even_quota,
    ids_from_file,
    ids_from_run,
    ids_sha256,
    label_one,
    label_settings,
    load_labels,
    new_manifest,
    pending_ids,
    preflight,
    stratified_subset,
    summarize_labels,
    topic_block,
    validate_labels,
)
from src.scoring import MODEL_SPECS, load_prompt_template, render_prompt

# --- the codebook prompt ------------------------------------------------------


def test_the_label_prompt_has_exactly_one_placeholder_and_renders() -> None:
    template = load_prompt_template(LABEL_PROMPT_PATH)

    assert template.count("{speech_text}") == 1
    # The only {word} field is the speech: the JSON example has no other.
    assert re.findall(r"\{[a-z_]+\}", template) == ["{speech_text}"]
    rendered = render_prompt(template, "Mr. Speaker, I rise today.")
    assert "Mr. Speaker, I rise today." in rendered
    assert "{speech_text}" not in rendered


def _codes(template: str, start: str, end: str) -> list[str]:
    """The label strings listed between two headings of the prompt."""
    block = template.split(start, 1)[1].split(end, 1)[0]
    return re.findall(r"^  ([a-z_]+)\s{2,}\S", block, flags=re.MULTILINE)


def test_the_prompts_labels_are_exactly_the_configured_ones() -> None:
    """Guards against the prompt and config.py drifting apart."""
    template = LABEL_PROMPT_PATH.read_text(encoding="utf-8")

    assert tuple(_codes(template, "QUESTION 1", "QUESTION 2")) == SPEECH_TYPES
    assert tuple(_codes(template, "QUESTION 2", "Rules:")) == TOPICS


def test_every_topic_is_in_exactly_one_block() -> None:
    blocked = [topic for topics in TOPIC_BLOCKS.values() for topic in topics]

    assert sorted(blocked) == sorted(TOPICS)
    assert len(blocked) == len(set(blocked))


def test_a_topic_maps_to_its_block_and_no_topic_to_none() -> None:
    assert topic_block("immigration") == "social_cultural"
    assert topic_block("health") == "economic"
    assert topic_block("law_crime") == "security_foreign"
    assert topic_block(None) is None


# --- validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"speech_type": "policy_debate", "topic": "health", "reason": "ACA"},
            ("policy_debate", "health", "ACA"),
        ),
        (
            {"speech_type": "procedural", "topic": None, "reason": "UC request"},
            ("procedural", None, "UC request"),
        ),
        ({"speech_type": "ceremonial", "topic": None}, ("ceremonial", None, "")),
        (
            {"speech_type": "other", "topic": None, "reason": 7, "extra": 1},
            ("other", None, "7"),
        ),
    ],
)
def test_valid_labels_pass(payload, expected) -> None:
    assert validate_labels(payload) == expected


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"topic": None}, "missing field 'speech_type'"),
        ({"speech_type": "speech", "topic": None}, "not in the codebook"),
        ({"speech_type": "Procedural", "topic": None}, "not in the codebook"),
        ({"speech_type": " procedural", "topic": None}, "not in the codebook"),
        ({"speech_type": 1, "topic": None}, "not in the codebook"),
        ({"speech_type": "policy_debate", "topic": None}, "needs one"),
        ({"speech_type": "policy_debate", "topic": "taxes"}, "not in the codebook"),
        ({"speech_type": "policy_debate", "topic": "Health"}, "not in the codebook"),
        ({"speech_type": "ceremonial", "topic": "health"}, "only policy_debate"),
        ({"speech_type": "ceremonial", "topic": "null"}, "only policy_debate"),
        ({"speech_type": "procedural"}, "missing field 'topic'"),
    ],
    ids=[
        "no-type", "unknown-type", "wrong-case", "padded", "not-a-string",
        "policy-without-topic", "unknown-topic", "topic-wrong-case",
        "topic-on-non-policy", "string-null", "no-topic-key",
    ],
)  # fmt: skip
def test_labels_outside_the_codebook_are_failures(payload, message) -> None:
    """Never mapped to the nearest code: a model ignoring the codebook is a finding."""
    with pytest.raises(LabelParseError, match=message):
        validate_labels(payload)


# --- one call ---------------------------------------------------------------------

ANSWER = '{"speech_type": "policy_debate", "topic": "health", "reason": "drug prices"}'


def _completion(content: str, finish: str = "stop", out: int = 30) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-flash",
            "system_fingerprint": "fp_label",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "logprobs": None,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 900,
                "completion_tokens": out,
                "total_tokens": 900 + out,
            },
        }
    )


def _client(response: ChatCompletion, calls: list | None = None, hang: int = 0):
    """Answers `response`; hangs on its first `hang` calls."""
    seen: list = [] if calls is None else calls

    async def create(**kwargs):
        seen.append(kwargs)
        if len(seen) <= hang:
            await asyncio.sleep(10)
        return response

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_a_valid_answer_becomes_labels() -> None:
    spec = LABEL_MODEL_SPECS[LABEL_MODEL]

    result = asyncio.run(label_one(spec, _client(_completion(ANSWER)), "p"))

    assert result.ok
    assert (result.speech_type, result.topic) == ("policy_debate", "health")
    assert (result.prompt_tokens, result.completion_tokens) == (900, 30)
    assert result.system_fingerprint == "fp_label"


def test_an_answer_outside_the_codebook_fails_but_keeps_its_tokens() -> None:
    """Those tokens were billed."""
    answer = '{"speech_type": "tribute", "topic": null, "reason": "r"}'
    spec = LABEL_MODEL_SPECS[LABEL_MODEL]

    result = asyncio.run(label_one(spec, _client(_completion(answer, out=25)), "p"))

    assert not result.ok
    assert "LabelParseError" in result.error and "tribute" in result.error
    assert result.speech_type is None
    assert (result.prompt_tokens, result.completion_tokens) == (900, 25)


def test_a_truncated_answer_fails_but_keeps_its_tokens() -> None:
    spec = LABEL_MODEL_SPECS[LABEL_MODEL]

    result = asyncio.run(
        label_one(
            spec, _client(_completion('{"speech_type": "pol', "length", 256)), "p"
        )
    )

    assert "TruncatedResponseError" in result.error
    assert result.completion_tokens == 256


def test_the_label_call_turns_deepseek_thinking_off() -> None:
    """DeepSeek's documented toggle; no effort is sent with thinking off (L3)."""
    calls: list = []

    asyncio.run(
        label_one(
            LABEL_MODEL_SPECS[LABEL_MODEL], _client(_completion(ANSWER), calls), "p"
        )
    )

    assert calls[0]["model"] == "deepseek-flash"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in calls[0]
    assert "response_format" not in calls[0]
    assert calls[0]["max_tokens"] == 256


def test_the_scoring_spec_still_thinks() -> None:
    """The label spec is separate: the spec that scored S8 is unchanged."""
    assert MODEL_SPECS["deepseek-flash"].request_options == {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert MODEL_SPECS["deepseek-flash"].max_output_tokens == 8192


def test_the_check_model_keeps_json_mode_under_the_label_cap() -> None:
    calls: list = []
    spec = LABEL_MODEL_SPECS[LABEL_CHECK_MODEL]

    asyncio.run(label_one(spec, _client(_completion(ANSWER), calls), "p"))

    assert calls[0]["model"] == "gpt-4o-2024-11-20"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["max_tokens"] == 256
    assert spec.effective_temperature == "provider default"


@pytest.fixture
def short_deadline(monkeypatch):
    """A 50 ms deadline and no back-off, set where scoring keeps them."""
    monkeypatch.setattr(scoring, "CALL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(scoring, "RETRY_WAIT", wait_none())


def test_a_hung_label_call_is_abandoned_and_retried(short_deadline) -> None:
    """The same deadline and retry as scoring (S14)."""
    calls: list = []
    spec = LABEL_MODEL_SPECS[LABEL_MODEL]

    result = asyncio.run(label_one(spec, _client(_completion(ANSWER), calls, 1), "p"))

    assert result.ok and len(calls) == 2


def test_a_label_call_that_never_answers_fails_after_three_tries(
    short_deadline,
) -> None:
    calls: list = []
    spec = LABEL_MODEL_SPECS[LABEL_MODEL]

    result = asyncio.run(label_one(spec, _client(_completion(ANSWER), calls, 99), "p"))

    assert not result.ok and "TimeoutError" in result.error
    assert len(calls) == 3


# --- which speeches ----------------------------------------------------------------


def test_ids_are_read_from_a_scoring_runs_ensemble(tmp_path: Path) -> None:
    run = "20261003T110419Z"
    (tmp_path / f"s8_manifest_{run}.json").write_text("{}")
    (tmp_path / f"s8_ensemble_{run}.jsonl").write_text(
        "".join(json.dumps({"speech_id": s, "n_models": 2}) + "\n" for s in ("b", "a"))
    )

    assert ids_from_run(run, tmp_path) == ["b", "a"]


def test_ids_are_read_from_a_csv_column_as_strings(tmp_path: Path) -> None:
    """Stanford ids look numeric; a leading zero must survive."""
    path = tmp_path / "sheet.csv"
    pd.DataFrame({"speech_id": ["0107000430", "gov-1"], "text": ["x", "y"]}).to_csv(
        path, index=False
    )

    assert ids_from_file(path) == ["0107000430", "gov-1"]


def test_ids_are_read_one_per_line_skipping_blanks_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# hand-picked\ns1\n\n  s2 \n")

    assert ids_from_file(path) == ["s1", "s2"]


def test_an_id_listed_twice_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("s1\ns2\ns1\n")

    with pytest.raises(ValueError, match="twice"):
        ids_from_file(path)


def test_the_id_hash_ignores_order() -> None:
    assert ids_sha256(["a", "b"]) == ids_sha256(["b", "a"])
    assert ids_sha256(["a", "b"]) != ids_sha256(["a", "c"])


def _strata_frame() -> pd.DataFrame:
    """Four chamber x source cells of 30 speeches each."""
    rows = [
        {
            "speech_id": f"{chamber}{source[0]}{n:02d}",
            "chamber": chamber,
            "source": source,
        }
        for chamber in ("H", "S")
        for source in ("stanford", "govinfo")
        for n in range(30)
    ]
    return pd.DataFrame(rows)


def test_a_limit_draws_evenly_over_the_strata() -> None:
    frame = _strata_frame()

    ids = stratified_subset(frame, 20, ["chamber", "source"], seed=42)

    drawn = frame[frame["speech_id"].isin(ids)]
    assert len(ids) == 20
    assert drawn.groupby(["chamber", "source"]).size().tolist() == [5, 5, 5, 5]


def test_a_limit_is_deterministic_and_depends_on_the_seed() -> None:
    frame = _strata_frame()

    first = stratified_subset(frame, 20, ["chamber", "source"], seed=42)

    assert stratified_subset(frame, 20, ["chamber", "source"], seed=42) == first
    assert stratified_subset(frame.sample(frac=1, random_state=1), 20,
                             ["chamber", "source"], seed=42) == first  # fmt: skip
    assert stratified_subset(frame, 20, ["chamber", "source"], seed=7) != first


def test_a_limit_at_least_the_frame_takes_everything() -> None:
    frame = _strata_frame()

    assert len(stratified_subset(frame, 500, ["chamber"], seed=42)) == 120


def test_a_short_stratum_gives_its_shortfall_to_the_others() -> None:
    # 10 over three strata is 4/3/3; "a" holds 1, so 9 go to "b" and "c".
    quota = even_quota({"a": 1, "b": 100, "c": 100}, 10)

    assert quota == {"a": 1, "b": 5, "c": 4}


# --- cost ----------------------------------------------------------------------------


def test_the_estimate_prices_tiktoken_input_and_assumed_output() -> None:
    estimate = preflight({"s1": 1000, "s2": 3000}, [LABEL_MODEL, LABEL_CHECK_MODEL], 45)

    # DeepSeek: 4,000 in at $0.30/1M + 90 out at $1.20/1M
    #   = (1,200 + 108) / 1M = $0.001308; per 1,000 speeches $0.654.
    deepseek = estimate["per_model"][LABEL_MODEL]
    assert (deepseek["prompt_tokens"], deepseek["completion_tokens"]) == (4000, 90)
    assert deepseek["cost_usd"] == pytest.approx(0.001308)
    assert deepseek["cost_per_1000_speeches_usd"] == pytest.approx(0.654)
    # GPT-4o: (4,000 * 2.50 + 90 * 10.00) / 1M = $0.0109.
    assert estimate["per_model"][LABEL_CHECK_MODEL]["cost_usd"] == pytest.approx(0.0109)
    assert estimate["speeches"] == 2


# --- the manifest ----------------------------------------------------------------------


def test_the_manifest_records_what_defines_the_labels(tmp_path: Path) -> None:
    template = "Code this: {speech_text}"
    settings = label_settings(template, "corpus-hash", LABEL_MODEL)

    manifest = new_manifest(
        "T", "s8", LABEL_MODEL, settings, template, Path("p.txt"), Path("c.parquet"), 42
    )

    assert manifest["codebook_version"] == "v1"
    assert manifest["codebook"] == {
        "speech_types": list(SPEECH_TYPES),
        "topics": list(TOPICS),
    }
    assert manifest["prompt_sha256"] == hashlib.sha256(template.encode()).hexdigest()
    assert manifest["prompt_template"] == template
    assert manifest["corpus_sha256"] == "corpus-hash"
    assert manifest["models"][LABEL_MODEL] == {
        "model": "deepseek-flash",
        "provider": "openai_compatible",
        "effective_temperature": "provider default",
        "max_output_tokens": 256,
        "request_options": {"extra_body": {"thinking": {"type": "disabled"}}},
    }
    assert manifest["pricing"] == {LABEL_MODEL: {"input": 0.30, "output": 1.20}}
    assert manifest["random_seed"] == 42 and manifest["stages"] == []


def test_a_changed_label_spec_changes_the_settings(monkeypatch) -> None:
    before = label_settings("t {speech_text}", "h", LABEL_MODEL)
    spec = replace(LABEL_MODEL_SPECS[LABEL_MODEL], max_output_tokens=512)
    monkeypatch.setitem(LABEL_MODEL_SPECS, LABEL_MODEL, spec)

    assert label_settings("t {speech_text}", "h", LABEL_MODEL) != before


# --- reading a run back -------------------------------------------------------------------


def _label_run(tmp_path: Path, rows: list[dict], run: str = "T1") -> Path:
    (tmp_path / f"label_manifest_s8_{run}.json").write_text(
        json.dumps({"model": LABEL_MODEL, "stages": []})
    )
    (tmp_path / f"label_s8_{LABEL_MODEL}_{run}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return tmp_path


def _row(sid, speech_type=None, topic=None, error=None, tokens=(900, 30)) -> dict:
    return {
        "speech_id": sid,
        "speech_type": speech_type,
        "topic": topic,
        "reason": "r",
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "reasoning_tokens": None,
        "served_model": "deepseek-flash",
        "system_fingerprint": "fp",
        "error": error,
    }


ROWS = [
    _row("s1", error="RateLimitError: slow down", tokens=(0, 0)),
    _row("s1", "policy_debate", "immigration"),
    _row("s2", error="LabelParseError: speech_type 'tribute' is not in the codebook"),
    _row("s3", "ceremonial"),
]


def test_load_labels_takes_the_attempt_that_counts(tmp_path: Path) -> None:
    labels = load_labels("T1", _label_run(tmp_path, ROWS)).set_index("speech_id")

    assert labels.loc["s1", "speech_type"] == "policy_debate"
    assert labels.loc["s1", "topic_block"] == "social_cultural"
    assert not labels.loc["s2", "label_ok"]
    assert pd.isna(labels.loc["s2", "speech_type"])
    assert pd.isna(labels.loc["s3", "topic"]) and pd.isna(
        labels.loc["s3", "topic_block"]
    )


def test_only_speeches_without_valid_labels_are_pending() -> None:
    assert pending_ids(["s1", "s2", "s3", "s4"], ROWS) == ["s2", "s4"]


def test_the_summary_counts_labels_failures_and_every_billed_attempt() -> None:
    speeches = [
        {"speech_id": "s1", "chamber": "H", "source": "stanford"},
        {"speech_id": "s2", "chamber": "S", "source": "govinfo"},
        {"speech_id": "s3", "chamber": "S", "source": "govinfo"},
    ]

    summary = summarize_labels(ROWS, speeches, LABEL_MODEL)

    assert (summary["speeches_labelled"], summary["failures"]) == (2, 1)
    assert (summary["calls"], summary["failed_attempts"]) == (4, 2)
    assert summary["errors_by_type"] == {"LabelParseError": 1}
    assert summary["speech_type_counts"]["ceremonial"] == 1
    assert summary["topic_counts"]["immigration"] == 1
    assert summary["topic_block_counts"]["social_cultural"] == 1
    # S: s3 labelled ceremonial, s2 failed -> 1 labelled, all ceremonial.
    senate = summary["speech_type_shares_by_chamber"]["S"]
    assert senate["n"] == 1 and senate["speech_type"]["ceremonial"] == 1.0
    # Three billed calls of 900 + 30 tokens, one of 0.
    assert summary["prompt_tokens"] == 2700 and summary["completion_tokens"] == 90
    assert summary["cost_usd"] == pytest.approx((2700 * 0.30 + 90 * 1.20) / 1e6)
    assert summary["system_fingerprints"] == ["fp"]
