"""Score congressional speeches through the model ensemble.

Provider wiring, prompt rendering, response parsing and retry logic. The pilot
(`code/scripts/pilot_run.py`) and any later sampled run import this, so the
clients exist in one place rather than being copied. There is no full-corpus
run (docs/decisions.md S11).

Each call asks one model for a JSON object with an ideological position, a tone
score and a short justification. What comes back is validated against the range
the prompt promised; anything outside it is recorded as a failure rather than
clipped, because a model ignoring the scale is a finding and not something to
quietly rescue.

No temperature is set on any model. The providers removed the control rather
than us declining to use it: DeepSeek ignores it, and the anthropic
SDK dropped the parameter outright (current models answer "`temperature` is
deprecated for this model"). Setting it on GPT-4o alone would imply an ensemble
tuned alike, so every model runs at its provider default and every run manifest
says so. `ModelSpec.effective_temperature` is what belongs in the write-up.

Anthropic returns JSON through structured outputs (`output_config.format`).
Assistant prefill -- the usual way to force JSON before structured outputs --
returns a 400 on current Claude models and is not an option.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import openai
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import (
    CALL_TIMEOUT_SECONDS,
    ENSEMBLE_MODELS,
    IDEOLOGY_SCORE_RANGE,
    MODEL_PRICING,
    SCORE_PROMPT_PATH,
    TONE_SCORE_RANGE,
)

# --- provider wiring ---------------------------------------------------

#: Placeholder in the prompt template that receives the speech text.
SPEECH_PLACEHOLDER = "{speech_text}"

#: Max tokens for the answer. The prompt asks for two floats and 1-2 sentences,
#: so this is ample for an instruction-following model.
MAX_OUTPUT_TOKENS = 1024

#: Reasoning models need far more headroom, because their reasoning tokens are
#: spent from the SAME budget as the answer. At 1024 on a 623-word speech,
#: the DeepSeek reasoner used the entire cap reasoning and returned empty content
#: with finish_reason="length". Raising the cap also costs LESS: given room to
#: finish, it reasoned to a natural stop in 486 tokens instead of being
#: truncated at 1024. Billing is on tokens used, not the cap.
REASONING_MAX_OUTPUT_TOKENS = 8192

#: The answer shape, as a JSON schema. Anthropic enforces it server-side via
#: `output_config.format`, so the reply is valid JSON by construction.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "ideology_score": {"type": "number"},
            "tone_score": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["ideology_score", "tone_score", "reasoning"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class ModelSpec:
    """How to reach one ensemble member, and how it must be called."""

    model: str
    provider: str  # "openai_compatible" | "anthropic"
    api_key_env: str
    base_url: str | None = None
    #: True where the provider supports OpenAI-style JSON mode.
    supports_json_mode: bool = False
    #: True where the provider supports a server-enforced JSON schema.
    supports_json_schema: bool = False
    #: Output cap. Reasoning models need headroom for reasoning tokens.
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    #: Extra arguments sent with every call. Recorded in the run manifest,
    #: because they change what the model does.
    request_options: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_temperature(self) -> str:
        """What this model actually runs at -- logged with every run.

        Always the provider default: no provider in the ensemble still accepts
        a temperature through its current SDK.
        """
        return "provider default"


#: Keyed by the entries of ENSEMBLE_MODELS in config.py. Change models there.
MODEL_SPECS: dict[str, ModelSpec] = {
    "deepseek-flash": ModelSpec(
        model="deepseek-flash",
        provider="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        # JSON mode is not used: the answer is parsed out of the content, as for
        # every run so far, so the scored output is produced the same way.
        supports_json_mode=False,
        # Reasoning is billed from the same budget as the answer.
        max_output_tokens=REASONING_MAX_OUTPUT_TOKENS,
        # Thinking mode at high effort. Both are DeepSeek's defaults today;
        # pinned so a changed default cannot silently change the scores (M3b).
        request_options={
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        },
    ),
    "gpt-4o-2024-11-20": ModelSpec(
        model="gpt-4o-2024-11-20",
        provider="openai_compatible",
        api_key_env="OPENAI_API_KEY",
        supports_json_mode=True,
    ),
    "claude-sonnet-4-6": ModelSpec(
        model="claude-sonnet-4-6",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        # Server-enforced schema. Prefill, the pre-structured-outputs way to
        # force JSON, returns a 400 on current Claude models.
        supports_json_schema=True,
    ),
}


@dataclass
class ScoreResult:
    """One model's answer for one speech. Nulls on failure, never an exception."""

    model: str
    ideology_score: float | None = None
    tone_score: float | None = None
    reasoning: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    #: The model id the provider says answered. A model name can be repointed
    #: by the provider, so the requested name alone does not say this.
    served_model: str | None = None
    #: OpenAI-compatible backends' build identifier; None for Anthropic.
    system_fingerprint: str | None = None
    #: Output tokens spent reasoning, where the provider reports them; shows
    #: that thinking mode was actually on.
    reasoning_tokens: int | None = None

    @property
    def ok(self) -> bool:
        """True when both scores parsed and validated."""
        return self.error is None and self.ideology_score is not None


@dataclass(frozen=True)
class RawResponse:
    """What one provider call returned, before parsing."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    served_model: str | None = None
    system_fingerprint: str | None = None
    reasoning_tokens: int | None = None


class ScoreParseError(ValueError):
    """A response could not be turned into two valid scores."""


class CallTimeoutError(TimeoutError):
    """A provider call gave no answer within CALL_TIMEOUT_SECONDS."""


class TruncatedResponseError(ScoreParseError):
    """The model hit its output cap before producing an answer.

    Carries the response, because the tokens it burned were billed.
    """

    def __init__(self, message: str, response: RawResponse) -> None:
        super().__init__(message)
        self.response = response


# --- prompt ------------------------------------------------------------


def load_prompt_template(path: Path = SCORE_PROMPT_PATH) -> str:
    """Read the scoring prompt template, checking it has the placeholder."""
    template = path.read_text(encoding="utf-8")
    if SPEECH_PLACEHOLDER not in template:
        raise ValueError(
            f"{path} does not contain {SPEECH_PLACEHOLDER}; nothing to substitute."
        )
    return template


def render_prompt(template: str, speech_text: str) -> str:
    """Substitute the speech into the template.

    Uses `str.replace`, deliberately: the template shows the expected answer as a
    literal JSON object, so it contains `{` and `}` that `str.format` would try
    to interpret as fields and raise on.
    """
    return template.replace(SPEECH_PLACEHOLDER, speech_text)


# --- response parsing --------------------------------------------------


def extract_json(raw: str) -> dict[str, Any]:
    """Return the first complete JSON object in `raw`.

    Not `json.loads(raw)`: responses are not reliably bare JSON. The reasoner
    emits its reasoning first, models add prose or fences, and no single
    response_format works across all three providers. Scanning brace depth finds
    the object wherever it sits.
    """
    start = raw.find("{")
    if start == -1:
        raise ScoreParseError(f"no JSON object in response: {raw[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : index + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as error:
                    raise ScoreParseError(
                        f"malformed JSON object: {error}: {candidate[:200]!r}"
                    ) from error
    raise ScoreParseError(f"unterminated JSON object in response: {raw[:200]!r}")


def _as_score(
    payload: dict[str, Any], field: str, bounds: tuple[float, float]
) -> float:
    """Pull one numeric field and check it against the range the prompt promised."""
    if field not in payload:
        raise ScoreParseError(f"missing field {field!r}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreParseError(f"{field} is not a number: {value!r}")
    low, high = bounds
    if not low <= value <= high:
        # Deliberately not clipped: a model outside the stated scale is a
        # finding about the model, and clipping would hide it.
        raise ScoreParseError(f"{field}={value} outside [{low}, {high}]")
    return float(value)


def validate_scores(payload: dict[str, Any]) -> tuple[float, float, str]:
    """Return (ideology, tone, reasoning), raising if the payload is unusable."""
    ideology = _as_score(payload, "ideology_score", IDEOLOGY_SCORE_RANGE)
    tone = _as_score(payload, "tone_score", TONE_SCORE_RANGE)
    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return ideology, tone, reasoning


# --- clients -----------------------------------------------------------


def build_clients(
    specs: dict[str, ModelSpec] | None = None,
) -> dict[str, openai.AsyncOpenAI | anthropic.AsyncAnthropic]:
    """Create one async client per model, raising if a key is missing.

    Checked up front so a missing key fails before any speech is scored, rather
    than after paying for the two providers that were configured.
    """
    specs = specs or MODEL_SPECS
    missing = [s.api_key_env for s in specs.values() if not os.getenv(s.api_key_env)]
    if missing:
        raise RuntimeError(
            "Missing API key(s): "
            + ", ".join(sorted(set(missing)))
            + ". Copy .env.example to .env and fill them in."
        )

    clients: dict[str, openai.AsyncOpenAI | anthropic.AsyncAnthropic] = {}
    for name, spec in specs.items():
        key = os.environ[spec.api_key_env]
        if spec.provider == "anthropic":
            clients[name] = anthropic.AsyncAnthropic(api_key=key)
        else:
            clients[name] = openai.AsyncOpenAI(api_key=key, base_url=spec.base_url)
    return clients


#: Transient failures worth retrying. Anything else is a real error and should
#: surface immediately rather than being retried three times.
RETRYABLE = (
    CallTimeoutError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


#: Back-off between attempts.
RETRY_WAIT = wait_exponential(multiplier=2, min=2, max=30)


def _retrying() -> AsyncRetrying:
    """Three attempts, exponential backoff from 2s, reraising the final error."""
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=RETRY_WAIT,
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )


async def _call_openai_compatible(
    spec: ModelSpec, client: openai.AsyncOpenAI, prompt: str
) -> RawResponse:
    """Call an OpenAI-compatible endpoint."""
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": spec.max_output_tokens,
    }
    if spec.supports_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs.update(spec.request_options)

    response = await client.chat.completions.create(**kwargs)
    usage = response.usage
    choice = response.choices[0]
    details = getattr(usage, "completion_tokens_details", None)
    raw = RawResponse(
        text=choice.message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        served_model=response.model,
        system_fingerprint=response.system_fingerprint,
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
    )

    # Say so explicitly. A reasoning model that runs out of budget returns
    # empty content, which would otherwise surface as a confusing "no JSON
    # object in response: ''".
    if choice.finish_reason == "length":
        raise TruncatedResponseError(
            f"{spec.model} hit its {spec.max_output_tokens}-token cap before "
            f"answering (used {raw.completion_tokens}); raise max_output_tokens",
            raw,
        )
    return raw


async def _call_anthropic(
    spec: ModelSpec, client: anthropic.AsyncAnthropic, prompt: str
) -> RawResponse:
    """Call Anthropic.

    JSON comes back through structured outputs, which the server enforces
    against RESPONSE_SCHEMA. Assistant prefill -- the older trick for forcing
    JSON -- returns a 400 on current Claude models. `thinking` is left off: the
    task is a short judgement, and adaptive thinking would roughly double the
    output tokens.
    """
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": spec.max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if spec.supports_json_schema:
        kwargs["output_config"] = {"format": RESPONSE_SCHEMA}
    kwargs.update(spec.request_options)

    response = await client.messages.create(**kwargs)
    raw = RawResponse(
        text="".join(block.text for block in response.content if block.type == "text"),
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        served_model=response.model,
    )
    if response.stop_reason == "max_tokens":
        raise TruncatedResponseError(
            f"{spec.model} hit its {spec.max_output_tokens}-token cap before "
            f"answering; raise max_output_tokens",
            raw,
        )
    return raw


async def score_one(
    spec: ModelSpec,
    client: openai.AsyncOpenAI | anthropic.AsyncAnthropic,
    prompt: str,
) -> ScoreResult:
    """Score one speech with one model. Never raises -- failures become nulls.

    A single bad response must not end a run that has already been paid for, so
    everything is caught and recorded. Token counts are kept even on a parse
    failure or a truncation, because those tokens were billed.
    """
    result = ScoreResult(model=spec.model)

    def record(raw: RawResponse) -> None:
        result.prompt_tokens = raw.prompt_tokens
        result.completion_tokens = raw.completion_tokens
        result.served_model = raw.served_model
        result.system_fingerprint = raw.system_fingerprint
        result.reasoning_tokens = raw.reasoning_tokens

    call = _call_anthropic if spec.provider == "anthropic" else _call_openai_compatible
    try:
        async for attempt in _retrying():
            with attempt:
                # Our own deadline: the SDK's timeout did not fire on a call
                # that hung for 20 minutes (S14). Tokens of an abandoned call
                # are unknown, so they cannot be recorded.
                try:
                    raw = await asyncio.wait_for(
                        call(spec, client, prompt), timeout=CALL_TIMEOUT_SECONDS
                    )
                except TimeoutError as error:
                    raise CallTimeoutError(
                        f"{spec.model} gave no answer within {CALL_TIMEOUT_SECONDS:g}s"
                    ) from error
    except TruncatedResponseError as error:
        record(error.response)
        result.error = f"{type(error).__name__}: {error}"
        return result
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        result.error = f"{type(error).__name__}: {error}"
        return result

    record(raw)
    try:
        ideology, tone, reasoning = validate_scores(extract_json(raw.text))
    except ScoreParseError as error:
        result.error = f"{type(error).__name__}: {error}"
        return result

    result.ideology_score = ideology
    result.tone_score = tone
    result.reasoning = reasoning
    return result


# --- cost --------------------------------------------------------------


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD for one model at the given token counts.

    An estimate against list prices in config.py, not an invoice: provider
    pricing changes, and discounts and cache pricing are not modelled.
    """
    if model not in MODEL_PRICING:
        raise KeyError(f"no pricing for {model!r}; add it to MODEL_PRICING")
    price = MODEL_PRICING[model]
    return (
        prompt_tokens * price["input"] + completion_tokens * price["output"]
    ) / 1_000_000


def specs_for(models: tuple[str, ...] = ENSEMBLE_MODELS) -> dict[str, ModelSpec]:
    """Return the specs for `models`, raising if one has no wiring."""
    missing = [m for m in models if m not in MODEL_SPECS]
    if missing:
        raise KeyError(
            f"no ModelSpec for {', '.join(missing)}; add it to MODEL_SPECS in "
            "code/src/scoring.py"
        )
    return {m: MODEL_SPECS[m] for m in models}
