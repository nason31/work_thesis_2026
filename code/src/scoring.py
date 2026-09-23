"""Score congressional speeches through the three-model ensemble.

Provider wiring, prompt rendering, response parsing and retry logic. The pilot
(`code/scripts/pilot_run.py`) and the eventual full run both import this, so
the clients exist in one place rather than being copied.

Each call asks one model for a JSON object with an ideological position, a tone
score and a short justification. What comes back is validated against the range
the prompt promised; anything outside it is recorded as a failure rather than
clipped, because a model ignoring the scale is a finding and not something to
quietly rescue.

No temperature is set on any model. The providers removed the control rather
than us declining to use it: `deepseek-reasoner` ignores it, and the anthropic
SDK dropped the parameter outright (current models answer "`temperature` is
deprecated for this model"). Setting it on GPT-4o alone would imply an ensemble
tuned alike, so all three run at their provider default and every run manifest
says so. `ModelSpec.effective_temperature` is what belongs in the write-up.

Anthropic returns JSON through structured outputs (`output_config.format`).
Assistant prefill -- the usual way to force JSON before structured outputs --
returns a 400 on current Claude models and is not an option.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
#: so this is generous; `deepseek-reasoner` also spends hidden reasoning tokens
#: that do not count against it.
MAX_OUTPUT_TOKENS = 1024

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

    @property
    def effective_temperature(self) -> str:
        """What this model actually runs at -- logged with every run.

        Always the provider default: no provider in the ensemble still accepts
        a temperature through its current SDK.
        """
        return "provider default"


#: Keyed by the entries of ENSEMBLE_MODELS in config.py. Change models there.
MODEL_SPECS: dict[str, ModelSpec] = {
    "deepseek-reasoner": ModelSpec(
        model="deepseek-reasoner",
        provider="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        # Reasoning output precedes the answer, so JSON mode does not apply.
        supports_json_mode=False,
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

    @property
    def ok(self) -> bool:
        """True when both scores parsed and validated."""
        return self.error is None and self.ideology_score is not None


class ScoreParseError(ValueError):
    """A response could not be turned into two valid scores."""


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
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


def _retrying() -> AsyncRetrying:
    """Three attempts, exponential backoff from 2s, reraising the final error."""
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    )


async def _call_openai_compatible(
    spec: ModelSpec, client: openai.AsyncOpenAI, prompt: str
) -> tuple[str, int, int]:
    """Call an OpenAI-compatible endpoint; return (text, prompt_tok, completion_tok)."""
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    if spec.supports_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    usage = response.usage
    return (
        response.choices[0].message.content or "",
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )


async def _call_anthropic(
    spec: ModelSpec, client: anthropic.AsyncAnthropic, prompt: str
) -> tuple[str, int, int]:
    """Call Anthropic; return (text, input_tok, output_tok).

    JSON comes back through structured outputs, which the server enforces
    against RESPONSE_SCHEMA. Assistant prefill -- the older trick for forcing
    JSON -- returns a 400 on current Claude models. `thinking` is left off: the
    task is a short judgement, and adaptive thinking would roughly double the
    output tokens.
    """
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if spec.supports_json_schema:
        kwargs["output_config"] = {"format": RESPONSE_SCHEMA}

    response = await client.messages.create(**kwargs)
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


async def score_one(
    spec: ModelSpec,
    client: openai.AsyncOpenAI | anthropic.AsyncAnthropic,
    prompt: str,
) -> ScoreResult:
    """Score one speech with one model. Never raises -- failures become nulls.

    A single bad response must not end a run that has already been paid for, so
    everything is caught and recorded. Token counts are kept even on a parse
    failure, because those tokens were billed.
    """
    result = ScoreResult(model=spec.model)
    raw, prompt_tokens, completion_tokens = "", 0, 0
    try:
        async for attempt in _retrying():
            with attempt:
                if spec.provider == "anthropic":
                    raw, prompt_tokens, completion_tokens = await _call_anthropic(
                        spec, client, prompt
                    )
                else:
                    raw, prompt_tokens, completion_tokens = (
                        await _call_openai_compatible(spec, client, prompt)
                    )
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        result.error = f"{type(error).__name__}: {error}"
        return result

    result.prompt_tokens = prompt_tokens
    result.completion_tokens = completion_tokens
    try:
        ideology, tone, reasoning = validate_scores(extract_json(raw))
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
