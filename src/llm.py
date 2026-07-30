"""LLM provider shim.

Single place the project talks to an LLM. OpenAI-backed, and deliberately
JSON-schema-only: every caller gets a validated dict or an exception, never a
string it has to parse and guess about.

That constraint is the point. The previous Anthropic path asked for JSON in
the prompt and regex-scraped the reply, so a response truncated by max_tokens
came back as unparseable text, was swallowed by a bare `except`, and surfaced
as a neutral-looking default — 5 of 20 screener rows on 2026-07-29 silently
read "wait / Analysis unavailable" that way. Strict structured output makes
malformed JSON impossible, and the errors below make the remaining failure
modes (truncation, refusal, missing key) loud instead of silent.
"""
import json
import os
import threading


class LLMError(RuntimeError):
    """Any failure to obtain a valid structured response.

    Callers decide whether to degrade or propagate — but they have to decide,
    which is the improvement over a swallowed exception.
    """


_client = None
_client_lock = threading.Lock()


def _get_client():
    """Lazily build one shared client. The OpenAI client is thread-safe and
    pools connections, so news.py's ThreadPoolExecutor shares this instance;
    the lock guards only the one-time construction."""
    global _client
    with _client_lock:
        if _client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMError("OPENAI_API_KEY not set")
            from openai import OpenAI
            _client = OpenAI(api_key=key)
        return _client


def client():
    """The shared OpenAI client, for callers that need an endpoint this module
    doesn't wrap (the Batch API in src/event_backtest.py)."""
    return _get_client()


def available() -> bool:
    """True when a call could plausibly succeed. Lets callers skip a stage
    cleanly instead of raising per item."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def object_schema(properties: dict, required: list[str] | None = None) -> dict:
    """Build a schema accepted by OpenAI strict mode.

    Strict mode demands additionalProperties:false and *every* property listed
    in required — optional fields are not allowed. Centralised here so callers
    can't forget and get a 400 at runtime.
    """
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


def complete_json(
    prompt: str,
    schema: dict,
    model: str,
    max_tokens: int = 2000,
    name: str = "result",
) -> dict:
    """Return a dict conforming to `schema`, or raise LLMError.

    `max_tokens` is a safety ceiling, not a target: truncation raises rather
    than returning a partial object, because a half-filled analysis is worse
    than a declared failure when the output feeds a buy/sell decision.
    """
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"{model} request failed: {e!r}") from e

    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise LLMError(f"{model} response truncated at max_completion_tokens={max_tokens}")
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        raise LLMError(f"{model} refused: {refusal}")
    content = choice.message.content
    if not content:
        raise LLMError(f"{model} returned empty content (finish_reason={choice.finish_reason})")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        # Should be unreachable under strict mode; if the guarantee ever
        # regresses, fail loudly rather than reintroducing silent fallbacks.
        raise LLMError(f"{model} returned non-JSON under strict schema: {e}") from e
    if not isinstance(parsed, dict):
        raise LLMError(f"{model} returned {type(parsed).__name__}, expected object")
    return parsed
