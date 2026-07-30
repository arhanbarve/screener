"""The shim's contract: a validated dict, or a loud LLMError. Never a silent
neutral-looking default — that behaviour is the bug it was written to kill.
"""
import pytest

from src.llm import LLMError, available, complete_json, object_schema


class _Msg:
    def __init__(self, content, refusal=None):
        self.content = content
        self.refusal = refusal


class _Choice:
    def __init__(self, content, finish_reason="stop", refusal=None):
        self.message = _Msg(content, refusal)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, *choices):
        self.choices = list(choices)


def _fake_client(monkeypatch, resp=None, exc=None):
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            if exc is not None:
                raise exc
            return resp

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr("src.llm._get_client", lambda: _Client())
    return captured


SCHEMA = object_schema({"a": {"type": "string"}})


class TestObjectSchema:
    def test_strict_mode_requirements_are_filled_in(self):
        s = object_schema({"x": {"type": "string"}, "y": {"type": "integer"}})
        assert s["additionalProperties"] is False
        assert set(s["required"]) == {"x", "y"}   # strict mode: ALL properties

    def test_required_can_be_narrowed_explicitly(self):
        s = object_schema({"x": {"type": "string"}, "y": {"type": "integer"}}, required=["x"])
        assert s["required"] == ["x"]


class TestCompleteJson:
    def test_returns_parsed_dict(self, monkeypatch):
        _fake_client(monkeypatch, _Resp(_Choice('{"a":"hi"}')))
        assert complete_json("p", SCHEMA, "m") == {"a": "hi"}

    def test_sends_strict_json_schema(self, monkeypatch):
        cap = _fake_client(monkeypatch, _Resp(_Choice('{"a":"hi"}')))
        complete_json("p", SCHEMA, "the-model", max_tokens=42, name="thing")
        rf = cap["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["name"] == "thing"
        assert cap["model"] == "the-model"
        # gpt-5 family takes max_completion_tokens, not max_tokens
        assert cap["max_completion_tokens"] == 42
        assert "max_tokens" not in cap

    def test_truncation_raises_instead_of_returning_a_partial(self, monkeypatch):
        # The exact failure that produced 5 silent "Analysis unavailable" rows
        _fake_client(monkeypatch, _Resp(_Choice('{"a":"hi', finish_reason="length")))
        with pytest.raises(LLMError, match="truncated"):
            complete_json("p", SCHEMA, "m", max_tokens=10)

    def test_refusal_raises(self, monkeypatch):
        _fake_client(monkeypatch, _Resp(_Choice(None, refusal="no thanks")))
        with pytest.raises(LLMError, match="refused"):
            complete_json("p", SCHEMA, "m")

    def test_empty_content_raises(self, monkeypatch):
        _fake_client(monkeypatch, _Resp(_Choice("")))
        with pytest.raises(LLMError, match="empty content"):
            complete_json("p", SCHEMA, "m")

    def test_transport_error_is_wrapped_not_leaked(self, monkeypatch):
        _fake_client(monkeypatch, exc=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="request failed"):
            complete_json("p", SCHEMA, "m")

    def test_non_object_json_raises(self, monkeypatch):
        _fake_client(monkeypatch, _Resp(_Choice('["not","an","object"]')))
        with pytest.raises(LLMError, match="expected object"):
            complete_json("p", SCHEMA, "m")

    def test_non_json_raises(self, monkeypatch):
        _fake_client(monkeypatch, _Resp(_Choice("sorry, here is prose")))
        with pytest.raises(LLMError, match="non-JSON"):
            complete_json("p", SCHEMA, "m")


class TestAvailable:
    def test_false_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert available() is False

    def test_true_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert available() is True


class TestMissingKey:
    def test_client_construction_raises_llmerror(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("src.llm._client", None)
        with pytest.raises(LLMError, match="OPENAI_API_KEY not set"):
            complete_json("p", SCHEMA, "m")
