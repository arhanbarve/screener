"""Entry-signal taxonomy.

The regression these guard: on 2026-07-29 every one of these distinct
situations came out of the pipeline as entry_signal="wait" —
  - no articles published
  - articles published, none material
  - the Finnhub fetch failed
  - the LLM call failed or its JSON didn't parse
so 13 of 20 screener rows showed a deliberate-looking "hold off" that nobody
had actually decided. "wait" must mean the model looked at real news and said
wait, and nothing else may render as it.
"""
import pandas as pd
import pytest

from src import news
from src.llm import LLMError


ARTICLE = {"headline": "Acme wins $400M contract", "summary": "Details.",
           "source": "Reuters", "datetime": 0}


class TestNoNewsIsNotWait:
    def test_zero_articles_is_no_news(self):
        r = news._analyze_stock("ACME", [], "ctx", "m", "pm")
        assert r["entry_signal"] == news.SIGNAL_NO_NEWS
        assert r["entry_signal"] != news.SIGNAL_WAIT
        assert "No news published" in r["reasoning"]

    def test_articles_but_none_material_is_no_news(self, monkeypatch):
        monkeypatch.setattr(news, "_prefilter_articles", lambda t, a, m: [])
        r = news._analyze_stock("ACME", [ARTICLE, ARTICLE], "ctx", "m", "pm")
        assert r["entry_signal"] == news.SIGNAL_NO_NEWS
        # says how many it looked at, so "no news" is auditable
        assert "2 articles" in r["reasoning"]

    def test_no_news_is_conviction_neutral(self):
        r = news._analyze_stock("ACME", [], "ctx", "m", "pm")
        assert r["conviction_delta"] == 0
        assert r["thesis_consistency"] == "neutral"
        assert r["catalyst"] == "none"


class TestFailureIsNotWait:
    def test_llm_failure_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(news, "_prefilter_articles", lambda t, a, m: [ARTICLE])

        def boom(*a, **k):
            raise LLMError("truncated at max_completion_tokens=600")

        monkeypatch.setattr(news, "complete_json", boom)
        r = news._analyze_stock("ACME", [ARTICLE], "ctx", "m", "pm")
        assert r["entry_signal"] == news.SIGNAL_UNAVAILABLE
        assert r["entry_signal"] != news.SIGNAL_WAIT
        # the actual cause survives into the row instead of being swallowed
        assert "truncated" in r["reasoning"]

    def test_unavailable_is_conviction_neutral(self, monkeypatch):
        monkeypatch.setattr(news, "_prefilter_articles", lambda t, a, m: [ARTICLE])
        monkeypatch.setattr(news, "complete_json",
                           lambda *a, **k: (_ for _ in ()).throw(LLMError("x")))
        r = news._analyze_stock("ACME", [ARTICLE], "ctx", "m", "pm")
        assert r["conviction_delta"] == 0


class TestPrefilterFailsOpen:
    def test_prefilter_error_keeps_all_articles(self, monkeypatch, capsys):
        """A broken filter must not masquerade as "nothing material" — that
        would be an editorial claim about the stock, not about our plumbing."""
        monkeypatch.setattr(news, "complete_json",
                           lambda *a, **k: (_ for _ in ()).throw(LLMError("boom")))
        out = news._prefilter_articles("ACME", [ARTICLE, ARTICLE], "pm")
        assert out == [ARTICLE, ARTICLE]
        assert "prefilter failed" in capsys.readouterr().out

    def test_prefilter_selects_by_index(self, monkeypatch):
        a, b, c = dict(ARTICLE, headline="a"), dict(ARTICLE, headline="b"), dict(ARTICLE, headline="c")
        monkeypatch.setattr(news, "complete_json", lambda *args, **k: {"material": [1, 3]})
        assert news._prefilter_articles("ACME", [a, b, c], "pm") == [a, c]

    def test_prefilter_empty_means_empty(self, monkeypatch):
        """Previously an empty result silently fell back to ALL articles,
        erasing the no-material-news case entirely."""
        monkeypatch.setattr(news, "complete_json", lambda *args, **k: {"material": []})
        assert news._prefilter_articles("ACME", [ARTICLE], "pm") == []


class TestRealSignalsSurvive:
    def test_model_verdict_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(news, "_prefilter_articles", lambda t, a, m: [ARTICLE])
        monkeypatch.setattr(news, "complete_json", lambda *a, **k: {
            "entry_signal": "avoid", "catalyst": "estimate_down", "priced_in": False,
            "duration": "weeks", "thesis_consistency": "contradicts",
            "conviction_delta": -1, "reasoning": "Downgrade.",
        })
        r = news._analyze_stock("ACME", [ARTICLE], "ctx", "m", "pm")
        assert r["entry_signal"] == "avoid"
        assert r["conviction_delta"] == -1

    def test_a_genuine_wait_is_still_wait(self, monkeypatch):
        monkeypatch.setattr(news, "_prefilter_articles", lambda t, a, m: [ARTICLE])
        monkeypatch.setattr(news, "complete_json", lambda *a, **k: {
            "entry_signal": "wait", "catalyst": "none", "priced_in": True,
            "duration": "days", "thesis_consistency": "neutral",
            "conviction_delta": 0, "reasoning": "Already ran on it.",
        })
        r = news._analyze_stock("ACME", [ARTICLE], "ctx", "m", "pm")
        assert r["entry_signal"] == news.SIGNAL_WAIT

    def test_conviction_delta_is_clamped(self, monkeypatch):
        monkeypatch.setattr(news, "_prefilter_articles", lambda t, a, m: [ARTICLE])
        monkeypatch.setattr(news, "complete_json", lambda *a, **k: {
            "entry_signal": "confirm_entry", "catalyst": "estimate_up", "priced_in": False,
            "duration": "weeks", "thesis_consistency": "confirms",
            "conviction_delta": 7, "reasoning": "Big.",
        })
        assert news._analyze_stock("ACME", [ARTICLE], "ctx", "m", "pm")["conviction_delta"] == 1


class TestMarketAnalysis:
    def test_sector_signal_list_becomes_a_lookup_dict(self, monkeypatch):
        """Strict schema can't express arbitrary keys, so the model returns a
        list; downstream code indexes by sector name."""
        monkeypatch.setattr(news, "complete_json", lambda *a, **k: {
            "regime_note": "Choppy.",
            "sector_signals": [
                {"sector": "Technology", "direction": "headwind",
                 "strength": "strong", "reason": "Rates."},
            ],
        })
        out = news._analyze_market([{"headline": "h"}], ["Technology"], "m")
        assert out["regime_note"] == "Choppy."
        assert out["sector_signals"]["Technology"]["direction"] == "headwind"

    def test_failure_yields_empty_signals_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(news, "complete_json",
                           lambda *a, **k: (_ for _ in ()).throw(LLMError("boom")))
        out = news._analyze_market([{"headline": "h"}], ["Technology"], "m")
        assert out == {"regime_note": "", "sector_signals": {}}


class TestFetchFailureIsDistinctFromNoNews:
    def test_finnhub_failure_is_unavailable_and_uncached(self, monkeypatch):
        put_calls = []
        monkeypatch.setattr(news, "get_news_sentiment", lambda *a, **k: None)
        monkeypatch.setattr(news, "put_news_sentiment",
                           lambda *a, **k: put_calls.append(a))
        monkeypatch.setattr(news, "_get_bucket", lambda rate: type("B", (), {"consume": lambda s: None})())

        def boom_client():
            raise RuntimeError("finnhub 503")

        monkeypatch.setattr(news, "_fh", boom_client)
        ticker, r = news._process_one(
            ("ACME", pd.Series({"ticker": "ACME"}), 1, 20, None, 60, ":memory:", 4, "m", "pm")
        )
        assert r["entry_signal"] == news.SIGNAL_UNAVAILABLE
        assert "Could not fetch news" in r["reasoning"]
        assert put_calls == [], "a transient fetch failure must not be cached"

    def test_llm_failure_is_not_cached(self, monkeypatch):
        put_calls = []
        monkeypatch.setattr(news, "get_news_sentiment", lambda *a, **k: None)
        monkeypatch.setattr(news, "put_news_sentiment",
                           lambda *a, **k: put_calls.append(a))
        monkeypatch.setattr(news, "_get_bucket", lambda rate: type("B", (), {"consume": lambda s: None})())
        monkeypatch.setattr(news, "_fh",
                            lambda: type("C", (), {"company_news": lambda s, *a, **k: [ARTICLE]})())
        monkeypatch.setattr(news, "_analyze_stock",
                            lambda *a, **k: news._unavailable("nope"))
        _, r = news._process_one(
            ("ACME", pd.Series({"ticker": "ACME"}), 1, 20, None, 60, ":memory:", 4, "m", "pm")
        )
        assert r["entry_signal"] == news.SIGNAL_UNAVAILABLE
        assert put_calls == [], "caching a failure would pin the row for the whole TTL"

    def test_real_result_is_cached(self, monkeypatch):
        put_calls = []
        monkeypatch.setattr(news, "get_news_sentiment", lambda *a, **k: None)
        monkeypatch.setattr(news, "put_news_sentiment",
                           lambda *a, **k: put_calls.append(a))
        monkeypatch.setattr(news, "_get_bucket", lambda rate: type("B", (), {"consume": lambda s: None})())
        monkeypatch.setattr(news, "_fh",
                            lambda: type("C", (), {"company_news": lambda s, *a, **k: [ARTICLE]})())
        monkeypatch.setattr(news, "_analyze_stock", lambda *a, **k: news._no_news("none material"))
        _, r = news._process_one(
            ("ACME", pd.Series({"ticker": "ACME"}), 1, 20, None, 60, ":memory:", 4, "m", "pm")
        )
        assert r["entry_signal"] == news.SIGNAL_NO_NEWS
        assert len(put_calls) == 1


class TestOverlaySkipsWithoutKey:
    def test_missing_key_returns_df_untouched(self, monkeypatch, capsys):
        monkeypatch.setattr(news, "llm_available", lambda: False)
        df = pd.DataFrame({"ticker": ["A"], "sector": ["Tech"], "conviction": [5]})
        out = news.attach_news_overlay(df, {}, ":memory:")
        assert out is df
        assert "OPENAI_API_KEY not set" in capsys.readouterr().out
