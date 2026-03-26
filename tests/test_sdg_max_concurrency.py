"""Tests for _resolve_max_concurrency precedence and validation."""

import pytest

from llama_stack_provider_trustyai_garak.sdg import _resolve_max_concurrency
from llama_stack_provider_trustyai_garak.constants import DEFAULT_SDG_MAX_CONCURRENCY


class TestResolveMaxConcurrency:
    """Precedence: explicit value >= 1 > SDG_MAX_CONCURRENCY env var > DEFAULT."""

    def test_explicit_value_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("SDG_MAX_CONCURRENCY", "99")
        assert _resolve_max_concurrency(5) == 5

    def test_explicit_value_takes_precedence_over_default(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        assert _resolve_max_concurrency(3) == 3

    def test_zero_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("SDG_MAX_CONCURRENCY", "7")
        assert _resolve_max_concurrency(0) == 7

    def test_negative_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("SDG_MAX_CONCURRENCY", "7")
        assert _resolve_max_concurrency(-3) == 7

    def test_missing_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        assert _resolve_max_concurrency(0) == DEFAULT_SDG_MAX_CONCURRENCY

    def test_no_args_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        assert _resolve_max_concurrency() == DEFAULT_SDG_MAX_CONCURRENCY

    def test_invalid_env_logs_warning_and_uses_default(self, monkeypatch, caplog):
        monkeypatch.setenv("SDG_MAX_CONCURRENCY", "not-an-int")
        with caplog.at_level("WARNING"):
            result = _resolve_max_concurrency(0)
        assert result == DEFAULT_SDG_MAX_CONCURRENCY
        assert any("SDG_MAX_CONCURRENCY" in r.message for r in caplog.records)

    def test_env_less_than_one_logs_warning_and_uses_default(self, monkeypatch, caplog):
        monkeypatch.setenv("SDG_MAX_CONCURRENCY", "0")
        with caplog.at_level("WARNING"):
            result = _resolve_max_concurrency(0)
        assert result == DEFAULT_SDG_MAX_CONCURRENCY
        assert any("SDG_MAX_CONCURRENCY" in r.message for r in caplog.records)

    def test_env_negative_logs_warning_and_uses_default(self, monkeypatch, caplog):
        monkeypatch.setenv("SDG_MAX_CONCURRENCY", "-5")
        with caplog.at_level("WARNING"):
            result = _resolve_max_concurrency(0)
        assert result == DEFAULT_SDG_MAX_CONCURRENCY
        assert any("SDG_MAX_CONCURRENCY" in r.message for r in caplog.records)
