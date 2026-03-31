"""Tests for SDG parameter resolution, flow block overrides, and plumbing."""

import pytest
from unittest.mock import MagicMock, patch, call

from llama_stack_provider_trustyai_garak.sdg import (
    _resolve_max_concurrency,
    _override_flow_block,
)
from llama_stack_provider_trustyai_garak.constants import (
    DEFAULT_SDG_MAX_CONCURRENCY,
    DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME,
    DEFAULT_SDG_MAX_TOKENS_BLOCK_NAME,
)


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


# ---------------------------------------------------------------------------
# _override_flow_block
# ---------------------------------------------------------------------------


def _make_mock_block(block_name: str, extra_config: dict | None = None):
    """Create a mock block with get_config/from_config support."""
    config = {"block_name": block_name}
    if extra_config:
        config.update(extra_config)
    block = MagicMock()
    block.get_config.return_value = dict(config)
    block.from_config.return_value = MagicMock(name=f"rebuilt-{block_name}")
    return block


class TestOverrideFlowBlock:
    """_override_flow_block finds blocks by name and patches config."""

    def test_overrides_matching_block(self):
        block = _make_mock_block(DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 10})
        flow = MagicMock()
        flow.blocks = [block]

        _override_flow_block(flow, DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 25})

        block.from_config.assert_called_once()
        patched_cfg = block.from_config.call_args[0][0]
        assert patched_cfg["num_samples"] == 25
        assert patched_cfg["block_name"] == DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME
        assert flow.blocks[0] is block.from_config.return_value

    def test_skips_non_matching_blocks(self):
        other = _make_mock_block("sample_demographics")
        target = _make_mock_block(DEFAULT_SDG_MAX_TOKENS_BLOCK_NAME, {"max_tokens": 2048})
        flow = MagicMock()
        flow.blocks = [other, target]

        _override_flow_block(flow, DEFAULT_SDG_MAX_TOKENS_BLOCK_NAME, {"max_tokens": 4096})

        other.from_config.assert_not_called()
        target.from_config.assert_called_once()
        patched_cfg = target.from_config.call_args[0][0]
        assert patched_cfg["max_tokens"] == 4096

    def test_warns_when_block_not_found(self, caplog):
        flow = MagicMock()
        flow.blocks = [_make_mock_block("other_block")]

        with caplog.at_level("WARNING"):
            _override_flow_block(flow, "nonexistent_block", {"key": "val"})

        assert any("not found" in r.message for r in caplog.records)

    def test_only_overrides_first_match(self):
        b1 = _make_mock_block(DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 10})
        b2 = _make_mock_block(DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 10})
        flow = MagicMock()
        flow.blocks = [b1, b2]

        _override_flow_block(flow, DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 5})

        b1.from_config.assert_called_once()
        b2.from_config.assert_not_called()


# ---------------------------------------------------------------------------
# generate_sdg_dataset plumbing
# ---------------------------------------------------------------------------


def _setup_sdg_mocks(monkeypatch):
    """Inject mock sdg_hub modules and return (mock_flow, mock_override) for assertions."""
    import types
    import pandas as pd

    mock_flow = MagicMock()
    mock_flow.generate.return_value = pd.DataFrame(
        {"policy_concept": ["test"], "prompt": ["prompt"], "concept_definition": ["desc"]}
    )

    mock_flow_cls = MagicMock()
    mock_flow_cls.from_yaml.return_value = mock_flow

    mock_registry = MagicMock()

    sdg_hub_module = types.ModuleType("sdg_hub")
    sdg_hub_module.Flow = mock_flow_cls
    sdg_hub_module.FlowRegistry = mock_registry
    monkeypatch.setitem(__import__("sys").modules, "sdg_hub", sdg_hub_module)

    nest_asyncio_module = types.ModuleType("nest_asyncio")
    nest_asyncio_module.apply = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "nest_asyncio", nest_asyncio_module)

    mock_override = MagicMock()
    monkeypatch.setattr("llama_stack_provider_trustyai_garak.sdg._override_flow_block", mock_override)

    return mock_flow, mock_override


class TestGenerateSDGDatasetParams:
    """Verify num_samples / max_tokens are applied to the flow."""

    def test_num_samples_override_applied(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        mock_flow, mock_override = _setup_sdg_mocks(monkeypatch)

        from llama_stack_provider_trustyai_garak.sdg import generate_sdg_dataset

        generate_sdg_dataset(model="m", api_base="http://x", num_samples=20)
        mock_override.assert_any_call(mock_flow, DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 20})

    def test_max_tokens_override_applied(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        mock_flow, mock_override = _setup_sdg_mocks(monkeypatch)

        from llama_stack_provider_trustyai_garak.sdg import generate_sdg_dataset

        generate_sdg_dataset(model="m", api_base="http://x", max_tokens=4096)
        mock_override.assert_any_call(mock_flow, DEFAULT_SDG_MAX_TOKENS_BLOCK_NAME, {"max_tokens": 4096})

    def test_zero_values_skip_overrides(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        mock_flow, mock_override = _setup_sdg_mocks(monkeypatch)

        from llama_stack_provider_trustyai_garak.sdg import generate_sdg_dataset

        generate_sdg_dataset(model="m", api_base="http://x", num_samples=0, max_tokens=0)
        mock_override.assert_not_called()

    def test_both_overrides_applied(self, monkeypatch):
        monkeypatch.delenv("SDG_MAX_CONCURRENCY", raising=False)
        mock_flow, mock_override = _setup_sdg_mocks(monkeypatch)

        from llama_stack_provider_trustyai_garak.sdg import generate_sdg_dataset

        generate_sdg_dataset(model="m", api_base="http://x", num_samples=15, max_tokens=8192)
        assert mock_override.call_count == 2
        mock_override.assert_any_call(mock_flow, DEFAULT_SDG_NUM_SAMPLES_BLOCK_NAME, {"num_samples": 15})
        mock_override.assert_any_call(mock_flow, DEFAULT_SDG_MAX_TOKENS_BLOCK_NAME, {"max_tokens": 8192})
