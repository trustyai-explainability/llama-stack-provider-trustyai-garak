"""Tests for inline provider implementation"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from llama_stack_provider_trustyai_garak.inline import get_provider_impl
from llama_stack_provider_trustyai_garak.inline.garak_eval import GarakInlineEvalAdapter
from llama_stack_provider_trustyai_garak.inline.provider import get_provider_spec
from llama_stack_provider_trustyai_garak.config import GarakInlineConfig
from llama_stack_provider_trustyai_garak.errors import GarakValidationError

from llama_stack_provider_trustyai_garak.compat import (
    Api,
    Benchmark,
    BenchmarkConfig,
    RunEvalRequest,
)


class TestInlineProvider:
    """Test cases for inline provider"""

    def test_get_provider_spec(self):
        """Test provider specification"""
        spec = get_provider_spec()

        assert spec.api == Api.eval
        assert spec.provider_type == "inline::trustyai_garak"
        assert any(pkg.startswith("garak") for pkg in spec.pip_packages)
        assert spec.config_class == "llama_stack_provider_trustyai_garak.config.GarakInlineConfig"
        assert spec.module == "llama_stack_provider_trustyai_garak.inline"
        assert Api.inference in spec.api_dependencies
        assert Api.files in spec.api_dependencies
        assert Api.benchmarks in spec.api_dependencies
        assert Api.safety in spec.optional_api_dependencies
        assert Api.shields in spec.optional_api_dependencies

    @pytest.mark.asyncio
    async def test_get_provider_impl_success(self):
        """Test successful provider implementation creation"""
        config = GarakInlineConfig()
        mock_deps = {Api.files: Mock(), Api.benchmarks: Mock()}

        with patch.object(GarakInlineEvalAdapter, "initialize", new_callable=AsyncMock):
            with patch.object(GarakInlineEvalAdapter, "_ensure_garak_installed"):
                with patch.object(GarakInlineEvalAdapter, "_get_all_probes", return_value=set()):
                    impl = await get_provider_impl(config, mock_deps)

                    assert isinstance(impl, GarakInlineEvalAdapter)
                    assert impl._config == config
                    impl.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_provider_impl_with_llama_stack_url(self):
        """Test provider implementation with llama_stack_url from config"""
        config = GarakInlineConfig(llama_stack_url="https://custom.api.com")

        # Provide required mock dependencies
        mock_deps = {Api.files: Mock(), Api.benchmarks: Mock()}

        with patch.object(GarakInlineEvalAdapter, "initialize", new_callable=AsyncMock):
            with patch.object(GarakInlineEvalAdapter, "_ensure_garak_installed"):
                with patch.object(GarakInlineEvalAdapter, "_get_all_probes", return_value=set()):
                    impl = await get_provider_impl(config, mock_deps)

                    assert impl._config.llama_stack_url == "https://custom.api.com"

    @pytest.mark.asyncio
    async def test_get_provider_impl_error_handling(self):
        """Test error handling in provider implementation creation"""
        config = GarakInlineConfig()

        # Provide required mock dependencies
        mock_deps = {Api.files: Mock(), Api.benchmarks: Mock()}

        with patch.object(GarakInlineEvalAdapter, "initialize", new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            with pytest.raises(Exception) as exc_info:
                await get_provider_impl(config, mock_deps)

            assert "Failed to create inline Garak implementation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_provider_impl_no_deps(self):
        """Test provider implementation with None dependencies (converted to empty dict internally)"""
        config = GarakInlineConfig()

        with patch.object(GarakInlineEvalAdapter, "initialize", new_callable=AsyncMock):
            with patch.object(GarakInlineEvalAdapter, "_ensure_garak_installed"):
                with patch.object(GarakInlineEvalAdapter, "_get_all_probes", return_value=set()):
                    # Pass None but the implementation should convert it to {}
                    # We'll need to mock the constructor to handle this
                    impl = await get_provider_impl(config, None)

                    assert isinstance(impl, GarakInlineEvalAdapter)

    @pytest.mark.asyncio
    async def test_get_provider_impl_with_optional_deps(self):
        """Test provider implementation with optional safety and shields dependencies"""
        config = GarakInlineConfig()

        # Provide all dependencies including optional ones
        mock_deps = {Api.files: Mock(), Api.benchmarks: Mock(), Api.safety: Mock(), Api.shields: Mock()}

        with patch.object(GarakInlineEvalAdapter, "initialize", new_callable=AsyncMock):
            with patch.object(GarakInlineEvalAdapter, "_ensure_garak_installed"):
                with patch.object(GarakInlineEvalAdapter, "_get_all_probes", return_value=set()):
                    impl = await get_provider_impl(config, mock_deps)

                    assert isinstance(impl, GarakInlineEvalAdapter)
                    assert impl._config == config
                    assert impl.safety_api is not None
                    assert impl.shields_api is not None
                    impl.initialize.assert_called_once()


class TestInlineIntentsFailFast:
    """Verify that intents benchmarks fail fast in inline mode."""

    @pytest.mark.asyncio
    async def test_art_intents_raises_in_inline_mode(self):
        """art_intents=True should be rejected by the inline provider."""
        config = GarakInlineConfig()
        mock_deps = {Api.files: Mock(), Api.benchmarks: Mock()}

        with patch.object(GarakInlineEvalAdapter, "initialize", new_callable=AsyncMock):
            with patch.object(GarakInlineEvalAdapter, "_ensure_garak_installed"):
                with patch.object(GarakInlineEvalAdapter, "_get_all_probes", return_value=set()):
                    adapter = await get_provider_impl(config, mock_deps)

        adapter._initialized = True

        # Mock _validate_run_eval_request to return art_intents=True
        mock_garak_config = Mock()
        provider_params = {"art_intents": True, "sdg_model": "m", "sdg_api_base": "http://api"}
        adapter._validate_run_eval_request = AsyncMock(return_value=(mock_garak_config, provider_params))

        request = RunEvalRequest(
            benchmark_id="test-benchmark",
            benchmark_config=Mock(spec=BenchmarkConfig),
        )

        with pytest.raises(GarakValidationError, match="not supported in inline mode"):
            await adapter.run_eval(request)
