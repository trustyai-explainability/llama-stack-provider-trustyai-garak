"""Tests for intents benchmarks with shields enabled.

This module tests the integration of shields (input/output filtering) with
intents benchmarks (ART intents mode). The primary use case is to verify
that shields work correctly with the intents workflow, which requires:
- KFP execution mode (intents not supported in inline mode)
- art_intents=True configuration
- intents_models configuration (judge, attack, sdg)
- shield_ids or shield_config for input/output filtering
"""

import pytest
from contextlib import contextmanager
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from llama_stack_provider_trustyai_garak.compat import (
    Api,
    Benchmark,
    BenchmarkConfig,
    SamplingParams,
    TopPSamplingStrategy,
    GetShieldRequest,
)
from llama_stack_provider_trustyai_garak.config import (
    GarakRemoteConfig,
    KubeflowConfig,
)
from llama_stack_provider_trustyai_garak.garak_command_config import (
    GarakCommandConfig,
    GarakPluginsConfig,
)


@pytest.fixture
def remote_config():
    """Create remote provider configuration."""
    kubeflow_config = KubeflowConfig(
        pipelines_endpoint="https://kfp.example.com",
        namespace="test-namespace",
        garak_base_image="test:latest",
    )
    return GarakRemoteConfig(
        kubeflow_config=kubeflow_config,
        llama_stack_url="http://test-llama-stack.com/v1"
    )


@pytest.fixture
def mock_deps_with_shields(mock_file_api, mock_benchmarks_api, mock_safety_api, mock_shields_api):
    """Create mock dependencies including shields."""
    # Configure shield to be available
    mock_shield = Mock()
    mock_shield.identifier = "llama-guard"
    mock_shields_api.get_shield = AsyncMock(return_value=mock_shield)

    return {
        Api.files: mock_file_api,
        Api.benchmarks: mock_benchmarks_api,
        Api.safety: mock_safety_api,
        Api.shields: mock_shields_api,
    }


@contextmanager
def create_adapter(remote_config, mock_deps):
    """Helper to create GarakRemoteEvalAdapter with standard patches."""
    from llama_stack_provider_trustyai_garak.remote.garak_remote_eval import GarakRemoteEvalAdapter

    with patch.object(GarakRemoteEvalAdapter, '_ensure_garak_installed'):
        with patch.object(GarakRemoteEvalAdapter, '_get_all_probes', return_value=set()):
            with patch.object(GarakRemoteEvalAdapter, '_create_kfp_client'):
                adapter = GarakRemoteEvalAdapter(remote_config, mock_deps)
                adapter._initialized = True
                yield adapter


def create_benchmark_config(model="test-model"):
    """Factory helper to create standard BenchmarkConfig for testing."""
    return BenchmarkConfig(
        eval_candidate={
            "type": "model",
            "model": model,
            "sampling_params": SamplingParams(
                strategy=TopPSamplingStrategy(temperature=0.7, top_p=0.95),
                max_tokens=100
            )
        }
    )


class TestIntentsWithShieldsConfiguration:
    """Test intents benchmark configuration with shields."""

    @pytest.mark.asyncio
    async def test_intents_benchmark_with_shield_ids(self, remote_config, mock_deps_with_shields):
        """Test intents benchmark with shield_ids configuration."""
        with create_adapter(remote_config, mock_deps_with_shields) as adapter:
            # Register intents benchmark with shields
            benchmark = Benchmark(
            identifier="test_intents_with_shield",
            dataset_id="garak",
            scoring_functions=["garak_scoring"],
            provider_id="trustyai_garak",
            provider_benchmark_id="trustyai_garak::intents",
            metadata={
                "art_intents": True,
                "shield_ids": ["llama-guard"],
                "sdg_model": "test-sdg-model",
                "sdg_api_base": "http://sdg.example.com/v1",
                "sdg_api_key": "test-key",
                "intents_models": {
                    "judge": {
                        "name": "test-judge-model",
                        "url": "http://judge.example.com/v1",
                        "api_key": "judge-key"
                    },
                    "attack": {
                        "name": "test-attack-model",
                        "url": "http://attack.example.com/v1",
                        "api_key": "attack-key"
                    },
                    "sdg": {
                        "name": "test-sdg-model",
                        "url": "http://sdg.example.com/v1",
                        "api_key": "test-key"
                    }
                },
                "garak_config": GarakCommandConfig().to_dict(),
            }
            )

            await adapter.register_benchmark(benchmark)

            # Verify benchmark was registered
            registered = await adapter.get_benchmark("test_intents_with_shield")
            assert registered is not None
            assert registered.metadata.get("art_intents") is True
            assert "llama-guard" in registered.metadata.get("shield_ids", [])

    @pytest.mark.asyncio
    async def test_intents_benchmark_with_shield_config(self, remote_config, mock_deps_with_shields):
        """Test intents benchmark with shield_config for input/output shields."""
        with create_adapter(remote_config, mock_deps_with_shields) as adapter:
            # Register intents benchmark with shield_config
            benchmark = Benchmark(
                identifier="test_intents_with_shield_config",
                dataset_id="garak",
                scoring_functions=["garak_scoring"],
                provider_id="trustyai_garak",
                provider_benchmark_id="trustyai_garak::intents",
                metadata={
                    "art_intents": True,
                    "shield_config": {
                        "input": ["llama-guard"],
                        "output": ["llama-guard"]
                    },
                    "sdg_model": "test-sdg-model",
                    "sdg_api_base": "http://sdg.example.com/v1",
                    "intents_models": {
                        "judge": {
                            "name": "test-judge-model",
                            "url": "http://judge.example.com/v1"
                        },
                        "sdg": {
                            "name": "test-sdg-model",
                            "url": "http://sdg.example.com/v1"
                        }
                    },
                    "garak_config": GarakCommandConfig().to_dict(),
                }
            )

            await adapter.register_benchmark(benchmark)

            # Verify benchmark was registered
            registered = await adapter.get_benchmark("test_intents_with_shield_config")
            assert registered is not None
            assert registered.metadata.get("art_intents") is True
            shield_config = registered.metadata.get("shield_config", {})
            assert "llama-guard" in shield_config.get("input", [])
            assert "llama-guard" in shield_config.get("output", [])

    @pytest.mark.asyncio
    async def test_intents_with_shields_build_command(self, remote_config, mock_deps_with_shields):
        """Test that _build_command properly configures shields for intents benchmark."""
        with create_adapter(remote_config, mock_deps_with_shields) as adapter:
            # Create benchmark config for intents with shields
            benchmark_config = create_benchmark_config()

            # Create garak config with judge detector (required for intents)
            garak_config = GarakCommandConfig(
                plugins=GarakPluginsConfig(
                    detectors={
                        "judge": {
                            "detector_model_type": "openai.OpenAICompatible",
                            "detector_model_name": "test-judge-model",
                            "detector_model_config": {
                                "uri": "http://judge.example.com/v1",
                                "api_key": "test-key",
                            }
                        }
                    }
                )
            )

            # Provider params with shields and intents
            provider_params = {
                "art_intents": True,
                "shield_config": {
                    "input": ["llama-guard"],
                    "output": ["llama-guard"]
                },
                "sdg_model": "test-sdg-model",
                "intents_file_id": None,  # Force SDG to run
            }

            # Build command
            cmd_config = await adapter._build_command(
                benchmark_config,
                garak_config,
                provider_params,
                scan_report_prefix="test_scan"
            )

            # Verify that function-based generator is configured (shields require this)
            assert cmd_config["plugins"]["target_type"] == "function.Single"
            assert "shield_scan" in cmd_config["plugins"]["target_name"]

            # Verify generator options include shield mapping
            generators = cmd_config["plugins"]["generators"]
            assert "function" in generators
            assert "Single" in generators["function"]

            kwargs = generators["function"]["Single"]["kwargs"]
            assert "llm_io_shield_mapping" in kwargs
            assert "llama-guard" in kwargs["llm_io_shield_mapping"]["input"]
            assert "llama-guard" in kwargs["llm_io_shield_mapping"]["output"]


class TestIntentsWithShieldsValidation:
    """Test validation logic for intents + shields combination."""

    @pytest.mark.asyncio
    async def test_intents_with_shields_requires_safety_api(self, remote_config, mock_file_api, mock_benchmarks_api):
        """Test that shields require safety API to be available."""
        from llama_stack_provider_trustyai_garak.errors import GarakConfigError

        # Create deps without safety/shields APIs
        mock_deps = {
            Api.files: mock_file_api,
            Api.benchmarks: mock_benchmarks_api,
        }

        with create_adapter(remote_config, mock_deps) as adapter:
            # Create benchmark config for intents with shields
            benchmark_config = create_benchmark_config()

            garak_config = GarakCommandConfig()
            provider_params = {
                "art_intents": True,
                "shield_ids": ["llama-guard"],
                "sdg_model": "test-sdg-model",
            }

            # Should raise error because shields API is not available
            with pytest.raises(GarakConfigError, match="Shields API is not available"):
                await adapter._build_command(
                    benchmark_config,
                    garak_config,
                    provider_params,
                )

    @pytest.mark.asyncio
    async def test_intents_with_unavailable_shield_raises_error(self, remote_config, mock_deps_with_shields):
        """Test that using unavailable shield raises validation error."""
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        # Configure shield to be unavailable
        mock_deps_with_shields[Api.shields].get_shield = AsyncMock(return_value=None)

        with create_adapter(remote_config, mock_deps_with_shields) as adapter:
            # Create benchmark config
            benchmark_config = create_benchmark_config()

            garak_config = GarakCommandConfig(
                plugins=GarakPluginsConfig(
                    detectors={
                        "judge": {
                            "detector_model_type": "openai.OpenAICompatible",
                            "detector_model_name": "test-judge-model",
                            "detector_model_config": {
                                "uri": "http://judge.example.com/v1",
                                "api_key": "test-key",
                            }
                        }
                    }
                )
            )
            provider_params = {
                "art_intents": True,
                "shield_ids": ["unavailable-shield"],
                "sdg_model": "test-sdg-model",
            }

            # Should raise validation error for unavailable shield
            with pytest.raises(GarakValidationError, match="shield.*is not available"):
                await adapter._build_command(
                    benchmark_config,
                    garak_config,
                    provider_params,
                )

    @pytest.mark.asyncio
    async def test_intents_with_shields_requires_sdg_model_when_no_bypass_file(
        self, remote_config, mock_deps_with_shields
    ):
        """Test that intents with shields requires sdg_model when no intents_file_id provided."""
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        with create_adapter(remote_config, mock_deps_with_shields) as adapter:
            # Register benchmark with missing sdg_model (but has judge detector)
            benchmark = Benchmark(
                identifier="test_intents_missing_config",
                dataset_id="garak",
                scoring_functions=["garak_scoring"],
                provider_id="trustyai_garak",
                provider_benchmark_id="trustyai_garak::intents",
                metadata={
                    "art_intents": True,
                    "shield_ids": ["llama-guard"],
                    # Missing sdg_model and intents_file_id
                    "garak_config": GarakCommandConfig(
                        plugins=GarakPluginsConfig(
                            detectors={
                                "judge": {
                                    "detector_model_type": "openai.OpenAICompatible",
                                    "detector_model_name": "test-judge-model",
                                    "detector_model_config": {
                                        "uri": "http://judge.example.com/v1",
                                        "api_key": "test-key",
                                    }
                                }
                            }
                        )
                    ).to_dict(),
                }
            )

            await adapter.register_benchmark(benchmark)

            benchmark_config = create_benchmark_config()

            garak_config, provider_params = await adapter._validate_run_eval_request(
                "test_intents_missing_config",
                benchmark_config
            )

            # Should raise error about missing sdg_model
            with pytest.raises(GarakValidationError, match="sdg_model is required"):
                await adapter._build_command(
                    benchmark_config,
                    garak_config,
                    provider_params,
                )


class TestIntentsWithShieldsIntegration:
    """Integration tests for intents + shields workflow."""

    @pytest.mark.asyncio
    async def test_intents_with_shields_full_workflow(self, remote_config, mock_deps_with_shields):
        """Test full workflow: register benchmark, validate, build command."""
        with create_adapter(remote_config, mock_deps_with_shields) as adapter:
            # Step 1: Register intents benchmark with shields
            benchmark = Benchmark(
                identifier="intents_shield_workflow",
                dataset_id="garak",
                scoring_functions=["garak_scoring"],
                provider_id="trustyai_garak",
                provider_benchmark_id="trustyai_garak::intents",
                metadata={
                    "art_intents": True,
                    "shield_config": {
                        "input": ["llama-guard"],
                        "output": ["llama-guard"]
                    },
                    "sdg_model": "sdg-model",
                    "sdg_api_base": "http://sdg.example.com/v1",
                    "intents_models": {
                        "judge": {
                            "name": "judge-model",
                            "url": "http://judge.example.com/v1"
                        },
                        "attack": {
                            "name": "attack-model",
                            "url": "http://attack.example.com/v1"
                        },
                        "sdg": {
                            "name": "sdg-model",
                            "url": "http://sdg.example.com/v1"
                        }
                    },
                    "garak_config": GarakCommandConfig(
                        plugins=GarakPluginsConfig(
                            detectors={
                                "judge": {
                                    "detector_model_type": "openai.OpenAICompatible",
                                    "detector_model_name": "judge-model",
                                    "detector_model_config": {
                                        "uri": "http://judge.example.com/v1",
                                        "api_key": "test-key",
                                    }
                                }
                            }
                        )
                    ).to_dict(),
                }
            )

            await adapter.register_benchmark(benchmark)

            # Step 2: Validate benchmark configuration
            benchmark_config = create_benchmark_config()

            garak_config, provider_params = await adapter._validate_run_eval_request(
                "intents_shield_workflow",
                benchmark_config
            )

            # Verify intents params
            assert provider_params.get("art_intents") is True
            assert provider_params.get("shield_config") is not None
            assert provider_params.get("sdg_model") == "sdg-model"

            # Step 3: Build command configuration
            cmd_config = await adapter._build_command(
                benchmark_config,
                garak_config,
                provider_params,
                scan_report_prefix="test_workflow"
            )

            # Step 4: Verify final configuration
            # Verify function-based generator for shields
            assert cmd_config["plugins"]["target_type"] == "function.Single"
            assert "shield_scan" in cmd_config["plugins"]["target_name"]

            # Verify shield mapping
            generators = cmd_config["plugins"]["generators"]
            kwargs = generators["function"]["Single"]["kwargs"]
            shield_mapping = kwargs["llm_io_shield_mapping"]
            assert "llama-guard" in shield_mapping["input"]
            assert "llama-guard" in shield_mapping["output"]

            # Verify intents-specific configuration is present
            assert cmd_config["plugins"]["detectors"] is not None
            assert "judge" in cmd_config["plugins"]["detectors"]

            # Verify report prefix
            assert cmd_config["reporting"]["report_prefix"] == "test_workflow"
