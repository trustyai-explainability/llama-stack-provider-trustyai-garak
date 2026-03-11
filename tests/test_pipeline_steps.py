"""Tests for core/pipeline_steps.py shared business logic and base_eval validation."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

from llama_stack_provider_trustyai_garak.core.pipeline_steps import (
    log_kfp_metrics,
    normalize_prompts,
    parse_and_build_results,
    resolve_taxonomy_data,
    run_sdg_generation,
    setup_and_run_garak,
    validate_scan_config,
)
from llama_stack_provider_trustyai_garak.errors import (
    GarakError,
    GarakValidationError,
)


class TestValidateScanConfig:

    def test_valid_config(self):
        config = json.dumps({"plugins": {"probes": []}})
        validate_scan_config(config)

    def test_invalid_json(self):
        with pytest.raises(GarakValidationError, match="not valid JSON"):
            validate_scan_config("not json")

    def test_missing_plugins(self):
        config = json.dumps({"other": "stuff"})
        with pytest.raises(GarakValidationError, match="plugins"):
            validate_scan_config(config)

    def test_dangerous_flags(self):
        config = json.dumps({"plugins": {}, "--rm": True})
        with pytest.raises(GarakValidationError, match="Dangerous flag"):
            validate_scan_config(config)

    @patch.dict("sys.modules", {"garak": None})
    def test_garak_not_installed(self):
        """If garak import fails, validation should fail."""
        config = json.dumps({"plugins": {"probes": []}})
        with patch("builtins.__import__", side_effect=ImportError("no garak")):
            with pytest.raises(GarakValidationError, match="garak is not installed"):
                validate_scan_config(config)


class TestResolveTaxonomyData:

    def test_default_taxonomy(self):
        df = resolve_taxonomy_data(None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "policy_concept" in df.columns

    def test_custom_taxonomy_csv(self):
        csv_content = "policy_concept,concept_definition\nTest,A test harm\n"
        df = resolve_taxonomy_data(csv_content.encode("utf-8"), format="csv")
        assert len(df) == 1
        assert df.iloc[0]["policy_concept"] == "Test"

    def test_custom_taxonomy_string(self):
        csv_content = "policy_concept,concept_definition\nTest,A test harm\n"
        df = resolve_taxonomy_data(csv_content, format="csv")
        assert len(df) == 1


class TestRunSdgGeneration:

    def test_missing_model_raises(self):
        df = pd.DataFrame({"policy_concept": ["test"], "concept_definition": ["test"]})
        with pytest.raises(GarakValidationError, match="sdg_model is required"):
            run_sdg_generation(df, sdg_model="", sdg_api_base="http://test")

    def test_missing_api_base_raises(self):
        df = pd.DataFrame({"policy_concept": ["test"], "concept_definition": ["test"]})
        with pytest.raises(GarakValidationError, match="sdg_api_base is required"):
            run_sdg_generation(df, sdg_model="test-model", sdg_api_base="")

    @patch("llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset")
    def test_calls_generate_sdg(self, mock_gen):
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        raw_df = pd.DataFrame({"policy_concept": ["x"], "prompt": ["test"]})
        norm_df = pd.DataFrame({"category": ["x"], "prompt": ["test"]})
        mock_gen.return_value = SDGResult(raw=raw_df, normalized=norm_df)

        taxonomy = pd.DataFrame({"policy_concept": ["test"], "concept_definition": ["test"]})
        result = run_sdg_generation(
            taxonomy, sdg_model="m", sdg_api_base="http://api", sdg_api_key="key"
        )
        assert len(result) == 1
        mock_gen.assert_called_once()


class TestNormalizePrompts:

    def test_normalize_csv(self):
        csv_content = "category,prompt,description\nHarm,Do bad,A bad thing\n"
        df = normalize_prompts(csv_content, format="csv")
        assert "category" in df.columns
        assert "prompt" in df.columns
        assert len(df) == 1


class TestSetupAndRunGarak:

    @patch("llama_stack_provider_trustyai_garak.core.garak_runner.convert_to_avid_report")
    @patch("llama_stack_provider_trustyai_garak.core.garak_runner.run_garak_scan")
    def test_success_flow(self, mock_scan, mock_avid):
        from llama_stack_provider_trustyai_garak.core.garak_runner import GarakScanResult

        scan_dir = Path(tempfile.mkdtemp())
        mock_result = GarakScanResult(
            returncode=0, stdout="ok", stderr="", report_prefix=scan_dir / "scan"
        )
        mock_scan.return_value = mock_result
        mock_avid.return_value = True

        config = json.dumps({"plugins": {"probes": []}, "reporting": {}})
        result = setup_and_run_garak(config, None, scan_dir, 300)
        assert result.success
        mock_scan.assert_called_once()

    @patch("llama_stack_provider_trustyai_garak.core.garak_runner.run_garak_scan")
    def test_failure_raises(self, mock_scan):
        from llama_stack_provider_trustyai_garak.core.garak_runner import GarakScanResult

        scan_dir = Path(tempfile.mkdtemp())
        mock_result = GarakScanResult(
            returncode=1, stdout="", stderr="error occurred",
            report_prefix=scan_dir / "scan",
        )
        mock_scan.return_value = mock_result

        config = json.dumps({"plugins": {"probes": []}, "reporting": {}})
        with pytest.raises(GarakError, match="Garak scan failed"):
            setup_and_run_garak(config, None, scan_dir, 300)

    @patch("llama_stack_provider_trustyai_garak.core.garak_runner.convert_to_avid_report")
    @patch("llama_stack_provider_trustyai_garak.core.garak_runner.run_garak_scan")
    def test_with_prompts_csv(self, mock_scan, mock_avid):
        from llama_stack_provider_trustyai_garak.core.garak_runner import GarakScanResult

        scan_dir = Path(tempfile.mkdtemp())
        prompts_path = scan_dir / "prompts.csv"
        pd.DataFrame({"category": ["harm"], "prompt": ["bad"], "description": ["d"]}).to_csv(
            prompts_path, index=False
        )

        mock_result = GarakScanResult(
            returncode=0, stdout="ok", stderr="", report_prefix=scan_dir / "scan"
        )
        mock_scan.return_value = mock_result
        mock_avid.return_value = True

        config = json.dumps({"plugins": {"probes": []}, "reporting": {}})

        with patch(
            "llama_stack_provider_trustyai_garak.intents.generate_intents_from_dataset"
        ) as mock_gen_intents:
            result = setup_and_run_garak(config, prompts_path, scan_dir, 300)
            mock_gen_intents.assert_called_once()
            assert result.success


class TestParseAndBuildResults:

    @patch("llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results")
    @patch("llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content")
    @patch("llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content")
    @patch("llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content")
    def test_calls_result_utils(self, mock_parse_gen, mock_parse_agg, mock_parse_dig, mock_combine):
        mock_parse_gen.return_value = ([], {}, {})
        mock_parse_agg.return_value = {}
        mock_parse_dig.return_value = {}
        mock_combine.return_value = {"scores": {}, "generations": []}

        result = parse_and_build_results(
            "report", "avid", art_intents=False, eval_threshold=0.5
        )
        mock_combine.assert_called_once()
        assert "scores" in result


class TestLogKfpMetrics:

    def test_intents_mode(self):
        mock_metrics = MagicMock()
        result_dict = {
            "scores": {
                "_overall": {
                    "aggregated_results": {
                        "attack_success_rate": 0.42,
                        "total_attempts": 10,
                    }
                }
            }
        }
        log_kfp_metrics(mock_metrics, result_dict, art_intents=True)
        mock_metrics.log_metric.assert_any_call("attack_success_rate", 0.42)
        # Should NOT log total_attempts in intents mode
        calls = [c[0] for c in mock_metrics.log_metric.call_args_list]
        assert ("total_attempts", 10) not in calls

    def test_native_mode(self):
        mock_metrics = MagicMock()
        result_dict = {
            "scores": {
                "_overall": {
                    "aggregated_results": {
                        "attack_success_rate": 0.3,
                        "total_attempts": 20,
                        "vulnerable_responses": 6,
                    }
                }
            }
        }
        log_kfp_metrics(mock_metrics, result_dict, art_intents=False)
        mock_metrics.log_metric.assert_any_call("attack_success_rate", 0.3)
        mock_metrics.log_metric.assert_any_call("total_attempts", 20)
        mock_metrics.log_metric.assert_any_call("vulnerable_responses", 6)

    def test_tbsa_logged_when_present(self):
        mock_metrics = MagicMock()
        result_dict = {
            "scores": {
                "_overall": {
                    "aggregated_results": {
                        "attack_success_rate": 0.5,
                        "tbsa": 0.8,
                    }
                }
            }
        }
        log_kfp_metrics(mock_metrics, result_dict, art_intents=True)
        mock_metrics.log_metric.assert_any_call("tbsa", 0.8)


class TestBaseEvalIntentsValidation:
    """Test mutual exclusivity and SDG requirement validation in register_benchmark."""

    @pytest.fixture
    def _make_adapter(self):
        """Create a minimal adapter for testing register_benchmark."""
        from llama_stack_provider_trustyai_garak.compat import Api, Benchmark

        def _factory():
            config = Mock()
            config.llama_stack_url = "http://localhost:8321"
            config.timeout = 300

            deps = {Api.files: Mock(), Api.benchmarks: Mock()}

            with patch(
                "llama_stack_provider_trustyai_garak.base_eval.GarakEvalBase._ensure_garak_installed"
            ):
                with patch(
                    "llama_stack_provider_trustyai_garak.base_eval.GarakEvalBase._get_all_probes",
                    return_value=set(),
                ):
                    from llama_stack_provider_trustyai_garak.base_eval import GarakEvalBase

                    adapter = GarakEvalBase.__new__(GarakEvalBase)
                    adapter._config = config
                    adapter.benchmarks = {}
                    adapter._benchmarks_lock = __import__("asyncio").Lock()
                    adapter.scan_config = Mock()
                    adapter.scan_config.SCAN_PROFILES = {}
                    adapter.scan_config.FRAMEWORK_PROFILES = {}
            return adapter

        return _factory

    def _make_benchmark(self, metadata):
        from llama_stack_provider_trustyai_garak.compat import Benchmark
        return Benchmark(
            identifier="test",
            provider_id="trustyai_garak",
            dataset_id="garak",
            scoring_functions=["garak_scoring"],
            metadata=metadata,
        )

    @pytest.mark.asyncio
    async def test_mutual_exclusivity_raises(self, _make_adapter):
        adapter = _make_adapter()
        benchmark = self._make_benchmark({
            "art_intents": True,
            "policy_file_id": "some-policy",
            "intents_file_id": "some-intents",
            "sdg_model": "m",
            "garak_config": {"run": {}, "plugins": {"detectors": {"judge": {}}}},
        })
        with pytest.raises(GarakValidationError, match="mutually exclusive"):
            await adapter.register_benchmark(benchmark)

    @pytest.mark.asyncio
    async def test_sdg_model_not_required_at_registration(self, _make_adapter):
        """sdg_model is validated at run-eval time, not registration time."""
        adapter = _make_adapter()
        benchmark = self._make_benchmark({
            "art_intents": True,
            "garak_config": {"run": {}, "plugins": {"detectors": {"judge": {}}}},
        })
        # Should NOT raise at registration — sdg_model is provided at run_eval time
        await adapter.register_benchmark(benchmark)
        assert "test" in adapter.benchmarks

    @pytest.mark.asyncio
    async def test_bypass_skips_sdg_model_check(self, _make_adapter):
        adapter = _make_adapter()
        benchmark = self._make_benchmark({
            "art_intents": True,
            "intents_file_id": "some-intents",
            "garak_config": {"run": {}, "plugins": {"detectors": {"judge": {}}}},
        })
        await adapter.register_benchmark(benchmark)
        assert "test" in adapter.benchmarks
