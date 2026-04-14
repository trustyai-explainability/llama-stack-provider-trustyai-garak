"""Tests for eval-hub Garak adapter alignment behavior."""

from __future__ import annotations

import importlib
import json
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from llama_stack_provider_trustyai_garak.core.config_resolution import (
    build_effective_garak_config,
    resolve_scan_profile,
    resolve_timeout_seconds,
)
from llama_stack_provider_trustyai_garak.core.garak_runner import convert_to_avid_report


def _load_evalhub_garak_adapter(monkeypatch) -> types.ModuleType:
    """Import the eval-hub adapter module with lightweight evalhub stubs."""

    class _SimpleModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FrameworkAdapter:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

    class JobStatus(str, Enum):
        RUNNING = "running"
        FAILED = "failed"

    class JobPhase(str, Enum):
        INITIALIZING = "initializing"
        RUNNING_EVALUATION = "running_evaluation"
        POST_PROCESSING = "post_processing"

    adapter_module = types.ModuleType("evalhub.adapter")
    adapter_models_job_module = types.ModuleType("evalhub.adapter.models.job")
    models_api_module = types.ModuleType("evalhub.models.api")
    evalhub_module = types.ModuleType("evalhub")
    adapter_models_module = types.ModuleType("evalhub.adapter.models")
    models_module = types.ModuleType("evalhub.models")

    adapter_module.DefaultCallbacks = _SimpleModel
    adapter_module.FrameworkAdapter = FrameworkAdapter
    adapter_module.JobCallbacks = _SimpleModel
    adapter_module.JobPhase = JobPhase
    adapter_module.JobResults = _SimpleModel
    adapter_module.JobSpec = _SimpleModel
    adapter_module.JobStatus = JobStatus
    adapter_module.JobStatusUpdate = _SimpleModel
    adapter_module.OCIArtifactSpec = _SimpleModel

    adapter_models_job_module.ErrorInfo = _SimpleModel
    adapter_models_job_module.MessageInfo = _SimpleModel
    models_api_module.EvaluationResult = _SimpleModel

    adapter_auth_module = types.ModuleType("evalhub.adapter.auth")
    adapter_auth_module.read_model_auth_key = lambda _name: None

    monkeypatch.setitem(sys.modules, "evalhub", evalhub_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter", adapter_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.auth", adapter_auth_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.models", adapter_models_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter.models.job", adapter_models_job_module)
    monkeypatch.setitem(sys.modules, "evalhub.models", models_module)
    monkeypatch.setitem(sys.modules, "evalhub.models.api", models_api_module)

    # Ensure this import uses the stubbed evalhub modules above.
    sys.modules.pop("llama_stack_provider_trustyai_garak.evalhub.garak_adapter", None)
    return importlib.import_module("llama_stack_provider_trustyai_garak.evalhub.garak_adapter")


def test_resolve_scan_profile_accepts_prefixed_and_unprefixed_ids():
    prefixed = resolve_scan_profile("trustyai_garak::owasp_llm_top10")
    unprefixed = resolve_scan_profile("owasp_llm_top10")

    assert prefixed["name"] == "OWASP LLM Top 10"
    assert unprefixed["name"] == "OWASP LLM Top 10"
    assert unprefixed["garak_config"]["run"]["probe_tags"] == prefixed["garak_config"]["run"]["probe_tags"]


def test_build_effective_garak_config_honors_precedence():
    profile = resolve_scan_profile("trustyai_garak::quick")
    benchmark_config = {
        "garak_config": {
            "run": {"generations": 3},
            "plugins": {"probe_spec": ["promptinject"], "extended_detectors": False},
        },
        # Legacy key should override explicit nested key where both are set.
        "generations": 5,
        "probe_tags": "owasp:llm",
    }

    resolved = build_effective_garak_config(benchmark_config, profile)
    resolved_dict = resolved.to_dict(exclude_none=True)

    assert resolved_dict["run"]["generations"] == 5
    assert resolved_dict["run"]["probe_tags"] == "owasp:llm"
    assert resolved_dict["plugins"]["probe_spec"] == "promptinject"
    assert resolved_dict["plugins"]["extended_detectors"] is False


def test_resolve_timeout_seconds_precedence():
    profile = {"timeout": 3600}

    assert resolve_timeout_seconds({"timeout_seconds": 120}, profile, 600) == 120
    assert resolve_timeout_seconds({"timeout": 180}, profile, 600) == 180
    assert resolve_timeout_seconds({}, profile, 600) == 3600
    assert resolve_timeout_seconds({}, {}, 600) == 600

    assert resolve_timeout_seconds({"timeout_seconds": 0}, profile, 600) == 0
    assert resolve_timeout_seconds({"timeout": 0}, profile, 600) == 0
    assert resolve_timeout_seconds({}, {"timeout": 0}, 600) == 0

    # Negative values are rejected and fall through to the next source
    assert resolve_timeout_seconds({"timeout_seconds": -1}, profile, 600) == 3600
    assert resolve_timeout_seconds({"timeout": -1}, {}, 600) == 600
    assert resolve_timeout_seconds({}, {"timeout": -5}, 600) == 600

    # Non-numeric values are ignored and fall through
    assert resolve_timeout_seconds({"timeout_seconds": "abc"}, profile, 600) == 3600
    assert resolve_timeout_seconds({"timeout": None}, profile, 600) == 3600


def test_run_benchmark_job_uses_garak_config_file(monkeypatch, tmp_path):
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run_garak_scan(
        config_file,
        timeout_seconds,
        report_prefix,
        env=None,
        log_file=None,
    ):
        _ = env, log_file
        captured["config_file"] = config_file
        captured["timeout_seconds"] = timeout_seconds
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
        return module.GarakScanResult(
            returncode=0,
            stdout="",
            stderr="",
            report_prefix=report_prefix,
        )

    monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
    monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
    monkeypatch.setattr(
        module.GarakAdapter,
        "_parse_results",
        lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="job-1",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters={"timeout_seconds": 42, "generations": 2},
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())

    config_path = Path(captured["config_file"])
    assert config_path.exists()

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_payload["reporting"]["report_prefix"].endswith("scan")
    assert config_payload["run"]["generations"] == 2
    assert captured["timeout_seconds"] == 42


def test_simple_mode_passes_hf_cache_env(monkeypatch, tmp_path):
    """When hf_cache_path is set, _run_simple passes HF_HUB_CACHE via env to run_garak_scan."""
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
        captured["env"] = env
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
        return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

    monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
    monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
    monkeypatch.setattr(
        module.GarakAdapter,
        "_parse_results",
        lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="hf-cache-job",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters={"hf_cache_path": "/test_data/hf-cache"},
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())
    assert captured["env"] == {"HF_HUB_CACHE": "/test_data/hf-cache"}


def test_simple_mode_no_hf_cache_passes_none_env(monkeypatch, tmp_path):
    """When hf_cache_path is not set, env=None is passed (default behavior)."""
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()
    monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
        captured["env"] = env
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
        return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

    monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
    monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
    monkeypatch.setattr(
        module.GarakAdapter,
        "_parse_results",
        lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="no-hf-cache-job",
        benchmark_id="trustyai_garak::quick",
        benchmark_index=0,
        model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
        parameters={},
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())
    assert captured["env"] is None


def test_parse_results_uses_overall_without_double_count(monkeypatch, tmp_path):
    module = _load_evalhub_garak_adapter(monkeypatch)
    adapter = module.GarakAdapter()

    report_prefix = tmp_path / "scan"
    report_prefix.with_suffix(".report.jsonl").write_text(
        '{"entry_type":"attempt","status":2}\n',
        encoding="utf-8",
    )
    report_prefix.with_suffix(".avid.jsonl").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_generations_from_report_content",
        lambda _content, _threshold: ([], {"probe.alpha": [{}]}, {"probe.alpha": [{}]}),
    )
    monkeypatch.setattr(
        module,
        "parse_aggregated_from_avid_content",
        lambda _content: {"probe.alpha": {"total_attempts": 10, "vulnerable_responses": 3}},
    )
    monkeypatch.setattr(module, "parse_digest_from_report_content", lambda _content: {"eval": {}})
    monkeypatch.setattr(
        module,
        "combine_parsed_results",
        lambda *_args, **_kwargs: {
            "scores": {
                "probe.alpha": {
                    "aggregated_results": {
                        "total_attempts": 10,
                        "vulnerable_responses": 3,
                        "benign_responses": 7,
                        "attack_success_rate": 30.0,
                        "metadata": {},
                    }
                },
                "_overall": {
                    "aggregated_results": {
                        "total_attempts": 10,
                        "vulnerable_responses": 3,
                        "attack_success_rate": 30.0,
                        "tbsa": 4.1,
                    }
                },
            }
        },
    )

    result = module.GarakScanResult(
        returncode=0,
        stdout="",
        stderr="",
        report_prefix=report_prefix,
    )
    metrics, overall_score, num_examples, overall_summary = adapter._parse_results(result, 0.5)

    assert len(metrics) == 2
    assert metrics[0].metric_name == "attack_success_rate"
    assert metrics[0].metric_type == "ratio"
    assert metrics[0].metric_value == 0.3
    assert metrics[0].num_samples == 10
    assert metrics[1].metric_name == "probe.alpha_asr"
    assert metrics[1].metric_type == "ratio"
    assert overall_score == 0.3
    assert num_examples == 10
    assert overall_summary["tbsa"] == 4.1


def test_convert_to_avid_report_imports_garak_report(monkeypatch, tmp_path):
    report_path = tmp_path / "scan.report.jsonl"
    report_path.write_text("{}", encoding="utf-8")

    called: dict[str, bool] = {"export": False}

    class _FakeReport:
        def __init__(self, _path):
            pass

        def load(self):
            return self

        def get_evaluations(self):
            return self

        def export(self):
            called["export"] = True

    fake_garak = types.ModuleType("garak")
    fake_report_module = types.ModuleType("garak.report")
    fake_report_module.Report = _FakeReport

    monkeypatch.setitem(sys.modules, "garak", fake_garak)
    monkeypatch.setitem(sys.modules, "garak.report", fake_report_module)

    assert convert_to_avid_report(report_path) is True
    assert called["export"] is True


# ---------------------------------------------------------------------------
# KFP mode tests
# ---------------------------------------------------------------------------


class TestResolveExecutionMode:
    """Tests for _resolve_execution_mode static method."""

    def test_defaults_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({}) == "simple"

    def test_benchmark_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_EXECUTION_MODE", "simple")
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": "kfp"}) == "kfp"

    def test_env_var_used_when_no_benchmark_config(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_EXECUTION_MODE", "kfp")
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({}) == "kfp"

    def test_unknown_mode_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": "unknown"}) == "simple"


class TestKFPConfig:
    """Tests for KFPConfig.from_env_and_config."""

    def test_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")
        monkeypatch.setenv("KFP_S3_SECRET_NAME", "my-data-connection")
        monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config()
        assert cfg.endpoint == "https://kfp.example.com"
        assert cfg.namespace == "test-ns"
        assert cfg.s3_secret_name == "my-data-connection"
        assert cfg.s3_bucket == "my-bucket"
        assert cfg.experiment_name == "evalhub-garak"

    def test_benchmark_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://env.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "env-ns")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config(
            {
                "kfp_config": {
                    "endpoint": "https://override.example.com",
                    "namespace": "override-ns",
                    "s3_secret_name": "custom-s3-conn",
                    "s3_bucket": "override-bucket",
                }
            }
        )
        assert cfg.endpoint == "https://override.example.com"
        assert cfg.namespace == "override-ns"
        assert cfg.s3_secret_name == "custom-s3-conn"
        assert cfg.s3_bucket == "override-bucket"

    def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("KFP_ENDPOINT", raising=False)
        monkeypatch.setenv("KFP_NAMESPACE", "ns")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig
        import pytest

        with pytest.raises(ValueError) as excinfo:
            KFPConfig.from_env_and_config()

        msg = str(excinfo.value)
        assert "KFP_ENDPOINT" in msg
        assert "kfp_config.endpoint" in msg

    def test_missing_namespace_raises(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.delenv("KFP_NAMESPACE", raising=False)

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig
        import pytest

        with pytest.raises(ValueError) as excinfo:
            KFPConfig.from_env_and_config()

        msg = str(excinfo.value)
        assert "KFP_NAMESPACE" in msg
        assert "kfp_config.namespace" in msg

    def test_verify_ssl_false(self, monkeypatch):
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "ns")
        monkeypatch.setenv("KFP_VERIFY_SSL", "false")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config()
        assert cfg.verify_ssl is False


class TestKFPModeExecution:
    """Tests for KFP mode in run_benchmark_job."""

    def test_kfp_mode_submits_pipeline(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        captured: dict[str, object] = {}

        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_via_kfp(
            self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5
        ):
            captured["called"] = True
            captured["timeout"] = timeout
            captured["config_json"] = garak_config_dict
            captured["intents_params"] = intents_params
            captured["eval_threshold"] = eval_threshold
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ), tmp_path

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-job-1",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={"execution_mode": "kfp", "timeout_seconds": 99},
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())
        assert captured["called"] is True
        assert captured["timeout"] == 99
        assert captured["eval_threshold"] == 0.5
        assert result.evaluation_metadata["execution_mode"] == "kfp"

    def test_simple_mode_does_not_call_kfp(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        kfp_called = {"value": False}

        def _fake_run_via_kfp(*args, **kwargs):
            kfp_called["value"] = True

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)

        report_prefix = tmp_path / "kfp-job-2" / "scan"
        report_prefix.parent.mkdir(parents=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            )

        monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-job-2",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={},
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())
        assert kfp_called["value"] is False
        assert result.evaluation_metadata["execution_mode"] == "simple"

    def _setup_mlflow_test(self, monkeypatch, tmp_path, job_id, mlflow_return_value):
        """Common setup for MLflow tests: mocks garak scan, parse, and injects fake mlflow module."""
        import types
        from unittest.mock import MagicMock

        mlflow_mod = types.ModuleType("evalhub.adapter.mlflow")
        mlflow_mod.MlflowArtifact = lambda name, data, mime: SimpleNamespace(name=name, data=data, mime_type=mime)
        monkeypatch.setitem(__import__("sys").modules, "evalhub.adapter.mlflow", mlflow_mod)

        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", lambda *a, **kw: None)

        report_prefix = tmp_path / job_id / "scan"
        report_prefix.parent.mkdir(parents=True)
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")
            return module.GarakScanResult(returncode=0, stdout="", stderr="", report_prefix=report_prefix)

        monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        mock_mlflow = MagicMock()
        mock_mlflow.save.return_value = mlflow_return_value

        class _Callbacks:
            mlflow = mock_mlflow

            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id=job_id,
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={},
            exports=None,
        )
        return adapter, job, _Callbacks(), mock_mlflow

    def test_mlflow_run_id_saved_on_results(self, monkeypatch, tmp_path):
        """When callbacks.mlflow.save() returns a run ID, it is stored on results."""
        adapter, job, callbacks, mock_mlflow = self._setup_mlflow_test(
            monkeypatch, tmp_path, "mlflow-job", "run-abc-123"
        )
        result = adapter.run_benchmark_job(job, callbacks)
        mock_mlflow.save.assert_called_once()
        assert result.mlflow_run_id == "run-abc-123"

    def test_mlflow_run_id_not_set_when_save_returns_none(self, monkeypatch, tmp_path):
        """When callbacks.mlflow.save() returns None, mlflow_run_id is not set."""
        adapter, job, callbacks, mock_mlflow = self._setup_mlflow_test(monkeypatch, tmp_path, "mlflow-job-2", None)
        result = adapter.run_benchmark_job(job, callbacks)
        mock_mlflow.save.assert_called_once()
        assert not hasattr(result, "mlflow_run_id") or result.mlflow_run_id is None

    def test_mlflow_save_exception_is_non_fatal(self, monkeypatch, tmp_path):
        """When callbacks.mlflow.save() raises, the job still succeeds and mlflow_run_id is unset."""
        adapter, job, callbacks, mock_mlflow = self._setup_mlflow_test(monkeypatch, tmp_path, "mlflow-job-3", None)
        mock_mlflow.save.side_effect = Exception("boom")
        result = adapter.run_benchmark_job(job, callbacks)
        assert result is not None
        assert result.id == "mlflow-job-3"
        assert not hasattr(result, "mlflow_run_id") or result.mlflow_run_id is None


class TestS3Download:
    """Tests for S3 download logic in the adapter."""

    def test_download_results_from_s3(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "http://minio:9000")

        downloaded_files = []

        class _FakeS3Client:
            def get_paginator(self, _method):
                return self

            def paginate(self, Bucket, Prefix):
                return [
                    {
                        "Contents": [
                            {"Key": f"{Prefix}/scan.report.jsonl"},
                            {"Key": f"{Prefix}/scan.avid.jsonl"},
                            {"Key": f"{Prefix}/scan.log"},
                            {"Key": f"{Prefix}/config.json"},
                        ]
                    }
                ]

            def download_file(self, bucket, key, local_path):
                downloaded_files.append(key)
                Path(local_path).write_text(f"content of {key}")

        monkeypatch.setattr(
            module.GarakAdapter,
            "_create_s3_client",
            staticmethod(lambda **kwargs: _FakeS3Client()),
        )

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3(
            "test-bucket",
            "evalhub-garak/job-123",
            local_dir,
        )

        assert len(downloaded_files) == 4
        assert (local_dir / "scan.report.jsonl").exists()
        assert (local_dir / "scan.avid.jsonl").exists()
        assert (local_dir / "scan.log").exists()
        assert (local_dir / "config.json").exists()

    def test_download_skips_when_no_bucket(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3("", "prefix", local_dir)
        assert list(local_dir.iterdir()) == []

    def test_download_handles_empty_listing(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

        class _FakeS3Client:
            def get_paginator(self, _method):
                return self

            def paginate(self, Bucket, Prefix):
                return [{"Contents": []}]

        monkeypatch.setattr(
            module.GarakAdapter,
            "_create_s3_client",
            staticmethod(lambda **kwargs: _FakeS3Client()),
        )

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3(
            "test-bucket",
            "evalhub-garak/job-empty",
            local_dir,
        )
        assert list(local_dir.iterdir()) == []


class TestPollKFPRun:
    """Tests for _poll_kfp_run static method."""

    def test_returns_on_succeeded(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        call_count = {"n": 0}

        class _FakeClient:
            def get_run(self, run_id):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    return SimpleNamespace(state="RUNNING")
                return SimpleNamespace(state="SUCCEEDED")

        class _Callbacks:
            statuses = []

            def report_status(self, update):
                self.statuses.append(update)

        # Use 0 poll interval for test speed
        monkeypatch.setattr("time.sleep", lambda _: None)

        callbacks = _Callbacks()
        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-123",
            callbacks,
            poll_interval=0,
        )
        assert state == "SUCCEEDED"
        assert call_count["n"] == 3
        assert len(callbacks.statuses) == 2  # two RUNNING updates before SUCCEEDED

    def test_returns_on_failed(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        class _FakeClient:
            def get_run(self, run_id):
                return SimpleNamespace(state="FAILED")

        class _Callbacks:
            def report_status(self, update):
                pass

        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-456",
            _Callbacks(),
            poll_interval=0,
        )
        assert state == "FAILED"

    @pytest.mark.parametrize("terminal_state", ["SKIPPED", "CANCELED", "CANCELING"])
    def test_returns_immediately_on_other_terminal_states(self, monkeypatch, terminal_state):
        module = _load_evalhub_garak_adapter(monkeypatch)

        call_count = {"n": 0}

        class _FakeClient:
            def get_run(self, run_id):
                call_count["n"] += 1
                return SimpleNamespace(state=terminal_state)

        class _Callbacks:
            statuses = []

            def report_status(self, update):
                self.statuses.append(update)

        monkeypatch.setattr("time.sleep", lambda _: None)

        callbacks = _Callbacks()
        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-789",
            callbacks,
            poll_interval=0,
        )
        assert state == terminal_state
        assert call_count["n"] == 1

    def test_times_out_when_deadline_exceeded(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        clock = {"now": 0.0}
        monkeypatch.setattr("time.monotonic", lambda: clock["now"])
        monkeypatch.setattr("time.sleep", lambda _: None)

        class _FakeClient:
            def get_run(self, run_id):
                clock["now"] += 50
                return SimpleNamespace(state="RUNNING")

        class _Callbacks:
            def report_status(self, update):
                pass

        state = module.GarakAdapter._poll_kfp_run(
            _FakeClient(),
            "run-timeout",
            _Callbacks(),
            poll_interval=0,
            timeout=100,
        )
        assert state == "TIMED_OUT"


class TestResolveExecutionModeNonString:
    """Test that non-string execution_mode values are handled safely."""

    def test_bool_execution_mode_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": True}) == "simple"

    def test_int_execution_mode_falls_back_to_simple(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_EXECUTION_MODE", raising=False)
        module = _load_evalhub_garak_adapter(monkeypatch)
        assert module.GarakAdapter._resolve_execution_mode({"execution_mode": 42}) == "simple"


class TestKFPMissingS3Secret:
    """Test that missing s3_secret_name is caught early."""

    def test_run_via_kfp_raises_without_s3_secret(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")
        monkeypatch.delenv("KFP_S3_SECRET_NAME", raising=False)

        class _Callbacks:
            def report_status(self, _update):
                return None

        job = SimpleNamespace(
            id="kfp-no-secret",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            parameters={"execution_mode": "kfp"},
            exports=None,
        )

        with pytest.raises(ValueError) as excinfo:
            adapter.run_benchmark_job(job, _Callbacks())

        msg = str(excinfo.value)
        assert "KFP_S3_SECRET_NAME" in msg
        assert "kfp_config.s3_secret_name" in msg


# ---------------------------------------------------------------------------
# Intents tests
# ---------------------------------------------------------------------------

_INTENTS_MODELS_SINGLE = {
    "intents_models": {
        "judge": {"url": "http://judge:8000/v1", "name": "judge-model"},
    }
}

_INTENTS_MODELS_ALL_ROLES = {
    "intents_models": {
        "judge": {"url": "http://judge:8000/v1", "name": "judge-model", "api_key": "jk"},
        "attacker": {"url": "http://attacker:9000/v1", "name": "atk-model", "api_key": "ak"},
        "evaluator": {"url": "http://evaluator:9001/v1", "name": "eval-model", "api_key": "ek"},
        "sdg": {"url": "http://sdg:7000/v1", "name": "sdg-model", "api_key": "sk"},
    }
}


class TestBuildConfigIntentsOverrides:
    """Tests for intents-specific config overrides in _build_config_from_spec."""

    def test_intents_profile_returns_art_intents_true(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={**_INTENTS_MODELS_SINGLE},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is True
        assert intents_params["sdg_flow_id"] == "major-sage-742"

    def test_single_role_fills_all_three(self, monkeypatch, tmp_path):
        """When only one of judge/attacker/evaluator is provided, all roles use it."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="intents-single-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_SINGLE},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "judge-model"
        assert tap_cfg["attack_model_config"]["uri"] == "http://judge:8000/v1"
        assert tap_cfg["evaluator_model_name"] == "judge-model"
        assert tap_cfg["evaluator_model_config"]["uri"] == "http://judge:8000/v1"

    def test_single_attacker_fills_all_three(self, monkeypatch, tmp_path):
        """Providing only 'attacker' should populate judge and evaluator too."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="intents-atk-only-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "intents_models": {
                    "attacker": {"url": "http://atk-model:9000/v1", "name": "atk-model"},
                }
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "atk-model"
        assert judge["detector_model_config"]["uri"] == "http://atk-model:9000/v1"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "atk-model"
        assert tap_cfg["evaluator_model_name"] == "atk-model"

    def test_separate_models_per_role(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="multi-model-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_ALL_ROLES},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert judge["detector_model_config"]["api_key"] == "__FROM_ENV__"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "atk-model"
        assert tap_cfg["attack_model_config"]["uri"] == "http://attacker:9000/v1"
        assert tap_cfg["attack_model_config"]["api_key"] == "__FROM_ENV__"
        assert tap_cfg["evaluator_model_name"] == "eval-model"
        assert tap_cfg["evaluator_model_config"]["uri"] == "http://evaluator:9001/v1"
        assert tap_cfg["evaluator_model_config"]["api_key"] == "__FROM_ENV__"

        assert intents_params["sdg_model"] == "sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"

    def test_no_roles_raises_clear_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="no-roles-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="requires model configuration"):
            adapter._build_config_from_spec(job, report_prefix)

    def test_two_roles_raises_ambiguous_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="two-roles-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "intents_models": {
                    "judge": {"url": "http://judge:8000/v1", "name": "judge-m"},
                    "attacker": {"url": "http://atk:9000/v1", "name": "atk-m"},
                }
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="Ambiguous"):
            adapter._build_config_from_spec(job, report_prefix)

    def test_missing_name_raises_clear_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="no-name-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "intents_models": {"judge": {"url": "http://judge:8000/v1"}},
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="intents_models.judge.name"):
            adapter._build_config_from_spec(job, report_prefix)

    # ------------------------------------------------------------------
    # Merge preservation tests (intents_models + garak_config overrides)
    # ------------------------------------------------------------------

    def test_multiclass_judge_config_preserved_with_intents_models(self, monkeypatch, tmp_path):
        """MulticlassJudge sub-config in garak_config survives intents_models override."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="mcj-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "MulticlassJudge": {
                                    "system_prompt": "Custom safety evaluator",
                                    "score_key": "complied",
                                    "score_scale": 100,
                                    "confidence_cutoff": 70,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert "MulticlassJudge" in judge
        assert judge["MulticlassJudge"]["system_prompt"] == "Custom safety evaluator"
        assert judge["MulticlassJudge"]["score_key"] == "complied"
        assert judge["MulticlassJudge"]["confidence_cutoff"] == 70

    def test_extra_attack_evaluator_config_keys_preserved(self, monkeypatch, tmp_path):
        """Extra keys in attack/evaluator model configs survive intents_models override."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="extra-keys-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probes": {
                            "tap": {
                                "TAPIntent": {
                                    "attack_model_config": {
                                        "temperature": 0.9,
                                        "top_p": 0.95,
                                    },
                                    "evaluator_model_config": {
                                        "top_p": 0.8,
                                        "presence_penalty": 0.5,
                                    },
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        tap = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]

        atk = tap["attack_model_config"]
        assert atk["uri"] == "http://attacker:9000/v1"
        assert atk["api_key"] == "__FROM_ENV__"
        assert atk["temperature"] == 0.9
        assert atk["top_p"] == 0.95

        ev = tap["evaluator_model_config"]
        assert ev["uri"] == "http://evaluator:9001/v1"
        assert ev["api_key"] == "__FROM_ENV__"
        assert ev["top_p"] == 0.8
        assert ev["presence_penalty"] == 0.5

    # ------------------------------------------------------------------
    # garak_config-only mode (no intents_models)
    # ------------------------------------------------------------------

    def test_garak_config_only_skips_intents_models_override(self, monkeypatch, tmp_path):
        """Full model config in garak_config works without intents_models."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="gc-only-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "sdg_model": "my-sdg-model",
                "sdg_api_base": "http://sdg:7000/v1",
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "detector_model_name": "my-judge",
                                "detector_model_config": {
                                    "uri": "http://judge:8000/v1",
                                    "api_key": "sk-judge",
                                },
                                "MulticlassJudge": {
                                    "system_prompt": "Custom prompt",
                                },
                            }
                        },
                        "probes": {
                            "tap": {
                                "TAPIntent": {
                                    "attack_model_name": "my-atk",
                                    "attack_model_config": {
                                        "uri": "http://atk:9000/v1",
                                        "api_key": "sk-atk",
                                    },
                                    "evaluator_model_name": "my-eval",
                                    "evaluator_model_config": {
                                        "uri": "http://eval:9001/v1",
                                        "api_key": "sk-eval",
                                    },
                                }
                            }
                        },
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "my-judge"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert judge["detector_model_config"]["api_key"] == "sk-judge"
        assert judge["MulticlassJudge"]["system_prompt"] == "Custom prompt"

        tap = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap["attack_model_name"] == "my-atk"
        assert tap["attack_model_config"]["uri"] == "http://atk:9000/v1"
        assert tap["attack_model_config"]["api_key"] == "sk-atk"
        assert tap["evaluator_model_name"] == "my-eval"
        assert tap["evaluator_model_config"]["uri"] == "http://eval:9001/v1"
        assert tap["evaluator_model_config"]["api_key"] == "sk-eval"

        assert intents_params["sdg_model"] == "my-sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"

    def test_partial_garak_config_missing_tap_raises(self, monkeypatch, tmp_path):
        """Judge configured in garak_config but TAP models missing raises error."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="partial-gc-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "detector_model_name": "my-judge",
                                "detector_model_config": {"uri": "http://judge:8000/v1"},
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="requires model configuration"):
            adapter._build_config_from_spec(job, report_prefix)

    # ------------------------------------------------------------------
    # Deep-merge granularity tests: override a single leaf without
    # clobbering siblings or ancestor dicts
    # ------------------------------------------------------------------

    def test_single_tapintent_field_override_preserves_profile_defaults(self, monkeypatch, tmp_path):
        """Override only ``depth`` in TAPIntent; all other profile defaults survive deep-merge."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="deep-tap-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "probes": {
                            "tap": {
                                "TAPIntent": {
                                    "depth": 20,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        tap = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap["depth"] == 20, "User override for depth must take effect"
        assert tap["width"] == 10, "Profile default width must survive"
        assert tap["attack_max_attempts"] == 5, "Profile default attack_max_attempts must survive"
        assert tap["branching_factor"] == 4, "Profile default branching_factor must survive"
        assert tap["pruning"] is True, "Profile default pruning must survive"
        assert tap["attack_model_config"]["uri"] == "http://attacker:9000/v1"

    def test_extra_keys_in_detector_model_config_preserved(self, monkeypatch, tmp_path):
        """Extra keys in ``detector_model_config`` survive ``_apply_intents_models`` overlay."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="det-cfg-extra-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "detector_model_config": {
                                    "max_tokens": 500,
                                    "temperature": 0.1,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        det_cfg = config_dict["plugins"]["detectors"]["judge"]["detector_model_config"]
        assert det_cfg["uri"] == "http://judge:8000/v1", "intents_models uri must be injected"
        assert det_cfg["api_key"] == "__FROM_ENV__", "placeholder api_key must be injected"
        assert det_cfg["max_tokens"] == 500, "User-provided max_tokens must survive overlay"
        assert det_cfg["temperature"] == 0.1, "User-provided temperature must survive overlay"

    def test_single_multiclass_judge_field_override(self, monkeypatch, tmp_path):
        """Override only ``system_prompt`` in MulticlassJudge via garak_config;
        other MulticlassJudge fields provided alongside must survive."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="mcj-partial-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={
                **_INTENTS_MODELS_ALL_ROLES,
                "garak_config": {
                    "plugins": {
                        "detectors": {
                            "judge": {
                                "MulticlassJudge": {
                                    "system_prompt": "Custom evaluator prompt",
                                    "score_key": "complied",
                                    "score_scale": 100,
                                    "confidence_cutoff": 70,
                                }
                            }
                        }
                    }
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        mcj = judge["MulticlassJudge"]
        assert mcj["system_prompt"] == "Custom evaluator prompt"
        assert mcj["score_key"] == "complied"
        assert mcj["score_scale"] == 100
        assert mcj["confidence_cutoff"] == 70

    def test_attack_model_config_extra_keys_survive_overlay(self, monkeypatch, tmp_path):
        """Profile's ``max_tokens`` in ``attack_model_config`` survives ``_apply_intents_models``."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="atk-max-tokens-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://target:8000", name="target-llm"),
            parameters={**_INTENTS_MODELS_ALL_ROLES},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, _ = adapter._build_config_from_spec(job, report_prefix)

        atk = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]["attack_model_config"]
        assert atk["uri"] == "http://attacker:9000/v1"
        assert atk["api_key"] == "__FROM_ENV__"
        assert atk["max_tokens"] == 500, "Profile's max_tokens must survive intents_models overlay"

        ev = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]["evaluator_model_config"]
        assert ev["uri"] == "http://evaluator:9001/v1"
        assert ev["api_key"] == "__FROM_ENV__"
        assert ev["max_tokens"] == 10, "Profile's max_tokens must survive intents_models overlay"
        assert ev["temperature"] == 0.0, "Profile's temperature must survive intents_models overlay"

    def test_non_intents_profile_returns_art_intents_false(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="native-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is False

    def test_benchmark_config_art_intents_false_overrides_profile(self, monkeypatch, tmp_path):
        """Explicit art_intents=False in benchmark_config wins over profile (intents profile sets True)."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="override-false-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={"art_intents": False},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is False

    def test_benchmark_config_art_intents_true_overrides_profile(self, monkeypatch, tmp_path):
        """Explicit art_intents=True in benchmark_config wins over non-intents profile."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="override-true-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "art_intents": True,
                **_INTENTS_MODELS_SINGLE,
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is True

    def test_sdg_params_from_intents_models(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="sdg-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                **_INTENTS_MODELS_SINGLE,
                "intents_models": {
                    **_INTENTS_MODELS_SINGLE["intents_models"],
                    "sdg": {"url": "http://sdg:7000/v1", "name": "sdg-model", "api_key": "sdg-key"},
                },
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["sdg_model"] == "sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"

    def test_sdg_fallback_to_flat_keys(self, monkeypatch, tmp_path):
        """When intents_models.sdg is absent, fall back to flat sdg_model/sdg_api_base."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="sdg-flat-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                **_INTENTS_MODELS_SINGLE,
                "sdg_model": "legacy-sdg",
                "sdg_api_base": "http://legacy-sdg:5000",
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["sdg_model"] == "legacy-sdg"
        assert intents_params["sdg_api_base"] == "http://legacy-sdg:5000"


class TestResolveIntentsApiKey:
    """Tests for _resolve_intents_api_key static method."""

    def test_direct_api_key_takes_priority(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "from-env")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key": "direct-key"},
        )
        assert result == "direct-key"

    def test_api_key_env_from_config(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-val")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key_env": "MY_CUSTOM_KEY"},
        )
        assert result == "custom-val"

    def test_convention_env_var(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "conv-key")

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "conv-key"

    def test_generic_env_fallback(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAICOMPATIBLE_API_KEY", "generic-key")

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "generic-key"

    def test_returns_dummy_when_nothing_found(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "DUMMY"

    def _install_auth_stub(self, monkeypatch, fake_read_fn):
        """Install a stub evalhub.adapter.auth module with a fake read_model_auth_key."""
        auth_module = types.ModuleType("evalhub.adapter.auth")
        auth_module.read_model_auth_key = fake_read_fn
        monkeypatch.setitem(sys.modules, "evalhub.adapter.auth", auth_module)

    def test_api_key_name_takes_precedence_over_env_vars(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "env-judge")
        monkeypatch.setenv("OPENAICOMPATIBLE_API_KEY", "env-openai")

        calls = []

        def fake_read(name):
            calls.append(name)
            return "secret-from-custom" if name == "custom-secret" else None

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key_name": "custom-secret"},
        )
        assert result == "secret-from-custom"
        assert calls == ["custom-secret"]

    def test_role_specific_secret_when_no_explicit_key_or_env(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)

        def fake_read(name):
            return "secret-judge" if name == "judge-api-key" else None

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "secret-judge"

    def test_generic_secret_when_role_specific_missing(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.delenv("JUDGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAICOMPATIBLE_API_KEY", raising=False)

        def fake_read(name):
            return "generic-secret" if name == "api-key" else None

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key("judge", {})
        assert result == "generic-secret"

    def test_api_key_env_takes_precedence_over_role_specific_env(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("JUDGE_API_KEY", "env-judge")
        monkeypatch.setenv("CUSTOM_KEY", "env-custom")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key_env": "CUSTOM_KEY"},
        )
        assert result == "env-custom"

    def test_direct_api_key_wins_over_secret(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)

        def fake_read(name):
            return "should-not-be-used"

        self._install_auth_stub(monkeypatch, fake_read)

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge",
            {"api_key": "direct-wins", "api_key_name": "some-secret"},
        )
        assert result == "direct-wins"


class TestIntentsRequiresKFP:
    """Intents benchmarks must reject non-KFP execution modes."""

    def test_intents_simple_mode_raises(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        class _Callbacks:
            def report_status(self, _update):
                return None

        job = SimpleNamespace(
            id="intents-simple-job",
            benchmark_id="trustyai_garak::intents_spo",
            benchmark_index=0,
            model=SimpleNamespace(
                url="http://localhost:8000",
                name="test-model",
            ),
            parameters={
                "execution_mode": "simple",
                "art_intents": True,
            },
            exports=None,
        )

        with pytest.raises(ValueError, match="Intents benchmarks are only supported in KFP"):
            adapter.run_benchmark_job(job, _Callbacks())


class TestKFPIntentsMode:
    """Tests for intents mode through run_benchmark_job with KFP."""

    def test_kfp_intents_passes_params_to_pipeline(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        captured: dict[str, object] = {}
        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_via_kfp(
            self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5
        ):
            captured["intents_params"] = intents_params
            captured["eval_threshold"] = eval_threshold
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ), tmp_path

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "execution_mode": "kfp",
                **_INTENTS_MODELS_ALL_ROLES,
            },
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())

        assert captured["intents_params"]["art_intents"] is True
        assert captured["intents_params"]["sdg_model"] == "sdg-model"
        assert result.evaluation_metadata["art_intents"] is True

    def test_art_html_generated_for_intents(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setenv("KFP_ENDPOINT", "http://kfp:8080")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")

        scan_dir = tmp_path / "intents-html-job"
        scan_dir.mkdir(parents=True)
        report_prefix = scan_dir / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text(
            '{"entry_type":"attempt","status":2}\n',
            encoding="utf-8",
        )

        def _fake_run_via_kfp(
            self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5
        ):
            return module.GarakScanResult(
                returncode=0,
                stdout="",
                stderr="",
                report_prefix=report_prefix,
            ), scan_dir

        monkeypatch.setattr(module.GarakAdapter, "_run_via_kfp", _fake_run_via_kfp)
        monkeypatch.setattr(
            module.GarakAdapter,
            "_parse_results",
            lambda self, result, eval_threshold, art_intents=False: ([], None, 0, {"total_attempts": 0}),
        )

        html_generated = {"called": False}

        def _fake_generate_art_report(content, **kwargs):
            html_generated["called"] = True
            return "<html><body>ART Report</body></html>"

        monkeypatch.setattr(module, "generate_art_report", _fake_generate_art_report)

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="intents-html-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={**_INTENTS_MODELS_SINGLE, "execution_mode": "kfp"},
            exports=None,
        )

        adapter.run_benchmark_job(job, _Callbacks())

        assert html_generated["called"] is True
        html_path = scan_dir / "scan.intents.html"
        assert html_path.exists()
        assert "ART Report" in html_path.read_text()

    def test_kfp_raises_when_art_intents_but_no_sdg_model(self, monkeypatch, tmp_path):
        """_run_via_kfp raises ValueError when art_intents=True with no sdg_model (sdg_model always required)."""
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))
        monkeypatch.setenv("KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("KFP_NAMESPACE", "test-ns")
        monkeypatch.setenv("KFP_S3_SECRET_NAME", "data-conn")

        class _Callbacks:
            def report_status(self, _update):
                return None

            def create_oci_artifact(self, _spec):
                return SimpleNamespace(reference="oci://ref", digest="sha256:test")

        job = SimpleNamespace(
            id="kfp-no-data-source",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            parameters={
                "execution_mode": "kfp",
                **_INTENTS_MODELS_SINGLE,
            },
            exports=None,
        )

        with pytest.raises(ValueError, match="sdg_model"):
            adapter.run_benchmark_job(job, _Callbacks())


class TestParseResultsIntentsMode:
    """Tests for _parse_results with art_intents=True."""

    def test_parse_results_with_art_intents(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()

        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text(
            '{"entry_type":"attempt","status":2}\n',
            encoding="utf-8",
        )
        report_prefix.with_suffix(".avid.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr(
            module,
            "parse_generations_from_report_content",
            lambda _content, _threshold: (
                [],
                {"spo.SPOIntent": [{}]},
                {"spo.SPOIntent": [{"detector_results": {}, "notes": {}}]},
            ),
        )
        monkeypatch.setattr(
            module,
            "parse_aggregated_from_avid_content",
            lambda _content: {},
        )
        monkeypatch.setattr(
            module,
            "parse_digest_from_report_content",
            lambda _content: {},
        )
        monkeypatch.setattr(
            module,
            "combine_parsed_results",
            lambda *_args, **kwargs: {
                "scores": {
                    "spo.SPOIntent": {
                        "aggregated_results": {
                            "total_attempts": 20,
                            "unsafe_stubs": 3,
                            "safe_stubs": 7,
                            "attack_success_rate": 30.0,
                            "intent_breakdown": {
                                "S001": {
                                    "total_attempts": 10,
                                    "total_stubs": 5,
                                    "unsafe_stubs": 2,
                                    "safe_stubs": 3,
                                    "attack_success_rate": 40.0,
                                },
                                "S002": {
                                    "total_attempts": 10,
                                    "total_stubs": 5,
                                    "unsafe_stubs": 1,
                                    "safe_stubs": 4,
                                    "attack_success_rate": 20.0,
                                },
                            },
                            "metadata": {},
                        }
                    },
                    "_overall": {
                        "aggregated_results": {
                            "total_attempts": 20,
                            "attack_success_rate": 30.0,
                            "intent_breakdown": {},
                        }
                    },
                }
            },
        )

        result = module.GarakScanResult(
            returncode=0,
            stdout="",
            stderr="",
            report_prefix=report_prefix,
        )
        metrics, overall_score, num_examples, _ = adapter._parse_results(
            result,
            0.5,
            art_intents=True,
        )

        assert len(metrics) == 2
        assert metrics[0].metric_name == "attack_success_rate"
        assert metrics[0].metric_type == "ratio"
        assert metrics[0].metric_value == 0.3
        assert metrics[0].num_samples == 20
        assert metrics[1].metric_name == "spo.SPOIntent_asr"
        assert metrics[1].metric_type == "ratio"
        assert metrics[1].metric_value == 0.3
        assert metrics[1].num_samples is None
        assert metrics[1].metadata["total_attempts"] == 20
        assert metrics[1].metadata["unsafe_stubs"] == 3
        assert metrics[1].metadata["safe_stubs"] == 7
        assert "intent_breakdown" in metrics[1].metadata
        assert metrics[1].metadata["intent_breakdown"]["S001"]["unsafe_stubs"] == 2
        assert overall_score == 0.3
        assert num_examples == 20


# ---------------------------------------------------------------------------
# Targeted KFP component tests
# ---------------------------------------------------------------------------


class _FakeArtifact:
    """Minimal stand-in for KFP dsl.Output / dsl.Input artifacts."""

    def __init__(self, path: str):
        self.path = path


class _FakeMetrics:
    """Minimal stand-in for dsl.Output[dsl.Metrics]."""

    def __init__(self):
        self.logged: dict[str, float] = {}

    def log_metric(self, name: str, value: float):
        self.logged[name] = value


def _get_component_fn(component_func):
    """Extract the raw Python function from a KFP @dsl.component."""
    return getattr(component_func, "python_func", component_func)


class TestValidateComponent:
    """Targeted tests for the validate KFP component."""

    def test_valid_config_and_s3_passes(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate

        # Mock garak module so import succeeds
        monkeypatch.setitem(sys.modules, "garak", types.ModuleType("garak"))

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")

        head_bucket_called = {}

        def _fake_create_s3_client():
            client = SimpleNamespace(
                head_bucket=lambda Bucket: head_bucket_called.update({"bucket": Bucket}),
            )
            return client

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        config_json = json.dumps({"plugins": {"probe_spec": ["test"]}, "reporting": {}})
        fn = _get_component_fn(validate)
        result = fn(config_json=config_json)
        assert result.valid is True
        assert head_bucket_called["bucket"] == "test-bucket"

    def test_malformed_config_json_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            return SimpleNamespace(head_bucket=lambda Bucket: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="not valid JSON"):
            fn(config_json="not-valid-json{{{")

    def test_missing_plugins_section_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            return SimpleNamespace(head_bucket=lambda Bucket: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="plugins"):
            fn(config_json=json.dumps({"reporting": {}}))

    def test_missing_s3_bucket_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        # Mock garak module so import succeeds
        monkeypatch.setitem(sys.modules, "garak", types.ModuleType("garak"))

        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="AWS_S3_BUCKET"):
            fn(config_json=json.dumps({"plugins": {}}))

    def test_unreachable_s3_bucket_raises(self, monkeypatch):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import validate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        # Mock garak module so import succeeds
        monkeypatch.setitem(sys.modules, "garak", types.ModuleType("garak"))

        monkeypatch.setenv("AWS_S3_BUCKET", "bad-bucket")

        def _fake_create_s3_client():
            def _fail(**kwargs):
                raise ConnectionError("unreachable")

            return SimpleNamespace(head_bucket=_fail)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        fn = _get_component_fn(validate)
        with pytest.raises(GarakValidationError, match="not reachable"):
            fn(config_json=json.dumps({"plugins": {}}))


class TestResolveTaxonomyComponent:
    """Tests for the resolve_taxonomy KFP component."""

    def test_no_policy_emits_base_taxonomy(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import resolve_taxonomy
        import pandas as pd

        artifact = _FakeArtifact(str(tmp_path / "taxonomy.csv"))

        fn = _get_component_fn(resolve_taxonomy)
        fn(policy_s3_key="", policy_format="csv", taxonomy_dataset=artifact)

        df = pd.read_csv(artifact.path)
        assert len(df) > 0
        assert "policy_concept" in df.columns

    def test_custom_taxonomy_from_s3(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import resolve_taxonomy
        import pandas as pd

        artifact = _FakeArtifact(str(tmp_path / "taxonomy.csv"))

        taxonomy_csv = "policy_concept,concept_definition\nCustom,Custom def\n"

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                return {"Body": SimpleNamespace(read=lambda: taxonomy_csv.encode())}

            return SimpleNamespace(get_object=_get_object)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        fn = _get_component_fn(resolve_taxonomy)
        fn(
            policy_s3_key="path/to/taxonomy.csv",
            policy_format="csv",
            taxonomy_dataset=artifact,
        )

        df = pd.read_csv(artifact.path)
        assert df["policy_concept"].iloc[0] == "Custom"

    def test_invalid_s3_uri_raises(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import resolve_taxonomy
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        artifact = _FakeArtifact(str(tmp_path / "taxonomy.csv"))

        fn = _get_component_fn(resolve_taxonomy)
        with pytest.raises(GarakValidationError, match="Invalid policy_s3_key"):
            fn(
                policy_s3_key="s3://bucket-only",
                policy_format="csv",
                taxonomy_dataset=artifact,
            )


class TestSdgGenerateComponent:
    """Tests for the sdg_generate KFP component."""

    def test_non_intents_writes_empty_marker(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=False,
            intents_s3_key="",
            sdg_model="",
            sdg_api_base="",
            sdg_flow_id="",
            sdg_max_concurrency=10,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        assert Path(sdg_out.path).read_text() == ""

    def test_bypass_mode_writes_empty_marker(self, monkeypatch, tmp_path):
        """When intents_s3_key is set, sdg_generate writes empty (bypass handled by prepare_prompts)."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="path/to/intents.csv",
            sdg_model="",
            sdg_api_base="",
            sdg_flow_id="",
            sdg_max_concurrency=10,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        assert Path(sdg_out.path).read_text() == ""

    def test_intents_no_sdg_model_raises(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        with pytest.raises(GarakValidationError, match="sdg_model"):
            fn(
                art_intents=True,
                intents_s3_key="",
                sdg_model="",
                sdg_api_base="",
                sdg_flow_id="",
                sdg_max_concurrency=10,
                sdg_num_samples=0,
                sdg_max_tokens=0,
                taxonomy_dataset=taxonomy,
                sdg_dataset=sdg_out,
            )

    def test_sdg_missing_api_base_raises(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.errors import GarakValidationError

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        fn = _get_component_fn(sdg_generate)
        with pytest.raises(GarakValidationError, match="sdg_api_base"):
            fn(
                art_intents=True,
                intents_s3_key="",
                sdg_model="some-model",
                sdg_api_base="",
                sdg_flow_id="",
                sdg_max_concurrency=10,
                sdg_num_samples=0,
                sdg_max_tokens=0,
                taxonomy_dataset=taxonomy,
                sdg_dataset=sdg_out,
            )

    def test_sdg_outputs_raw_artifact(self, monkeypatch, tmp_path):
        """SDG path writes only the raw artifact (normalisation is in prepare_prompts)."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        import pandas as pd

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nHarm,Harm def\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        raw_df = pd.DataFrame(
            {
                "policy_concept": ["Harm"],
                "concept_definition": ["Harm def"],
                "prompt": ["generated"],
                "demographics_pool": [["Teens"]],
            }
        )
        norm_df = pd.DataFrame(
            {
                "category": ["harm"],
                "prompt": ["generated"],
                "description": ["Harm def"],
            }
        )

        def _fake_generate_sdg(
            model,
            api_base,
            flow_id,
            api_key="dummy",
            taxonomy=None,
            max_concurrency=10,
            num_samples=0,
            max_tokens=0,
        ):
            return SDGResult(raw=raw_df, normalized=norm_df)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset",
            _fake_generate_sdg,
        )

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="",
            sdg_model="test-model",
            sdg_api_base="http://sdg:8000",
            sdg_flow_id="test-flow",
            sdg_max_concurrency=10,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        raw_result = pd.read_csv(sdg_out.path)
        assert "policy_concept" in raw_result.columns
        assert "prompt" in raw_result.columns
        assert len(raw_result) == 1

    def test_non_default_sdg_params_plumbed_through(self, monkeypatch, tmp_path):
        """Non-default sdg_max_concurrency/num_samples/max_tokens reach generate_sdg_dataset."""
        from unittest.mock import MagicMock
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        import pandas as pd

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        raw_df = pd.DataFrame({"policy_concept": ["A"], "prompt": ["p"], "concept_definition": ["B"]})
        mock_generate = MagicMock(return_value=SDGResult(raw=raw_df, normalized=raw_df))
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset",
            mock_generate,
        )

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="",
            sdg_model="test-model",
            sdg_api_base="http://sdg:8000",
            sdg_flow_id="test-flow",
            sdg_max_concurrency=3,
            sdg_num_samples=25,
            sdg_max_tokens=4096,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        mock_generate.assert_called_once()
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["max_concurrency"] == 3
        assert kwargs["num_samples"] == 25
        assert kwargs["max_tokens"] == 4096

    def test_default_sdg_params_when_zeros(self, monkeypatch, tmp_path):
        """When sdg params are 0 (default), they are passed as 0 and generate_sdg_dataset handles the fallback."""
        from unittest.mock import MagicMock
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import sdg_generate
        from llama_stack_provider_trustyai_garak.sdg import SDGResult
        import pandas as pd

        taxonomy = _FakeArtifact(str(tmp_path / "taxonomy.csv"))
        Path(taxonomy.path).write_text("policy_concept,concept_definition\nA,B\n")
        sdg_out = _FakeArtifact(str(tmp_path / "sdg.csv"))

        raw_df = pd.DataFrame({"policy_concept": ["A"], "prompt": ["p"], "concept_definition": ["B"]})
        mock_generate = MagicMock(return_value=SDGResult(raw=raw_df, normalized=raw_df))
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.sdg.generate_sdg_dataset",
            mock_generate,
        )

        fn = _get_component_fn(sdg_generate)
        fn(
            art_intents=True,
            intents_s3_key="",
            sdg_model="test-model",
            sdg_api_base="http://sdg:8000",
            sdg_flow_id="test-flow",
            sdg_max_concurrency=0,
            sdg_num_samples=0,
            sdg_max_tokens=0,
            taxonomy_dataset=taxonomy,
            sdg_dataset=sdg_out,
        )

        mock_generate.assert_called_once()
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["max_concurrency"] == 0
        assert kwargs["num_samples"] == 0
        assert kwargs["max_tokens"] == 0


class TestPreparePromptsComponent:
    """Tests for the prepare_prompts KFP component."""

    def test_non_intents_writes_empty_marker(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts

        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text("")
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=False,
            s3_prefix="evalhub-garak-kfp/job1",
            intents_s3_key="",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        assert Path(prompts_out.path).read_text() == ""

    def test_sdg_path_normalises_and_uploads(self, monkeypatch, tmp_path):
        """SDG ran: reads raw artifact, uploads raw + normalised to S3, outputs normalised."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts
        import pandas as pd

        raw_csv = "policy_concept,concept_definition,prompt\nHarm,Harmful stuff,Do bad\n"
        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text(raw_csv)
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        uploaded: dict[str, str] = {}

        def _fake_create_s3_client():
            def _put_object(Bucket, Key, Body):
                uploaded[Key] = Body.decode("utf-8") if isinstance(Body, bytes) else Body

            def _upload_file(filepath, bucket, key):
                uploaded[key] = Path(filepath).read_text()

            return SimpleNamespace(put_object=_put_object, upload_file=_upload_file)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=True,
            s3_prefix="evalhub-garak-kfp/job123",
            intents_s3_key="",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        assert "evalhub-garak-kfp/job123/sdg_raw_output.csv" in uploaded
        assert "evalhub-garak-kfp/job123/sdg_normalized_output.csv" in uploaded

        result_df = pd.read_csv(prompts_out.path)
        assert "category" in result_df.columns
        assert "prompt" in result_df.columns
        assert len(result_df) == 1

        raw_uploaded = uploaded["evalhub-garak-kfp/job123/sdg_raw_output.csv"]
        assert "policy_concept" in raw_uploaded

    def test_bypass_fetches_and_preserves_raw(self, monkeypatch, tmp_path):
        """Bypass mode: fetches user file from S3, uploads original as raw, normalises."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts
        import pandas as pd

        user_csv = "policy_concept,concept_definition,prompt,demographics_pool\nHarm,Bad stuff,Do harm,\"['Teens']\"\n"

        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text("")
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        uploaded: dict[str, str] = {}

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                return {"Body": SimpleNamespace(read=lambda: user_csv.encode())}

            def _put_object(Bucket, Key, Body):
                uploaded[Key] = Body.decode("utf-8") if isinstance(Body, bytes) else Body

            def _upload_file(filepath, bucket, key):
                uploaded[key] = Path(filepath).read_text()

            return SimpleNamespace(
                get_object=_get_object,
                put_object=_put_object,
                upload_file=_upload_file,
            )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=True,
            s3_prefix="evalhub-garak-kfp/job456",
            intents_s3_key="user/intents.csv",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        raw_uploaded = uploaded.get("evalhub-garak-kfp/job456/sdg_raw_output.csv", "")
        assert "demographics_pool" in raw_uploaded

        result_df = pd.read_csv(prompts_out.path)
        assert list(result_df.columns) == ["category", "prompt", "description"]
        assert len(result_df) == 1

    def test_empty_sdg_and_no_bypass_writes_empty(self, monkeypatch, tmp_path):
        """No SDG output and no bypass key: writes empty marker."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import prepare_prompts

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        sdg_in = _FakeArtifact(str(tmp_path / "sdg.csv"))
        Path(sdg_in.path).write_text("")
        prompts_out = _FakeArtifact(str(tmp_path / "prompts.csv"))

        fn = _get_component_fn(prepare_prompts)
        fn(
            art_intents=True,
            s3_prefix="evalhub-garak-kfp/job1",
            intents_s3_key="",
            intents_format="csv",
            sdg_dataset=sdg_in,
            prompts_dataset=prompts_out,
        )

        assert Path(prompts_out.path).read_text() == ""


class TestAdapterMutualExclusivity:
    """Tests for mutual exclusivity of policy_s3_key and intents_s3_key in adapter."""

    _KFP_ENV = {
        "KFP_ENDPOINT": "http://kfp:8080",
        "KFP_NAMESPACE": "test-ns",
        "KFP_S3_SECRET_NAME": "test-secret",
        "AWS_S3_BUCKET": "test-bucket",
    }

    def test_both_policy_and_intents_s3_key_raises(self, monkeypatch):
        """Both policy_s3_key and intents_s3_key set raises ValueError."""
        from llama_stack_provider_trustyai_garak.evalhub.garak_adapter import GarakAdapter

        for k, v in self._KFP_ENV.items():
            monkeypatch.setenv(k, v)

        adapter = GarakAdapter.__new__(GarakAdapter)

        with pytest.raises(ValueError, match="mutually exclusive"):
            adapter._run_via_kfp(
                config=SimpleNamespace(
                    id="test-job",
                    benchmark_id="test",
                    parameters={},
                ),
                callbacks=SimpleNamespace(report_status=lambda *a, **kw: None),
                garak_config_dict={"plugins": {}},
                timeout=300,
                intents_params={
                    "art_intents": True,
                    "policy_s3_key": "path/to/policy.csv",
                    "intents_s3_key": "path/to/intents.csv",
                    "sdg_model": "model",
                    "sdg_api_base": "http://base",
                },
                eval_threshold=0.5,
            )

    def test_bypass_mode_no_sdg_params_accepted(self, monkeypatch):
        """intents_s3_key set without sdg_model/sdg_api_base should not raise SDG validation."""
        from llama_stack_provider_trustyai_garak.evalhub.garak_adapter import GarakAdapter

        for k, v in self._KFP_ENV.items():
            monkeypatch.setenv(k, v)

        adapter = GarakAdapter.__new__(GarakAdapter)

        ip = {
            "art_intents": True,
            "policy_s3_key": "",
            "intents_s3_key": "path/to/intents.csv",
            "intents_format": "csv",
            "sdg_model": "",
            "sdg_api_base": "",
        }

        try:
            adapter._run_via_kfp(
                config=SimpleNamespace(
                    id="test-job",
                    benchmark_id="test",
                    parameters={},
                ),
                callbacks=SimpleNamespace(report_status=lambda *a, **kw: None),
                garak_config_dict={"plugins": {}},
                timeout=300,
                intents_params=ip,
                eval_threshold=0.5,
            )
        except ValueError as e:
            assert "sdg_model" not in str(e), f"Should not require sdg_model in bypass mode: {e}"
            assert "sdg_api_base" not in str(e), f"Should not require sdg_api_base in bypass mode: {e}"
        except Exception:
            pass  # Other errors (KFP client creation, etc.) are expected

    def test_sdg_mode_requires_sdg_params(self, monkeypatch):
        """art_intents=True without intents_s3_key requires sdg_model."""
        from llama_stack_provider_trustyai_garak.evalhub.garak_adapter import GarakAdapter

        for k, v in self._KFP_ENV.items():
            monkeypatch.setenv(k, v)

        adapter = GarakAdapter.__new__(GarakAdapter)

        with pytest.raises(ValueError, match="sdg_model"):
            adapter._run_via_kfp(
                config=SimpleNamespace(
                    id="test-job",
                    benchmark_id="test",
                    parameters={},
                ),
                callbacks=SimpleNamespace(report_status=lambda *a, **kw: None),
                garak_config_dict={"plugins": {}},
                timeout=300,
                intents_params={
                    "art_intents": True,
                    "policy_s3_key": "",
                    "intents_s3_key": "",
                    "sdg_model": "",
                    "sdg_api_base": "",
                },
                eval_threshold=0.5,
            )


class TestArtifactMetadataSurfacing:
    """Tests that artifact S3 paths are surfaced in evaluation_metadata."""

    def test_intents_kfp_job_includes_only_verified_artifacts(self, monkeypatch):
        """art_intents=True + KFP mode -> only artifacts verified in S3 are included."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import DEFAULT_S3_PREFIX
        from llama_stack_provider_trustyai_garak.constants import EXECUTION_MODE_KFP
        from unittest.mock import MagicMock
        import os

        job_id = "test-job-123"
        bucket = "my-bucket"
        s3_prefix = f"{DEFAULT_S3_PREFIX}/{job_id}"

        monkeypatch.setenv("AWS_S3_BUCKET", bucket)

        existing_keys = {
            f"{s3_prefix}/sdg_raw_output.csv",
            f"{s3_prefix}/sdg_normalized_output.csv",
            f"{s3_prefix}/scan.report.jsonl",
        }

        mock_s3 = MagicMock()

        def _head_object(Bucket, Key):
            if Key not in existing_keys:
                raise Exception("Not found")

        mock_s3.head_object = MagicMock(side_effect=_head_object)

        mock_create = MagicMock(return_value=mock_s3)
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.garak_adapter.create_s3_client",
            mock_create,
            raising=False,
        )

        _kfp_ov = {}
        _prefix = _kfp_ov.get("s3_prefix", os.getenv("KFP_S3_PREFIX", DEFAULT_S3_PREFIX))
        _bucket = _kfp_ov.get("s3_bucket", os.getenv("AWS_S3_BUCKET", ""))

        artifact_keys = {
            "sdg_raw_output": f"{s3_prefix}/sdg_raw_output.csv",
            "sdg_normalized_output": f"{s3_prefix}/sdg_normalized_output.csv",
            "intents_html_report": f"{s3_prefix}/scan.intents.html",
            "scan_report": f"{s3_prefix}/scan.report.jsonl",
        }
        verified = {}
        if _bucket:
            s3 = mock_create()
            for name, key in artifact_keys.items():
                try:
                    s3.head_object(Bucket=_bucket, Key=key)
                    verified[name] = f"s3://{_bucket}/{key}"
                except Exception:
                    pass

        assert len(verified) == 3
        assert "sdg_raw_output" in verified
        assert "sdg_normalized_output" in verified
        assert "scan_report" in verified
        assert "intents_html_report" not in verified
        assert verified["sdg_raw_output"] == f"s3://{bucket}/{s3_prefix}/sdg_raw_output.csv"

    def test_intents_kfp_job_bucket_from_kfp_config(self, monkeypatch):
        """s3_bucket from kfp_config takes precedence over env var."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import DEFAULT_S3_PREFIX
        import os

        monkeypatch.setenv("AWS_S3_BUCKET", "env-bucket")
        kfp_ov = {"s3_bucket": "config-bucket"}
        _bucket = kfp_ov.get("s3_bucket", os.getenv("AWS_S3_BUCKET", ""))

        assert _bucket == "config-bucket"

    def test_non_intents_job_no_artifacts(self):
        """art_intents=False -> no artifacts dict in evaluation_metadata."""
        from llama_stack_provider_trustyai_garak.constants import EXECUTION_MODE_KFP

        art_intents = False
        execution_mode = EXECUTION_MODE_KFP

        eval_meta = {
            "framework": "garak",
            "art_intents": art_intents,
            "execution_mode": execution_mode,
        }

        if art_intents and execution_mode == EXECUTION_MODE_KFP:
            eval_meta["artifacts"] = {}

        assert "artifacts" not in eval_meta


class TestWriteKfpOutputsComponent:
    """Targeted tests for the write_kfp_outputs KFP component."""

    def test_missing_s3_bucket_skips_gracefully(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.delenv("AWS_S3_BUCKET", raising=False)

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=False,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged == {}

    def test_empty_report_skips_gracefully(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            def _get_object(**kwargs):
                return {"Body": SimpleNamespace(read=lambda: b"")}

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=False,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged == {}

    def test_native_probes_logs_metrics_and_html(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        report_content = '{"entry_type":"attempt","status":2}\n'
        html_content = "<html><body>Native Report</body></html>"

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: report_content.encode())}
                if Key.endswith(".avid.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: b"")}
                if Key.endswith(".report.html"):
                    return {"Body": SimpleNamespace(read=lambda: html_content.encode())}
                raise Exception(f"unexpected key: {Key}")

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold: ([], {}, {}),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results",
            lambda *args, **kwargs: {
                "scores": {
                    "_overall": {
                        "aggregated_results": {
                            "total_attempts": 100,
                            "vulnerable_responses": 15,
                            "attack_success_rate": 15.0,
                        }
                    }
                }
            },
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=False,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged["total_attempts"] == 100
        assert metrics.logged["vulnerable_responses"] == 15
        assert metrics.logged["attack_success_rate"] == 15.0
        assert "Native Report" in Path(html.path).read_text()

    def test_intents_mode_uploads_html_to_s3(self, monkeypatch, tmp_path):
        """Intents mode: uploads intents HTML to S3 job folder."""
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        report_content = '{"entry_type":"attempt","status":2}\n'

        uploaded_keys: list[str] = []

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: report_content.encode())}
                return {"Body": SimpleNamespace(read=lambda: b"")}

            def _upload_file(filepath, bucket, key):
                uploaded_keys.append(key)

            return SimpleNamespace(get_object=_get_object, upload_file=_upload_file)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold: ([], {}, {}),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results",
            lambda *args, **kwargs: {"scores": {"_overall": {"aggregated_results": {"attack_success_rate": 25.0}}}},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.generate_art_report",
            lambda content, **kw: "<html>ART</html>",
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="evalhub-garak/job123",
            eval_threshold=0.5,
            art_intents=True,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged["attack_success_rate"] == 25.0
        assert "ART" in Path(html.path).read_text()
        assert "evalhub-garak/job123/scan.intents.html" in uploaded_keys

    def test_intents_mode_logs_asr_metric(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        report_content = '{"entry_type":"attempt","status":2}\n'

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: report_content.encode())}
                return {"Body": SimpleNamespace(read=lambda: b"")}

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold: ([], {}, {}),
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_aggregated_from_avid_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_digest_from_report_content",
            lambda content: {},
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.combine_parsed_results",
            lambda *args, **kwargs: {
                "scores": {
                    "_overall": {
                        "aggregated_results": {
                            "attack_success_rate": 25.0,
                        }
                    }
                }
            },
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.generate_art_report",
            lambda content, **kw: "<html>ART</html>",
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=True,
            summary_metrics=metrics,
            html_report=html,
        )

        assert metrics.logged["attack_success_rate"] == 25.0
        assert "total_attempts" not in metrics.logged
        assert "ART" in Path(html.path).read_text()

    def test_parse_failure_writes_fallback_html(self, monkeypatch, tmp_path):
        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import write_kfp_outputs

        monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")

        def _fake_create_s3_client():
            def _get_object(Bucket, Key):
                if Key.endswith(".report.jsonl"):
                    return {"Body": SimpleNamespace(read=lambda: b'{"data": true}\n')}
                return {"Body": SimpleNamespace(read=lambda: b"")}

            return SimpleNamespace(get_object=_get_object, upload_file=lambda *a: None)

        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.evalhub.s3_utils.create_s3_client",
            _fake_create_s3_client,
        )
        monkeypatch.setattr(
            "llama_stack_provider_trustyai_garak.result_utils.parse_generations_from_report_content",
            lambda content, threshold: (_ for _ in ()).throw(RuntimeError("parse boom")),
        )

        metrics = _FakeMetrics()
        html = _FakeArtifact(str(tmp_path / "report.html"))

        fn = _get_component_fn(write_kfp_outputs)
        fn(
            s3_prefix="test/prefix",
            eval_threshold=0.5,
            art_intents=False,
            summary_metrics=metrics,
            html_report=html,
        )

        html_content = Path(html.path).read_text()
        assert "Report generation failed" in html_content
