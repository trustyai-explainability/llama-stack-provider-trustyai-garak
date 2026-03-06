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

    monkeypatch.setitem(sys.modules, "evalhub", evalhub_module)
    monkeypatch.setitem(sys.modules, "evalhub.adapter", adapter_module)
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
    assert (
        unprefixed["garak_config"]["run"]["probe_tags"]
        == prefixed["garak_config"]["run"]["probe_tags"]
    )


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
        benchmark_config={"timeout_seconds": 42, "generations": 2},
        exports=None,
    )

    adapter.run_benchmark_job(job, _Callbacks())

    config_path = Path(captured["config_file"])
    assert config_path.exists()

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_payload["reporting"]["report_prefix"].endswith("scan")
    assert config_payload["run"]["generations"] == 2
    assert captured["timeout_seconds"] == 42


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

    assert len(metrics) == 1
    assert metrics[0].metric_name == "probe.alpha_asr"
    assert overall_score == 30.0
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
        monkeypatch.setenv("EVALHUB_KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("EVALHUB_KFP_NAMESPACE", "test-ns")
        monkeypatch.setenv("EVALHUB_KFP_S3_SECRET_NAME", "my-data-connection")
        monkeypatch.setenv("AWS_S3_BUCKET", "my-bucket")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config()
        assert cfg.endpoint == "https://kfp.example.com"
        assert cfg.namespace == "test-ns"
        assert cfg.s3_secret_name == "my-data-connection"
        assert cfg.s3_bucket == "my-bucket"
        assert cfg.experiment_name == "evalhub-garak"

    def test_benchmark_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_KFP_ENDPOINT", "https://env.example.com")
        monkeypatch.setenv("EVALHUB_KFP_NAMESPACE", "env-ns")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig

        cfg = KFPConfig.from_env_and_config({
            "kfp_config": {
                "endpoint": "https://override.example.com",
                "namespace": "override-ns",
                "s3_secret_name": "custom-s3-conn",
                "s3_bucket": "override-bucket",
            }
        })
        assert cfg.endpoint == "https://override.example.com"
        assert cfg.namespace == "override-ns"
        assert cfg.s3_secret_name == "custom-s3-conn"
        assert cfg.s3_bucket == "override-bucket"

    def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("EVALHUB_KFP_ENDPOINT", raising=False)
        monkeypatch.setenv("EVALHUB_KFP_NAMESPACE", "ns")

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig
        import pytest

        with pytest.raises(ValueError, match="KFP endpoint is required"):
            KFPConfig.from_env_and_config()

    def test_missing_namespace_raises(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.delenv("EVALHUB_KFP_NAMESPACE", raising=False)

        from llama_stack_provider_trustyai_garak.evalhub.kfp_pipeline import KFPConfig
        import pytest

        with pytest.raises(ValueError, match="KFP namespace is required"):
            KFPConfig.from_env_and_config()

    def test_verify_ssl_false(self, monkeypatch):
        monkeypatch.setenv("EVALHUB_KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("EVALHUB_KFP_NAMESPACE", "ns")
        monkeypatch.setenv("EVALHUB_KFP_VERIFY_SSL", "false")

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

        def _fake_run_via_kfp(self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5):
            captured["called"] = True
            captured["timeout"] = timeout
            captured["config_json"] = garak_config_dict
            captured["intents_params"] = intents_params
            captured["eval_threshold"] = eval_threshold
            return module.GarakScanResult(
                returncode=0, stdout="", stderr="", report_prefix=report_prefix,
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
            benchmark_config={"execution_mode": "kfp", "timeout_seconds": 99},
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
                returncode=0, stdout="", stderr="", report_prefix=report_prefix,
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
            benchmark_config={},
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())
        assert kfp_called["value"] is False
        assert result.evaluation_metadata["execution_mode"] == "simple"


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
                return [{
                    "Contents": [
                        {"Key": f"{Prefix}/scan.report.jsonl"},
                        {"Key": f"{Prefix}/scan.avid.jsonl"},
                        {"Key": f"{Prefix}/scan.log"},
                        {"Key": f"{Prefix}/config.json"},
                    ]
                }]

            def download_file(self, bucket, key, local_path):
                downloaded_files.append(key)
                Path(local_path).write_text(f"content of {key}")

        monkeypatch.setattr(
            module.GarakAdapter, "_create_s3_client",
            staticmethod(lambda: _FakeS3Client()),
        )

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3(
            "test-bucket", "evalhub-garak/job-123", local_dir,
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
            module.GarakAdapter, "_create_s3_client",
            staticmethod(lambda: _FakeS3Client()),
        )

        local_dir = tmp_path / "results"
        local_dir.mkdir()

        module.GarakAdapter._download_results_from_s3(
            "test-bucket", "evalhub-garak/job-empty", local_dir,
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
            _FakeClient(), "run-123", callbacks, poll_interval=0,
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
            _FakeClient(), "run-456", _Callbacks(), poll_interval=0,
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
            _FakeClient(), "run-789", callbacks, poll_interval=0,
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
            _FakeClient(), "run-timeout", _Callbacks(), poll_interval=0, timeout=100,
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
        monkeypatch.setenv("EVALHUB_KFP_ENDPOINT", "https://kfp.example.com")
        monkeypatch.setenv("EVALHUB_KFP_NAMESPACE", "test-ns")
        monkeypatch.delenv("EVALHUB_KFP_S3_SECRET_NAME", raising=False)

        class _Callbacks:
            def report_status(self, _update):
                return None

        job = SimpleNamespace(
            id="kfp-no-secret",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://localhost:8000", name="test-model"),
            benchmark_config={"execution_mode": "kfp"},
            exports=None,
        )

        with pytest.raises(ValueError, match="S3 data-connection secret name is required"):
            adapter.run_benchmark_job(job, _Callbacks())


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
            benchmark_config={**_INTENTS_MODELS_SINGLE},
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
            benchmark_config={**_INTENTS_MODELS_SINGLE},
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
            benchmark_config={
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
            benchmark_config={**_INTENTS_MODELS_ALL_ROLES},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        config_dict, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        judge = config_dict["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "judge-model"
        assert judge["detector_model_config"]["uri"] == "http://judge:8000/v1"
        assert judge["detector_model_config"]["api_key"] == "jk"

        tap_cfg = config_dict["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap_cfg["attack_model_name"] == "atk-model"
        assert tap_cfg["attack_model_config"]["uri"] == "http://attacker:9000/v1"
        assert tap_cfg["attack_model_config"]["api_key"] == "ak"
        assert tap_cfg["evaluator_model_name"] == "eval-model"
        assert tap_cfg["evaluator_model_config"]["uri"] == "http://evaluator:9001/v1"
        assert tap_cfg["evaluator_model_config"]["api_key"] == "ek"

        assert intents_params["sdg_model"] == "sdg-model"
        assert intents_params["sdg_api_base"] == "http://sdg:7000/v1"
        assert intents_params["sdg_api_key"] == "sk"

    def test_no_roles_raises_clear_error(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="no-roles-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            benchmark_config={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="at least one of"):
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
            benchmark_config={
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
            benchmark_config={
                "intents_models": {"judge": {"url": "http://judge:8000/v1"}},
            },
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        with pytest.raises(ValueError, match="intents_models.judge.name"):
            adapter._build_config_from_spec(job, report_prefix)

    def test_non_intents_profile_returns_art_intents_false(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="native-job",
            benchmark_id="trustyai_garak::quick",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            benchmark_config={},
            exports=None,
        )

        report_prefix = tmp_path / "scan"
        _, _, intents_params = adapter._build_config_from_spec(job, report_prefix)

        assert intents_params["art_intents"] is False

    def test_sdg_params_from_intents_models(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        job = SimpleNamespace(
            id="sdg-intents-job",
            benchmark_id="trustyai_garak::intents",
            benchmark_index=0,
            model=SimpleNamespace(url="http://model:8000", name="my-llm"),
            benchmark_config={
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
        assert intents_params["sdg_api_key"] == "sdg-key"

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
            benchmark_config={
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
            "judge", {"api_key": "direct-key"},
        )
        assert result == "direct-key"

    def test_api_key_env_from_config(self, monkeypatch):
        module = _load_evalhub_garak_adapter(monkeypatch)
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-val")

        result = module.GarakAdapter._resolve_intents_api_key(
            "judge", {"api_key_env": "MY_CUSTOM_KEY"},
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


class TestKFPIntentsMode:
    """Tests for intents mode through run_benchmark_job with KFP."""

    def test_kfp_intents_passes_params_to_pipeline(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        captured: dict[str, object] = {}
        report_prefix = tmp_path / "scan"
        report_prefix.with_suffix(".report.jsonl").write_text("{}", encoding="utf-8")

        def _fake_run_via_kfp(self, config, callbacks, garak_config_dict, timeout, intents_params=None, eval_threshold=0.5):
            captured["intents_params"] = intents_params
            captured["eval_threshold"] = eval_threshold
            return module.GarakScanResult(
                returncode=0, stdout="", stderr="", report_prefix=report_prefix,
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
            benchmark_config={
                "execution_mode": "kfp",
                **_INTENTS_MODELS_ALL_ROLES,
            },
            exports=None,
        )

        result = adapter.run_benchmark_job(job, _Callbacks())

        assert captured["intents_params"]["art_intents"] is True
        assert captured["intents_params"]["sdg_model"] == "sdg-model"
        assert captured["intents_params"]["sdg_api_key"] == "sk"
        assert result.evaluation_metadata["art_intents"] is True

    def test_art_html_generated_for_intents(self, monkeypatch, tmp_path):
        module = _load_evalhub_garak_adapter(monkeypatch)
        adapter = module.GarakAdapter()
        monkeypatch.setenv("GARAK_SCAN_DIR", str(tmp_path))

        report_prefix = tmp_path / "intents-html-job" / "scan"
        report_prefix.parent.mkdir(parents=True)
        report_prefix.with_suffix(".report.jsonl").write_text(
            '{"entry_type":"attempt","status":2}\n',
            encoding="utf-8",
        )

        def _fake_run_garak_scan(config_file, timeout_seconds, report_prefix, env=None, log_file=None):
            report_prefix.with_suffix(".report.jsonl").write_text(
                '{"entry_type":"attempt","status":2}\n',
                encoding="utf-8",
            )
            return module.GarakScanResult(
                returncode=0, stdout="", stderr="", report_prefix=report_prefix,
            )

        monkeypatch.setattr(module, "run_garak_scan", _fake_run_garak_scan)
        monkeypatch.setattr(module, "convert_to_avid_report", lambda _path: True)
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
            benchmark_config={**_INTENTS_MODELS_SINGLE},
            exports=None,
        )

        adapter.run_benchmark_job(job, _Callbacks())

        assert html_generated["called"] is True
        html_path = tmp_path / "intents-html-job" / "scan.intents.html"
        assert html_path.exists()
        assert "ART Report" in html_path.read_text()


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
            module, "parse_aggregated_from_avid_content", lambda _content: {},
        )
        monkeypatch.setattr(
            module, "parse_digest_from_report_content", lambda _content: {},
        )
        monkeypatch.setattr(
            module,
            "combine_parsed_results",
            lambda *_args, **kwargs: {
                "scores": {
                    "spo.SPOIntent": {
                        "aggregated_results": {
                            "total_attacks": 20,
                            "successful_attacks": 5,
                            "total_prompts": 10,
                            "safe_prompts": 7,
                            "attack_success_rate": 30.0,
                            "metadata": {},
                        }
                    },
                    "_overall": {
                        "aggregated_results": {
                            "total_attacks": 20,
                            "attack_success_rate": 30.0,
                        }
                    },
                }
            },
        )

        result = module.GarakScanResult(
            returncode=0, stdout="", stderr="", report_prefix=report_prefix,
        )
        metrics, overall_score, num_examples, _ = adapter._parse_results(
            result, 0.5, art_intents=True,
        )

        assert len(metrics) == 1
        assert metrics[0].metric_name == "spo.SPOIntent_asr"
        assert metrics[0].metric_value == 30.0
        assert metrics[0].metadata["total_prompts"] == 10
        assert metrics[0].metadata["safe_prompts"] == 7
        assert overall_score == 30.0
