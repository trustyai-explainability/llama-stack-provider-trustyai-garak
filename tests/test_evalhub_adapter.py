"""Tests for eval-hub Garak adapter alignment behavior."""

from __future__ import annotations

import importlib
import json
import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

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
        lambda self, result, eval_threshold: ([], None, 0, {"total_attempts": 0}),
    )

    class _Callbacks:
        def report_status(self, _update):
            return None

        def create_oci_artifact(self, _spec):
            return SimpleNamespace(reference="oci://ref", digest="sha256:test")

    job = SimpleNamespace(
        id="job-1",
        benchmark_id="trustyai_garak::quick",
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
        lambda _content, _threshold: ([], {"probe.alpha": [{}]}),
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
