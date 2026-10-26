"""Run real subprocesses to check liveness, deadlines, and cleanup without Garak."""

import signal
import subprocess
import sys
import time

import pytest

from llama_stack_provider_trustyai_garak.core import garak_runner


def _run_script(monkeypatch, tmp_path, script, **kwargs):
    config = tmp_path / "config.json"
    config.write_text("{}")
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        garak_runner.subprocess,
        "Popen",
        lambda _cmd, **options: real_popen([sys.executable, "-c", script], **options),
    )
    return garak_runner.run_garak_scan(
        config, report_prefix=tmp_path / "scan", progress_interval_seconds=0.03, **kwargs
    )


@pytest.mark.parametrize("timeout_seconds", [0, 5])
def test_silent_scan_reports_liveness_until_completion(monkeypatch, tmp_path, timeout_seconds):
    updates = []
    result = _run_script(
        monkeypatch,
        tmp_path,
        "import time; time.sleep(0.25)",
        timeout_seconds=timeout_seconds,
        on_progress=lambda: updates.append(time.monotonic()),
    )
    assert result.success
    assert updates


def test_liveness_does_not_extend_deadline_and_timeout_kills_group(monkeypatch, tmp_path):
    updates = []
    signals = []
    real_killpg = garak_runner.os.killpg

    def killpg(pgid, sig):
        signals.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(garak_runner.os, "killpg", killpg)
    started = time.monotonic()
    result = _run_script(
        monkeypatch,
        tmp_path,
        "import time; time.sleep(60)",
        timeout_seconds=0.25,
        on_progress=lambda: updates.append(time.monotonic()),
    )
    assert result.timed_out
    assert not result.success
    assert result.returncode < 0
    assert signal.SIGTERM in signals
    assert updates
    assert time.monotonic() - started < 5


def test_status_failure_does_not_interrupt_scan(monkeypatch, tmp_path, caplog):
    def report():
        raise RuntimeError("sidecar temporarily unavailable")

    result = _run_script(
        monkeypatch,
        tmp_path,
        "import time; time.sleep(0.15)",
        timeout_seconds=5,
        on_progress=report,
    )
    assert result.success
    assert "Could not report Garak scan progress" in caplog.text


def test_timeout_kills_worker_after_leader_exits(monkeypatch, tmp_path):
    ready = tmp_path / "worker-ready"
    survived = tmp_path / "worker-survived"
    worker = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).touch()\n"
        "time.sleep(1.5)\n"
        f"Path({str(survived)!r}).touch()\n"
    )
    leader = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {worker!r}])\ntime.sleep(60)"
    result = _run_script(monkeypatch, tmp_path, leader, timeout_seconds=1)
    assert result.timed_out
    assert ready.exists()
    time.sleep(1)
    assert not survived.exists(), "A worker outlived its timed-out scan"
