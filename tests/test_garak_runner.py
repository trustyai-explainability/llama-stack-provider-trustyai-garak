"""Run real subprocesses to check completion, deadlines, and cleanup without Garak."""

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
    return garak_runner.run_garak_scan(config, report_prefix=tmp_path / "scan", **kwargs)


@pytest.mark.parametrize("timeout_seconds", [0, 5])
def test_silent_scan_completes(monkeypatch, tmp_path, timeout_seconds):
    result = _run_script(
        monkeypatch,
        tmp_path,
        "import time; time.sleep(0.25)",
        timeout_seconds=timeout_seconds,
    )
    assert result.success


def test_timeout_kills_process_group(monkeypatch, tmp_path):
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
    )
    assert result.timed_out
    assert not result.success
    assert result.returncode < 0
    assert signal.SIGTERM in signals
    assert time.monotonic() - started < 5


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
