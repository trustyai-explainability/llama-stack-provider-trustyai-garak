"""This module provides a sync subprocess runner that can be used by:
- eval-hub adapter (K8s Job)
- KFP components
"""

import logging
import os
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)
_STDERR_WARNING_PATTERN = re.compile(
    r"(?:\berror\b|\bexception\b|\btraceback\b|\bfailed\b|\bfatal\b|\bwarn(?:ing)?\b|⚠)",
    re.IGNORECASE,
)


@dataclass
class GarakScanResult:
    """Result of a Garak scan execution."""
    
    returncode: int
    stdout: str
    stderr: str
    report_prefix: Path
    timed_out: bool = False
    
    @property
    def success(self) -> bool:
        """Check if the scan completed successfully."""
        return self.returncode == 0 and not self.timed_out
    
    @property
    def report_jsonl(self) -> Path:
        """Path to the main report file."""
        return self.report_prefix.with_suffix(".report.jsonl")
    
    @property
    def avid_jsonl(self) -> Path:
        """Path to the AVID format report."""
        return self.report_prefix.with_suffix(".avid.jsonl")
    
    @property
    def hitlog_jsonl(self) -> Path:
        """Path to the hitlog (vulnerable attempts only)."""
        return self.report_prefix.with_suffix(".hitlog.jsonl")
    
    @property
    def report_html(self) -> Path:
        """Path to the HTML report."""
        return self.report_prefix.with_suffix(".report.html")
    
    def get_all_output_files(self) -> list[Path]:
        """Get all output files that exist."""
        candidates = [
            self.report_jsonl,
            self.avid_jsonl,
            self.hitlog_jsonl,
            self.report_html,
        ]
        return [f for f in candidates if f.exists()]


def run_garak_scan(
    config_file: Path,
    timeout_seconds: int,
    report_prefix: Path,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> GarakScanResult:
    """Run a Garak scan with proper process group handling.
    
    This function:
    1. Creates a new process group for the garak process
    2. Handles timeout with graceful shutdown (SIGTERM → SIGKILL)
    3. Kills the entire process tree on timeout
    4. Streams process output to logs in real time
    
    Args:
        config_file: The path to the Garak command config file
        timeout_seconds: Maximum execution time
        report_prefix: Expected report prefix path (for results)
        env: Environment variables (merged with os.environ)
        log_file: Path to write garak logs
    
    Returns:
        GarakScanResult with execution details
    
    Raises:
        FileNotFoundError: If garak is not installed
        PermissionError: If garak cannot be executed
    """
    # Merge environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    # Improves real-time output behavior for Python subprocesses.
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    
    # Set log file if provided
    if log_file:
        run_env["GARAK_LOG_FILE"] = str(log_file)
    
    
    timed_out = False
    output_tail_lines = int(run_env.get("GARAK_OUTPUT_TAIL_LINES", "500"))
    stdout_tail: deque[str] = deque(maxlen=output_tail_lines)
    stderr_tail: deque[str] = deque(maxlen=output_tail_lines)
    stdout = ""
    stderr = ""
    returncode = -1
    
    if not config_file.exists():
        raise FileNotFoundError(f"Garak command config file not found: {config_file}")
    
    cmd = ["garak", "--config", str(config_file)]

    # Start process in new process group
    # argument list with shell=False prevents shell injection
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=run_env,
        preexec_fn=os.setsid,
        shell=False,
    )

    def _stream_pipe(
        pipe: subprocess.PIPE,
        tail: deque[str],
        level: int,
        stream_name: str,
    ) -> None:
        """Read pipe output line-by-line and stream to logger."""
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                tail.append(line)
                text = line.rstrip()
                if text:
                    # Garak/tqdm writes progress bars to stderr; avoid classifying
                    # these as warnings unless the line looks like an actual issue.
                    is_stderr_issue = bool(
                        stream_name == "stderr" and _STDERR_WARNING_PATTERN.search(text)
                    )
                    line_level = logging.WARNING if is_stderr_issue else level
                    stream_label = (
                        "stderr"
                        if is_stderr_issue
                        else ("progress" if stream_name == "stderr" else stream_name)
                    )
                    logger.log(line_level, "garak[%s] %s", stream_label, text)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, stdout_tail, logging.INFO, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, stderr_tail, logging.INFO, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    
    try:
        process.wait(timeout=timeout_seconds)
        returncode = process.returncode
        
        if returncode == 0:
            logger.info("Garak scan completed successfully")
        else:
            logger.error("Garak scan failed with return code %s", returncode)
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Garak scan timed out after {timeout_seconds} seconds")
        timed_out = True
        
        # Kill the entire process group
        try:
            pgid = os.getpgid(process.pid)
            logger.info(f"Sending SIGTERM to process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)
            
            try:
                process.wait(timeout=5)
                logger.info("Process terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate, sending SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
                process.wait()
                
        except ProcessLookupError:
            logger.debug("Process already terminated")
        except Exception as e:
            logger.error(f"Error killing process group: {e}")
            # Try to kill just the main process
            try:
                process.kill()
                process.wait()
            except Exception:
                pass
        
        returncode = process.returncode if process.returncode is not None else -1
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    stdout = "".join(stdout_tail)
    stderr = "".join(stderr_tail)
    if timed_out and not stderr:
        stderr = "Scan timed out"
    
    return GarakScanResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        report_prefix=report_prefix,
        timed_out=timed_out,
    )


def convert_to_avid_report(report_path: Path) -> bool:
    """Convert a Garak report to AVID format.
    
    Args:
        report_path: Path to the .report.jsonl file
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return False
    
    try:
        from garak.report import Report

        report = Report(str(report_path)).load().get_evaluations()
        report.export()  # Creates .avid.jsonl file at same location as report_path
        logger.info(f"Converted report to AVID format: {report_path}")
        return True
        
    except ImportError:
        logger.error("AVID report module not found, cannot convert to AVID format")
        return False
    except Exception as e:
        logger.error(f"Failed to convert report to AVID format: {e}")
        return False
