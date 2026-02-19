"""This module provides a sync subprocess runner that can be used by:
- eval-hub adapter (K8s Job)
- KFP components
"""

import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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
    cmd: list[str],
    timeout_seconds: int,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    report_prefix: Path | None = None,
) -> GarakScanResult:
    """Run a Garak scan with proper process group handling.
    
    This function:
    1. Creates a new process group for the garak process
    2. Handles timeout with graceful shutdown (SIGTERM → SIGKILL)
    3. Kills the entire process tree on timeout
    
    Args:
        cmd: The garak command to execute
        timeout_seconds: Maximum execution time
        env: Environment variables (merged with os.environ)
        log_file: Path to write garak logs
        report_prefix: Expected report prefix path (for result)
    
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
    
    # Set log file if provided
    if log_file:
        run_env["GARAK_LOG_FILE"] = str(log_file)
    
    # Determine report prefix from command
    if report_prefix is None:
        # Try to extract from command
        try:
            prefix_idx = cmd.index("--report_prefix")
            report_prefix = Path(cmd[prefix_idx + 1])
        except (ValueError, IndexError):
            report_prefix = Path("/tmp/garak_scan")
    
    logger.debug(f"Starting Garak scan with command: {' '.join(cmd)}")
    
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = -1
    
    # Start process in new process group
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
        preexec_fn=os.setsid,
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        returncode = process.returncode
        
        if returncode == 0:
            logger.info("Garak scan completed successfully")
        else:
            logger.error(f"Garak scan failed with return code {returncode}, stderr: {stderr}")
            
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
        
        # Get any output that was produced
        try:
            stdout, stderr = process.communicate(timeout=1)
        except Exception:
            stdout, stderr = "", "Scan timed out"
        
        returncode = process.returncode if process.returncode is not None else -1
    
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
        from ..avid_report import Report
        
        report = Report(str(report_path)).load().get_evaluations()
        report.export()  # Creates .avid.jsonl file
        logger.info(f"Converted report to AVID format: {report_path}")
        return True
        
    except ImportError:
        logger.error("AVID report module not found, cannot convert to AVID format")
        return False
    except Exception as e:
        logger.error(f"Failed to convert report to AVID format: {e}")
        return False
