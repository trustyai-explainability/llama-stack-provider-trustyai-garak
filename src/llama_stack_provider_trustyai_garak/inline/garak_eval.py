from ..compat import (
    EvaluateResponse, 
    BenchmarkConfig, 
    ProviderSpec, 
    Api, 
    OpenAIFilePurpose, 
    OpenAIFileObject, Job, JobStatus, ScoringResult
)
from fastapi import UploadFile
from typing import List, Dict, Optional, Any, Union
import os
import logging
import json
from pathlib import Path
from ..config import GarakInlineConfig
from ..base_eval import GarakEvalBase
from datetime import datetime
import asyncio
import signal
import shutil
from llama_stack_provider_trustyai_garak import shield_scan
from ..result_utils import (
    parse_generations_from_report_content, 
    parse_aggregated_from_avid_content, 
    combine_parsed_results
)

logger = logging.getLogger(__name__)

class GarakInlineEvalAdapter(GarakEvalBase):
    """Inline Garak evaluation adapter for running scans locally."""

    def __init__(self, config: GarakInlineConfig, deps: dict[Api, ProviderSpec]):
        super().__init__(config, deps)
        self._config: GarakInlineConfig = config
        
        self._running_tasks: Dict[str, asyncio.Task] = {}  # job_id -> running task
        self._job_semaphore: Optional[asyncio.Semaphore] = None  # Concurrency control
        self._jobs_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the inline Garak provider."""

        self._initialize()

        try:
            self.scan_config.scan_dir.mkdir(exist_ok=True, parents=True)
            # Test write permissions
            test_file = self.scan_config.scan_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
            logger.info(f"Scan directory initialized: {self.scan_config.scan_dir}")
        except PermissionError as e:
            from ..errors import GarakConfigError
            logger.error(
                f"Permission denied creating scan directory: {self.scan_config.scan_dir}. "
                f"XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME', 'not set')}"
            )
            raise GarakConfigError(
                f"Permission denied: {self.scan_config.scan_dir}. "
                f"Set GARAK_SCAN_DIR or XDG_CACHE_HOME to a writable directory."
            ) from e
        except OSError as e:
            from ..errors import GarakConfigError
            raise GarakConfigError(
                f"Failed to create scan directory: {self.scan_config.scan_dir}. Error: {e}"
            ) from e

        self._job_semaphore = asyncio.Semaphore(self._config.max_concurrent_jobs)

        self._initialized = True
        logger.info("Initialized Garak inline provider.")
    
    async def run_eval(self, benchmark_id: str, benchmark_config: BenchmarkConfig) -> Dict[str, Union[str, Dict[str, str]]]:
        """Run an evaluation for a specific benchmark and configuration.

        Args:
            benchmark_id: The benchmark id
            benchmark_config: Configuration for the evaluation task
        """
        if not self._initialized:
            await self.initialize()
        
        await self._validate_run_eval_request(benchmark_id, benchmark_config)
        
        job_id = self._get_job_id()
        job = Job(
            job_id=job_id,
            status=JobStatus.scheduled
        )
        
        async with self._jobs_lock:
            self._jobs[job_id] = job
            self._job_metadata[job_id] = {"created_at": datetime.now().isoformat()}
            self._running_tasks[job_id] = asyncio.create_task(
                self._run_scan_with_semaphore(job, benchmark_id, benchmark_config),
                name=job_id
            )
        
        return {"job_id": job_id, "status": job.status, "metadata": self._job_metadata.get(job_id, {})}
    
    async def _run_scan_with_semaphore(self, job: Job, benchmark_id: str, benchmark_config: BenchmarkConfig):
        """Wrapper to run the scan with semaphore"""
        async with self._job_semaphore:
            # logger.info(f"Starting job {job.job_id} (Slots used: {self._config.max_concurrent_jobs - self._job_semaphore._value}/{self._config.max_concurrent_jobs})")
            async with self._jobs_lock:
                active_jobs = len([j for j in self._jobs.values() if j.status in [JobStatus.in_progress, JobStatus.scheduled]])
            logger.info(f"Starting job {job.job_id} (Slots used: {active_jobs}/{self._config.max_concurrent_jobs})")
            await self._run_scan(job, benchmark_id, benchmark_config)
        
    async def _run_scan(self, job: Job, benchmark_id: str, benchmark_config: BenchmarkConfig):
        """Run the scan with the given command.

        Args:
            job: The job object
            benchmark_id: The benchmark id
            benchmark_config: The benchmark configuration
        """
        stored_benchmark = await self.get_benchmark(benchmark_id)
        benchmark_metadata: dict = getattr(stored_benchmark, "metadata", {})

        async with self._jobs_lock:
            job.status = JobStatus.in_progress
            self._job_metadata[job.job_id]["started_at"] = datetime.now().isoformat()

        job_scan_dir: Path = self.scan_config.scan_dir / job.job_id
        job_scan_dir.mkdir(exist_ok=True, parents=True)

        scan_log_file: Path = job_scan_dir / "scan.log"
        scan_log_file.touch(exist_ok=True)
        scan_report_prefix: Path = job_scan_dir / "scan"
        
        try:
            scan_profile_config: dict = {
                "probes": benchmark_metadata["probes"],
                "timeout": benchmark_metadata.get("timeout", self._config.timeout)
            }

            cmd: List[str] = await self._build_command(benchmark_config, benchmark_id, scan_profile_config, scan_report_prefix=str(scan_report_prefix))
            logger.info(f"Running scan with command: {' '.join(cmd)}")

            env = os.environ.copy()
            env["GARAK_LOG_FILE"] = str(scan_log_file)
            env["GARAK_TLS_VERIFY"] = str(self._config.tls_verify)

            process = await asyncio.create_subprocess_exec(*cmd, 
                                                           stdout=asyncio.subprocess.PIPE, 
                                                           stderr=asyncio.subprocess.PIPE, 
                                                           env=env)
            
            async with self._jobs_lock:
                self._job_metadata[job.job_id]["process_id"] = str(process.pid)
            timeout: int = scan_profile_config.get("timeout", self._config.timeout)
            
            _, stderr = await asyncio.wait_for(process.communicate(), 
                                                    timeout=timeout)

            if process.returncode == 0:
                # convert report to avid report
                report_file = scan_report_prefix.with_suffix(".report.jsonl")
                try:
                    from ..avid_report import Report
                    
                    if not report_file.exists():
                        logger.error(f"Report file not found: {report_file}")
                    else:
                        report = Report(str(report_file)).load().get_evaluations()
                        report.export()  # this will create a new file - scan_report_prefix.with_suffix(".avid.jsonl")
                        logger.info(f"Successfully converted report to AVID format for job {job.job_id}")
                        
                except FileNotFoundError as e:
                    logger.error(f"Report file not found during AVID conversion for job {job.job_id}: {e}")
                except PermissionError as e:
                    logger.error(f"Permission denied reading report file for job {job.job_id}: {e}")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse report file for job {job.job_id}: {e}", exc_info=True)
                except ImportError as e:
                    logger.error(f"Failed to import AVID report module for job {job.job_id}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error converting report to AVID format for job {job.job_id}: {e}", exc_info=True)

                # Upload scan files to file storage
                upload_scan_report: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".report.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_report:
                    async with self._jobs_lock:
                        self._job_metadata[job.job_id]["scan.report.jsonl"] = upload_scan_report.id
                
                upload_avid_report: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".avid.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_avid_report:
                    async with self._jobs_lock:
                        self._job_metadata[job.job_id]["scan.avid.jsonl"] = upload_avid_report.id

                upload_scan_log: OpenAIFileObject = await self._upload_file(
                    file=scan_log_file, 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_log:
                    async with self._jobs_lock:
                        self._job_metadata[job.job_id]["scan.log"] = upload_scan_log.id

                upload_scan_hitlog: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".hitlog.jsonl"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_hitlog:
                    async with self._jobs_lock:
                        self._job_metadata[job.job_id]["scan.hitlog.jsonl"] = upload_scan_hitlog.id

                upload_scan_report_html: OpenAIFileObject = await self._upload_file(
                    file=scan_report_prefix.with_suffix(".report.html"), 
                    purpose=OpenAIFilePurpose.ASSISTANTS)
                if upload_scan_report_html:
                    async with self._jobs_lock:
                        self._job_metadata[job.job_id]["scan.report.html"] = upload_scan_report_html.id
                
                # parse results
                scan_report_file_id: str = self._job_metadata[job.job_id].get("scan.report.jsonl", "")
                avid_report_file_id: str = self._job_metadata[job.job_id].get("scan.avid.jsonl", "")
                
                if scan_report_file_id:
                    scan_result = await self._parse_combined_results(
                        scan_report_file_id, avid_report_file_id, benchmark_metadata
                    )

                    # save file and upload results to llama stack
                    scan_result_file = Path(job_scan_dir) / "scan_result.json"
                    with open(scan_result_file, 'w') as f:
                        json.dump(scan_result.model_dump(), f)
                    
                    upload_scan_result: OpenAIFileObject = await self._upload_file(
                        file=scan_result_file, 
                        purpose=OpenAIFilePurpose.ASSISTANTS)
                    if upload_scan_result:
                        async with self._jobs_lock:
                            self._job_metadata[job.job_id]["scan_result.json"] = upload_scan_result.id

                async with self._jobs_lock:
                    job.status = JobStatus.completed

            else:
                async with self._jobs_lock:
                    job.status = JobStatus.failed
                    self._job_metadata[job.job_id]["error"] = f"Scan failed with return code {process.returncode} - {stderr.decode('utf-8')}"
        except asyncio.TimeoutError:
            async with self._jobs_lock:
                job.status = JobStatus.failed
                self._job_metadata[job.job_id]["error"] = f"Scan timed out after {timeout} seconds."
        except Exception as e:
            async with self._jobs_lock:
                job.status = JobStatus.failed
                self._job_metadata[job.job_id]["error"] = str(e)
        finally:
            async with self._jobs_lock:
                self._job_metadata[job.job_id]["completed_at"] = datetime.now().isoformat()
                self._running_tasks.pop(job.job_id, None)
            if 'process' in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            # cleanup the tmp job dir
            if Path(job_scan_dir).exists():
                shutil.rmtree(job_scan_dir, ignore_errors=True)

    async def _upload_file(self, file: Path, purpose: OpenAIFilePurpose) -> Optional[OpenAIFileObject]:
        """Upload a file to the file storage and return the file object.

        Args:
            file: The file to upload
            purpose: The purpose of the file
        """
        if file.exists():
            with open(file, "rb") as f:
                upload_file: OpenAIFileObject = await self.file_api.openai_upload_file(
                    # file: The File object (not file name) to be uploaded.
                    file=UploadFile(file=f, filename=file.name), 
                    purpose=purpose
                )
                return upload_file
        else:
            logger.warning(f"File {file} does not exist")
            return None
        
    async def _parse_generations_from_report(
        self, report_file_id: str, eval_threshold: float
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """Parse enhanced generations and score rows from report.jsonl.
        
        Returns:
            Tuple of (generations, score_rows_by_probe)
        """
        report_content = await self.file_api.openai_retrieve_file_content(report_file_id)
        if not report_content:
            return [], {}
        
        report_str = report_content.body.decode("utf-8")
        
        # Use shared parsing utility
        return parse_generations_from_report_content(report_str, eval_threshold)

    async def _parse_aggregated_from_avid(self, avid_file_id: str) -> Dict[str, Dict[str, Any]]:
        """Parse probe-level aggregated info from AVID report.
        
        Wrapper that fetches content and delegates to shared utility.
        """
        if not avid_file_id:
            return {}
        
        avid_content = await self.file_api.openai_retrieve_file_content(avid_file_id)
        if not avid_content:
            return {}
        
        avid_str = avid_content.body.decode("utf-8")
        return parse_aggregated_from_avid_content(avid_str)

    async def _parse_combined_results(
        self, report_file_id: str, avid_file_id: str, benchmark_metadata: dict
    ) -> EvaluateResponse:
        """Parse results using hybrid approach: report.jsonl for generations, AVID for taxonomy.
        
        Wrapper that fetches content and delegates to shared utilities.
        """
        eval_threshold = float(benchmark_metadata.get("eval_threshold", self.scan_config.VULNERABLE_SCORE))
        
        # Fetch and parse both reports using shared utilities
        generations, score_rows_by_probe = await self._parse_generations_from_report(report_file_id, eval_threshold)
        aggregated_by_probe = await self._parse_aggregated_from_avid(avid_file_id)
        
        # Combine using shared utility
        result_dict = combine_parsed_results(
            generations,
            score_rows_by_probe,
            aggregated_by_probe,
            eval_threshold
        )
        
        scores = {
            probe_name: ScoringResult(
                score_rows=score_data["score_rows"],
                aggregated_results=score_data["aggregated_results"]
            )
            for probe_name, score_data in result_dict["scores"].items()
        }
        
        return EvaluateResponse(generations=result_dict["generations"], scores=scores)

    async def job_status(self, benchmark_id: str, job_id: str) -> Dict[str, Union[str, Dict[str, str]]]:
        """Get the status of a job.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return {"status": "not_found", "job_id": job_id}
            
            metadata: dict = self._job_metadata.get(job_id, {}).copy()

            if self._job_semaphore:
                # metadata["running_jobs"] = str(self._config.max_concurrent_jobs - self._job_semaphore._value)
                active_jobs = len([j for j in self._jobs.values() if j.status in [JobStatus.in_progress, JobStatus.scheduled]])
                metadata["running_jobs"] = str(active_jobs)
                metadata["max_concurrent_jobs"] = str(self._config.max_concurrent_jobs)

        return {"job_id": job_id, "status": job.status, "metadata": metadata}
    
    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        """Cancel a job and kill the process.

        Args:
            benchmark_id: The benchmark id
            job_id: The job id
        """
        async with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found")
                return
            
            if job.status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
                logger.warning(f"Job {job_id} is not running")
                return

            if job.status not in [JobStatus.in_progress, JobStatus.scheduled]:
                logger.warning(f"Job {job_id} has an unknown status: {job.status}")
                return
            
            process_id: str = self._job_metadata[job_id].get("process_id", None)
        
        # Kill process outside the lock to avoid blocking
        if process_id:
            process_id: int = int(process_id)
            logger.info(f"Killing process {process_id} for job {job_id}")
            try:
                os.kill(process_id, signal.SIGTERM)
                # Give process time to gracefully shutdown
                await asyncio.sleep(2)
                try:
                    os.kill(process_id, 0)
                    # process is still running, kill it with SIGKILL
                    os.kill(process_id, signal.SIGKILL)
                except ProcessLookupError:
                    # process is not running (killed gracefully), no need to kill it
                    pass
            except ProcessLookupError:
                logger.warning(f"Process {process_id} not found for job {job_id}")
            except Exception as e:
                logger.error(f"Error killing process {process_id} for job {job_id}: {e}")
        
        async with self._jobs_lock:
            job.status = JobStatus.cancelled
            self._jobs[job_id] = job
            self._job_metadata[job_id]["cancelled_at"] = datetime.now().isoformat()
            self._job_metadata[job_id]["error"] = "Job cancelled"
    
    async def shutdown(self) -> None:
        """Clean up resources when shutting down."""
        logger.info("Shutting down Garak provider")
        
        # Get snapshot of running tasks to cancel
        async with self._jobs_lock:
            tasks_to_cancel = list(self._running_tasks.items())
        
        # Cancel all running asyncio tasks
        for job_id, task in tasks_to_cancel:
            if not task.done():
                logger.info(f"Cancelling running task {task.get_name()} for job {job_id}")
                task.cancel()
        
        # Wait for tasks to be cancelled (with timeout)
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[task for _, task in tasks_to_cancel], return_exceptions=True),
                    timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks didn't cancel within timeout")
        
        # Kill all running jobs
        async with self._jobs_lock:
            jobs_to_cancel = [(job_id, job) for job_id, job in self._jobs.items() 
                             if job.status in [JobStatus.in_progress, JobStatus.scheduled]]
        
        for job_id, job in jobs_to_cancel:
            await self.job_cancel("placeholder", job_id)
        
        # Clear all running tasks, jobs and job metadata
        async with self._jobs_lock:
            self._running_tasks.clear()
            self._jobs.clear()
            self._job_metadata.clear()

        # Close the shield scanning HTTP client
        shield_scan.simple_shield_orchestrator.close()
        
        # Cleanup the scan directory
        shutil.rmtree(self.scan_config.scan_dir, ignore_errors=True)