"""Garak Framework Adapter for eval-hub.

This adapter integrates NVIDIA's Garak red-teaming framework with the
eval-hub evaluation platform. It runs Garak scans as K8s Jobs and reports
results via the eval-hub sidecar.

The adapter:
1. Reads JobSpec from mounted ConfigMap (/etc/eval-job/spec.json)
2. Builds and executes Garak CLI commands
3. Parses results from Garak's JSONL reports
4. Persists artifacts to OCI registry
5. Reports results back to eval-hub via sidecar callbacks
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from evalhub.adapter import (
    DefaultCallbacks,
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
    OCIArtifactSpec,
)
from evalhub.adapter.models.job import ErrorInfo, MessageInfo
from evalhub.models.api import EvaluationResult

from ..core.command_builder import build_garak_command, build_generator_options
from ..core.garak_runner import convert_to_avid_report, GarakScanResult, run_garak_scan
from ..result_utils import (
    combine_parsed_results,
    parse_aggregated_from_avid_content,
    parse_generations_from_report_content,
)
from ..utils import get_scan_base_dir

logger = logging.getLogger(__name__)

# Default model type for OpenAI-compatible endpoints
DEFAULT_MODEL_TYPE = "openai.OpenAICompatible"

# Default evaluation threshold
DEFAULT_EVAL_THRESHOLD = 0.5


class GarakAdapter(FrameworkAdapter):
    """Garak red-teaming framework adapter for eval-hub.
    
    This adapter runs Garak security scans against LLM models and reports
    vulnerability metrics back to eval-hub.
    
    Benchmark Configuration:
        The adapter expects benchmark_config in JobSpec to contain:
        - probes: List of Garak probe names
        - probe_tags: Optional probe tag filters
        - eval_threshold: Threshold for vulnerability detection (default: 0.5)
        - Other Garak CLI options (detectors, buffs, etc.)
    
    Results:
        Returns one EvaluationResult per probe with:
        - attack_success_rate: Percentage of successful attacks
        - vulnerable_responses: Count of vulnerable responses
        - total_attempts: Total number of probe attempts
    """

    def run_benchmark_job(
        self, config: JobSpec, callbacks: JobCallbacks
    ) -> JobResults:
        """Run a Garak security scan job.
        
        Args:
            config: Job specification with model and benchmark configuration
            callbacks: Callbacks for status updates and artifact persistence
        
        Returns:
            JobResults with vulnerability metrics per probe
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If scan execution fails
        """
        start_time = time.time()
        logger.info(f"Starting Garak job {config.id} for benchmark {config.benchmark_id}")

        try:
            # Phase 1: Initialize
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.INITIALIZING,
                progress=0.0,
                message=MessageInfo(
                    message="Validating configuration and building scan command",
                    message_code="initializing",
                ),
            ))

            self._validate_config(config)

            # Setup scan directories
            scan_dir = get_scan_base_dir() / config.id
            scan_dir.mkdir(parents=True, exist_ok=True)

            log_file = scan_dir / "scan.log"
            report_prefix = scan_dir / "scan"

            # Build command from config
            cmd = self._build_command_from_spec(config, report_prefix)
            logger.info(f"Built Garak command: {' '.join(cmd)}")

            # Phase 2: Run scan
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.1,
                message=MessageInfo(
                    message=f"Running Garak scan for {config.benchmark_id}",
                    message_code="running_scan",
                ),
                current_step="Executing probes",
            ))

            timeout = config.benchmark_config.get("timeout_seconds", 600)

            result = run_garak_scan(
                cmd=cmd,
                timeout_seconds=timeout,
                log_file=log_file,
                report_prefix=report_prefix,
            )

            if not result.success:
                error_msg = f"Garak scan failed: {result.stderr}" if result.stderr else "Unknown error"
                if result.timed_out:
                    error_msg = f"Scan timed out after {timeout} seconds"

                callbacks.report_status(JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error=ErrorInfo(message=error_msg, message_code="scan_failed"),
                ))
                raise RuntimeError(error_msg)

            # Phase 3: Convert to AVID format
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.POST_PROCESSING,
                progress=0.7,
                message=MessageInfo(
                    message="Converting results to AVID format",
                    message_code="post_processing",
                ),
            ))

            convert_to_avid_report(result.report_jsonl)

            # Phase 4: Parse results
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.POST_PROCESSING,
                progress=0.8,
                message=MessageInfo(
                    message="Parsing scan results",
                    message_code="parsing_results",
                ),
            ))

            eval_threshold = config.benchmark_config.get("eval_threshold", DEFAULT_EVAL_THRESHOLD)
            metrics, overall_score, num_examples = self._parse_results(
                result, eval_threshold
            )

            logger.info(f"Parsed {len(metrics)} probe metrics, overall score: {overall_score}")

            # Phase 5: Persist artifacts
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.PERSISTING_ARTIFACTS,
                progress=0.9,
                message=MessageInfo(
                    message="Uploading artifacts to OCI registry",
                    message_code="persisting_artifacts",
                ),
            ))

            # Collect all output files
            output_files = self._collect_output_files(log_file, result)

            oci_artifact = None
            if output_files:
                oci_artifact = callbacks.create_oci_artifact(OCIArtifactSpec(
                    files=output_files,
                    base_path=scan_dir,
                    title=f"Garak scan results for {config.benchmark_id}",
                    description=f"Red-teaming results from job {config.id}",
                    annotations={
                        "job_id": config.id,
                        "benchmark_id": config.benchmark_id,
                        "model_name": config.model.name,
                        "overall_score": str(overall_score) if overall_score else "N/A",
                        "framework": "garak",
                    },
                    id=config.id,
                    benchmark_id=config.benchmark_id,
                    model_name=config.model.name,
                ))
                logger.info(f"Persisted {len(output_files)} artifacts")

            # Compute duration
            duration = time.time() - start_time

            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                model_name=config.model.name,
                results=metrics,
                overall_score=overall_score,
                num_examples_evaluated=num_examples,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "framework": "garak",
                    "framework_version": self._get_garak_version(),
                    "eval_threshold": eval_threshold,
                    "timed_out": result.timed_out,
                },
                oci_artifact=oci_artifact,
            )

        except Exception as e:
            logger.exception(f"Garak job {config.id} failed")
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.FAILED,
                error=ErrorInfo(message=str(e), message_code="job_failed"),
                error_details={"exception_type": type(e).__name__},
            ))
            raise

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration."""
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")
        
        if not config.model.url:
            raise ValueError("model.url is required")
        
        if not config.model.name:
            raise ValueError("model.name is required")
        
        probes = config.benchmark_config.get("probes", [])
        probe_tags = config.benchmark_config.get("probe_tags", [])
        
        if not probes and not probe_tags:
            logger.warning("No probes or probe_tags provided, using all probes")
            
        logger.debug("Configuration validated successfully")

    def _build_command_from_spec(
        self, config: JobSpec, report_prefix: Path
    ) -> list[str]:
        """Build Garak CLI command from JobSpec."""
        # Build generator options for the model
        model_params = config.benchmark_config.get("model_parameters") or {}
        generator_options = build_generator_options(
            model_endpoint=self._normalize_url(config.model.url),
            model_name=config.model.name,
            api_key=getattr(config.model, "api_key", None) or os.getenv("OPENAICOMPATIBLE_API_KEY", "DUMMY"),
            extra_params=model_params,
        )
        
        # Extract benchmark config options
        bc = config.benchmark_config
        
        return build_garak_command(
            model_type=bc.get("model_type", DEFAULT_MODEL_TYPE),
            model_name=config.model.name,
            generator_options=generator_options,
            probes=bc.get("probes", ["all"]),
            report_prefix=str(report_prefix),
            parallel_attempts=bc.get("parallel_attempts", 8),
            generations=bc.get("generations", 1),
            parallel_requests=bc.get("parallel_requests"),
            skip_unknown=bc.get("skip_unknown"),
            seed=bc.get("seed"),
            deprefix=bc.get("deprefix"),
            eval_threshold=bc.get("eval_threshold"),
            probe_tags=bc.get("probe_tags"),
            probe_options=bc.get("probe_options"),
            detectors=bc.get("detectors"),
            extended_detectors=bc.get("extended_detectors"),
            detector_options=bc.get("detector_options"),
            buffs=bc.get("buffs"),
            buff_options=bc.get("buff_options"),
            harness_options=bc.get("harness_options"),
            taxonomy=bc.get("taxonomy"),
            generate_autodan=bc.get("generate_autodan"),
        )

    def _normalize_url(self, url: str) -> str:
        """Normalize model URL to include /v1 suffix if needed."""
        import re
        url = url.strip().rstrip("/")
        if not re.match(r"^.*\/v\d+$", url):
            url = f"{url}/v1"
        return url

    # def _build_environment(self, config: JobSpec) -> dict[str, str]:
    #     """Build environment variables for Garak execution."""
    #     env = {
    #         "GARAK_TLS_VERIFY": os.getenv("GARAK_TLS_VERIFY", "True"),
    #     }
        
    #     # Pass through relevant env vars
    #     for var in ["MODEL_API_KEY", "OPENAI_API_KEY", "OPENAICOMPATIBLE_API_KEY"]:
    #         if val := os.getenv(var):
    #             env[var] = val
        
    #     return env

    def _parse_results(
        self, result: GarakScanResult, eval_threshold: float
    ) -> tuple[list[EvaluationResult], float | None, int]:
        """Parse Garak results into EvaluationResult metrics.
        
        Returns:
            Tuple of (metrics list, overall_score, num_examples)
        """
        metrics: list[EvaluationResult] = []
        total_attempts = 0
        total_vulnerable = 0
        
        # Read report content
        report_content = ""
        if result.report_jsonl.exists():
            report_content = result.report_jsonl.read_text()
            if not report_content.strip():
                logger.warning("Report file is empty")
                return metrics, None, 0
        
        avid_content = ""
        if result.avid_jsonl.exists():
            avid_content = result.avid_jsonl.read_text()
        
        # Use shared parsing utilities
        generations, score_rows_by_probe = parse_generations_from_report_content(
            report_content, eval_threshold
        )
        aggregated_by_probe = parse_aggregated_from_avid_content(avid_content)
        
        # Combine results
        combined = combine_parsed_results(
            generations, score_rows_by_probe, aggregated_by_probe, eval_threshold
        )
        
        # Convert to EvaluationResult format (one per probe)
        for probe_name, score_data in combined["scores"].items():
            agg = score_data["aggregated_results"]
            
            probe_attempts = agg.get("total_attempts", 0)
            probe_vulnerable = agg.get("vulnerable_responses", 0)
            attack_success_rate = agg.get("attack_success_rate", 0.0)
            
            total_attempts += probe_attempts
            total_vulnerable += probe_vulnerable
            
            # Create metric for this probe
            metrics.append(EvaluationResult(
                metric_name=f"{probe_name}_asr",
                metric_value=attack_success_rate,
                metric_type="percentage",
                num_samples=probe_attempts,
                metadata={
                    "probe": probe_name,
                    "vulnerable_responses": probe_vulnerable,
                    "benign_responses": agg.get("benign_responses", 0),
                    "detector_scores": agg.get("detector_scores", {}),
                    "avid_taxonomy": agg.get("metadata", {}).get("avid_taxonomy", {}),
                },
            ))
        
        # Calculate overall attack success rate
        overall_score = None
        if total_attempts > 0:
            overall_score = round((total_vulnerable / total_attempts) * 100, 2)
        
        return metrics, overall_score, total_attempts

    def _collect_output_files(
        self, log_file: Path, result: GarakScanResult
    ) -> list[Path]:
        """Collect all output files for artifact persistence."""
        files = []
        
        # Main report files
        if result.report_jsonl.exists():
            files.append(result.report_jsonl)
        
        if result.avid_jsonl.exists():
            files.append(result.avid_jsonl)
        
        if result.hitlog_jsonl.exists():
            files.append(result.hitlog_jsonl)
        
        if result.report_html.exists():
            files.append(result.report_html)
        
        # Log file
        if log_file.exists():
            files.append(log_file)
        
        return files

    def _get_garak_version(self) -> str:
        """Get Garak version string."""
        try:
            from ..version_utils import get_garak_version
            return get_garak_version()
        except Exception:
            return "unknown"


def main() -> None:
    """Entry point for running the adapter as a K8s Job.

    Reads JobSpec from mounted ConfigMap and executes the Garak scan.
    The callback URL comes from job_spec.callback_url (set by the service).
    Registry credentials come from AdapterSettings (environment variables).
    """
    import sys

    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = GarakAdapter(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            provider_id=adapter.job_spec.provider_id,
            sidecar_url=adapter.job_spec.callback_url,
            registry_url=adapter.settings.registry_url,
            registry_username=adapter.settings.registry_username,
            registry_password=adapter.settings.registry_password,
            insecure=adapter.settings.registry_insecure,
        )

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall attack success rate: {results.overall_score}%")

        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Job spec not found: {e}")
        logger.error("Set EVALHUB_JOB_SPEC_PATH or ensure job spec exists at default location")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
