"""Garak Framework Adapter for eval-hub.

This adapter integrates NVIDIA's Garak red-teaming framework with the
eval-hub evaluation platform. It supports two execution modes:

- **simple** (default): Runs garak as a subprocess inside the K8s Job pod.
- **kfp**: Delegates scan execution to a Kubeflow Pipeline, using S3
  (via a Data Connection secret) for artifact transfer. The adapter
  submits the pipeline, polls for completion, and downloads results
  from the shared S3 bucket.

The adapter:
1. Reads JobSpec from mounted ConfigMap (/etc/eval-job/spec.json)
2. Builds and executes Garak CLI commands (simple) or submits KFP pipeline (kfp)
3. Parses results from Garak's JSONL reports
4. Persists artifacts to OCI registry
5. Reports results back to eval-hub via sidecar callbacks
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .kfp_pipeline import KFPConfig

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

from ..core.command_builder import build_generator_options
from ..core.config_resolution import (
    build_effective_garak_config,
    resolve_scan_profile,
    resolve_timeout_seconds,
)
from ..core.garak_runner import convert_to_avid_report, GarakScanResult, run_garak_scan
from ..garak_command_config import GarakCommandConfig
from ..result_utils import (
    combine_parsed_results,
    parse_aggregated_from_avid_content,
    parse_digest_from_report_content,
    parse_generations_from_report_content,
)
from ..utils import get_scan_base_dir
from ..constants import DEFAULT_TIMEOUT, DEFAULT_MODEL_TYPE, DEFAULT_EVAL_THRESHOLD, EXECUTION_MODE_SIMPLE, EXECUTION_MODE_KFP
logger = logging.getLogger(__name__)





class GarakAdapter(FrameworkAdapter):
    """Garak red-teaming framework adapter for eval-hub.

    Supports two execution modes:
    - **simple**: Runs garak as a local subprocess (default).
    - **kfp**: Submits a Kubeflow Pipeline, polls for completion, and
      downloads results from S3 via a Data Connection secret.

    Benchmark Configuration:
        The adapter expects benchmark_config in JobSpec to contain:
        - probes: List of Garak probe names
        - probe_tags: Optional probe tag filters
        - eval_threshold: Threshold for vulnerability detection (default: 0.5)
        - execution_mode: "simple" or "kfp" (default: "simple")
        - kfp_config: dict with KFP connection details (for kfp mode)
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

        Dispatches to either local subprocess execution or KFP pipeline
        execution depending on the resolved execution_mode.
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

            benchmark_config = config.benchmark_config or {}
            execution_mode = self._resolve_execution_mode(benchmark_config)
            logger.info("Execution mode: %s", execution_mode)

            # Build merged garak config (common to both modes)
            scan_dir = get_scan_base_dir() / config.id
            scan_dir.mkdir(parents=True, exist_ok=True)
            report_prefix = scan_dir / "scan"

            garak_config_dict, profile = self._build_config_from_spec(config, report_prefix)
            if not garak_config_dict:
                raise ValueError("Garak command config is empty")

            timeout = resolve_timeout_seconds(
                benchmark_config, profile, default_timeout=DEFAULT_TIMEOUT,
            )
            logger.info("Using timeout=%ss for benchmark=%s", timeout, config.benchmark_id)

            eval_threshold = float(
                garak_config_dict.get("run", {}).get("eval_threshold", DEFAULT_EVAL_THRESHOLD)
            )

            # Phase 2: Execute scan (mode-dependent)
            if execution_mode == EXECUTION_MODE_KFP:
                result, scan_dir = self._run_via_kfp(
                    config=config,
                    callbacks=callbacks,
                    garak_config_dict=garak_config_dict,
                    timeout=timeout,
                )
            else:
                result = self._run_simple(
                    config=config,
                    callbacks=callbacks,
                    garak_config_dict=garak_config_dict,
                    scan_dir=scan_dir,
                    timeout=timeout,
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

            # Phase 3: Parse results (common to both modes)
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.POST_PROCESSING,
                progress=0.8,
                message=MessageInfo(
                    message="Parsing scan results",
                    message_code="parsing_results",
                ),
            ))

            metrics, overall_score, num_examples, overall_summary = self._parse_results(
                result, eval_threshold,
            )
            logger.info(f"Parsed {len(metrics)} probe metrics, overall score: {overall_score}")

            # Phase 4: Persist artifacts
            oci_artifact = None
            if config.exports and config.exports.oci:
                oci_artifact = callbacks.create_oci_artifact(OCIArtifactSpec(
                    files_path=scan_dir,
                    coordinates=config.exports.oci.coordinates,
                ))
                logger.info(f"Persisted scan artifacts: {oci_artifact.reference}")

            # Compute duration
            duration = time.time() - start_time

            return JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
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
                    "execution_mode": execution_mode,
                    "overall": overall_summary,
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

    # ------------------------------------------------------------------
    # Execution mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_execution_mode(benchmark_config: dict) -> str:
        """Resolve execution mode from benchmark_config or env var.

        Priority: benchmark_config > env var > default ("simple").
        """
        mode = (
            benchmark_config.get("execution_mode")
            or os.getenv("EVALHUB_EXECUTION_MODE", EXECUTION_MODE_SIMPLE)
        )
        mode = str(mode).strip().lower()
        if mode not in (EXECUTION_MODE_SIMPLE, EXECUTION_MODE_KFP):
            logger.warning("Unknown execution_mode '%s', falling back to simple", mode)
            mode = EXECUTION_MODE_SIMPLE
        return mode

    # ------------------------------------------------------------------
    # Simple (subprocess) execution
    # ------------------------------------------------------------------

    def _run_simple(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        garak_config_dict: dict,
        scan_dir: Path,
        timeout: int,
    ) -> GarakScanResult:
        """Run garak as a local subprocess."""
        log_file = scan_dir / "scan.log"
        report_prefix = scan_dir / "scan"

        config_file = scan_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(garak_config_dict, f, indent=1)

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

        result = run_garak_scan(
            config_file=config_file,
            timeout_seconds=timeout,
            log_file=log_file,
            report_prefix=report_prefix,
        )

        # AVID conversion
        if result.success:
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

        return result

    # ------------------------------------------------------------------
    # KFP execution
    # ------------------------------------------------------------------

    def _run_via_kfp(
        self,
        config: JobSpec,
        callbacks: JobCallbacks,
        garak_config_dict: dict,
        timeout: int,
    ) -> tuple[GarakScanResult, Path]:
        """Submit a KFP pipeline, poll until done, download results from S3.

        The KFP component uploads scan output to S3 under
        ``{DEFAULT_S3_PREFIX}/{job_id}/``. After the run completes, this
        method downloads those files to a local scan_dir for parsing.

        Both sides get S3 credentials from the same Data Connection secret:
        - KFP pod: injected via ``kubernetes.use_secret_as_env``
        - Adapter pod: mounted via ``envFrom`` by the eval-hub service

        Returns:
            Tuple of (GarakScanResult, local scan_dir with downloaded results).
        """
        from .kfp_pipeline import (
            KFPConfig, DEFAULT_S3_PREFIX, evalhub_garak_pipeline, _resolve_base_image,
        )

        benchmark_config = config.benchmark_config or {}
        kfp_config = KFPConfig.from_env_and_config(benchmark_config)

        if not kfp_config.s3_secret_name:
            raise ValueError(
                "S3 data-connection secret name is required for KFP mode. "
                "Set EVALHUB_KFP_S3_SECRET_NAME or provide "
                "kfp_config.s3_secret_name in benchmark_config."
            )

        s3_prefix = f"{DEFAULT_S3_PREFIX}/{config.id}"
        scan_dir = get_scan_base_dir() / config.id
        scan_dir.mkdir(parents=True, exist_ok=True)

        config_json = json.dumps(garak_config_dict)

        callbacks.report_status(JobStatusUpdate(
            status=JobStatus.RUNNING,
            phase=JobPhase.RUNNING_EVALUATION,
            progress=0.1,
            message=MessageInfo(
                message=f"Submitting KFP pipeline for {config.benchmark_id}",
                message_code="kfp_submitting",
            ),
            current_step="Submitting to Kubeflow Pipelines",
        ))

        kfp_client = self._create_kfp_client(kfp_config)

        base_image = _resolve_base_image(kfp_config)
        logger.info("Using base image for KFP: %s", base_image)

        run = kfp_client.create_run_from_pipeline_func(
            evalhub_garak_pipeline,
            arguments={
                "config_json": config_json,
                "s3_prefix": s3_prefix,
                "timeout_seconds": timeout,
                "s3_secret_name": kfp_config.s3_secret_name,
            },
            run_name=f"evalhub-garak-{config.id}",
            namespace=kfp_config.namespace,
            experiment_name=kfp_config.experiment_name,
        )

        kfp_run_id = run.run_id
        logger.info("Submitted KFP run %s", kfp_run_id)

        poll_timeout = int(timeout * 2) if timeout > 0 else 0
        final_state = self._poll_kfp_run(
            kfp_client, kfp_run_id, callbacks, kfp_config.poll_interval_seconds,
            timeout=poll_timeout,
        )

        timed_out = final_state == "TIMED_OUT"
        success = final_state == "SUCCEEDED"

        if success:
            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.POST_PROCESSING,
                progress=0.7,
                message=MessageInfo(
                    message="Downloading scan results from S3",
                    message_code="downloading_results",
                ),
            ))
            s3_bucket = kfp_config.s3_bucket or os.getenv("AWS_S3_BUCKET", "")
            self._download_results_from_s3(s3_bucket, s3_prefix, scan_dir)

        report_prefix = scan_dir / "scan"

        return GarakScanResult(
            returncode=0 if success else 1,
            stdout="",
            stderr="" if success else f"KFP run ended with state: {final_state}",
            report_prefix=report_prefix,
            timed_out=timed_out,
        ), scan_dir

    @staticmethod
    def _create_kfp_client(kfp_config: "KFPConfig"):
        """Create a KFP Client from KFPConfig."""
        from kfp import Client

        ssl_ca_cert = kfp_config.ssl_ca_cert or None
        token = kfp_config.auth_token or None

        # If no explicit token, try to get one from the cluster service account
        if not token:
            sa_token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
            if sa_token_path.exists():
                token = sa_token_path.read_text().strip()
                logger.debug("Using service account token for KFP auth")

        return Client(
            host=kfp_config.endpoint,
            existing_token=token,
            verify_ssl=kfp_config.verify_ssl,
            ssl_ca_cert=ssl_ca_cert,
        )

    @staticmethod
    def _poll_kfp_run(
        kfp_client,
        run_id: str,
        callbacks: JobCallbacks,
        poll_interval: int,
        timeout: int = 0,
    ) -> str:
        """Poll a KFP run until it reaches a terminal state or times out.

        Relays progress updates to the eval-hub sidecar via callbacks.

        Args:
            timeout: Maximum wall-clock seconds to wait. 0 means no limit.

        Returns:
            Final run state string (e.g. "SUCCEEDED", "FAILED", "TIMED_OUT").
        """
        terminal_states = {"SUCCEEDED", "FAILED", "SKIPPED", "CANCELED", "CANCELING"}
        deadline = (time.monotonic() + timeout) if timeout > 0 else None

        while True:
            run = kfp_client.get_run(run_id)
            state = run.state or "UNKNOWN"
            logger.info("KFP run %s state: %s", run_id, state)

            if state in terminal_states:
                return state

            if deadline and time.monotonic() >= deadline:
                logger.error(
                    "KFP run %s timed out after %ss (last state: %s)",
                    run_id, timeout, state,
                )
                return "TIMED_OUT"

            callbacks.report_status(JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.3,
                message=MessageInfo(
                    message=f"KFP pipeline running (state: {state})",
                    message_code="kfp_running",
                ),
                current_step=f"KFP state: {state}",
            ))

            time.sleep(poll_interval)

    @staticmethod
    def _create_s3_client():
        """Create a boto3 S3 client from Data Connection environment variables.

        Expects standard Data Connection env vars (``AWS_ACCESS_KEY_ID``,
        ``AWS_SECRET_ACCESS_KEY``, ``AWS_S3_ENDPOINT``, etc.) injected from
        the same secret that the KFP component uses.
        """
        import boto3
        from botocore.config import Config as BotoConfig

        s3_endpoint = os.getenv("AWS_S3_ENDPOINT", "")
        client_kwargs: dict = {
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        }
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint
            client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

        return boto3.client("s3", **client_kwargs)

    @staticmethod
    def _download_results_from_s3(bucket: str, prefix: str, local_dir: Path) -> None:
        """Download all scan result files from S3 to a local directory.

        Uses pagination to handle prefixes with many objects.
        """
        if not bucket:
            logger.warning("No S3 bucket configured; skipping result download")
            return

        s3 = GarakAdapter._create_s3_client()

        paginator = s3.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(prefix):].lstrip("/")
                if not relative:
                    continue
                local_path = local_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path))
                downloaded += 1

        logger.info(
            "Downloaded %d files from s3://%s/%s to %s",
            downloaded, bucket, prefix, local_dir,
        )

    def _resolve_profile(self, benchmark_id: str) -> dict:
        """Resolve a benchmark_id to a GarakScanConfig profile dict.

        Checks both FRAMEWORK_PROFILES and SCAN_PROFILES, with and without
        the 'trustyai_garak::' prefix, so that bare IDs like 'owasp_llm_top10'
        and fully-qualified IDs like 'trustyai_garak::owasp_llm_top10' both work.
        """
        return resolve_scan_profile(benchmark_id)

    def _validate_config(self, config: JobSpec) -> None:
        """Validate job configuration."""
        if not config.benchmark_id:
            raise ValueError("benchmark_id is required")

        if not config.model.url:
            raise ValueError("model.url is required")

        if not config.model.name:
            raise ValueError("model.name is required")

        profile = self._resolve_profile(config.benchmark_id)
        benchmark_config = config.benchmark_config or {}
        explicit_garak_cfg = benchmark_config.get("garak_config", {})
        if not isinstance(explicit_garak_cfg, dict):
            explicit_garak_cfg = {}

        explicit_probes = (
            benchmark_config.get("probes")
            or explicit_garak_cfg.get("plugins", {}).get("probe_spec")
        )
        explicit_tags = (
            benchmark_config.get("probe_tags")
            or explicit_garak_cfg.get("run", {}).get("probe_tags")
        )

        if not explicit_probes and not explicit_tags and not profile:
            logger.warning(
                "benchmark_id '%s' does not match a known profile and no probes or "
                "probe_tags provided in parameters — all probes will run",
                config.benchmark_id,
            )

        logger.debug("Configuration validated successfully")

    def _build_config_from_spec(
        self, config: JobSpec, report_prefix: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Garak command config dict from JobSpec."""
        benchmark_config = config.benchmark_config or {}
        profile = self._resolve_profile(config.benchmark_id)
        garak_config: GarakCommandConfig = build_effective_garak_config(
            benchmark_config=benchmark_config,
            profile=profile,
        )

        if profile:
            logger.info(
                "Resolved benchmark_id '%s' to profile '%s'",
                config.benchmark_id,
                profile.get("name"),
            )

        model_params = benchmark_config.get("model_parameters") or {}
        model_type = benchmark_config.get("model_type", DEFAULT_MODEL_TYPE)
        # set generators if not already set by user (because there's a eval-hub config.model for this)
        if model_type == DEFAULT_MODEL_TYPE and not garak_config.plugins.generators:
            garak_config.plugins.generators = build_generator_options(
                model_endpoint=self._normalize_url(config.model.url),
                model_name=config.model.name,
                api_key=getattr(config.model, "api_key", None)
                or os.getenv("OPENAICOMPATIBLE_API_KEY", "DUMMY"),
                extra_params=model_params,
            )

        garak_config.plugins.target_type = model_type
        garak_config.plugins.target_name = config.model.name
        garak_config.reporting.report_prefix = str(report_prefix)

        return garak_config.to_dict(exclude_none=True), profile

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
    ) -> tuple[list[EvaluationResult], float | None, int, dict[str, Any]]:
        """Parse Garak results into EvaluationResult metrics.
        
        Returns:
            Tuple of (metrics list, overall_score, num_examples, overall summary)
        """
        metrics: list[EvaluationResult] = []
        total_attempts = 0
        
        # Read report content
        report_content = ""
        if result.report_jsonl.exists():
            report_content = result.report_jsonl.read_text()
            if not report_content.strip():
                logger.warning("Report file is empty")
                return metrics, None, 0, {}
        
        avid_content = ""
        if result.avid_jsonl.exists():
            avid_content = result.avid_jsonl.read_text()
        
        # Use shared parsing utilities
        generations, score_rows_by_probe = parse_generations_from_report_content(
            report_content, eval_threshold
        )
        aggregated_by_probe = parse_aggregated_from_avid_content(avid_content)
        digest = parse_digest_from_report_content(report_content)
        
        # Combine results
        combined = combine_parsed_results(
            generations, score_rows_by_probe, aggregated_by_probe, eval_threshold, digest
        )
        overall_summary = (
            combined.get("scores", {})
            .get("_overall", {})
            .get("aggregated_results", {})
        )
        
        # Convert to EvaluationResult format (one per probe)
        for probe_name, score_data in combined["scores"].items():
            if probe_name == "_overall":
                continue

            agg = score_data["aggregated_results"]
            
            probe_attempts = agg.get("total_attempts", 0)
            probe_vulnerable = agg.get("vulnerable_responses", 0)
            attack_success_rate = agg.get("attack_success_rate", 0.0)
            
            total_attempts += probe_attempts
            
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
        
        overall_score = overall_summary.get("attack_success_rate")
        if overall_score is not None:
            try:
                overall_score = float(overall_score)
            except (TypeError, ValueError):
                overall_score = None

        if overall_score is None and total_attempts > 0:
            total_vulnerable = sum(
                score_data["aggregated_results"].get("vulnerable_responses", 0)
                for probe_name, score_data in combined["scores"].items()
                if probe_name != "_overall"
            )
            overall_score = round((total_vulnerable / total_attempts) * 100, 2)

        num_examples = overall_summary.get("total_attempts", total_attempts)
        try:
            num_examples = int(num_examples)
        except (TypeError, ValueError):
            num_examples = total_attempts
        
        return metrics, overall_score, num_examples, overall_summary


    def _get_garak_version(self) -> str:
        """Get Garak version string."""
        try:
            from ..version_utils import get_garak_version
            return get_garak_version()
        except Exception:
            return "unknown"


def main(adapter_cls: type[GarakAdapter] = GarakAdapter) -> None:
    """Entry point for running the adapter as a K8s Job.

    Reads JobSpec from mounted ConfigMap and executes the Garak scan.
    The callback URL comes from job_spec.callback_url (set by the service).
    Registry credentials come from AdapterSettings (environment variables).

    Args:
        adapter_cls: The adapter class to instantiate. Defaults to
            GarakAdapter (simple mode). Pass a subclass to override
            behaviour (e.g. GarakKFPAdapter for forced KFP execution).
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
        adapter = adapter_cls(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        oci_auth_config = os.getenv("OCI_AUTH_CONFIG_PATH")
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            provider_id=adapter.job_spec.provider_id,
            sidecar_url=adapter.job_spec.callback_url,
            oci_auth_config_path=Path(oci_auth_config) if oci_auth_config else None,
            oci_insecure=os.getenv("OCI_REGISTRY_INSECURE", "false").lower() == "true",
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
