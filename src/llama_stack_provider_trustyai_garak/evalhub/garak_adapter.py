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
    generate_art_report,
    parse_aggregated_from_avid_content,
    parse_digest_from_report_content,
    parse_generations_from_report_content,
)
from ..utils import get_scan_base_dir, as_bool, safe_int
from ..constants import (
    DEFAULT_TIMEOUT,
    DEFAULT_MODEL_TYPE,
    DEFAULT_EVAL_THRESHOLD,
    EXECUTION_MODE_SIMPLE,
    EXECUTION_MODE_KFP,
    DEFAULT_SDG_FLOW_ID,
    DEFAULT_SDG_MAX_CONCURRENCY,
    DEFAULT_SDG_NUM_SAMPLES,
    DEFAULT_SDG_MAX_TOKENS,
)

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

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        """Run a Garak security scan job.

        Dispatches to either local subprocess execution or KFP pipeline
        execution depending on the resolved execution_mode.
        """
        start_time = time.time()
        logger.info(f"Starting Garak job {config.id} for benchmark {config.benchmark_id}")

        try:
            # Phase 1: Initialize
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Validating configuration and building scan command",
                        message_code="initializing",
                    ),
                )
            )

            self._validate_config(config)

            benchmark_config = config.parameters or {}
            execution_mode = self._resolve_execution_mode(benchmark_config)
            logger.info("Execution mode: %s", execution_mode)

            # Early check: intents benchmarks require KFP mode
            _intents_required = benchmark_config.get("art_intents")
            if _intents_required is None:
                _profile = self._resolve_profile(config.benchmark_id)
                _intents_required = _profile.get("art_intents", False) if _profile else False
            if _intents_required and execution_mode != EXECUTION_MODE_KFP:
                raise ValueError("Intents benchmarks are only supported in KFP execution mode. ")

            # Build merged garak config (common to both modes)
            scan_dir = get_scan_base_dir() / config.id
            scan_dir.mkdir(parents=True, exist_ok=True)
            report_prefix = scan_dir / "scan"

            garak_config_dict, profile, intents_params = self._build_config_from_spec(config, report_prefix)
            if not garak_config_dict:
                raise ValueError("Garak command config is empty")

            art_intents = intents_params.get("art_intents", False)

            timeout = resolve_timeout_seconds(
                benchmark_config,
                profile,
                default_timeout=DEFAULT_TIMEOUT,
            )
            logger.info("Using timeout=%ss for benchmark=%s", timeout, config.benchmark_id)

            eval_threshold = float(garak_config_dict.get("run", {}).get("eval_threshold", DEFAULT_EVAL_THRESHOLD))

            # Phase 2: Execute scan (mode-dependent)
            if execution_mode == EXECUTION_MODE_KFP:
                result, scan_dir = self._run_via_kfp(
                    config=config,
                    callbacks=callbacks,
                    garak_config_dict=garak_config_dict,
                    timeout=timeout,
                    intents_params=intents_params,
                    eval_threshold=eval_threshold,
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

                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.FAILED,
                        error=ErrorInfo(message=error_msg, message_code="scan_failed"),
                    )
                )
                raise RuntimeError(error_msg)

            # Phase 3: Parse results (common to both modes)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.8,
                    message=MessageInfo(
                        message="Parsing scan results",
                        message_code="parsing_results",
                    ),
                )
            )

            metrics, overall_score, num_examples, overall_summary = self._parse_results(
                result,
                eval_threshold,
                art_intents=art_intents,
            )
            logger.info(f"Parsed {len(metrics)} probe metrics, overall score: {overall_score}")

            # Phase 3b: Generate ART HTML report for intents scans
            if art_intents and result.report_jsonl.exists():
                try:
                    report_content = result.report_jsonl.read_text()
                    if report_content.strip():
                        art_html_path = scan_dir / "scan.intents.html"
                        if not art_html_path.exists():
                            art_html = generate_art_report(report_content)
                            art_html_path.write_text(art_html)
                            logger.info("Generated ART HTML report: %s", art_html_path)
                except Exception as e:
                    logger.warning("Failed to generate ART HTML report: %s", e)

            # Redact api_key values from config.json before OCI export
            config_file = scan_dir / "config.json"
            if config_file.exists():
                try:
                    from ..core.pipeline_steps import redact_api_keys

                    cfg = json.loads(config_file.read_text())
                    config_file.write_text(json.dumps(redact_api_keys(cfg), indent=1))
                except Exception as exc:
                    logger.warning("Could not redact config.json: %s", exc)

            # Phase 4: Persist artifacts
            oci_artifact = None
            if config.exports and config.exports.oci:
                oci_artifact = callbacks.create_oci_artifact(
                    OCIArtifactSpec(
                        files_path=scan_dir,
                        coordinates=config.exports.oci.coordinates,
                    )
                )
                logger.info(f"Persisted scan artifacts: {oci_artifact.reference}")

            # Compute duration
            duration = time.time() - start_time

            eval_meta: dict[str, Any] = {
                "framework": "garak",
                "eval_threshold": eval_threshold,
                "timed_out": result.timed_out,
                "execution_mode": execution_mode,
                "art_intents": art_intents,
                "overall": overall_summary,
            }

            if execution_mode == EXECUTION_MODE_KFP:
                from .kfp_pipeline import DEFAULT_S3_PREFIX

                _bc = config.parameters or {}
                _kfp_ov = _bc.get("kfp_config", {}) if isinstance(_bc.get("kfp_config"), dict) else {}
                _prefix = _kfp_ov.get("s3_prefix", os.getenv("KFP_S3_PREFIX", DEFAULT_S3_PREFIX))
                _bucket = _kfp_ov.get("s3_bucket", os.getenv("AWS_S3_BUCKET", ""))
                s3_prefix = f"{_prefix}/{config.id}"

                artifact_keys: dict[str, str] = {
                    "scan_report": f"{s3_prefix}/scan.report.jsonl",
                    "scan_html_report": f"{s3_prefix}/scan.report.html",
                }
                if art_intents:
                    artifact_keys["sdg_raw_output"] = f"{s3_prefix}/sdg_raw_output.csv"
                    artifact_keys["sdg_normalized_output"] = f"{s3_prefix}/sdg_normalized_output.csv"
                    artifact_keys["intents_html_report"] = f"{s3_prefix}/scan.intents.html"

                s3_artifacts: dict[str, str] = {}
                for name, key in artifact_keys.items():
                    local_file = scan_dir / key.split("/")[-1]
                    if local_file.exists():
                        s3_artifacts[name] = f"s3://{_bucket}/{key}" if _bucket else key
                    else:
                        logger.debug("Artifact not downloaded locally, skipping: %s", key)

                eval_meta["artifacts"] = s3_artifacts

            results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=metrics,
                overall_score=overall_score,
                num_examples_evaluated=num_examples,
                duration_seconds=duration,
                completed_at=datetime.now(UTC),
                evaluation_metadata=eval_meta,
                oci_artifact=oci_artifact,
            )

            # Phase 5: Save to MLflow (if experiment_name configured)
            try:
                from evalhub.adapter.mlflow import MlflowArtifact

                mlflow_artifacts: list[MlflowArtifact] = []
                if result.report_html.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.report.html",
                            result.report_html.read_bytes(),
                            "text/html",
                        )
                    )
                art_html_path = scan_dir / "scan.intents.html"
                if art_html_path.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.intents.html",
                            art_html_path.read_bytes(),
                            "text/html",
                        )
                    )
                if result.report_jsonl.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "scan.report.jsonl",
                            result.report_jsonl.read_bytes(),
                            "application/jsonl",
                        )
                    )
                sdg_raw = scan_dir / "sdg_raw_output.csv"
                if sdg_raw.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "sdg_raw_output.csv",
                            sdg_raw.read_bytes(),
                            "text/csv",
                        )
                    )
                sdg_norm = scan_dir / "sdg_normalized_output.csv"
                if sdg_norm.exists():
                    mlflow_artifacts.append(
                        MlflowArtifact(
                            "sdg_normalized_output.csv",
                            sdg_norm.read_bytes(),
                            "text/csv",
                        )
                    )

                rid = callbacks.mlflow.save(results, config, artifacts=mlflow_artifacts)
                if rid:
                    results.mlflow_run_id = rid
                logger.info("Saved results and %d artifacts to MLflow with run ID: %s", len(mlflow_artifacts), rid)
            except Exception as mlflow_exc:
                logger.warning("MLflow save failed (non-fatal): %s", mlflow_exc)

            return results

        except Exception as e:
            logger.exception(f"Garak job {config.id} failed")
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error=ErrorInfo(message=str(e), message_code="job_failed"),
                    error_details={"exception_type": type(e).__name__},
                )
            )
            raise

    # ------------------------------------------------------------------
    # Execution mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_execution_mode(benchmark_config: dict) -> str:
        """Resolve execution mode from benchmark_config or env var.

        Priority: benchmark_config > env var > default ("simple").
        """
        mode = benchmark_config.get("execution_mode") or os.getenv("EVALHUB_EXECUTION_MODE", EXECUTION_MODE_SIMPLE)
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

        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.1,
                message=MessageInfo(
                    message=f"Running Garak scan for {config.benchmark_id}",
                    message_code="running_scan",
                ),
                current_step="Executing probes",
            )
        )

        env: dict[str, str] = {}
        hf_cache = (config.parameters or {}).get("hf_cache_path", "")
        if hf_cache:
            env["HF_HUB_CACHE"] = hf_cache
            logger.info("Using HF cache from mounted path: %s", hf_cache)

        result = run_garak_scan(
            config_file=config_file,
            timeout_seconds=timeout,
            log_file=log_file,
            report_prefix=report_prefix,
            env=env if env else None,
        )

        # AVID conversion
        if result.success:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.7,
                    message=MessageInfo(
                        message="Converting results to AVID format",
                        message_code="post_processing",
                    ),
                )
            )
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
        intents_params: dict[str, Any] | None = None,
        eval_threshold: float = DEFAULT_EVAL_THRESHOLD,
    ) -> tuple[GarakScanResult, Path]:
        """Submit a KFP pipeline, poll until done, download results from S3.

        The KFP component uploads scan output to S3 under
        ``{kfp_config.s3_prefix}/{job_id}/``. After the run completes,
        this method downloads those files to a local scan_dir for parsing.

        Both sides get S3 credentials from the same Data Connection secret:
        - KFP pod: injected via ``kubernetes.use_secret_as_env``
        - Adapter pod: mounted via ``envFrom`` by the eval-hub service

        Returns:
            Tuple of (GarakScanResult, local scan_dir with downloaded results).
        """
        from .kfp_pipeline import KFPConfig, evalhub_garak_pipeline

        benchmark_config = config.parameters or {}
        kfp_config = KFPConfig.from_env_and_config(benchmark_config)

        if not kfp_config.s3_secret_name:
            raise ValueError(
                "S3 data-connection secret name is required for KFP mode. "
                "Set KFP_S3_SECRET_NAME or provide "
                "kfp_config.s3_secret_name in benchmark_config."
            )

        s3_prefix = f"{kfp_config.s3_prefix}/{config.id}"
        scan_dir = get_scan_base_dir() / config.id
        scan_dir.mkdir(parents=True, exist_ok=True)

        from ..core.pipeline_steps import redact_api_keys

        sanitised_config = redact_api_keys(garak_config_dict)
        config_json = json.dumps(sanitised_config)

        ip = intents_params or {}

        if ip.get("art_intents"):
            if ip.get("policy_s3_key") and ip.get("intents_s3_key"):
                raise ValueError(
                    "policy_s3_key and intents_s3_key are mutually exclusive. "
                    "Provide a taxonomy for SDG (policy_s3_key) OR "
                    "pre-generated prompts to bypass SDG (intents_s3_key), not both."
                )
            if not ip.get("intents_s3_key"):
                if not ip.get("sdg_model"):
                    raise ValueError(
                        "Intents benchmark (art_intents=True) requires "
                        "sdg_model for prompt generation when intents_s3_key "
                        "is not provided."
                    )
                if not ip.get("sdg_api_base"):
                    raise ValueError(
                        "Intents benchmark (art_intents=True) requires "
                        "sdg_api_base for prompt generation when intents_s3_key "
                        "is not provided."
                    )

        callbacks.report_status(
            JobStatusUpdate(
                status=JobStatus.RUNNING,
                phase=JobPhase.RUNNING_EVALUATION,
                progress=0.1,
                message=MessageInfo(
                    message=f"Submitting KFP pipeline for {config.benchmark_id}",
                    message_code="kfp_submitting",
                ),
                current_step="Submitting to Kubeflow Pipelines",
            )
        )

        # Resolve model auth secret from EvalHub SDK model.auth.secret_ref
        # Falls back to pipeline default ("model-auth") when not specified.
        model_auth_secret = ""
        try:
            model_auth = getattr(config.model, "auth", None)
            if model_auth:
                model_auth_secret = getattr(model_auth, "secret_ref", "") or ""
        except Exception:
            pass
        if model_auth_secret:
            logger.info("Using model auth secret: %s", model_auth_secret)

        kfp_client = self._create_kfp_client(kfp_config)

        pipeline_args: dict[str, Any] = {
            "config_json": config_json,
            "s3_prefix": s3_prefix,
            "timeout_seconds": timeout,
            "s3_secret_name": kfp_config.s3_secret_name,
            "eval_threshold": eval_threshold,
            "art_intents": ip.get("art_intents", False),
            "policy_s3_key": ip.get("policy_s3_key", ""),
            "policy_format": ip.get("policy_format", "csv"),
            "intents_s3_key": ip.get("intents_s3_key", ""),
            "intents_format": ip.get("intents_format", "csv"),
            "sdg_model": ip.get("sdg_model", ""),
            "sdg_api_base": ip.get("sdg_api_base", ""),
            "sdg_flow_id": ip.get("sdg_flow_id", DEFAULT_SDG_FLOW_ID),
            "sdg_max_concurrency": ip.get("sdg_max_concurrency", DEFAULT_SDG_MAX_CONCURRENCY),
            "sdg_num_samples": ip.get("sdg_num_samples", DEFAULT_SDG_NUM_SAMPLES),
            "sdg_max_tokens": ip.get("sdg_max_tokens", DEFAULT_SDG_MAX_TOKENS),
            "hf_cache_path": benchmark_config.get("hf_cache_path", ""),
        }
        if model_auth_secret:
            pipeline_args["model_auth_secret_name"] = model_auth_secret

        disable_cache = as_bool(ip.get("disable_cache", False))
        run = kfp_client.create_run_from_pipeline_func(
            evalhub_garak_pipeline,
            arguments=pipeline_args,
            run_name=f"evalhub-garak-{config.id}",
            namespace=kfp_config.namespace,
            experiment_name=kfp_config.experiment_name,
            enable_caching=not disable_cache,
        )

        kfp_run_id = run.run_id
        logger.info("Submitted KFP run %s", kfp_run_id)

        poll_timeout = int(timeout * 2) if timeout > 0 else 0
        final_state = self._poll_kfp_run(
            kfp_client,
            kfp_run_id,
            callbacks,
            kfp_config.poll_interval_seconds,
            timeout=poll_timeout,
        )

        timed_out = final_state == "TIMED_OUT"
        success = final_state == "SUCCEEDED"

        if success:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.POST_PROCESSING,
                    progress=0.7,
                    message=MessageInfo(
                        message="Downloading scan results from S3",
                        message_code="downloading_results",
                    ),
                )
            )
            creds = (
                self._read_s3_credentials_from_secret(
                    kfp_config.s3_secret_name,
                    kfp_config.namespace,
                )
                if kfp_config.s3_secret_name
                else {}
            )
            if kfp_config.s3_secret_name and not creds:
                logger.warning(
                    "S3 credentials from secret '%s/%s' are empty. "
                    "Ensure the secret exists and the Job pod's service account "
                    "has RBAC permissions to read secrets in namespace '%s'. "
                    "Falling back to environment variables for S3 access."
                    "If no environment variables are set, the job will fail",
                    "as it will not be able to interact with S3.",
                    kfp_config.namespace,
                    kfp_config.s3_secret_name,
                    kfp_config.namespace,
                )
            s3_bucket = kfp_config.s3_bucket or creds.pop("bucket", "") or os.getenv("AWS_S3_BUCKET", "")
            s3_endpoint = kfp_config.s3_endpoint or creds.pop("endpoint_url", "") or None
            self._download_results_from_s3(
                s3_bucket,
                s3_prefix,
                scan_dir,
                endpoint_url=s3_endpoint,
                **creds,
            )

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
                    run_id,
                    timeout,
                    state,
                )
                return "TIMED_OUT"

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.3,
                    message=MessageInfo(
                        message=f"KFP pipeline running (state: {state})",
                        message_code="kfp_running",
                    ),
                    current_step=f"KFP state: {state}",
                )
            )

            time.sleep(poll_interval)

    @staticmethod
    def _read_s3_credentials_from_secret(secret_name: str, namespace: str) -> dict:
        """Read S3 credentials from a Kubernetes secret.

        Falls back gracefully if the secret cannot be read (e.g. outside a
        cluster or missing RBAC), returning an empty dict so env-var fallback
        in ``create_s3_client`` still applies.

        .. note:: **RBAC requirement** — The Job pod's service account must
           have ``get`` permission on Secrets in the target namespace.
           Example Role/RoleBinding::

               apiVersion: rbac.authorization.k8s.io/v1
               kind: Role
               metadata:
                 name: secret-reader
                 namespace: <namespace>
               rules:
               - apiGroups: [""]
                 resources: ["secrets"]
                 verbs: ["get"]
               ---
               apiVersion: rbac.authorization.k8s.io/v1
               kind: RoleBinding
               metadata:
                 name: evalhub-secret-reader
                 namespace: <namespace>
               subjects:
               - kind: ServiceAccount
                 name: <job-service-account>
                 namespace: <namespace>
               roleRef:
                 kind: Role
                 name: secret-reader
                 apiGroup: rbac.authorization.k8s.io

           Without this, the call returns an empty dict and S3 operations
           fall back to environment variables (which may also be empty,
           causing job failures).
        """
        import base64

        try:
            from kubernetes import client as k8s_client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()
            v1 = k8s_client.CoreV1Api()
            secret = v1.read_namespaced_secret(secret_name, namespace)
            data = secret.data or {}

            def _decode(key: str) -> str:
                val = data.get(key, "")
                return base64.b64decode(val).decode() if val else ""

            return {
                "access_key": _decode("AWS_ACCESS_KEY_ID"),
                "secret_key": _decode("AWS_SECRET_ACCESS_KEY"),
                "region": _decode("AWS_DEFAULT_REGION"),
                "bucket": _decode("AWS_S3_BUCKET"),
                "endpoint_url": _decode("AWS_S3_ENDPOINT"),
            }
        except Exception as exc:
            logger.warning("Could not read S3 credentials from secret %s/%s: %s", namespace, secret_name, exc)
            return {}

    @staticmethod
    def _create_s3_client(
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ):
        """Create a boto3 S3 client.

        Explicit parameters take precedence over environment variables, which
        allows the adapter pod to supply credentials read from a k8s secret
        when they are not present in its own environment.
        """
        from .s3_utils import create_s3_client

        return create_s3_client(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )

    @staticmethod
    def _download_results_from_s3(
        bucket: str,
        prefix: str,
        local_dir: Path,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ) -> None:
        """Download all scan result files from S3 to a local directory.

        Uses pagination to handle prefixes with many objects.
        Explicit credential parameters take precedence over environment variables.
        """
        if not bucket:
            logger.warning("No S3 bucket configured; skipping result download")
            return

        s3 = GarakAdapter._create_s3_client(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )

        paginator = s3.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative = key[len(prefix) :].lstrip("/")
                if not relative:
                    continue
                local_path = local_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path))
                downloaded += 1

        logger.info(
            "Downloaded %d files from s3://%s/%s to %s",
            downloaded,
            bucket,
            prefix,
            local_dir,
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
        benchmark_config = config.parameters or {}
        explicit_garak_cfg = benchmark_config.get("garak_config", {})
        if not isinstance(explicit_garak_cfg, dict):
            explicit_garak_cfg = {}

        explicit_probes = benchmark_config.get("probes") or explicit_garak_cfg.get("plugins", {}).get("probe_spec")
        explicit_tags = benchmark_config.get("probe_tags") or explicit_garak_cfg.get("run", {}).get("probe_tags")

        if not explicit_probes and not explicit_tags and not profile:
            logger.warning(
                "benchmark_id '%s' does not match a known profile and no probes or "
                "probe_tags provided in parameters — all probes will run",
                config.benchmark_id,
            )

        logger.debug("Configuration validated successfully")

    def _build_config_from_spec(
        self, config: JobSpec, report_prefix: Path
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Build Garak command config dict from JobSpec.

        Returns:
            Tuple of (garak_config_dict, profile, intents_params) where
            intents_params contains art_intents and related settings
            extracted from the profile and benchmark_config.
        """
        benchmark_config = config.parameters or {}
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
            from evalhub.adapter.auth import read_model_auth_key

            api_key = (
                getattr(config.model, "api_key", None)
                or read_model_auth_key("api-key")
                or os.getenv("OPENAICOMPATIBLE_API_KEY")
                or "DUMMY"
            )
            garak_config.plugins.generators = build_generator_options(
                model_endpoint=self._normalize_url(config.model.url),
                model_name=config.model.name,
                api_key=api_key,
                extra_params=model_params,
            )

        garak_config.plugins.target_type = model_type
        garak_config.plugins.target_name = config.model.name
        garak_config.reporting.report_prefix = str(report_prefix)

        art_intents = (
            bool(benchmark_config.get("art_intents"))
            if "art_intents" in benchmark_config
            else bool(profile.get("art_intents", False))
        )
        intents_params: dict[str, Any] = {
            "art_intents": art_intents,
            "policy_s3_key": benchmark_config.get("policy_s3_key", profile.get("policy_s3_key", "")),
            "policy_format": benchmark_config.get("policy_format", profile.get("policy_format", "csv")),
            "intents_s3_key": benchmark_config.get("intents_s3_key", profile.get("intents_s3_key", "")),
            "intents_format": benchmark_config.get("intents_format", profile.get("intents_format", "csv")),
            "sdg_flow_id": benchmark_config.get("sdg_flow_id", profile.get("sdg_flow_id", DEFAULT_SDG_FLOW_ID)),
            "sdg_max_concurrency": safe_int(
                benchmark_config.get(
                    "sdg_max_concurrency", profile.get("sdg_max_concurrency", DEFAULT_SDG_MAX_CONCURRENCY)
                ),
                DEFAULT_SDG_MAX_CONCURRENCY,
            ),
            "sdg_num_samples": safe_int(
                benchmark_config.get("sdg_num_samples", profile.get("sdg_num_samples", DEFAULT_SDG_NUM_SAMPLES)),
                DEFAULT_SDG_NUM_SAMPLES,
            ),
            "sdg_max_tokens": safe_int(
                benchmark_config.get("sdg_max_tokens", profile.get("sdg_max_tokens", DEFAULT_SDG_MAX_TOKENS)),
                DEFAULT_SDG_MAX_TOKENS,
            ),
            "disable_cache": as_bool(benchmark_config.get("disable_cache", False)),
        }

        if art_intents:
            sdg_params = self._apply_intents_model_config(garak_config, benchmark_config, profile)
            intents_params.update(sdg_params)

        return garak_config.to_dict(exclude_none=True), profile, intents_params

    # ------------------------------------------------------------------
    # Intents model configuration
    # ------------------------------------------------------------------

    def _apply_intents_model_config(
        self,
        garak_config: GarakCommandConfig,
        benchmark_config: dict,
        profile: dict,
    ) -> dict[str, Any]:
        """Configure judge/attacker/evaluator/SDG models from ``intents_models``.

        Users provide per-role model endpoints in ``benchmark_config``:

        .. code-block:: json

            {
                "intents_models": {
                    "judge":     {"url": "...", "name": "..."},
                    "attacker":  {"url": "...", "name": "..."},
                    "evaluator": {"url": "...", "name": "..."},
                    "sdg":       {"url": "...", "name": "..."}
                }
            }

        Exactly 1 or all 3 of judge/attacker/evaluator must have a ``url``:

        - **1 provided**: the configured role is used for all three.
        - **3 provided**: each role uses its own config.
        - **0 provided, but models pre-configured in garak_config**: the
          override is skipped and models from garak_config are used as-is.
          SDG params are extracted from flat keys (``sdg_model``,
          ``sdg_api_base``).
        - **0 provided, no garak_config models**: raises ``ValueError``.
        - **2 provided**: raises ``ValueError`` — ambiguous which should fill
          the missing role.

        ``sdg`` has no fallback — it must be provided explicitly if SDG
        generation is needed.

        API keys are **no longer** embedded in the config dict.  They are
        injected at pod level via a Kubernetes Secret
        (``model_auth_secret_name``).  Placeholder values (``__FROM_ENV__``)
        are written into the config and resolved inside the KFP pod by
        ``core.pipeline_steps._resolve_config_api_keys``.

        Returns:
            Dict with SDG-related keys (``sdg_model``, ``sdg_api_base``)
            extracted from the ``sdg`` role, or empty strings if SDG is
            not configured.
        """
        intents_models = benchmark_config.get("intents_models", {})
        if not isinstance(intents_models, dict):
            intents_models = {}

        judge_cfg = intents_models.get("judge") or {}
        attacker_cfg = intents_models.get("attacker") or {}
        evaluator_cfg = intents_models.get("evaluator") or {}
        sdg_cfg = intents_models.get("sdg") or {}

        provided = {
            role: cfg
            for role, cfg in [("judge", judge_cfg), ("attacker", attacker_cfg), ("evaluator", evaluator_cfg)]
            if cfg.get("url")
        }

        if len(provided) == 0:
            if garak_config.plugins and self._models_preconfigured_in_garak_config(garak_config.plugins):
                logger.info(
                    "No intents_models provided but models are already "
                    "configured in garak_config — skipping intents_models "
                    "override. API keys will be resolved by "
                    "_resolve_config_api_keys in the KFP pod."
                )
                return self._extract_sdg_params(sdg_cfg, benchmark_config, profile)

            raise ValueError(
                "Intents benchmark requires model configuration for "
                "judge/attacker/evaluator roles. Either:\n"
                "  1. Provide intents_models with at least one role "
                "(url + name), or\n"
                "  2. Configure models directly in garak_config "
                "(plugins.detectors.judge with detector_model_name and "
                "detector_model_config.uri)."
            )
        if len(provided) == 2:
            missing = {"judge", "attacker", "evaluator"} - set(provided)
            raise ValueError(
                f"Ambiguous intents_models config: {', '.join(sorted(provided))} "
                f"are configured but {', '.join(sorted(missing))} is missing. "
                f"Provide all three, or provide exactly one to use for all roles."
            )

        if len(provided) == 1:
            base_cfg = next(iter(provided.values()))
            judge_cfg = {**base_cfg, **judge_cfg}
            attacker_cfg = {**base_cfg, **attacker_cfg}
            evaluator_cfg = {**base_cfg, **evaluator_cfg}

        for role, cfg in [("judge", judge_cfg), ("attacker", attacker_cfg), ("evaluator", evaluator_cfg)]:
            if not cfg.get("name"):
                raise ValueError(f"intents_models.{role}.name is required. Provide a model identifier for the {role}.")

        judge_url = judge_cfg["url"]
        judge_name = judge_cfg["name"]

        attacker_url = attacker_cfg["url"]
        attacker_name = attacker_cfg["name"]

        evaluator_url = evaluator_cfg["url"]
        evaluator_name = evaluator_cfg["name"]

        # Use placeholder — real keys are injected via K8s Secret at pod level
        _PLACEHOLDER = "__FROM_ENV__"

        plugins = garak_config.plugins

        plugins.detectors = plugins.detectors or {}
        existing_judge = plugins.detectors.get("judge", {})
        existing_judge["detector_model_type"] = existing_judge.get("detector_model_type") or "openai.OpenAICompatible"
        existing_judge["detector_model_name"] = existing_judge.get("detector_model_name") or judge_name
        existing_det_cfg = existing_judge.get("detector_model_config", {})
        existing_det_cfg["uri"] = existing_det_cfg.get("uri") or judge_url
        existing_det_cfg["api_key"] = _PLACEHOLDER
        existing_judge["detector_model_config"] = existing_det_cfg
        plugins.detectors["judge"] = existing_judge

        if plugins.probes and plugins.probes.get("tap"):
            tap_cfg = plugins.probes["tap"].get("TAPIntent", {})
            if isinstance(tap_cfg, dict):
                tap_cfg["attack_model_name"] = tap_cfg.get("attack_model_name") or attacker_name
                existing_attack_cfg = tap_cfg.get("attack_model_config", {})
                existing_attack_cfg.setdefault("max_tokens", 500)
                existing_attack_cfg["uri"] = existing_attack_cfg.get("uri") or attacker_url
                existing_attack_cfg["api_key"] = _PLACEHOLDER
                tap_cfg["attack_model_config"] = existing_attack_cfg

                tap_cfg["evaluator_model_name"] = tap_cfg.get("evaluator_model_name") or evaluator_name
                existing_eval_cfg = tap_cfg.get("evaluator_model_config", {})
                existing_eval_cfg.setdefault("max_tokens", 10)
                existing_eval_cfg.setdefault("temperature", 0.0)
                existing_eval_cfg["uri"] = existing_eval_cfg.get("uri") or evaluator_url
                existing_eval_cfg["api_key"] = _PLACEHOLDER
                tap_cfg["evaluator_model_config"] = existing_eval_cfg

                plugins.probes["tap"]["TAPIntent"] = tap_cfg

        return self._extract_sdg_params(sdg_cfg, benchmark_config, profile)

    @staticmethod
    def _models_preconfigured_in_garak_config(plugins: Any) -> bool:
        """Check if intents models are already configured in garak_config.

        Returns True when the judge detector has a non-empty
        ``detector_model_name`` and ``detector_model_config.uri``.
        If TAPIntent is present in the probes, also requires non-empty
        attack and evaluator model names and URIs.
        """
        detectors = plugins.detectors or {}
        judge = detectors.get("judge", {})
        if not judge.get("detector_model_name") or not judge.get("detector_model_config", {}).get("uri"):
            return False

        probes = plugins.probes or {}
        tap_cfg = probes.get("tap", {}).get("TAPIntent")
        if tap_cfg and isinstance(tap_cfg, dict):
            for name_key, cfg_key in [
                ("attack_model_name", "attack_model_config"),
                ("evaluator_model_name", "evaluator_model_config"),
            ]:
                if not tap_cfg.get(name_key) or not tap_cfg.get(cfg_key, {}).get("uri"):
                    return False

        return True

    @staticmethod
    def _extract_sdg_params(
        sdg_cfg: dict,
        benchmark_config: dict,
        profile: dict,
    ) -> dict[str, str]:
        """Extract SDG model params from intents_models.sdg or flat keys."""
        sdg_params: dict[str, str] = {
            "sdg_model": "",
            "sdg_api_base": "",
        }
        if sdg_cfg.get("url") and sdg_cfg.get("name"):
            sdg_params["sdg_model"] = sdg_cfg["name"]
            sdg_params["sdg_api_base"] = sdg_cfg["url"]
        else:
            sdg_model = benchmark_config.get("sdg_model", profile.get("sdg_model", ""))
            sdg_api_base = benchmark_config.get("sdg_api_base", profile.get("sdg_api_base", ""))
            if sdg_model and sdg_api_base:
                sdg_params["sdg_model"] = sdg_model
                sdg_params["sdg_api_base"] = sdg_api_base
        return sdg_params

    @staticmethod
    def _resolve_intents_api_key(role: str, model_cfg: dict) -> str:
        """Resolve API key for an intents model role.

        Resolution order:

        1. Direct ``api_key`` value in ``model_cfg``
        2. ``api_key_name`` -> read from mounted model auth secret
        3. ``api_key_env`` -> read from named environment variable
        4. Convention: ``{role}-api-key`` from mounted secret
        5. Convention: ``{ROLE}_API_KEY`` from environment
        6. Generic: ``api-key`` from mounted secret
        7. Generic: ``OPENAICOMPATIBLE_API_KEY`` from environment
        8. ``"DUMMY"``
        """
        if model_cfg.get("api_key"):
            return model_cfg["api_key"]

        _read_secret: Any = None
        try:
            from evalhub.adapter.auth import read_model_auth_key

            _read_secret = read_model_auth_key
        except ImportError:
            pass

        if _read_secret and model_cfg.get("api_key_name"):
            key = _read_secret(model_cfg["api_key_name"])
            if key:
                return key

        if model_cfg.get("api_key_env"):
            key = os.getenv(model_cfg["api_key_env"])
            if key:
                return key

        if _read_secret:
            key = _read_secret(f"{role}-api-key")
            if key:
                return key

        key = os.getenv(f"{role.upper()}_API_KEY")
        if key:
            return key

        if _read_secret:
            key = _read_secret("api-key")
            if key:
                return key

        key = os.getenv("OPENAICOMPATIBLE_API_KEY")
        if key:
            return key

        return "DUMMY"

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

    @staticmethod
    def _pct_to_ratio(value: float) -> float:
        """Convert a 0-100 percentage to a 0-1 ratio."""
        return round(value / 100, 4)

    def _parse_results(
        self,
        result: GarakScanResult,
        eval_threshold: float,
        art_intents: bool = False,
    ) -> tuple[list[EvaluationResult], float | None, int, dict[str, Any]]:
        """Parse Garak results into EvaluationResult metrics.

        Args:
            result: Scan result with report file paths.
            eval_threshold: Threshold for vulnerability detection (0-1).
            art_intents: When True, use prompt-level intents aggregation
                instead of AVID-based aggregation.

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

        generations, score_rows_by_probe, raw_entries_by_probe = parse_generations_from_report_content(
            report_content, eval_threshold
        )
        aggregated_by_probe = parse_aggregated_from_avid_content(avid_content)
        digest = parse_digest_from_report_content(report_content)

        # Combine results
        combined = combine_parsed_results(
            generations,
            score_rows_by_probe,
            aggregated_by_probe,
            eval_threshold,
            digest,
            art_intents=art_intents,
            raw_entries_by_probe=raw_entries_by_probe,
        )
        overall_summary = combined.get("scores", {}).get("_overall", {}).get("aggregated_results", {})

        overall_asr = overall_summary.get("attack_success_rate")
        if overall_asr is not None:
            try:
                overall_asr = float(overall_asr)
            except (TypeError, ValueError):
                overall_asr = None
        if overall_asr is not None:
            metrics.append(
                EvaluationResult(
                    metric_name="attack_success_rate",
                    metric_value=self._pct_to_ratio(overall_asr),
                    metric_type="ratio",
                    num_samples=overall_summary.get("total_attempts"),
                )
            )

        # Convert to EvaluationResult format (one per probe)
        for probe_name, score_data in combined["scores"].items():
            if probe_name == "_overall":
                continue

            agg = score_data["aggregated_results"]

            attack_success_rate = agg.get("attack_success_rate", 0.0)

            probe_metadata: dict[str, Any] = {"probe": probe_name}
            if art_intents:
                probe_metadata.update(
                    {
                        "total_attempts": agg.get("total_attempts", 0),
                        "unsafe_stubs": agg.get("unsafe_stubs", 0),
                        "safe_stubs": agg.get("safe_stubs", 0),
                        "intent_breakdown": agg.get("intent_breakdown", {}),
                    }
                )
            else:
                probe_attempts = agg.get("total_attempts", agg.get("total_attacks", 0))
                total_attempts += probe_attempts
                probe_metadata.update(
                    {
                        "vulnerable_responses": agg.get("vulnerable_responses", 0),
                        "benign_responses": agg.get("benign_responses", 0),
                        "detector_scores": agg.get("detector_scores", {}),
                        "avid_taxonomy": agg.get("metadata", {}).get("avid_taxonomy", {}),
                    }
                )

            metrics.append(
                EvaluationResult(
                    metric_name=f"{probe_name}_asr",
                    metric_value=self._pct_to_ratio(attack_success_rate),
                    metric_type="ratio",
                    num_samples=probe_attempts if not art_intents else None,
                    metadata=probe_metadata,
                )
            )

        overall_score = overall_summary.get("attack_success_rate")
        if overall_score is not None:
            try:
                overall_score = self._pct_to_ratio(float(overall_score))
            except (TypeError, ValueError):
                overall_score = None

        if overall_score is None and total_attempts > 0:
            total_vulnerable = sum(
                score_data["aggregated_results"].get("vulnerable_responses", 0)
                for probe_name, score_data in combined["scores"].items()
                if probe_name != "_overall"
            )
            overall_score = self._pct_to_ratio((total_vulnerable / total_attempts) * 100)

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


class _GarakCallbacks(DefaultCallbacks):
    """Extends DefaultCallbacks to forward evaluation_metadata artifacts.

    Workaround until the SDK natively forwards evaluation_metadata into
    the status event's ``artifacts`` field (tracked for SDK 3.4).

    Overrides ``report_results`` to merge ``evaluation_metadata["artifacts"]``
    into the single COMPLETED status event alongside OCI refs and metrics.
    """

    def report_results(self, results: JobResults) -> None:
        eval_artifacts = (results.evaluation_metadata or {}).get("artifacts", {})

        if not eval_artifacts:
            super().report_results(results)
            return

        error: str | None = None

        if self.sidecar_url and self._httpx_available and self._http_client:
            try:
                url = f"{self.sidecar_url}{self._events_path_template.format(job_id=self.job_id)}"

                metrics = {}
                for result in results.results:
                    metrics[result.metric_name] = result.metric_value

                status_event: dict[str, Any] = {
                    "id": self.benchmark_id,
                    "benchmark_index": self.benchmark_index,
                    "state": JobStatus.COMPLETED.value,
                    "status": JobStatus.COMPLETED.value,
                    "message": {
                        "message": "Evaluation completed successfully",
                        "message_code": "evaluation_completed",
                    },
                    "metrics": metrics,
                    "completed_at": results.completed_at.isoformat(),
                    "duration_seconds": int(results.duration_seconds),
                }

                if self.provider_id:
                    status_event["provider_id"] = self.provider_id

                if results.mlflow_run_id:
                    status_event["mlflow_run_id"] = results.mlflow_run_id

                artifacts_payload: dict[str, Any] = {}
                if results.oci_artifact:
                    artifacts_payload["oci_reference"] = results.oci_artifact.reference
                    artifacts_payload["oci_digest"] = results.oci_artifact.digest
                artifacts_payload.update(eval_artifacts)
                status_event["artifacts"] = artifacts_payload

                data = {"benchmark_status_event": status_event}
                logger.debug("Events report_results body: %s", data)

                response = self._http_client.post(
                    url,
                    json=data,
                    headers=self._request_headers(),
                    timeout=10.0,
                )
                response.raise_for_status()

                logger.info(
                    "Results reported to evalhub | Metrics: %d | Score: %s | Artifacts: %d",
                    len(metrics),
                    results.overall_score,
                    len(eval_artifacts),
                )

            except self.httpx.HTTPStatusError as e:
                error = f"Failed to send results to evalhub (HTTP {e.response.status_code}): {e}"
                logger.exception(error)
            except Exception as e:
                error = f"Failed to send results to evalhub: {e}"
                logger.exception(error)

        logger.info(
            "Job %s completed | Benchmark: %s | Model: %s | Score: %s | Examples: %s | Duration: %.2fs",
            results.id,
            results.benchmark_id,
            results.model_name,
            results.overall_score,
            results.num_examples_evaluated,
            results.duration_seconds,
        )

        self._signal_termination(error)


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

    callbacks: _GarakCallbacks | None = None
    exit_error: str | None = None
    try:
        job_spec_path = os.getenv("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
        adapter = adapter_cls(job_spec_path=job_spec_path)
        logger.info(f"Loaded job {adapter.job_spec.id}")
        logger.info(f"Benchmark: {adapter.job_spec.benchmark_id}")
        logger.info(f"Model: {adapter.job_spec.model.name}")

        from evalhub.adapter.config import DEFAULT_TERMINATION_FILE_PATH, EvalHubMode
        from evalhub.adapter.oci import DEFAULT_OCI_PROXY_HOST

        callbacks = _GarakCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            benchmark_index=adapter.job_spec.benchmark_index,
            provider_id=adapter.job_spec.provider_id,
            sidecar_url=adapter.job_spec.callback_url,
            insecure=adapter.settings.evalhub_insecure,
            oci_auth_config_path=adapter.settings.oci_auth_config_path,
            oci_insecure=adapter.settings.oci_insecure,
            oci_proxy_host=(DEFAULT_OCI_PROXY_HOST if adapter.settings.mode == EvalHubMode.K8S else None),
            termination_file_path=(DEFAULT_TERMINATION_FILE_PATH if adapter.settings.mode == EvalHubMode.K8S else None),
            mlflow_backend=adapter.settings.mlflow_backend,
        )

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info(f"Job completed successfully: {results.id}")
        logger.info(f"Overall attack success rate (ratio): {results.overall_score}")

        callbacks.report_results(results)
        sys.exit(0)

    except FileNotFoundError as e:
        exit_error = f"Job spec not found: {e}"
        logger.exception(f"{exit_error}. Set EVALHUB_JOB_SPEC_PATH or ensure job spec exists at default location")
        sys.exit(1)
    except ValueError as e:
        exit_error = f"Configuration error: {e}"
        logger.exception(exit_error)
        sys.exit(1)
    except Exception as e:
        exit_error = f"Job failed: {e}"
        logger.exception(exit_error)
        sys.exit(1)
    finally:
        if callbacks:
            callbacks._signal_termination(exit_error)


if __name__ == "__main__":
    main()
