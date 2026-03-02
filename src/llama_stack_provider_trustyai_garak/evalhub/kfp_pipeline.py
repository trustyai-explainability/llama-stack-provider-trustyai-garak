"""Eval-hub-specific KFP pipeline for running Garak scans.

This module provides a lightweight Kubeflow Pipeline that executes a Garak
scan inside a KFP pod, uploading all output to S3 via a Data Connection
secret. The eval-hub adapter submits this pipeline, polls for completion,
and downloads results from the same S3 bucket.

Artifact transfer uses an OpenShift AI "Data Connection"
-- a Kubernetes Secret with standard S3 credential keys. The secret is
injected into the KFP pod via ``kubernetes.use_secret_as_env``, and into
the eval-hub Job pod via ``envFrom`` (configured by the eval-hub service).

The pipeline has a single component for now (track 1 - native probes).
Multi-component pipelines (validate -> resolve dataset -> scan -> post-process)
will be added in track 2 for intents/SDG support.
"""

import logging
import os
from dataclasses import dataclass
from typing import NamedTuple

from kfp import dsl, kubernetes

logger = logging.getLogger(__name__)

DEFAULT_KFP_EXPERIMENT = "evalhub-garak"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_S3_PREFIX = "evalhub-garak"

S3_DATA_CONNECTION_KEYS = {
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
    "AWS_S3_BUCKET": "AWS_S3_BUCKET",
    "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
}


@dataclass
class KFPConfig:
    """Configuration for connecting to and running pipelines on KFP.

    Resolved from environment variables with benchmark_config overrides.
    """

    endpoint: str
    namespace: str
    auth_token: str = ""
    s3_secret_name: str = ""
    s3_bucket: str = ""
    experiment_name: str = DEFAULT_KFP_EXPERIMENT
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL
    base_image: str = ""
    verify_ssl: bool = True
    ssl_ca_cert: str = ""

    @classmethod
    def from_env_and_config(cls, benchmark_config: dict | None = None) -> "KFPConfig":
        """Build KFPConfig from env vars, with benchmark_config overrides.

        Env var precedence (lowest to highest):
        1. Dataclass defaults
        2. Environment variables (EVALHUB_KFP_*)
        3. benchmark_config dict keys
        """
        bc = benchmark_config or {}
        kfp_overrides = bc.get("kfp_config", {})
        if not isinstance(kfp_overrides, dict):
            kfp_overrides = {}

        def _resolve(key: str, env_var: str, default: str = "") -> str:
            return str(kfp_overrides.get(key, os.getenv(env_var, default)))

        endpoint = _resolve("endpoint", "EVALHUB_KFP_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "KFP endpoint is required. Set EVALHUB_KFP_ENDPOINT or "
                "provide kfp_config.endpoint in benchmark_config."
            )

        namespace = _resolve("namespace", "EVALHUB_KFP_NAMESPACE")
        if not namespace:
            raise ValueError(
                "KFP namespace is required. Set EVALHUB_KFP_NAMESPACE or "
                "provide kfp_config.namespace in benchmark_config."
            )

        verify_ssl_raw = _resolve("verify_ssl", "EVALHUB_KFP_VERIFY_SSL", "true")
        verify_ssl = verify_ssl_raw.lower() not in ("false", "0", "no", "off")

        return cls(
            endpoint=endpoint,
            namespace=namespace,
            auth_token=_resolve("auth_token", "EVALHUB_KFP_AUTH_TOKEN"),
            s3_secret_name=_resolve("s3_secret_name", "EVALHUB_KFP_S3_SECRET_NAME"),
            s3_bucket=_resolve("s3_bucket", "AWS_S3_BUCKET"),
            experiment_name=_resolve(
                "experiment_name", "EVALHUB_KFP_EXPERIMENT", DEFAULT_KFP_EXPERIMENT,
            ),
            poll_interval_seconds=int(
                _resolve("poll_interval_seconds", "EVALHUB_KFP_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))
            ),
            base_image=_resolve("base_image", "KUBEFLOW_GARAK_BASE_IMAGE"),
            verify_ssl=verify_ssl,
            ssl_ca_cert=_resolve("ssl_ca_cert", "EVALHUB_KFP_SSL_CA_CERT"),
        )


def _resolve_base_image(kfp_config: KFPConfig | None = None) -> str:
    """Resolve the container image for KFP components.

    Priority: KFPConfig.base_image > env var > k8s ConfigMap > default.
    """
    if kfp_config and kfp_config.base_image:
        return kfp_config.base_image

    from ..remote.kfp_utils.components import get_base_image
    return get_base_image()


# ---------------------------------------------------------------------------
# KFP Component
# ---------------------------------------------------------------------------

@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def evalhub_garak_scan(
    config_json: str,
    s3_prefix: str,
    timeout_seconds: int,
) -> NamedTuple("Outputs", [("success", bool), ("return_code", int)]):
    """Run a Garak scan and upload output to S3 via Data Connection credentials.

    S3 credentials are injected as environment variables by the pipeline
    using ``kubernetes.use_secret_as_env`` from a Data Connection secret.
    """
    import json
    import logging
    import os
    import tempfile
    from collections import namedtuple
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("evalhub_garak_scan")

    from llama_stack_provider_trustyai_garak.core.garak_runner import (
        convert_to_avid_report,
        run_garak_scan,
    )
    from llama_stack_provider_trustyai_garak.errors import GarakError

    scan_dir = Path(tempfile.mkdtemp(prefix="garak-scan-"))

    config: dict = json.loads(config_json)
    config.setdefault("reporting", {})["report_prefix"] = str(scan_dir / "scan")
    config_file = scan_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=1))

    report_prefix = scan_dir / "scan"
    log_file = scan_dir / "scan.log"

    log.info("Starting garak scan (timeout=%ss, output=%s)", timeout_seconds, scan_dir)

    result = run_garak_scan(
        config_file=config_file,
        timeout_seconds=timeout_seconds,
        report_prefix=report_prefix,
        log_file=log_file,
    )

    if result.success:
        log.info("Garak scan completed successfully (rc=%s)", result.returncode)
        convert_to_avid_report(result.report_jsonl)
    else:
        error_msg = f"Garak scan failed (rc={result.returncode}, timed_out={result.timed_out}): {result.stderr[:500] if result.stderr else 'no stderr'}"
        log.error(error_msg)
        raise GarakError(error_msg)


    # Upload results to S3
    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
    s3_endpoint = os.environ.get("AWS_S3_ENDPOINT", "")

    if s3_bucket and s3_prefix:
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            client_kwargs = {
                "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            }
            if s3_endpoint:
                client_kwargs["endpoint_url"] = s3_endpoint
                client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

            s3 = boto3.client("s3", **client_kwargs)

            uploaded = 0
            for file_path in scan_dir.rglob("*"):
                if file_path.is_file():
                    key = f"{s3_prefix}/{file_path.relative_to(scan_dir)}"
                    s3.upload_file(str(file_path), s3_bucket, key)
                    uploaded += 1

            log.info("Uploaded %d files to s3://%s/%s", uploaded, s3_bucket, s3_prefix)
        except Exception as e:
            error_msg = f"Failed to upload results to S3: {e}"
            log.error(error_msg)
            raise GarakError(error_msg)
    else:
        error_msg = f"S3 not configured (bucket={s3_bucket}, prefix={s3_prefix}); results not uploaded"
        log.error(error_msg)
        raise GarakError(error_msg)


    Outputs = namedtuple("Outputs", ["success", "return_code"])
    return Outputs(success=result.success, return_code=result.returncode)


# ---------------------------------------------------------------------------
# KFP Pipeline
# ---------------------------------------------------------------------------

@dsl.pipeline(name="evalhub-garak-scan")
def evalhub_garak_pipeline(
    config_json: str,
    s3_prefix: str,
    timeout_seconds: int,
    s3_secret_name: str,
):
    """Single-component pipeline that runs a Garak scan with S3 artifact transfer.

    ``s3_secret_name`` is required, the Data Connection secret is injected
    as environment variables into the scan pod, giving it access to S3 for
    uploading results.
    """
    scan_task = evalhub_garak_scan(
        config_json=config_json,
        s3_prefix=s3_prefix,
        timeout_seconds=timeout_seconds,
    )
    scan_task.set_caching_options(False)

    kubernetes.use_secret_as_env(
        scan_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )
