"""Eval-hub-specific KFP pipeline for running Garak scans.

This module provides a Kubeflow Pipeline that executes a Garak scan inside
a KFP pod, uploading all output to S3 via a Data Connection secret. The
eval-hub adapter submits this pipeline, polls for completion, and downloads
results from the same S3 bucket.

Artifact transfer uses an OpenShift AI "Data Connection" -- a Kubernetes
Secret with standard S3 credential keys. The secret is injected into the
KFP pods via ``kubernetes.use_secret_as_env``, and into the eval-hub Job
pod via ``envFrom`` (configured by the eval-hub service).

The pipeline has six steps:
1. **validate** -- pre-flight checks (garak install, config, S3, model).
2. **resolve_taxonomy** -- resolve the taxonomy input (custom from S3 or
   default BASE_TAXONOMY).
3. **sdg_generate** -- run SDG on the taxonomy to produce raw prompts,
   or emit an empty marker for non-intents / bypass-sdg scans.
4. **prepare_prompts** -- fetch bypass data from S3 as raw dataset (if applicable),
   upload raw output (either from SDG or bypass data) to S3, normalise, upload normalised output to S3,
   and emit the normalised prompts for garak.
5. **garak_scan** -- run the scan and upload results to S3.
6. **write_kfp_outputs** -- parse results from S3 and write KFP Metrics /
   HTML artifacts so results are visible in the KFP dashboard.
"""

import logging
import os
from dataclasses import dataclass
from typing import NamedTuple

from kfp import dsl, kubernetes

from ..constants import DEFAULT_SDG_FLOW_ID

logger = logging.getLogger(__name__)

DEFAULT_KFP_EXPERIMENT = "evalhub-garak"
DEFAULT_POLL_INTERVAL = 30
DEFAULT_S3_PREFIX = "evalhub-garak-kfp"

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
    s3_endpoint: str = ""
    experiment_name: str = DEFAULT_KFP_EXPERIMENT
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL
    s3_prefix: str = DEFAULT_S3_PREFIX
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
            s3_endpoint=_resolve("s3_endpoint", "AWS_S3_ENDPOINT"),
            experiment_name=_resolve(
                "experiment_name", "EVALHUB_KFP_EXPERIMENT", DEFAULT_KFP_EXPERIMENT,
            ),
            poll_interval_seconds=int(
                _resolve("poll_interval_seconds", "EVALHUB_KFP_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL))
            ),
            s3_prefix=_resolve(
                "s3_prefix", "EVALHUB_KFP_S3_PREFIX", DEFAULT_S3_PREFIX,
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
# KFP Components
# ---------------------------------------------------------------------------

@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def validate(
    config_json: str,
) -> NamedTuple("Outputs", [("valid", bool)]):
    """Pre-flight validation before running the scan pipeline.

    Checks:
    - garak is importable (installed in the base image).
    - ``config_json`` is valid JSON with required sections.
    - S3 credentials are present and the bucket is reachable.

    Raises ``GarakValidationError`` on any hard failure, which
    short-circuits all downstream tasks.
    """
    import json
    import logging
    import os
    from collections import namedtuple

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("validate")

    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    errors: list[str] = []

    # 1. garak installed
    try:
        import garak  # noqa: F401
        log.info("garak is installed")
    except ImportError:
        errors.append("garak is not installed in the base image")

    # 2. config_json parseable with required sections
    try:
        config = json.loads(config_json)
        if "plugins" not in config:
            errors.append("config_json missing required 'plugins' section")
        log.info("config_json is valid JSON")
    except json.JSONDecodeError as exc:
        errors.append(f"config_json is not valid JSON: {exc}")

    # 3. S3 credentials and bucket reachable
    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
    if not s3_bucket:
        errors.append("AWS_S3_BUCKET env var is not set")
    else:
        try:
            from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

            s3 = create_s3_client()
            s3.head_bucket(Bucket=s3_bucket)
            log.info("S3 bucket '%s' is reachable", s3_bucket)
        except Exception as exc:
            errors.append(f"S3 bucket '{s3_bucket}' is not reachable: {exc}")
    if errors:
        raise GarakValidationError(
            "Pre-flight validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    log.info("All pre-flight checks passed")
    Outputs = namedtuple("Outputs", ["valid"])
    return Outputs(valid=True)


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def resolve_taxonomy(
    taxonomy_dataset: dsl.Output[dsl.Dataset],
    policy_s3_key: str = "",
    policy_format: str = "csv"
):
    """Resolve the taxonomy input for SDG.

    If ``policy_s3_key`` is provided, the referenced S3 object is
    downloaded and validated via ``load_taxonomy_dataset()``.  Otherwise
    the built-in ``BASE_TAXONOMY`` is emitted.

    The output artifact is always a CSV-formatted taxonomy DataFrame.
    """
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("resolve_taxonomy")

    import pandas as pd
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    if policy_s3_key and policy_s3_key.strip():
        from llama_stack_provider_trustyai_garak.intents import load_taxonomy_dataset
        from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

        if policy_s3_key.startswith("s3://"):
            parts = policy_s3_key[len("s3://"):].split("/", 1)
            s3_bucket = parts[0]
            s3_key = parts[1] if len(parts) > 1 else ""
            if not s3_key:
                raise GarakValidationError(
                    f"Invalid policy_s3_key '{policy_s3_key}': "
                    "expected s3://bucket/key format"
                )
        else:
            s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
            s3_key = policy_s3_key

        if not s3_bucket:
            raise GarakValidationError(
                "Cannot determine S3 bucket for policy taxonomy. "
                "Provide a full s3://bucket/key URI in policy_s3_key, "
                "or set AWS_S3_BUCKET."
            )

        log.info(
            "Fetching policy taxonomy from S3 (bucket=%s, key=%s, format=%s)",
            s3_bucket, s3_key, policy_format,
        )

        s3 = create_s3_client()
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        file_content = response["Body"].read().decode("utf-8")

        taxonomy = load_taxonomy_dataset(content=file_content, format=policy_format)
        log.info(
            "Loaded custom taxonomy: %d entries, columns: %s",
            len(taxonomy), list(taxonomy.columns),
        )
    else:
        from llama_stack_provider_trustyai_garak.sdg import BASE_TAXONOMY

        taxonomy = pd.DataFrame(BASE_TAXONOMY)
        log.info("Using BASE_TAXONOMY: %d entries", len(taxonomy))

    taxonomy.to_csv(taxonomy_dataset.path, index=False)


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def sdg_generate(
    art_intents: bool,
    intents_s3_key: str,
    sdg_model: str,
    sdg_api_base: str,
    sdg_api_key: str,
    sdg_flow_id: str,
    taxonomy_dataset: dsl.Input[dsl.Dataset],
    sdg_dataset: dsl.Output[dsl.Dataset],
):
    """Run Synthetic Data Generation on a taxonomy to produce raw prompts.

    Two modes:

    * ``art_intents=False`` or ``intents_s3_key`` set -- writes an empty
      marker (native probes or bypass; downstream ``prepare_prompts``
      handles the bypass fetch).
    * Otherwise -- run SDG on the taxonomy artifact, emit the raw
      (un-normalised) output including all pool columns.

    Only depends on taxonomy + SDG model params so KFP caching works
    across multiple garak runs with identical inputs.
    """
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("sdg_generate")

    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    if not art_intents:
        log.info("Non-intents scan — writing empty dataset marker")
        with open(sdg_dataset.path, "w") as f:
            f.write("")
        return

    if intents_s3_key and intents_s3_key.strip():
        log.info("Bypass mode (intents_s3_key set) — SDG not needed, writing empty marker")
        with open(sdg_dataset.path, "w") as f:
            f.write("")
        return

    # --- SDG path ---
    if not sdg_model or not sdg_model.strip():
        raise GarakValidationError(
            "sdg_model is required for intents scans when intents_s3_key "
            "is not provided (SDG must run)."
        )
    if not sdg_api_base or not sdg_api_base.strip():
        raise GarakValidationError(
            "sdg_api_base is required for intents scans when intents_s3_key "
            "is not provided (SDG must run)."
        )

    import pandas as pd
    from llama_stack_provider_trustyai_garak.sdg import generate_sdg_dataset
    from llama_stack_provider_trustyai_garak.constants import DEFAULT_SDG_FLOW_ID as _DEFAULT_FLOW

    if sdg_api_key and sdg_api_key.strip():
        os.environ["OPENAI_API_KEY"] = sdg_api_key

    effective_flow_id = sdg_flow_id.strip() if sdg_flow_id else ""
    if not effective_flow_id:
        effective_flow_id = _DEFAULT_FLOW

    taxonomy = pd.read_csv(taxonomy_dataset.path)
    log.info(
        "Read taxonomy from artifact: %d entries, columns: %s",
        len(taxonomy), list(taxonomy.columns),
    )

    log.info(
        "Running SDG: model=%s, api_base=%s, flow=%s",
        sdg_model, sdg_api_base, effective_flow_id,
    )
    sdg_result = generate_sdg_dataset(
        model=sdg_model,
        api_base=sdg_api_base,
        flow_id=effective_flow_id,
        api_key=sdg_api_key if sdg_api_key and sdg_api_key.strip() else "dummy",
        taxonomy=taxonomy,
    )

    sdg_result.raw.to_csv(sdg_dataset.path, index=False)
    log.info(
        "SDG produced %d raw rows across %d categories",
        len(sdg_result.raw), sdg_result.raw["policy_concept"].nunique(),
    )


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def prepare_prompts(
    art_intents: bool,
    s3_prefix: str,
    intents_s3_key: str,
    intents_format: str,
    sdg_dataset: dsl.Input[dsl.Dataset],
    prompts_dataset: dsl.Output[dsl.Dataset],
):
    """Prepare normalised prompts for the Garak scan.

    Handles three scenarios:

    * **Non-intents** (``art_intents=False``): passes through an empty
      marker — garak uses its native probes.
    * **Bypass SDG** (``intents_s3_key`` set): fetches the user's
      pre-generated dataset from S3, uploads the original file as the
      raw artifact, normalises it via :func:`load_intents_dataset`,
      uploads the normalised version, and outputs it for garak.
    * **SDG ran** (``sdg_dataset`` non-empty): uploads the raw SDG
      output, normalises it, uploads the normalised version, and
      outputs it for garak.

    Raw and normalised CSVs are persisted to the S3 job folder
    *before* the scan starts, so they are available for review while
    the scan is still running.
    """
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("prepare_prompts")

    if not art_intents:
        log.info("Non-intents scan — passing through empty marker")
        with open(prompts_dataset.path, "w") as f:
            f.write("")
        return

    from llama_stack_provider_trustyai_garak.intents import load_intents_dataset
    from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")
    if not s3_bucket:
        raise GarakValidationError(
            "AWS_S3_BUCKET is required. "
            "Ensure the Data Connection secret is configured."
        )
    if not s3_prefix:
        raise GarakValidationError(
            "s3_prefix is required. "
            "Ensure the s3_prefix is configured in the benchmark_config."
        )

    s3 = create_s3_client()

    raw_content: str | None = None

    # --- Bypass SDG: fetch the user's pre-generated dataset from S3 ---
    if intents_s3_key and intents_s3_key.strip():
        if intents_s3_key.startswith("s3://"):
            parts = intents_s3_key[len("s3://"):].split("/", 1)
            fetch_bucket = parts[0]
            fetch_key = parts[1] if len(parts) > 1 else ""
            if not fetch_key:
                raise GarakValidationError(
                    f"Invalid intents_s3_key '{intents_s3_key}': "
                    "expected s3://bucket/key format"
                )
        else:
            fetch_bucket = s3_bucket
            fetch_key = intents_s3_key

        log.info(
            "Bypass mode — fetching pre-generated intents from S3 "
            "(bucket=%s, key=%s, format=%s)",
            fetch_bucket, fetch_key, intents_format,
        )
        response = s3.get_object(Bucket=fetch_bucket, Key=fetch_key)
        raw_content = response["Body"].read().decode("utf-8")

    # --- SDG ran: read the raw artifact ---
    elif os.path.getsize(sdg_dataset.path) > 0:
        with open(sdg_dataset.path) as f:
            raw_content = f.read()
        log.info("Read raw SDG output from artifact (%d bytes)", len(raw_content))
    else:
        log.warning("No SDG output and no intents_s3_key — writing empty marker")
        with open(prompts_dataset.path, "w") as f:
            f.write("")
        return

    # Upload raw (user-provided or SDG output) to S3 job folder
    raw_key = f"{s3_prefix}/sdg_raw_output.csv"
    s3.put_object(Bucket=s3_bucket, Key=raw_key, Body=raw_content.encode("utf-8"))
    log.info("Uploaded raw output to s3://%s/%s", s3_bucket, raw_key)

    # Normalise
    fmt = intents_format if (intents_s3_key and intents_s3_key.strip()) else "csv"
    normalized = load_intents_dataset(content=raw_content, format=fmt)
    normalized.to_csv(prompts_dataset.path, index=False)
    log.info(
        "Normalised %d prompts across %d categories",
        len(normalized), normalized["category"].nunique(),
    )

    # Upload normalised to S3 job folder
    norm_key = f"{s3_prefix}/sdg_normalized_output.csv"
    s3.upload_file(prompts_dataset.path, s3_bucket, norm_key)
    log.info("Uploaded normalised output to s3://%s/%s", s3_bucket, norm_key)


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def garak_scan(
    config_json: str,
    s3_prefix: str,
    timeout_seconds: int,
    prompts_dataset: dsl.Input[dsl.Dataset],
) -> NamedTuple("Outputs", [("success", bool), ("return_code", int)]):
    """Run a Garak scan and upload output to S3 via Data Connection credentials.

    If the ``prompts_dataset`` artifact contains data, intent stubs are
    generated from it before launching garak. S3 credentials are injected
    as environment variables by the pipeline using
    ``kubernetes.use_secret_as_env`` from a Data Connection secret.
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

    if os.path.getsize(prompts_dataset.path) > 0:
        import pandas as pd
        from llama_stack_provider_trustyai_garak.intents import generate_intents_from_dataset

        df = pd.read_csv(prompts_dataset.path)
        if not df.empty:
            desc_col = "description" if "description" in df.columns else None
            generate_intents_from_dataset(
                df,
                category_description_column_name=desc_col,
            )
            log.info(
                "Generated intent stubs for %d prompts across %d categories",
                len(df), df["category"].nunique(),
            )
        else:
            log.info("Policy dataset artifact is empty DataFrame — skipping intent generation")

    log.info("Starting garak scan (timeout=%ss, output=%s)", timeout_seconds, scan_dir)

    result = run_garak_scan(
        config_file=config_file,
        timeout_seconds=timeout_seconds,
        report_prefix=report_prefix,
        log_file=log_file,
    )

    if result.success:
        log.info("Garak scan completed successfully (rc=%s)", result.returncode)
        try:
            convert_to_avid_report(result.report_jsonl)
        except Exception as e:
            log.warning("AVID conversion failed: %s", e)
    else:
        error_msg = f"Garak scan failed (rc={result.returncode}, timed_out={result.timed_out}): {result.stderr[:500] if result.stderr else 'no stderr'}"
        log.error(error_msg)
        raise GarakError(error_msg)


    # Upload results to S3
    s3_bucket = os.environ.get("AWS_S3_BUCKET", "")

    if s3_bucket and s3_prefix:
        try:
            from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

            s3 = create_s3_client()

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


@dsl.component(
    install_kfp_package=False,
    packages_to_install=[],
    base_image=_resolve_base_image(),
)
def write_kfp_outputs(
    s3_prefix: str,
    eval_threshold: float,
    art_intents: bool,
    summary_metrics: dsl.Output[dsl.Metrics],
    html_report: dsl.Output[dsl.HTML],
):
    """Download scan results from S3, parse them, write KFP artifacts,
    and upload the intents HTML report to the S3 job folder.

    This is a best-effort component: the adapter pod performs the
    authoritative parse. This component exists so that key metrics and
    an HTML report are visible directly in the KFP dashboard UI.
    """
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("write_kfp_outputs")

    try:
        from llama_stack_provider_trustyai_garak.evalhub.s3_utils import create_s3_client

        s3_bucket = os.environ.get("AWS_S3_BUCKET", "")

        if not s3_bucket:
            log.warning("AWS_S3_BUCKET not set; skipping KFP output generation")
            return

        s3 = create_s3_client()

        def _download_text(key: str) -> str:
            try:
                resp = s3.get_object(Bucket=s3_bucket, Key=key)
                return resp["Body"].read().decode("utf-8")
            except Exception:
                return ""

        report_key = f"{s3_prefix}/scan.report.jsonl"
        avid_key = f"{s3_prefix}/scan.avid.jsonl"

        report_content = _download_text(report_key)
        if not report_content.strip():
            log.warning("Report file empty or not found at s3://%s/%s", s3_bucket, report_key)
            return

        avid_content = _download_text(avid_key)

        # Parse using shared utilities
        from llama_stack_provider_trustyai_garak.result_utils import (
            combine_parsed_results,
            generate_art_report,
            parse_aggregated_from_avid_content,
            parse_digest_from_report_content,
            parse_generations_from_report_content,
        )

        generations, score_rows_by_probe, raw_entries_by_probe = (
            parse_generations_from_report_content(report_content, eval_threshold)
        )
        aggregated_by_probe = parse_aggregated_from_avid_content(avid_content)
        digest = parse_digest_from_report_content(report_content)

        combined = combine_parsed_results(
            generations,
            score_rows_by_probe,
            aggregated_by_probe,
            eval_threshold,
            digest,
            art_intents=art_intents,
            raw_entries_by_probe=raw_entries_by_probe,
        )

        # Log metrics
        overall = (
            combined.get("scores", {})
            .get("_overall", {})
            .get("aggregated_results", {})
        )

        if art_intents:
            summary_metrics.log_metric(
                "attack_success_rate",
                overall.get("attack_success_rate", 0),
            )
        else:
            summary_metrics.log_metric(
                "total_attempts",
                overall.get("total_attempts", 0),
            )
            summary_metrics.log_metric(
                "vulnerable_responses",
                overall.get("vulnerable_responses", 0),
            )
            summary_metrics.log_metric(
                "attack_success_rate",
                overall.get("attack_success_rate", 0),
            )

        if overall.get("tbsa"):
            summary_metrics.log_metric("tbsa", overall["tbsa"])

        log.info("Logged metrics to KFP: %s", {k: v for k, v in overall.items() if k in ("attack_success_rate", "total_attempts", "tbsa")})

        # Generate HTML report
        html_content = None
        if art_intents:
            try:
                html_content = generate_art_report(report_content)
                log.info("Generated ART intents HTML report")
            except Exception as exc:
                log.warning("Failed to generate ART HTML report: %s", exc)
        else:
            html_key = f"{s3_prefix}/scan.report.html"
            html_content = _download_text(html_key)
            if html_content:
                log.info("Downloaded native probe HTML report from S3")
            else:
                log.warning("No native probe HTML report found at s3://%s/%s", s3_bucket, html_key)

        if html_content:
            with open(html_report.path, "w") as f:
                f.write(html_content)

            # Upload intents HTML to S3 job folder
            if art_intents and s3_prefix:
                try:
                    s3.upload_file(
                        html_report.path, s3_bucket,
                        f"{s3_prefix}/scan.intents.html",
                    )
                    log.info(
                        "Uploaded intents HTML report to s3://%s/%s/scan.intents.html",
                        s3_bucket, s3_prefix,
                    )
                except Exception as exc:
                    log.warning("Failed to upload intents HTML to S3: %s", exc)
        else:
            with open(html_report.path, "w") as f:
                f.write("<html><body><p>No HTML report available.</p></body></html>")

    except Exception as exc:
        log.warning("KFP output generation failed (non-fatal): %s", exc)
        with open(html_report.path, "w") as f:
            f.write(f"<html><body><p>Report generation failed: {exc}</p></body></html>")


# ---------------------------------------------------------------------------
# KFP Pipeline
# ---------------------------------------------------------------------------

@dsl.pipeline(name="evalhub-garak-scan")
def evalhub_garak_pipeline(
    config_json: str,
    s3_prefix: str,
    timeout_seconds: int,
    s3_secret_name: str,
    eval_threshold: float = 0.5,
    art_intents: bool = False,
    policy_s3_key: str = "",
    policy_format: str = "csv",
    intents_s3_key: str = "",
    intents_format: str = "csv",
    sdg_model: str = "",
    sdg_api_base: str = "",
    sdg_api_key: str = "",
    sdg_flow_id: str = DEFAULT_SDG_FLOW_ID,
):
    """Six-step pipeline: validate, resolve taxonomy, SDG, prepare prompts, scan, write outputs.

    ``s3_secret_name`` is required -- the Data Connection secret is injected
    as environment variables into all pipeline pods.

    Three intents modes:
    - Default taxonomy + SDG: no file params, requires ``sdg_model``/``sdg_api_base``.
    - Custom taxonomy + SDG: ``policy_s3_key`` set, requires ``sdg_model``/``sdg_api_base``.
    - Bypass SDG: ``intents_s3_key`` set, SDG params NOT required.

    ``policy_s3_key`` and ``intents_s3_key`` are mutually exclusive.
    """
    # Step 1: Pre-flight validation
    validate_task = validate(
        config_json=config_json,
    )
    validate_task.set_caching_options(False)

    kubernetes.use_secret_as_env(
        validate_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 2: Resolve taxonomy (custom from S3 or BASE_TAXONOMY)
    taxonomy_task = resolve_taxonomy(
        policy_s3_key=policy_s3_key,
        policy_format=policy_format,
    )
    taxonomy_task.set_caching_options(True)
    taxonomy_task.after(validate_task)

    kubernetes.use_secret_as_env(
        taxonomy_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 3: Run SDG (or no-op for non-intents / bypass)
    sdg_task = sdg_generate(
        art_intents=art_intents,
        intents_s3_key=intents_s3_key,
        sdg_model=sdg_model,
        sdg_api_base=sdg_api_base,
        sdg_api_key=sdg_api_key,
        sdg_flow_id=sdg_flow_id,
        taxonomy_dataset=taxonomy_task.outputs["taxonomy_dataset"],
    )
    sdg_task.set_caching_options(True)

    kubernetes.use_secret_as_env(
        sdg_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 4: Prepare prompts (normalise, persist raw+normalised to S3)
    prep_task = prepare_prompts(
        art_intents=art_intents,
        s3_prefix=s3_prefix,
        intents_s3_key=intents_s3_key,
        intents_format=intents_format,
        sdg_dataset=sdg_task.outputs["sdg_dataset"],
    )
    prep_task.set_caching_options(False)

    kubernetes.use_secret_as_env(
        prep_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 5: Run garak scan
    scan_task = garak_scan(
        config_json=config_json,
        s3_prefix=s3_prefix,
        timeout_seconds=timeout_seconds,
        prompts_dataset=prep_task.outputs["prompts_dataset"],
    )
    scan_task.set_caching_options(False)
    scan_task.after(prep_task)

    kubernetes.use_secret_as_env(
        scan_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )

    # Step 6: Write KFP Metrics + HTML to dashboard, upload intents HTML to S3
    outputs_task = write_kfp_outputs(
        s3_prefix=s3_prefix,
        eval_threshold=eval_threshold,
        art_intents=art_intents,
    )
    outputs_task.set_caching_options(False)
    outputs_task.after(scan_task)

    kubernetes.use_secret_as_env(
        outputs_task,
        secret_name=s3_secret_name,
        secret_key_to_env=S3_DATA_CONNECTION_KEYS,
    )
