"""Llama Stack KFP pipeline components.

Six-step pipeline mirroring the EvalHub KFP pipeline, using the
Llama Stack Files API for I/O and shared ``core.pipeline_steps``
for business logic.

Steps: validate -> resolve_taxonomy -> sdg_generate -> prepare_prompts
       -> garak_scan -> parse_results
"""

from kfp import dsl
from typing import NamedTuple, List, Dict
import os
from .utils import _load_kube_config
from ...constants import (
    GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME,
    GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY,
    KUBEFLOW_CANDIDATE_NAMESPACES,
    DEFAULT_GARAK_PROVIDER_IMAGE
    )
from kubernetes import client
from kubernetes.client.exceptions import ApiException
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_base_image() -> str:
    """Get base image from env, fallback to k8s ConfigMap, fallback to default image.

    This function is called at module import time, so it must handle cases where
    kubernetes config is not available (e.g., in tests or non-k8s environments).
    """
    if (base_image := os.environ.get("KUBEFLOW_GARAK_BASE_IMAGE")) is not None:
        return base_image

    try:
        _load_kube_config()
        api = client.CoreV1Api()

        for candidate_namespace in KUBEFLOW_CANDIDATE_NAMESPACES:
            try:
                configmap = api.read_namespaced_config_map(
                    name=GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME,
                    namespace=candidate_namespace,
                )

                data: dict[str, str] | None = configmap.data
                if data and GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY in data:
                    return data[GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY]
            except ApiException as api_exc:
                if api_exc.status == 404:
                    continue
                else:
                    logger.warning(f"Warning: Could not read from ConfigMap: {api_exc}")
            except Exception as e:
                logger.warning(f"Warning: Could not read from ConfigMap: {e}")
        else:
            logger.debug(
                f"ConfigMap '{GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME}' with key "
                f"'{GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY}' not found in any of the namespaces: "
                f"{KUBEFLOW_CANDIDATE_NAMESPACES}. Using default image."
            )
            return DEFAULT_GARAK_PROVIDER_IMAGE
    except Exception as e:
        logger.debug(f"Kubernetes config not available: {e}. Using default image.")
        return DEFAULT_GARAK_PROVIDER_IMAGE


# ---------------------------------------------------------------------------
# Helper: parse verify_ssl string from pipeline param
# ---------------------------------------------------------------------------

def _parse_verify_ssl(verify_ssl: str):
    """Convert pipeline string param to bool or cert path."""
    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        return True
    if verify_ssl.lower() in ("false", "0", "no", "off"):
        return False
    return verify_ssl


# ---------------------------------------------------------------------------
# Component 1: Validation
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,  # All dependencies pre-installed in base image
    packages_to_install=[]  # No additional packages needed
)
def validate(
    command: str,
    llama_stack_url: str,
    verify_ssl: str,
) -> NamedTuple("Outputs", [("valid", bool)]):
    """Pre-flight validation: garak, config JSON, flags, Llama Stack connectivity."""
    import logging
    from collections import namedtuple

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("validate")

    from llama_stack_provider_trustyai_garak.core.pipeline_steps import validate_scan_config
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError

    validate_scan_config(command)
    log.info("Core config validation passed")

    # Llama Stack connectivity check
    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls

    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        parsed_ssl = True
    elif verify_ssl.lower() in ("false", "0", "no", "off"):
        parsed_ssl = False
    else:
        parsed_ssl = verify_ssl

    try:
        ls_client = LlamaStackClient(
            base_url=llama_stack_url,
            http_client=get_http_client_with_tls(parsed_ssl),
        )
        ls_client.files.list(limit=1)
        log.info("Llama Stack Files API is reachable")
    except Exception as exc:
        raise GarakValidationError(
            f"Cannot connect to Llama Stack at {llama_stack_url}: {exc}"
        ) from exc
    finally:
        try:
            ls_client.close()
        except Exception:
            pass

    log.info("All pre-flight checks passed")
    Outputs = namedtuple("Outputs", ["valid"])
    return Outputs(valid=True)


# ---------------------------------------------------------------------------
# Component 2: Resolve taxonomy
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,
    packages_to_install=[],
)
def resolve_taxonomy(
    art_intents: bool,
    policy_file_id: str,
    policy_format: str,
    llama_stack_url: str,
    verify_ssl: str,
    taxonomy_dataset: dsl.Output[dsl.Dataset],
):
    """Resolve taxonomy: custom from Files API or built-in BASE_TAXONOMY."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("resolve_taxonomy")

    from llama_stack_provider_trustyai_garak.core.pipeline_steps import resolve_taxonomy_data

    if not art_intents:
        log.info("Non-intents scan — writing empty taxonomy marker")
        with open(taxonomy_dataset.path, "w") as f:
            f.write("")
        return

    policy_content = None
    if policy_file_id and policy_file_id.strip():
        from llama_stack_client import LlamaStackClient
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls

        if verify_ssl.lower() in ("true", "1", "yes", "on"):
            parsed_ssl = True
        elif verify_ssl.lower() in ("false", "0", "no", "off"):
            parsed_ssl = False
        else:
            parsed_ssl = verify_ssl

        log.info("Fetching policy taxonomy (file_id=%s)", policy_file_id)
        ls_client = LlamaStackClient(
            base_url=llama_stack_url,
            http_client=get_http_client_with_tls(parsed_ssl),
        )
        try:
            raw = ls_client.files.content(policy_file_id)
            policy_content = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
        finally:
            try:
                ls_client.close()
            except Exception:
                pass

    taxonomy_df = resolve_taxonomy_data(policy_content, format=policy_format)
    taxonomy_df.to_csv(taxonomy_dataset.path, index=False)
    log.info("Wrote taxonomy (%d entries) to artifact", len(taxonomy_df))


# ---------------------------------------------------------------------------
# Component 3: SDG generation
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,
    packages_to_install=[],
)
def sdg_generate(
    art_intents: bool,
    intents_file_id: str,
    sdg_model: str,
    sdg_api_base: str,
    sdg_flow_id: str,
    taxonomy_dataset: dsl.Input[dsl.Dataset],
    sdg_dataset: dsl.Output[dsl.Dataset],
):
    """Run SDG on taxonomy to produce raw prompts, or emit empty marker.

    No API key parameter — Llama Stack models handle their own auth
    at the server level.  ``run_sdg_generation`` falls back to
    ``resolve_api_key("sdg")`` which returns ``"DUMMY"`` when no
    Secret is injected.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("sdg_generate")

    if not art_intents:
        log.info("Non-intents scan — writing empty dataset marker")
        with open(sdg_dataset.path, "w") as f:
            f.write("")
        return

    if intents_file_id and intents_file_id.strip():
        log.info("Bypass mode (intents_file_id set) — SDG not needed")
        with open(sdg_dataset.path, "w") as f:
            f.write("")
        return

    import pandas as pd
    from llama_stack_provider_trustyai_garak.core.pipeline_steps import run_sdg_generation

    taxonomy = pd.read_csv(taxonomy_dataset.path)
    log.info("Read taxonomy: %d entries", len(taxonomy))

    raw_df = run_sdg_generation(
        taxonomy_df=taxonomy,
        sdg_model=sdg_model,
        sdg_api_base=sdg_api_base,
        sdg_flow_id=sdg_flow_id,
    )
    raw_df.to_csv(sdg_dataset.path, index=False)
    log.info("Wrote %d raw SDG rows to artifact", len(raw_df))


# ---------------------------------------------------------------------------
# Component 4: Prepare prompts (normalise + persist artifacts)
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,
    packages_to_install=[],
)
def prepare_prompts(
    art_intents: bool,
    job_id: str,
    intents_file_id: str,
    intents_format: str,
    llama_stack_url: str,
    verify_ssl: str,
    sdg_dataset: dsl.Input[dsl.Dataset],
    prompts_dataset: dsl.Output[dsl.Dataset],
):
    """Normalise raw prompts and upload raw/normalised artifacts to Files API."""
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

    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
    from llama_stack_provider_trustyai_garak.core.pipeline_steps import normalize_prompts

    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        parsed_ssl = True
    elif verify_ssl.lower() in ("false", "0", "no", "off"):
        parsed_ssl = False
    else:
        parsed_ssl = verify_ssl

    ls_client = LlamaStackClient(
        base_url=llama_stack_url,
        http_client=get_http_client_with_tls(parsed_ssl),
    )

    try:
        raw_content: str | None = None

        # Bypass SDG: fetch user's pre-generated dataset from Files API
        if intents_file_id and intents_file_id.strip():
            log.info("Bypass mode — fetching intents dataset (file_id=%s)", intents_file_id)
            raw_bytes = ls_client.files.content(intents_file_id)
            raw_content = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else str(raw_bytes)

        # SDG ran: read the raw artifact
        elif os.path.getsize(sdg_dataset.path) > 0:
            with open(sdg_dataset.path) as f:
                raw_content = f.read()
            log.info("Read raw SDG output from artifact (%d bytes)", len(raw_content))
        else:
            log.warning("No SDG output and no intents_file_id — writing empty marker")
            with open(prompts_dataset.path, "w") as f:
                f.write("")
            return

        # Upload raw dataset
        raw_filename = f"{job_id}_sdg_raw_output.csv"
        uploaded = ls_client.files.create(
            file=(raw_filename, raw_content.encode("utf-8")),
            purpose="assistants",
        )
        log.info("Uploaded raw output: %s (ID: %s)", raw_filename, uploaded.id)

        # Normalise
        fmt = intents_format if (intents_file_id and intents_file_id.strip()) else "csv"
        normalized_df = normalize_prompts(raw_content, format=fmt)
        normalized_df.to_csv(prompts_dataset.path, index=False)

        # Upload normalised dataset
        norm_filename = f"{job_id}_sdg_normalized_output.csv"
        with open(prompts_dataset.path, "rb") as nf:
            uploaded_norm = ls_client.files.create(
                file=(norm_filename, nf.read()),
                purpose="assistants",
            )
        log.info("Uploaded normalised output: %s (ID: %s)", norm_filename, uploaded_norm.id)

    finally:
        try:
            ls_client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Component 5: Garak scan
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,
    packages_to_install=[],
)
def garak_scan(
    command: str,
    llama_stack_url: str,
    job_id: str,
    timeout_seconds: int,
    verify_ssl: str,
    prompts_dataset: dsl.Input[dsl.Dataset],
) -> NamedTuple("Outputs", [
    ("exit_code", int),
    ("success", bool),
    ("file_id_mapping", Dict[str, str]),
]):
    """Run Garak scan via core runner and upload outputs to Files API."""
    import json
    import logging
    import tempfile
    from collections import namedtuple
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("garak_scan")

    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
    from llama_stack_provider_trustyai_garak.core.pipeline_steps import (
        setup_and_run_garak,
        redact_api_keys,
    )

    scan_dir = Path(tempfile.mkdtemp(prefix="garak-scan-"))

    prompts_path = Path(prompts_dataset.path)
    has_prompts = prompts_path.exists() and prompts_path.stat().st_size > 0

    result = setup_and_run_garak(
        config_json=command,
        prompts_csv_path=prompts_path if has_prompts else None,
        scan_dir=scan_dir,
        timeout_seconds=timeout_seconds,
    )

    # Redact api_key values from config.json before uploading to Files API
    config_file = scan_dir / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            config_file.write_text(json.dumps(redact_api_keys(cfg), indent=1))
            log.info("Redacted api_key values from config.json before upload")
        except Exception as exc:
            log.warning("Could not redact config.json: %s", exc)

    # Upload scan outputs to Files API
    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        parsed_ssl = True
    elif verify_ssl.lower() in ("false", "0", "no", "off"):
        parsed_ssl = False
    else:
        parsed_ssl = verify_ssl

    ls_client = LlamaStackClient(
        base_url=llama_stack_url,
        http_client=get_http_client_with_tls(parsed_ssl),
    )

    file_id_mapping: dict[str, str] = {}
    try:
        for file_path in scan_dir.rglob("*"):
            if file_path.is_file():
                upload_name = f"{job_id}_{file_path.name}" if not file_path.name.startswith(job_id) else file_path.name
                with open(file_path, "rb") as f:
                    uploaded = ls_client.files.create(
                        file=(upload_name, f.read()),
                        purpose="assistants",
                    )
                    file_id_mapping[uploaded.filename] = uploaded.id

        # Upload raw mapping early as safety net
        raw_mapping_filename = f"{job_id}_mapping_raw.json"
        raw_mapping_content = json.dumps(file_id_mapping).encode("utf-8")
        ls_client.files.create(
            file=(raw_mapping_filename, raw_mapping_content),
            purpose="batch",
        )
        log.info("Uploaded raw mapping: %s (%d files)", raw_mapping_filename, len(file_id_mapping))

    finally:
        try:
            ls_client.close()
        except Exception:
            pass

    Outputs = namedtuple("Outputs", ["exit_code", "success", "file_id_mapping"])
    return Outputs(
        exit_code=result.returncode,
        success=result.success,
        file_id_mapping=file_id_mapping,
    )


# ---------------------------------------------------------------------------
# Component 6: Parse results
# ---------------------------------------------------------------------------

@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,  # All dependencies pre-installed in base image
    packages_to_install=[]  # No additional packages needed
)
def parse_results(
    file_id_mapping: Dict[str, str],
    llama_stack_url: str,
    eval_threshold: float,
    job_id: str,
    verify_ssl: str,
    art_intents: bool,
    summary_metrics: dsl.Output[dsl.Metrics],
    html_report: dsl.Output[dsl.HTML],
):
    """Parse scan results, build EvaluateResponse, log KFP metrics, upload artifacts."""
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("parse_results")

    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.compat import ScoringResult, EvaluateResponse
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls, get_scan_base_dir
    from llama_stack_provider_trustyai_garak.core.pipeline_steps import (
        parse_and_build_results,
        log_kfp_metrics,
    )
    from llama_stack_provider_trustyai_garak import result_utils

    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        parsed_ssl = True
    elif verify_ssl.lower() in ("false", "0", "no", "off"):
        parsed_ssl = False
    else:
        parsed_ssl = verify_ssl

    ls_client = LlamaStackClient(
        base_url=llama_stack_url,
        http_client=get_http_client_with_tls(parsed_ssl),
    )

    def _content_to_str(content) -> str:
        return content.decode("utf-8") if isinstance(content, bytes) else str(content)

    try:
        # Fetch report files
        report_file_id = file_id_mapping.get(f"{job_id}_scan.report.jsonl", "")
        avid_file_id = file_id_mapping.get(f"{job_id}_scan.avid.jsonl", "")

        if not report_file_id:
            raise GarakValidationError("No report file found in file_id_mapping")

        log.info("Fetching report.jsonl (ID: %s)", report_file_id)
        report_content = _content_to_str(ls_client.files.content(report_file_id))

        avid_content = ""
        if avid_file_id:
            log.info("Fetching avid.jsonl (ID: %s)", avid_file_id)
            avid_content = _content_to_str(ls_client.files.content(avid_file_id))

        # Parse using core function
        result_dict = parse_and_build_results(
            report_content=report_content,
            avid_content=avid_content,
            art_intents=art_intents,
            eval_threshold=eval_threshold,
        )

        # Build EvaluateResponse
        scores_with_sr = {
            probe_name: ScoringResult(
                score_rows=score_data["score_rows"],
                aggregated_results=score_data["aggregated_results"],
            )
            for probe_name, score_data in result_dict["scores"].items()
        }
        scan_result = EvaluateResponse(
            generations=result_dict["generations"],
            scores=scores_with_sr,
        ).model_dump()

        # Save and upload EvaluateResponse
        scan_dir = get_scan_base_dir()
        scan_dir.mkdir(exist_ok=True, parents=True)

        result_filename = f"{job_id}_scan_result.json"
        result_path = scan_dir / result_filename
        result_path.write_text(json.dumps(scan_result))

        with open(result_path, "rb") as rf:
            uploaded = ls_client.files.create(
                file=(result_filename, rf.read()),
                purpose="assistants",
            )
            file_id_mapping[uploaded.filename] = uploaded.id
        log.info("Uploaded scan result: %s (ID: %s)", result_filename, uploaded.id)

        # Upload intents HTML if applicable
        if art_intents:
            html_filename = f"{job_id}_scan.intents.html"
            html_content = result_utils.generate_art_report(report_content)
            uploaded_html = ls_client.files.create(
                file=(html_filename, html_content.encode("utf-8")),
                purpose="assistants",
            )
            file_id_mapping[uploaded_html.filename] = uploaded_html.id
            log.info("Uploaded intents HTML: %s (ID: %s)", html_filename, uploaded_html.id)

        # Upload final file_id_mapping
        mapping_filename = f"{job_id}_mapping.json"
        mapping_content = json.dumps(file_id_mapping).encode("utf-8")
        mapping_uploaded = ls_client.files.create(
            file=(mapping_filename, mapping_content),
            purpose="batch",
        )
        log.info("Uploaded mapping: %s (ID: %s)", mapping_filename, mapping_uploaded.id)

        # Log KFP metrics
        log_kfp_metrics(summary_metrics, result_dict, art_intents)

        # Write HTML report to KFP artifact
        html_output_content = None
        if art_intents:
            html_report_id = file_id_mapping.get(f"{job_id}_scan.intents.html")
            if html_report_id:
                html_output_content = _content_to_str(ls_client.files.content(html_report_id))
        else:
            html_report_id = file_id_mapping.get(f"{job_id}_scan.report.html")
            if html_report_id:
                html_output_content = _content_to_str(ls_client.files.content(html_report_id))

        if html_output_content:
            with open(html_report.path, "w") as hf:
                hf.write(html_output_content)
        else:
            with open(html_report.path, "w") as hf:
                hf.write("<html><body><p>No HTML report available.</p></body></html>")

    finally:
        try:
            ls_client.close()
        except Exception:
            pass
