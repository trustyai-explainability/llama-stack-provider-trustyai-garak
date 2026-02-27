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
    # Check environment variable first (highest priority)
    if (base_image := os.environ.get("KUBEFLOW_GARAK_BASE_IMAGE")) is not None:
        return base_image

    # Try to load from kubernetes ConfigMap
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
            # None of the candidate namespaces had the required ConfigMap/key
            logger.debug(
                f"ConfigMap '{GARAK_PROVIDER_IMAGE_CONFIGMAP_NAME}' with key "
                f"'{GARAK_PROVIDER_IMAGE_CONFIGMAP_KEY}' not found in any of the namespaces: "
                f"{KUBEFLOW_CANDIDATE_NAMESPACES}. Using default image."
            )
            return DEFAULT_GARAK_PROVIDER_IMAGE
    except Exception as e:
        # Kubernetes config not available (e.g., in tests or non-k8s environment)
        logger.debug(f"Kubernetes config not available: {e}. Using default image.")
        return DEFAULT_GARAK_PROVIDER_IMAGE

# Component 1: Validation Step
@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,  # All dependencies pre-installed in base image
    packages_to_install=[]  # No additional packages needed
)
def validate_inputs(
    command: str,
    llama_stack_url: str,
    verify_ssl: str
) -> NamedTuple('outputs', [
    ('is_valid', bool),
    ('validation_errors', List[str])
]):
    """Validate inputs before running expensive scan"""

    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError
    import json

    validation_errors = []
    
    # Validate Llama Stack connectivity
    try:
        client = LlamaStackClient(base_url=llama_stack_url,
                                  http_client=get_http_client_with_tls(verify_ssl))
        # Test connection
        client.files.list(limit=1)
    except Exception as e:
        validation_errors.append(f"Cannot connect to Llama Stack: {str(e)}")
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
    
    # Check garak is installed
    try:
        import garak
    except ImportError as e:
        validation_errors.append(f"Garak is not installed. Please install it using 'pip install garak': {e}")
        raise e

    # Validate command
    try:
        _ = json.loads(command)
    except json.JSONDecodeError as e:
        validation_errors.append(f"Invalid command: {e}")
        raise e
    
    # Check for dangerous flags
    dangerous_flags = ['--rm', '--force', '--no-limit']
    for flag in dangerous_flags:
        if flag in command:
            validation_errors.append(f"Dangerous flag detected: {flag}")
    
    if len(validation_errors) > 0:
        raise GarakValidationError("\n".join(validation_errors))
    
    return (
        len(validation_errors) == 0,
        validation_errors
    )

# Component 2: Resolve Intents Dataset
@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,
    packages_to_install=[]
)
def resolve_intents_dataset(
    art_intents: bool,
    intents_file_id: str,
    intents_format: str,
    llama_stack_url: str,
    category_column: str,
    prompt_column: str,
    description_column: str,
    verify_ssl: str,
    intents_dataset: dsl.Output[dsl.Dataset],
):
    """Resolve intents dataset from the Files API, or write an empty marker for non-intents scans.

    For art_intents=True with a file ID: fetches the user-uploaded dataset,
    validates required columns, normalizes to (category, prompt, description),
    and writes the CSV to the output artifact.

    For art_intents=False: writes an empty file (native probes need no dataset).

    When SDG (Synthetic Data Generation) is available, the art_intents=True
    path without a file ID will invoke the SDG component instead of raising.
    """
    import logging
    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
    from llama_stack_provider_trustyai_garak.intents import load_intents_dataset

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not art_intents:
        logger.info("Non-intents scan — writing empty dataset marker")
        with open(intents_dataset.path, 'w') as f:
            f.write("")
        return

    if not intents_file_id:
        raise NotImplementedError(
            "Synthetic Data Generation (SDG) for intents is not yet available. "
            "Please upload an intents dataset via the Llama Stack Files API and "
            "pass the file ID as 'intents_file_id' in the benchmark metadata."
        )

    logger.info(f"Fetching intents dataset (file_id={intents_file_id}, format={intents_format})")
    client = LlamaStackClient(
        base_url=llama_stack_url,
        http_client=get_http_client_with_tls(verify_ssl),
    )

    try:
        file_content = client.files.content(intents_file_id)
        if isinstance(file_content, bytes):
            file_content = file_content.decode("utf-8")
        file_content = str(file_content)

        normalized = load_intents_dataset(
            content=file_content,
            format=intents_format,
            category_column=category_column,
            prompt_column=prompt_column,
            description_column=description_column or None,
        )

        normalized.to_csv(intents_dataset.path, index=False)
        logger.info(
            f"Resolved {len(normalized)} intent prompts across "
            f"{normalized['category'].nunique()} categories"
        )
    finally:
        try:
            client.close()
        except Exception:
            pass


# Component 3: Garak Scan
@dsl.component(
    base_image=get_base_image(),
    install_kfp_package=False,
    packages_to_install=[]
)
def garak_scan(
    command: str,
    llama_stack_url: str,
    job_id: str,
    max_retries: int,
    timeout_seconds: int,
    verify_ssl: str,
    intents_dataset: dsl.Input[dsl.Dataset],
) -> NamedTuple('outputs', [
    ('exit_code', int),
    ('success', bool),
    ('file_id_mapping', Dict[str, str])
]):
    """Run a Garak scan. If the intents_dataset artifact contains data,
    intent stubs are generated from it before launching garak."""
    import subprocess
    import time
    import os
    import signal
    import json
    from llama_stack_client import LlamaStackClient
    import logging
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls, get_scan_base_dir
    from llama_stack_provider_trustyai_garak.intents import generate_intents_from_dataset
    import pandas as pd

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    scan_dir = get_scan_base_dir()
    scan_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Scan directory: {scan_dir}")

    scan_log_file = scan_dir / f"{job_id}_scan.log"
    scan_report_prefix = scan_dir / f"{job_id}_scan"

    scan_cmd_config_file = scan_dir / f"config.json"
    scan_cmd_config = json.loads(command)
    scan_cmd_config['reporting']['report_prefix'] = str(scan_report_prefix)

    if os.path.getsize(intents_dataset.path) > 0:
        df = pd.read_csv(intents_dataset.path)
        if not df.empty:
            generate_intents_from_dataset(df)
            logger.info(
                f"Generated intent stubs for {len(df)} prompts "
                f"across {df['category'].nunique()} categories"
            )
        else:
            logger.info("Intents dataset artifact is empty — skipping intent generation")

    with open(scan_cmd_config_file, 'w') as f:
        json.dump(scan_cmd_config, f)
    
    command = ['garak', '--config', str(scan_cmd_config_file)]
    env = os.environ.copy()
    env["GARAK_LOG_FILE"] = str(scan_log_file)
    
    file_id_mapping = {}
    client = None
    
    ## TODO: why not use dsl.PipelineTask.set_retry()..?
    for attempt in range(max_retries):
        
        try:
            logger.info(f"Starting Garak scan (attempt {attempt + 1}/{max_retries})")

            process = subprocess.Popen(
                command,
                stdout=None,  # (show progress in KFP pod logs)
                stderr=None,  # (show progress in KFP pod logs)
                env=env,
                preexec_fn=os.setsid  # Create new process group
            )
            
            try:
                # Wait for completion
                process.wait(timeout=timeout_seconds)
                
            except subprocess.TimeoutExpired:
                logger.error(f"Garak scan timed out after {timeout_seconds} seconds")
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # process is still running, kill it with SIGKILL
                    logger.warning("Process did not terminate gracefully, forcing kill")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                raise
            
            
            if process.returncode != 0:
                logger.error(f"Garak scan failed with exit code {process.returncode}")
                raise subprocess.CalledProcessError(
                    process.returncode, command
                )
            
            logger.info("Garak scan completed successfully")
            
            # Best-effort AVID conversion: works for native probes, skipped
            # gracefully for intents probes (which don't produce eval entries yet)
            report_file = scan_report_prefix.with_suffix(".report.jsonl")
            try:
                from garak.report import Report
                if report_file.exists():
                    report = Report(str(report_file)).load().get_evaluations()
                    report.export()  # this will create a new file - scan_report_prefix.with_suffix(".avid.jsonl")
                    logger.info("Successfully converted report to AVID format")
                else:
                    logger.warning(f"Report file not found: {report_file} — skipping AVID conversion")
            except Exception as e:
                logger.warning(f"AVID conversion skipped (expected for intents probes): {e}")

            # Upload files to llama stack
            client = LlamaStackClient(base_url=llama_stack_url,
                                      http_client=get_http_client_with_tls(verify_ssl))
            
            for file_path in scan_dir.glob('*'):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        uploaded_file = client.files.create(
                            file=f, 
                            purpose='assistants'
                        )
                        file_id_mapping[uploaded_file.filename] = uploaded_file.id

            # Upload raw mapping early so server can find scan files
            # even if the downstream parse_results task fails
            raw_mapping_filename = f"{job_id}_mapping_raw.json"
            raw_mapping_path = scan_dir / raw_mapping_filename
            with open(raw_mapping_path, 'w') as f:
                json.dump(file_id_mapping, f)
            with open(raw_mapping_path, 'rb') as f:
                client.files.create(file=f, purpose='batch')
            logger.info(f"Uploaded raw mapping: {raw_mapping_filename}")

            return (0, True, file_id_mapping)
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(5 * (2 ** attempt), 60)  # Exponential backoff
                logger.error(f"Error: {e}", exc_info=True)
                logger.error(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
        finally:
            if client:
                try:
                    client.close()
                except Exception as e:
                    logger.warning(f"Error closing client: {e}")

# Component 4: Results Parser
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
    html_report: dsl.Output[dsl.HTML]
):
    """Parse Garak scan results using shared result_utils.
    
    Uploads file_id_mapping with predictable filename pattern: {job_id}_mapping.json
    Server retrieves it by searching Files API for this filename.
    """
    import os
    import json
    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.compat import ScoringResult, EvaluateResponse
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError
    from llama_stack_provider_trustyai_garak import result_utils
    import logging
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls, get_scan_base_dir

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Parse verify_ssl string back to bool or keep as path
    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        parsed_verify_ssl = True
    elif verify_ssl.lower() in ("false", "0", "no", "off"):
        parsed_verify_ssl = False
    else:
        # It's a path to a certificate file
        parsed_verify_ssl = verify_ssl

    client = LlamaStackClient(base_url=llama_stack_url,
                              http_client=get_http_client_with_tls(parsed_verify_ssl))
    
    # Get report files
    report_file_id = file_id_mapping.get(f"{job_id}_scan.report.jsonl", "")
    avid_file_id = file_id_mapping.get(f"{job_id}_scan.avid.jsonl", "")
    
    if not report_file_id:
        raise GarakValidationError("No report file found")
    
    def _content_to_str(content) -> str:
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return str(content)

    # Fetch content
    logger.info(f"Fetching report.jsonl (ID: {report_file_id})...")
    report_content = _content_to_str(client.files.content(report_file_id))
    
    avid_content = ""
    if avid_file_id:
        logger.info(f"Fetching avid.jsonl (ID: {avid_file_id})...")
        avid_content = _content_to_str(client.files.content(avid_file_id))
    else:
        logger.warning("No AVID report - will not have taxonomy info")
    
    # Parse using shared utilities
    logger.debug("Parsing generations from report.jsonl...")
    generations, score_rows_by_probe = result_utils.parse_generations_from_report_content(
        report_content, eval_threshold
    )
    logger.info(f"Parsed {len(generations)} attempts")
    
    logger.debug("Parsing aggregated info from AVID report...")
    aggregated_by_probe = result_utils.parse_aggregated_from_avid_content(avid_content)
    logger.info(f"Parsed {len(aggregated_by_probe)} probe summaries")
    
    logger.debug("Parsing digest from report.jsonl...")
    digest = result_utils.parse_digest_from_report_content(report_content)
    logger.info(f"Digest parsed: {bool(digest)}")
    
    logger.debug("Combining results...")
    result_dict = result_utils.combine_parsed_results(
        generations,
        score_rows_by_probe,
        aggregated_by_probe,
        eval_threshold,
        digest
    )
    
    # Convert to EvaluateResponse
    scores_with_scoring_result = {
        probe_name: ScoringResult(
            score_rows=score_data["score_rows"],
            aggregated_results=score_data["aggregated_results"]
        )
        for probe_name, score_data in result_dict["scores"].items()
    }
    
    scan_result = EvaluateResponse(
        generations=result_dict["generations"],
        scores=scores_with_scoring_result
    ).model_dump()
    
    # # Save file using shared XDG-based directory
    logger.info("Saving scan result...")
    scan_dir = get_scan_base_dir()
    scan_dir.mkdir(exist_ok=True, parents=True)
    scan_result_file = scan_dir / f"{job_id}_scan_result.json"
    with open(scan_result_file, 'w') as f:
        json.dump(scan_result, f)
    
    with open(scan_result_file, 'rb') as result_file:
        uploaded_file = client.files.create(
            file=result_file,
            purpose='assistants'
        )
        file_id_mapping[uploaded_file.filename] = uploaded_file.id
    
    logger.info(f"Updated file_id_mapping: {file_id_mapping}")

    # Upload file_id_mapping to Files API with predictable filename
    # Server will retrieve by searching for this filename pattern
    mapping_filename = f"{job_id}_mapping.json"
    mapping_file_path = scan_dir / mapping_filename
    with open(mapping_file_path, 'w') as f:
        json.dump(file_id_mapping, f)
    
    logger.info(f"Uploading file_id_mapping to Files API as '{mapping_filename}'...")
    with open(mapping_file_path, 'rb') as f:
        mapping_file = client.files.create(
            file=f,
            purpose='batch'
        )
        logger.info(f"File mapping uploaded: {mapping_file.filename} (ID: {mapping_file.id})")
    
    # combine scores of all probes into a single metric to log
    scoring_result_overall = scores_with_scoring_result.get("_overall")
    if scoring_result_overall:
        overall_metrics = scoring_result_overall.aggregated_results
    else:
        overall_metrics = {}
    log_metrics = {
        "total_attempts": overall_metrics.get("total_attempts", 0),
        "vulnerable_responses": overall_metrics.get("vulnerable_responses", 0),
        "attack_success_rate": overall_metrics.get("attack_success_rate", 0),
    }
    if overall_metrics.get("tbsa"):
        log_metrics["tbsa"] = overall_metrics["tbsa"]
    if log_metrics["total_attempts"] > 0:
        for key, value in log_metrics.items():
            summary_metrics.log_metric(key, value)

    # HTML report: intents scans generate an ART report from report.jsonl,
    # native probe scans use garak's own HTML report uploaded during the scan step
    html_content = None
    if art_intents:
        logger.info("Generating ART HTML report for garak scan (intents)")
        html_content = result_utils.generate_art_report(report_content)
    else:
        html_report_id = file_id_mapping.get(f"{job_id}_scan.report.html")
        if html_report_id:
            logger.info(f"Generating HTML report for garak scan")
            html_content = _content_to_str(client.files.content(html_report_id))
        else:
            logger.warning("No HTML report ID found in file mapping")
    if html_content:
        with open(html_report.path, 'w') as f:
            f.write(html_content)
    else:
        logger.warning("No HTML content found for native probe report")
    
    if client:
        try:
            client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
