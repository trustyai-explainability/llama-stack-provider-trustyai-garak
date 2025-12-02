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
    if (base_image := os.environ.get("KUBEFLOW_BASE_IMAGE")) is not None:
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
    base_image=get_base_image()
)
def validate_inputs(
    command: List[str],
    llama_stack_url: str,
    verify_ssl: str
) -> NamedTuple('outputs', [
    ('is_valid', bool),
    ('validation_errors', List[str])
]):
    """Validate inputs before running expensive scan"""

    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
    
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

    # Validate command structure
    if not command or command[0] != 'garak':
        validation_errors.append("Invalid command: must start with 'garak'")
    
    # Check for dangerous flags
    dangerous_flags = ['--rm', '--force', '--no-limit']
    for flag in dangerous_flags:
        if flag in command:
            validation_errors.append(f"Dangerous flag detected: {flag}")
    
    return (
        len(validation_errors) == 0,
        validation_errors
    )

# Component 2: Garak Scan
@dsl.component(
    base_image=get_base_image()
)
def garak_scan(
    command: List[str],
    llama_stack_url: str,
    max_retries: int,
    timeout_seconds: int,
    verify_ssl: str
) -> NamedTuple('outputs', [
    ('exit_code', int),
    ('success', bool),
    ('file_id_mapping', Dict[str, str])
]):
    """Actual Garak Scan"""
    import subprocess
    import time
    import os
    import signal
    import json
    from pathlib import Path
    from llama_stack_client import LlamaStackClient
    import logging
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Setup directories
    scan_dir = Path(os.getcwd()) / 'scan_files'
    scan_dir.mkdir(exist_ok=True, parents=True)
    
    scan_log_file = scan_dir / "scan.log"
    scan_report_prefix = scan_dir / "scan"
    
    command = command + ['--report_prefix', str(scan_report_prefix)]
    env = os.environ.copy()
    env["GARAK_LOG_FILE"] = str(scan_log_file)
    
    file_id_mapping = {}
    
    ## TODO: why not use dsl.PipelineTask.set_retry()..?
    for attempt in range(max_retries):
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                preexec_fn=os.setsid  # Create new process group
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                
            except subprocess.TimeoutExpired:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # process is still running, kill it with SIGKILL
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                raise
            
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, command, stdout, stderr
                )
            
            # create avid report file
            report_file = scan_report_prefix.with_suffix(".report.jsonl")
            try:
                from llama_stack_provider_trustyai_garak.avid_report import Report
                
                if not report_file.exists():
                    logger.error(f"Report file not found: {report_file}")
                else:
                    report = Report(str(report_file)).load().get_evaluations()
                    report.export()  # this will create a new file - scan_report_prefix.with_suffix(".avid.jsonl")
                    logger.info("Successfully converted report to AVID format")
                    
            except FileNotFoundError as e:
                logger.error(f"Report file not found during AVID conversion: {e}")
            except PermissionError as e:
                logger.error(f"Permission denied reading report file: {e}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse report file: {e}", exc_info=True)
            except ImportError as e:
                logger.error(f"Failed to import AVID report module: {e}")
            except Exception as e:
                logger.error(f"Unexpected error converting report to AVID format: {e}", exc_info=True)

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
                            
            
            return (0, True, file_id_mapping)
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(5 * (2 ** attempt), 60)  # Exponential backoff
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

    return (-1, False, file_id_mapping)

# Component 3: Results Parser
@dsl.component(
    base_image=get_base_image()
)
def parse_results(
    file_id_mapping: Dict[str, str],
    llama_stack_url: str,
    eval_threshold: float,
    job_id: str,
    verify_ssl: str,
    s3_bucket: str,
    s3_prefix: str,
    html_report: dsl.Output[dsl.HTML]
):
    """Parse Garak scan results using shared result_utils."""
    import boto3
    from botocore.exceptions import ClientError
    import os
    import json
    from pathlib import Path
    from llama_stack_client import LlamaStackClient
    from llama_stack_provider_trustyai_garak.compat import ScoringResult, EvaluateResponse
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError
    from llama_stack_provider_trustyai_garak import result_utils
    import logging
    from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls

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
    report_file_id = file_id_mapping.get("scan.report.jsonl", "")
    avid_file_id = file_id_mapping.get("scan.avid.jsonl", "")
    
    if not report_file_id:
        raise GarakValidationError("No report file found")
    
    # Fetch content
    logger.info(f"Fetching report.jsonl (ID: {report_file_id})...")
    report_content = client.files.content(report_file_id)
    
    avid_content = ""
    if avid_file_id:
        logger.info(f"Fetching avid.jsonl (ID: {avid_file_id})...")
        avid_content = client.files.content(avid_file_id)
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
    
    logger.debug("Combining results...")
    result_dict = result_utils.combine_parsed_results(
        generations,
        score_rows_by_probe,
        aggregated_by_probe,
        eval_threshold
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
    
    # Save file and upload results to llama stack
    logger.info("Saving scan result...")
    scan_result_file = Path(os.getcwd()) / "scan_result.json"
    with open(scan_result_file, 'w') as f:
        json.dump(scan_result, f)
    
    with open(scan_result_file, 'rb') as result_file:
        uploaded_file = client.files.create(
            file=result_file,
            purpose='assistants'
        )
        file_id_mapping[uploaded_file.filename] = uploaded_file.id
    
    logger.info(f"Updated file_id_mapping: {file_id_mapping}")

    # Configure S3 client with endpoint URL for MinIO
    s3_endpoint = os.environ.get('AWS_S3_ENDPOINT', '')
    
    if s3_endpoint:
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=s3_endpoint,
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
                use_ssl=s3_endpoint.startswith('https'),
                verify=parsed_verify_ssl,
            )
        except Exception as e:
            logger.error(f"Error creating S3 client: {e}")
            s3_client = boto3.client('s3')
    else:
        s3_client = boto3.client('s3')
    
    # Check if bucket exists
    try:
        s3_client.head_bucket(Bucket=s3_bucket)
        logger.info(f"Bucket '{s3_bucket}' exists. Proceeding with upload.")
    except ClientError as e:
        # If a ClientError is returned, the bucket either doesn't exist or lack permission
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404':
            logger.info(f"Bucket '{s3_bucket}' not found. Creating bucket now...")
            s3_client.create_bucket(Bucket=s3_bucket, CreateBucketConfiguration={'LocationConstraint': s3_client.meta.region_name})
            logger.info("Bucket created. Proceeding with upload.")
        else:
            if error_code == '403':
                logger.error(f"Permission denied for bucket '{s3_bucket}'. Check IAM policies.")
            raise e
    except Exception as e:
        logger.error(f"Unexpected error checking bucket '{s3_bucket}': {e}")
        raise e

    # Construct S3 key with prefix
    s3_prefix = s3_prefix.rstrip("/")
    s3_key = f"{s3_prefix}/{job_id}.json" if s3_prefix else f"{job_id}.json"
    
    # Upload file mapping to S3
    try:
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=json.dumps(file_id_mapping)
        )
        logger.info(f"Successfully uploaded file mapping to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        logger.error(f"Failed to upload file mapping to S3: {e}")
        raise GarakValidationError(f"Failed to upload results to S3: {e}") from e

    html_report_id = file_id_mapping.get("scan.report.html")
    if html_report_id:
        html_content = client.files.content(html_report_id)
        if html_content:
            with open(html_report.path, 'w') as f:
                f.write(html_content)
        else:
            logger.warning("No HTML content found")
    else:
        logger.warning("No HTML report ID found")
    
    if client:
        try:
            client.close()
        except Exception as e:
            logger.warning(f"Error closing client: {e}")
