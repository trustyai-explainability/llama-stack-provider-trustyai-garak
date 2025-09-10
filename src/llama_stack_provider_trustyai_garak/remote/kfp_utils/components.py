from kfp import dsl
from typing import NamedTuple, List, Dict
import os

CPU_BASE_IMAGE = 'quay.io/rh-ee-spandraj/trustyai-garak-provider-dsp:cpu'

# Component 1: Validation Step
@dsl.component(
    base_image=os.getenv('KUBEFLOW_BASE_IMAGE', CPU_BASE_IMAGE)
)
def validate_inputs(
    command: List[str],
    llama_stack_url: str
) -> NamedTuple('outputs', [
    ('is_valid', bool),
    ('validation_errors', List[str])
]):
    """Validate inputs before running expensive scan"""

    from llama_stack_client import LlamaStackClient
    
    validation_errors = []
    
    # Validate Llama Stack connectivity
    try:
        client = LlamaStackClient(base_url=llama_stack_url)
        # Test connection
        client.files.list(limit=1)
    except Exception as e:
        validation_errors.append(f"Cannot connect to Llama Stack: {str(e)}")
    
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
    base_image=os.getenv('KUBEFLOW_BASE_IMAGE', CPU_BASE_IMAGE)
)
def garak_scan(
    command: List[str],
    llama_stack_url: str,
    max_retries: int,
    timeout_seconds: int,
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
    from pathlib import Path
    from llama_stack_client import LlamaStackClient
    
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
            
            # Upload files to llama stack
            client = LlamaStackClient(base_url=llama_stack_url)
            
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
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    
    return (-1, False, file_id_mapping)

# Component 3: Results Parser
@dsl.component(
    base_image=os.getenv('KUBEFLOW_BASE_IMAGE', CPU_BASE_IMAGE)
)
def parse_results(
    file_id_mapping: Dict[str, str],
    llama_stack_url: str,
    eval_threshold: float,
    job_id: str,
    verify_ssl: str,
):

    """Parse results and provide analysis"""
    import boto3
    import os
    import json
    from pathlib import Path
    from llama_stack_client import LlamaStackClient
    from llama_stack.apis.scoring import ScoringResult
    from llama_stack.apis.eval import EvaluateResponse
    from llama_stack_provider_trustyai_garak.errors import GarakValidationError
    from typing import List, Dict, Any

    # Parse verify_ssl string back to bool or keep as path
    if verify_ssl.lower() in ("true", "1", "yes", "on"):
        parsed_verify_ssl = True
    elif verify_ssl.lower() in ("false", "0", "no", "off"):
        parsed_verify_ssl = False
    else:
        # It's a path to a certificate file
        parsed_verify_ssl = verify_ssl

    client = LlamaStackClient(base_url=llama_stack_url)
    
    # Get report file
    report_file_id = file_id_mapping.get("scan.report.jsonl", "")
    if not report_file_id:
        raise GarakValidationError("No report file found")
    
    else:
        generations: List[Dict[str, Any]] = []
        # Each key is a garak probe/attack_type and 
        # value is a ScoringResult object containing scores from respective garak detectors
        scores: Dict[str, ScoringResult] = {} 

        score_rows: Dict[str, List[Dict[str, float]]] = {}
        aggregated_results_by_probe: Dict[str, Dict[str, List[float]]] = {}
        report_file_content_str:str = client.files.content(report_file_id)

        for line in report_file_content_str.split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                generation: Dict[str, Any] = {}

                # Handle attempt entries
                if (entry.get("entry_type") == "attempt" and entry.get("status") == 2):
                    probe_name: str = entry.get("probe_classname", "unknown")

                    detector_results: dict = entry.get("detector_results", {})
                    detector_maxscore_map: Dict[str, float] = {}
                    is_vulnerable: bool = False
                    
                    for detector, _scores in detector_results.items():
                        _scores = _scores if isinstance(_scores, list) else [_scores]
                        if any(score >= eval_threshold for score in _scores):
                            is_vulnerable = True
                        detector_maxscore_map[detector] = max(_scores) if _scores else 0
                    outputs: list = entry.get("outputs", [])

                    generation["probe"] = probe_name
                    generation["vulnerable"] = is_vulnerable
                    generation["prompt"] = entry.get("prompt", "")
                    generation["response"] = outputs[0] if len(outputs) > 0 and outputs[0] else ""
                    generations.append(generation)

                    if probe_name not in score_rows:
                        score_rows[probe_name] = []
                    score_rows[probe_name].append(detector_maxscore_map)

                    if probe_name not in aggregated_results_by_probe:
                        aggregated_results_by_probe[probe_name] = {}
                    for detector, score in detector_maxscore_map.items():
                        if detector not in aggregated_results_by_probe[probe_name]:
                            aggregated_results_by_probe[probe_name][detector] = []
                        aggregated_results_by_probe[probe_name][detector].append(score)
            
            except json.JSONDecodeError as e:
                print(f"Invalid JSON line in report file for job {job_id}: {line} - {e}")
                continue
            except Exception as e:
                print(f"Error parsing line in report file for job {job_id}: {line} - {e}")
                continue
            
        # Calculate the mean of the scores for each probe
        aggregated_results_mean: Dict[str, Dict[str, float]] = {}
        for probe_name, results in aggregated_results_by_probe.items():
            aggregated_results_mean[probe_name] = {}
            for detector, _scores in results.items():
                detector_mean_score: float = round(sum(_scores) / len(_scores), 3) if _scores else 0
                aggregated_results_mean[probe_name][f"{detector}_mean"] = detector_mean_score
                
        if len(aggregated_results_mean.keys()) != len(score_rows.keys()):
            raise GarakValidationError(f"Number of probes in aggregated results ({len(aggregated_results_mean.keys())}) "
                                f"does not match number of probes in score rows ({len(score_rows.keys())})")
        
        all_probes: List[str] = list(aggregated_results_mean.keys())
        for probe_name in all_probes:
            scores[probe_name] = ScoringResult(
                score_rows=score_rows[probe_name],
                aggregated_results=aggregated_results_mean[probe_name]
                )

        scan_result = EvaluateResponse(generations=generations, scores=scores).model_dump()

        # save file and upload results to llama stack
        scan_result_file = Path(os.getcwd()) / "scan_result.json"
        with open(scan_result_file, 'w') as f:
            json.dump(scan_result, f)
        
        with open(scan_result_file, 'rb') as result_file:
            uploaded_file = client.files.create(
                file=result_file,
                purpose='assistants'
            )
            file_id_mapping[uploaded_file.filename] = uploaded_file.id
    
    print(f"file_id_mapping: {file_id_mapping}")

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
            print(f"Error creating S3 client: {e}")
            s3_client = boto3.client('s3')
    else:
        s3_client = boto3.client('s3')
    
    if not s3_client:
        raise GarakValidationError("S3 client not found")
    
    s3_client.put_object(
        Bucket=os.getenv('AWS_S3_BUCKET', 'pipeline-artifacts'),
        Key=f"{job_id}.json",
        Body=json.dumps(file_id_mapping)
    )