from kfp import dsl, kubernetes
from typing import NamedTuple, List, Dict

@dsl.component(
    # base_image='quay.io/modh/runtime-images:runtime-cuda-pytorch-ubi9-python-3.11-20250813',
    base_image='quay.io/spandraj/trustyai-garak-provider:latest',
    packages_to_install=['boto3']
)
def garak_scan(
    command: List[str],
    llama_stack_url: str,
    max_retries: int,
    job_id: str,
    ) -> NamedTuple('outputs', [('exit_code', int), ('success', bool), ('file_id_mapping', Dict[str, str])]):
    import subprocess
    import time
    import os
    import json
    import boto3
    from pathlib import Path
    from llama_stack_client import LlamaStackClient

    client = LlamaStackClient(base_url=llama_stack_url)

    # create scan directory in current working directory
    scan_dir:Path = Path(os.getcwd()) / 'scan_files'
    scan_dir.mkdir(exist_ok=True, parents=True)

    scan_log_file: Path = scan_dir / "scan.log"
    scan_log_file.touch(exist_ok=True)
    scan_report_prefix: Path = scan_dir / "scan"
    
    command = command + ['--report_prefix', str(scan_report_prefix)]
    env = os.environ.copy()
    env["GARAK_LOG_FILE"] = str(scan_log_file)

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            
            # upload all files to llama stack
            file_id_mapping:Dict[str, str] = {}
            for file_path in scan_dir.glob('*'):
                if file_path.is_file():
                    try:
                        with open(file_path, 'rb') as f:
                            uploaded_file = client.files.create(file=f, purpose='assistants')
                            file_id_mapping[uploaded_file.filename] = uploaded_file.id
                            # print(f"Uploaded {uploaded_file.filename} -> {uploaded_file.id}")
                    except Exception as e:
                        print(f"Failed to upload {file_path.name}: {e}")
            
            # print(f"File mapping: {file_id_mapping}")

            # Configure S3 client with endpoint URL for MinIO
            s3_endpoint = os.environ.get('AWS_S3_ENDPOINT', '')
            
            if s3_endpoint:
                s3_client = boto3.client(
                    's3',
                    endpoint_url=s3_endpoint,
                    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
                    use_ssl=s3_endpoint.startswith('https'),
                    verify=False  # Set to True if valid certs exist
                )
            else:
                s3_client = boto3.client('s3')
            
            s3_client.put_object(
                Bucket=os.getenv('AWS_S3_BUCKET', 'pipeline-artifacts'),
                Key=f"{job_id}.json",
                Body=json.dumps(file_id_mapping)
            )
            
            return (result.returncode, result.returncode == 0, file_id_mapping)

        except subprocess.CalledProcessError as e:
            print(f"Error output: {e.stderr}")
            print(f"Attempt {attempt + 1} failed with exit code {e.returncode}")
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retry
            else:
                raise
    
    # Should never reach here
    raise RuntimeError("Command failed after all retries")


@dsl.pipeline()
def garak_scan_pipeline(
    command: List[str],
    llama_stack_url: str,
    job_id: str,
    max_retries: int = 3
    ) -> NamedTuple('PipelineOutputs', [
        ('exit_code', int), 
        ('success', bool), 
        ('file_id_mapping', Dict[str, str])
    ]):
    scan_task = garak_scan(
        command=command,
        llama_stack_url=llama_stack_url,
        max_retries=max_retries,
        job_id=job_id,
    )

    kubernetes.use_secret_as_env(
        scan_task,
        secret_name="aws-connection-pipeline-artifacts",
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            "AWS_S3_BUCKET": "AWS_S3_BUCKET",  # if bucket name is in secret
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",  # if using MinIO/custom endpoint
        },
    )

    
    from collections import namedtuple
    PipelineOutputs = namedtuple('PipelineOutputs', ['exit_code', 'success', 'file_id_mapping'])
    
    return PipelineOutputs(
        exit_code=scan_task.outputs['exit_code'],
        success=scan_task.outputs['success'],
        file_id_mapping=scan_task.outputs['file_id_mapping']
    )