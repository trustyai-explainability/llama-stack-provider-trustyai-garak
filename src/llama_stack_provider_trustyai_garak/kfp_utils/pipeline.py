from typing import List
from kfp import dsl, kubernetes
from .components import validate_inputs, garak_scan, parse_results


@dsl.pipeline()
def garak_scan_pipeline(
    command: List[str],
    llama_stack_url: str,
    job_id: str,
    eval_threshold: float,
    timeout_seconds: int,
    max_retries: int = 3
    ):

    # AWS connection secret
    aws_connection_secret = "aws-connection-pipeline-artifacts"
    secret_key_to_env = {
        "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        "AWS_S3_BUCKET": "AWS_S3_BUCKET",
        "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
    }

    # Step 1: Validate inputs
    validate_task = validate_inputs(
        command=command,
        llama_stack_url=llama_stack_url
    )
    
    # Step 2: Run the garak scan (only if inputs are valid)
    with dsl.If(validate_task.outputs['is_valid'] == True, name="validate_inputs_success"):
        scan_task = garak_scan(
            command=command,
            llama_stack_url=llama_stack_url,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

        kubernetes.use_secret_as_env(
            scan_task,
            secret_name=aws_connection_secret,
            secret_key_to_env=secret_key_to_env,
        )

        # Step 3: Parse the scan results (only if scan succeeded)
        with dsl.If(scan_task.outputs['success'] == True, name="scan_success"):
            parse_task = parse_results(
                file_id_mapping=scan_task.outputs['file_id_mapping'],
                llama_stack_url=llama_stack_url,
                eval_threshold=eval_threshold,
                job_id=job_id,
                )
        
            kubernetes.use_secret_as_env(
                parse_task,
                secret_name=aws_connection_secret,
                secret_key_to_env=secret_key_to_env,
            )
