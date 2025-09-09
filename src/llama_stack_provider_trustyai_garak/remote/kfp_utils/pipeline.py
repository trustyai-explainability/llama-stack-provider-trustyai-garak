from typing import List
from kfp import dsl, kubernetes
from .components import validate_inputs, garak_scan, parse_results

# def add_gpu_toleration(task: dsl.PipelineTask, accelerator_type: str, accelerator_limit: int):
#     print(f"Adding GPU tolerations: {accelerator_type}({accelerator_limit})")
#     task.set_accelerator_type(accelerator=accelerator_type)
#     task.set_accelerator_limit(accelerator_limit)
#     kubernetes.add_toleration(task, key=accelerator_type, operator="Exists", effect="NoSchedule")

# def add_resource_limits(task: dsl.PipelineTask, resource_config: dict):
#     print(f"Adding resource limits: {resource_config}")
#     task.set_cpu_request(str(resource_config.get("cpu_request", "500m")))
#     task.set_cpu_limit(str(resource_config.get("cpu_limit", "4")))
#     task.set_memory_request(str(resource_config.get("memory_request", "2Gi")))
#     task.set_memory_limit(str(resource_config.get("memory_limit", "6Gi")))


@dsl.pipeline()
def garak_scan_pipeline(
    command: List[str],
    llama_stack_url: str,
    job_id: str,
    eval_threshold: float,
    timeout_seconds: int,
    max_retries: int = 3,
    use_gpu: bool = False,
    verify_ssl: str = "True",
    resource_config: dict = {}, # TODO: parameterize gpu and cpu resource limits
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
    validate_task: dsl.PipelineTask = validate_inputs(
        command=command,
        llama_stack_url=llama_stack_url
    )
    validate_task.set_caching_options(False)
    
    # Step 2: Run the garak scan ONLY if validation passes
    with dsl.If(validate_task.outputs['is_valid'] == True, name="validation_passed"):
        
        with dsl.If(use_gpu == True, name="USE_GPU"):
            scan_task_gpu: dsl.PipelineTask = garak_scan(
                command=command,
                llama_stack_url=llama_stack_url,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
            )
            scan_task_gpu.after(validate_task)
            scan_task_gpu.set_caching_options(False)
            scan_task_gpu.set_accelerator_type(accelerator="nvidia.com/gpu")
            scan_task_gpu.set_accelerator_limit(limit=1)
            kubernetes.add_toleration(scan_task_gpu, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")

            # Step 3: Parse the scan results ONLY if scan succeeds
            with dsl.If(scan_task_gpu.outputs['success'] == True, name="gpu_scan_succeeded"):
                parse_task_gpu: dsl.PipelineTask = parse_results(
                    file_id_mapping=scan_task_gpu.outputs['file_id_mapping'],
                    llama_stack_url=llama_stack_url,
                    eval_threshold=eval_threshold,
                    job_id=job_id,
                    verify_ssl=verify_ssl,
                )
                parse_task_gpu.set_caching_options(False)
                
                kubernetes.use_secret_as_env(
                    parse_task_gpu,
                    secret_name=aws_connection_secret,
                    secret_key_to_env=secret_key_to_env,
                )
            # add_gpu_toleration(scan_task_gpu, "nvidia.com/gpu", 1)
            # add_resource_limits(scan_task_gpu, resource_config)
        
        with dsl.Else(name="USE_CPU"):
            scan_task_cpu: dsl.PipelineTask = garak_scan(
                command=command,
                llama_stack_url=llama_stack_url,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
            )
            scan_task_cpu.after(validate_task)
            scan_task_cpu.set_caching_options(False)
            # add_resource_limits(scan_task_cpu, resource_config)

            # # Step 3: Parse the scan results ONLY if scan succeeds
            with dsl.If(scan_task_cpu.outputs['success'] == True, name="cpu_scan_succeeded"):
                parse_task_cpu: dsl.PipelineTask = parse_results(
                    file_id_mapping=scan_task_cpu.outputs['file_id_mapping'],
                    llama_stack_url=llama_stack_url,
                    eval_threshold=eval_threshold,
                    job_id=job_id,
                    verify_ssl=verify_ssl,
                )
                parse_task_cpu.set_caching_options(False)
                
                kubernetes.use_secret_as_env(
                    parse_task_cpu,
                    secret_name=aws_connection_secret,
                    secret_key_to_env=secret_key_to_env,
                )

        
        ## TODO: this is not working as expected
        # merged_file_id_mapping = dsl.OneOf(
        #     scan_task_gpu.outputs['file_id_mapping'], 
        #     scan_task_cpu.outputs['file_id_mapping']
        # )
        # merged_success = dsl.OneOf(
        #     scan_task_gpu.outputs['success'],
        #     scan_task_cpu.outputs['success']
        # )
        
        # with dsl.If(merged_success == True, name="scan_succeeded"):
        #     parse_task:dsl.PipelineTask = parse_results(
        #         file_id_mapping=merged_file_id_mapping,
        #         llama_stack_url=llama_stack_url,
        #         eval_threshold=eval_threshold,
        #         job_id=job_id,
        #     )
        #     parse_task.set_caching_options(False)
            
        #     kubernetes.use_secret_as_env(
        #         parse_task,
        #         secret_name=aws_connection_secret,
        #         secret_key_to_env=secret_key_to_env,
        #     )
