from typing import List
from kfp import dsl, kubernetes
from .components import validate_inputs, garak_scan, parse_results
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dsl.pipeline()
def garak_scan_pipeline(
    command: str,
    llama_stack_url: str,
    job_id: str,
    eval_threshold: float,
    timeout_seconds: int,
    max_retries: int = 3,
    use_gpu: bool = False,
    verify_ssl: str = "True",
    resource_config: dict = {}, # TODO: parameterize gpu and cpu resource limits
    ):

    # Step 1: Validate inputs
    validate_task: dsl.PipelineTask = validate_inputs(
        command=command,
        llama_stack_url=llama_stack_url,
        verify_ssl=verify_ssl
    )
    validate_task.set_caching_options(False)
    
    # Step 2: Run the garak scan ONLY if validation passes
    with dsl.If(validate_task.outputs['is_valid'] == True, name="validation_passed"):
        
        with dsl.If(use_gpu == True, name="USE_GPU"):
            scan_task_gpu: dsl.PipelineTask = garak_scan(
                command=command,
                llama_stack_url=llama_stack_url,
                job_id=job_id,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                verify_ssl=verify_ssl
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
        
        with dsl.Else(name="USE_CPU"):
            scan_task_cpu: dsl.PipelineTask = garak_scan(
                command=command,
                llama_stack_url=llama_stack_url,
                job_id=job_id,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                verify_ssl=verify_ssl
            )
            scan_task_cpu.after(validate_task)
            scan_task_cpu.set_caching_options(False)

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
