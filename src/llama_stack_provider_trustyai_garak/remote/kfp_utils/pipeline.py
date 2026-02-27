from kfp import dsl, kubernetes
from .components import validate_inputs, resolve_intents_dataset, garak_scan, parse_results
from dotenv import load_dotenv
import logging

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
    art_intents: bool = False,
    intents_file_id: str = "",
    intents_format: str = "csv",
    category_column: str = "category",
    prompt_column: str = "prompt",
    description_column: str = "",
):

    # Step 1: Validate inputs (raises on failure, short-circuiting the pipeline)
    validate_task = validate_inputs(
        command=command,
        llama_stack_url=llama_stack_url,
        verify_ssl=verify_ssl,
    )
    validate_task.set_caching_options(False)

    # Step 2: Resolve intents dataset (writes empty file for native probe scans)
    resolve_task = resolve_intents_dataset(
        art_intents=art_intents,
        intents_file_id=intents_file_id,
        intents_format=intents_format,
        llama_stack_url=llama_stack_url,
        category_column=category_column,
        prompt_column=prompt_column,
        description_column=description_column,
        verify_ssl=verify_ssl,
    )
    resolve_task.set_caching_options(False)
    resolve_task.after(validate_task)

    # Common arguments
    scan_kwargs = dict(
        command=command,
        llama_stack_url=llama_stack_url,
        job_id=job_id,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
        intents_dataset=resolve_task.outputs['intents_dataset'],
    )
    parse_kwargs = dict(
        llama_stack_url=llama_stack_url,
        eval_threshold=eval_threshold,
        job_id=job_id,
        verify_ssl=verify_ssl,
        art_intents=art_intents,
    )

    # Step 3 & 4: Scan then parse (GPU / CPU split is required because
    # KFP accelerator settings are compile-time, not runtime).
    # Both scan and parse raise on failure, so no success-check wrappers needed —
    # KFP skips downstream tasks automatically when an upstream task fails.
    with dsl.If(use_gpu == True, name="USE_GPU"):
        scan_task_gpu = garak_scan(**scan_kwargs)
        scan_task_gpu.set_caching_options(False)
        scan_task_gpu.set_accelerator_type(accelerator="nvidia.com/gpu")
        scan_task_gpu.set_accelerator_limit(limit=1)
        kubernetes.add_toleration(
            scan_task_gpu, key="nvidia.com/gpu",
            operator="Exists", effect="NoSchedule",
        )

        parse_gpu = parse_results(
            file_id_mapping=scan_task_gpu.outputs['file_id_mapping'],
            **parse_kwargs,
        )
        parse_gpu.set_caching_options(False)

    with dsl.Else(name="USE_CPU"):
        scan_task_cpu = garak_scan(**scan_kwargs)
        scan_task_cpu.set_caching_options(False)

        parse_cpu = parse_results(
            file_id_mapping=scan_task_cpu.outputs['file_id_mapping'],
            **parse_kwargs,
        )
        parse_cpu.set_caching_options(False)
