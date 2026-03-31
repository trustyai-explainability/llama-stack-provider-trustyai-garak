"""Llama Stack KFP pipeline — 6-step linear DAG.

Mirrors the EvalHub KFP pipeline structure:

1. validate — pre-flight checks (garak, config, Llama Stack connectivity)
2. resolve_taxonomy — custom taxonomy from Files API or BASE_TAXONOMY
3. sdg_generate — run SDG on taxonomy (or no-op for bypass / non-intents)
4. prepare_prompts — normalise, upload raw + normalised to Files API
5. garak_scan — run scan, upload outputs to Files API
6. parse_results — parse reports, log KFP metrics, upload EvaluateResponse
"""

from kfp import dsl
from dotenv import load_dotenv
import logging
from ...constants import (
    DEFAULT_SDG_FLOW_ID,
    DEFAULT_SDG_MAX_CONCURRENCY,
    DEFAULT_SDG_NUM_SAMPLES,
    DEFAULT_SDG_MAX_TOKENS,
)

from .components import (
    validate,
    resolve_taxonomy,
    sdg_generate,
    prepare_prompts,
    garak_scan,
    parse_results,
)

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dsl.pipeline(name="llama-stack-garak-scan")
def garak_scan_pipeline(
    command: str,
    llama_stack_url: str,
    job_id: str,
    eval_threshold: float,
    timeout_seconds: int,
    verify_ssl: str = "True",
    art_intents: bool = False,
    policy_file_id: str = "",
    policy_format: str = "csv",
    intents_file_id: str = "",
    intents_format: str = "csv",
    sdg_model: str = "",
    sdg_api_base: str = "",
    sdg_flow_id: str = DEFAULT_SDG_FLOW_ID,
    sdg_max_concurrency: int = DEFAULT_SDG_MAX_CONCURRENCY,
    sdg_num_samples: int = DEFAULT_SDG_NUM_SAMPLES,
    sdg_max_tokens: int = DEFAULT_SDG_MAX_TOKENS,
):
    """Six-step pipeline: validate, resolve taxonomy, SDG, prepare prompts, scan, parse.

    No API key parameters — Llama Stack models handle their own auth
    at the server level.  The ``core.pipeline_steps`` resolve functions
    fall back to ``"DUMMY"`` when no Secret is injected.

    Three intents modes:
    - Default taxonomy + SDG: no file params, requires sdg_model/sdg_api_base.
    - Custom taxonomy + SDG: policy_file_id set, requires sdg_model/sdg_api_base.
    - Bypass SDG: intents_file_id set, SDG params NOT required.

    policy_file_id and intents_file_id are mutually exclusive.
    """

    # Step 1: Pre-flight validation
    validate_task = validate(
        command=command,
        llama_stack_url=llama_stack_url,
        verify_ssl=verify_ssl,
    )
    validate_task.set_caching_options(False)

    # Step 2: Resolve taxonomy
    taxonomy_task = resolve_taxonomy(
        art_intents=art_intents,
        policy_file_id=policy_file_id,
        policy_format=policy_format,
        llama_stack_url=llama_stack_url,
        verify_ssl=verify_ssl,
    )
    taxonomy_task.set_caching_options(True)
    taxonomy_task.after(validate_task)

    # Step 3: SDG generation (or no-op)
    sdg_task = sdg_generate(
        art_intents=art_intents,
        intents_file_id=intents_file_id,
        sdg_model=sdg_model,
        sdg_api_base=sdg_api_base,
        sdg_flow_id=sdg_flow_id,
        sdg_max_concurrency=sdg_max_concurrency,
        sdg_num_samples=sdg_num_samples,
        sdg_max_tokens=sdg_max_tokens,
        taxonomy_dataset=taxonomy_task.outputs["taxonomy_dataset"],
    )
    sdg_task.set_caching_options(True)

    # Step 4: Prepare prompts (normalise + persist)
    prep_task = prepare_prompts(
        art_intents=art_intents,
        job_id=job_id,
        intents_file_id=intents_file_id,
        intents_format=intents_format,
        llama_stack_url=llama_stack_url,
        verify_ssl=verify_ssl,
        sdg_dataset=sdg_task.outputs["sdg_dataset"],
    )
    prep_task.set_caching_options(False)

    # Step 5: Garak scan
    scan_task = garak_scan(
        command=command,
        llama_stack_url=llama_stack_url,
        job_id=job_id,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
        prompts_dataset=prep_task.outputs["prompts_dataset"],
    )
    scan_task.set_caching_options(False)

    # Step 6: Parse results, log metrics, upload EvaluateResponse
    results_task = parse_results(
        file_id_mapping=scan_task.outputs["file_id_mapping"],
        llama_stack_url=llama_stack_url,
        eval_threshold=eval_threshold,
        job_id=job_id,
        verify_ssl=verify_ssl,
        art_intents=art_intents,
    )
    results_task.set_caching_options(False)
