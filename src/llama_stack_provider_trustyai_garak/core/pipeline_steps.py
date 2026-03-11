"""Shared pipeline step logic for both EvalHub KFP and Llama Stack KFP pipelines.

These functions contain framework-agnostic business logic without I/O adapter
dependencies (no S3, no Files API).  They operate on local file paths, content
strings, and DataFrames.

Each KFP adapter (EvalHub/S3, Llama Stack/Files API) calls these functions
and handles its own I/O.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants import DEFAULT_SDG_FLOW_ID
from ..errors import GarakError, GarakValidationError

from .garak_runner import GarakScanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Validation
# ---------------------------------------------------------------------------

_DANGEROUS_FLAGS = ("--rm", "--force", "--no-limit")


def validate_scan_config(config_json: str) -> None:
    """Validate config JSON structure, garak importability, and flag safety.

    Raises GarakValidationError on any failure.  Adapter-specific
    connectivity checks (S3, Files API) are done by the caller.
    """
    errors: list[str] = []

    try:
        import garak  # noqa: F401
    except ImportError:
        errors.append("garak is not installed in the base image")

    try:
        config = json.loads(config_json)
        if "plugins" not in config:
            errors.append("config_json missing required 'plugins' section")
    except json.JSONDecodeError as exc:
        errors.append(f"config_json is not valid JSON: {exc}")

    for flag in _DANGEROUS_FLAGS:
        if flag in config_json:
            errors.append(f"Dangerous flag detected: {flag}")

    if errors:
        raise GarakValidationError(
            "Pre-flight validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Step 2: Taxonomy resolution
# ---------------------------------------------------------------------------

def resolve_taxonomy_data(
    content: bytes | None,
    format: str = "csv",
) -> pd.DataFrame:
    """Parse raw taxonomy bytes or return the built-in BASE_TAXONOMY.

    The caller is responsible for fetching the bytes from its storage
    backend (S3, Files API, local FS) before calling this function.
    """
    if content is not None:
        from ..intents import load_taxonomy_dataset

        text = content.decode("utf-8") if isinstance(content, bytes) else content
        taxonomy = load_taxonomy_dataset(content=text, format=format)
        logger.info(
            "Loaded custom taxonomy: %d entries, columns: %s",
            len(taxonomy),
            list(taxonomy.columns),
        )
        return taxonomy

    from ..sdg import BASE_TAXONOMY

    taxonomy = pd.DataFrame(BASE_TAXONOMY)
    logger.info("Using BASE_TAXONOMY: %d entries", len(taxonomy))
    return taxonomy


# ---------------------------------------------------------------------------
# Step 3: SDG generation
# ---------------------------------------------------------------------------

def run_sdg_generation(
    taxonomy_df: pd.DataFrame,
    sdg_model: str,
    sdg_api_base: str,
    sdg_api_key: str = "",
    sdg_flow_id: str = "",
) -> pd.DataFrame:
    """Run Synthetic Data Generation on a taxonomy.  Returns the raw DataFrame."""
    from ..sdg import generate_sdg_dataset

    if not sdg_model or not sdg_model.strip():
        raise GarakValidationError(
            "sdg_model is required for intents scans when SDG must run."
        )
    if not sdg_api_base or not sdg_api_base.strip():
        raise GarakValidationError(
            "sdg_api_base is required for intents scans when SDG must run."
        )

    if sdg_api_key and sdg_api_key.strip():
        os.environ["OPENAI_API_KEY"] = sdg_api_key

    effective_flow_id = (sdg_flow_id.strip() if sdg_flow_id else "") or DEFAULT_SDG_FLOW_ID

    logger.info(
        "Running SDG: model=%s, api_base=%s, flow=%s",
        sdg_model,
        sdg_api_base,
        effective_flow_id,
    )
    sdg_result = generate_sdg_dataset(
        model=sdg_model,
        api_base=sdg_api_base,
        flow_id=effective_flow_id,
        api_key=sdg_api_key if sdg_api_key and sdg_api_key.strip() else "dummy",
        taxonomy=taxonomy_df,
    )
    logger.info(
        "SDG produced %d raw rows across %d categories",
        len(sdg_result.raw),
        sdg_result.raw["policy_concept"].nunique(),
    )
    return sdg_result.raw


# ---------------------------------------------------------------------------
# Step 4: Prompt normalisation
# ---------------------------------------------------------------------------

def normalize_prompts(
    raw_content: str,
    format: str = "csv",
) -> pd.DataFrame:
    """Normalise raw intents data into the (category, prompt, description) schema."""
    from ..intents import load_intents_dataset

    normalized = load_intents_dataset(content=raw_content, format=format)
    logger.info(
        "Normalised %d prompts across %d categories",
        len(normalized),
        normalized["category"].nunique(),
    )
    return normalized


# ---------------------------------------------------------------------------
# Step 5: Garak scan execution
# ---------------------------------------------------------------------------

def setup_and_run_garak(
    config_json: str,
    prompts_csv_path: Path | str | None,
    scan_dir: Path,
    timeout_seconds: int,
) -> "GarakScanResult":
    """Configure and run a Garak scan with optional intent-stub generation.

    1. Parse *config_json*, set ``report_prefix`` inside *scan_dir*.
    2. If *prompts_csv_path* points to a non-empty CSV, generate intent stubs.
    3. Run ``garak --config ...`` via :func:`core.garak_runner.run_garak_scan`.
    4. Attempt AVID conversion on success.

    Returns a :class:`GarakScanResult`.  Raises :class:`GarakError` on failure.
    """
    from .garak_runner import convert_to_avid_report, run_garak_scan

    scan_dir.mkdir(parents=True, exist_ok=True)

    config: dict = json.loads(config_json)
    config.setdefault("reporting", {})["report_prefix"] = str(scan_dir / "scan")
    config_file = scan_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=1))

    report_prefix = scan_dir / "scan"
    log_file = scan_dir / "scan.log"

    if prompts_csv_path is not None:
        prompts_csv_path = Path(prompts_csv_path)
        if prompts_csv_path.exists() and prompts_csv_path.stat().st_size > 0:
            from ..intents import generate_intents_from_dataset

            df = pd.read_csv(prompts_csv_path)
            if not df.empty:
                desc_col = "description" if "description" in df.columns else None
                generate_intents_from_dataset(
                    df,
                    category_description_column_name=desc_col,
                )
                logger.info(
                    "Generated intent stubs for %d prompts across %d categories",
                    len(df),
                    df["category"].nunique(),
                )

    logger.info(
        "Starting garak scan (timeout=%ss, output=%s)",
        timeout_seconds,
        scan_dir,
    )

    result = run_garak_scan(
        config_file=config_file,
        timeout_seconds=timeout_seconds,
        report_prefix=report_prefix,
        log_file=log_file,
    )

    if result.success:
        logger.info("Garak scan completed successfully (rc=%s)", result.returncode)
        try:
            convert_to_avid_report(result.report_jsonl)
        except Exception as exc:
            logger.warning("AVID conversion failed: %s", exc)
    else:
        error_msg = (
            f"Garak scan failed (rc={result.returncode}, "
            f"timed_out={result.timed_out}): "
            f"{result.stderr[:500] if result.stderr else 'no stderr'}"
        )
        logger.error(error_msg)
        raise GarakError(error_msg)

    return result


# ---------------------------------------------------------------------------
# Step 6: Result parsing & metrics
# ---------------------------------------------------------------------------

def parse_and_build_results(
    report_content: str,
    avid_content: str | None,
    art_intents: bool,
    eval_threshold: float,
    raw_entries_by_probe: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Parse Garak reports and build the combined result dictionary.

    Calls the result_utils parsing pipeline and returns a dict suitable
    for constructing an EvaluateResponse.
    """
    from .. import result_utils

    generations, score_rows_by_probe, parsed_raw = (
        result_utils.parse_generations_from_report_content(
            report_content, eval_threshold
        )
    )
    aggregated_by_probe = result_utils.parse_aggregated_from_avid_content(
        avid_content or ""
    )
    digest = result_utils.parse_digest_from_report_content(report_content)

    effective_raw = raw_entries_by_probe if raw_entries_by_probe is not None else parsed_raw

    return result_utils.combine_parsed_results(
        generations,
        score_rows_by_probe,
        aggregated_by_probe,
        eval_threshold,
        digest,
        art_intents=art_intents,
        raw_entries_by_probe=effective_raw,
    )


def log_kfp_metrics(
    metrics_output: Any,
    result_dict: dict[str, Any],
    art_intents: bool,
) -> None:
    """Log summary metrics to a KFP Metrics artifact.

    Works with ``dsl.Output[dsl.Metrics]`` — call ``metrics_output.log_metric``.
    """
    overall = (
        result_dict.get("scores", {})
        .get("_overall", {})
        .get("aggregated_results", {})
    )

    if art_intents:
        metrics_output.log_metric(
            "attack_success_rate",
            overall.get("attack_success_rate", 0),
        )
    else:
        metrics_output.log_metric(
            "total_attempts",
            overall.get("total_attempts", 0),
        )
        metrics_output.log_metric(
            "vulnerable_responses",
            overall.get("vulnerable_responses", 0),
        )
        metrics_output.log_metric(
            "attack_success_rate",
            overall.get("attack_success_rate", 0),
        )

    if overall.get("tbsa"):
        metrics_output.log_metric("tbsa", overall["tbsa"])

    logger.info(
        "Logged KFP metrics: %s",
        {k: v for k, v in overall.items() if k in ("attack_success_rate", "total_attempts", "tbsa")},
    )
