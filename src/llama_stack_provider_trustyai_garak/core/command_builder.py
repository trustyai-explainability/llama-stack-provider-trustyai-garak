"""Pure functions for building Garak CLI commands.

This module provides framework-agnostic command building utilities
that can be used by Llama Stack providers and eval-hub adapters.
"""

import json
from typing import Any, Union


def _normalize_list_arg(arg: Union[str, list[str]]) -> str:
    """Normalize a list argument to a comma-separated string."""
    return arg if isinstance(arg, str) else ",".join(arg)


def build_generator_options(
    model_endpoint: str,
    model_name: str,
    api_key: str = "DUMMY",
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build generator options for OpenAI-compatible endpoints.
    
    Args:
        model_endpoint: The model API endpoint URL (must end with /v1 or similar)
        model_name: The model name/identifier
        api_key: API key for authentication (default: "DUMMY" for local endpoints)
        extra_params: Additional parameters (temperature, max_tokens, etc.)
    
    Returns:
        Generator options dict for garak's --generator_options flag
    """
    options = {
        "openai": {
            "OpenAICompatible": {
                "uri": model_endpoint,
                "model": model_name,
                "api_key": api_key,
                "suppressed_params": ["n"]
            }
        }
    }
    
    if extra_params:
        options["openai"]["OpenAICompatible"].update(extra_params)
    
    return options


def build_garak_command(
    model_type: str,
    model_name: str,
    generator_options: dict[str, Any],
    probes: list[str] | None = None,
    report_prefix: str | None = None,
    parallel_attempts: int = 8,
    generations: int = 1,
    # Optional benchmark config parameters
    parallel_requests: int | None = None,
    skip_unknown: bool | None = None,
    seed: int | None = None,
    deprefix: str | None = None,
    eval_threshold: float | None = None,
    probe_tags: list[str] | str | None = None,
    probe_options: dict[str, Any] | None = None,
    detectors: list[str] | str | None = None,
    extended_detectors: bool | None = None,
    detector_options: dict[str, Any] | None = None,
    buffs: list[str] | str | None = None,
    buff_options: dict[str, Any] | None = None,
    harness_options: dict[str, Any] | None = None,
    taxonomy: str | None = None,
    generate_autodan: str | None = None,
) -> list[str]:
    """Build a garak CLI command from configuration.
    
    This is a pure function with no framework dependencies.
    
    Args:
        model_type: Garak model type (e.g., 'openai.OpenAICompatible')
        model_name: Model name or function path
        generator_options: Generator configuration dict
        probes: List of probe names to run (None or ['all'] for all probes)
        report_prefix: Prefix for output report files
        parallel_attempts: Number of parallel probe attempts
        generations: Number of generations per prompt
        parallel_requests: Number of parallel requests
        skip_unknown: Skip unknown probes
        seed: Random seed for reproducibility
        deprefix: Prefix to remove from prompts
        eval_threshold: Threshold for vulnerability detection
        probe_tags: Tags to filter probes
        probe_options: Probe-specific configuration
        detectors: Detector names to use
        extended_detectors: Use extended detectors
        detector_options: Detector-specific configuration
        buffs: Buff names to apply
        buff_options: Buff-specific configuration
        harness_options: Harness-specific configuration
        taxonomy: Taxonomy to use for reporting
        generate_autodan: AutoDAN generation flag
    
    Returns:
        List of command-line arguments for garak
    """
    cmd = [
        "garak",
        "--model_type", model_type,
        "--model_name", model_name,
        "--generator_options", json.dumps(generator_options),
        "--parallel_attempts", str(parallel_attempts),
        "--generations", str(generations),
    ]
    
    # Add report prefix if provided
    if report_prefix:
        cmd.extend(["--report_prefix", report_prefix.strip()])
    
    # Optional parameters
    if parallel_requests is not None:
        cmd.extend(["--parallel_requests", str(parallel_requests)])
    
    if skip_unknown:
        cmd.append("--skip_unknown")
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    if deprefix is not None:
        cmd.extend(["--deprefix", deprefix])
    
    if eval_threshold is not None:
        cmd.extend(["--eval_threshold", str(eval_threshold)])
    
    if probe_tags is not None:
        cmd.extend(["--probe_tags", _normalize_list_arg(probe_tags)])
    
    if probe_options is not None:
        cmd.extend(["--probe_options", json.dumps(probe_options)])
    
    if detectors is not None:
        cmd.extend(["--detectors", _normalize_list_arg(detectors)])
    
    if extended_detectors:
        cmd.append("--extended_detectors")
    
    if detector_options is not None:
        cmd.extend(["--detector_options", json.dumps(detector_options)])
    
    if buffs is not None:
        cmd.extend(["--buffs", _normalize_list_arg(buffs)])
    
    if buff_options is not None:
        cmd.extend(["--buff_options", json.dumps(buff_options)])
    
    if harness_options is not None:
        cmd.extend(["--harness_options", json.dumps(harness_options)])
    
    if taxonomy is not None:
        cmd.extend(["--taxonomy", taxonomy])
    
    if generate_autodan is not None:
        cmd.extend(["--generate_autodan", generate_autodan])
    
    # Add probes (if not 'all' as 'all' is the default)
    if not probe_tags and probes and probes != ["all"]:
        # Normalize probes to list
        if isinstance(probes, str):
            probes = probes.split(",") if "," in probes else [probes]
        cmd.extend(["--probes", ",".join(probes)])
    
    return cmd
