"""Core module with shared, framework-agnostic utilities.

This module contains pure functions that can be used by:
- Llama Stack inline/remote providers
- eval-hub adapter
- KFP components
"""

from .command_builder import build_garak_command, build_generator_options
from .config_resolution import (
    build_effective_garak_config,
    deep_merge_dicts,
    legacy_overrides_from_benchmark_config,
    resolve_scan_profile,
    resolve_timeout_seconds,
)
from .garak_runner import run_garak_scan, GarakScanResult

__all__ = [
    "build_garak_command",
    "build_generator_options",
    "build_effective_garak_config",
    "deep_merge_dicts",
    "legacy_overrides_from_benchmark_config",
    "resolve_scan_profile",
    "resolve_timeout_seconds",
    "run_garak_scan",
    "GarakScanResult",
]
