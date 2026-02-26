"""Shared helpers for resolving Garak benchmark/runtime configuration.

These helpers are framework-agnostic and intended for reuse by
multiple execution paths (eval-hub, inline, remote).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..config import GarakScanConfig
from ..garak_command_config import GarakCommandConfig


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge two mappings where override values take precedence."""
    merged = deepcopy(dict(base))

    for key, value in dict(override).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def resolve_scan_profile(
    benchmark_id: str,
) -> dict[str, Any]:
    """Resolve a benchmark id to a predefined scan/framework profile."""
    _cfg = GarakScanConfig()
    all_profiles = {**_cfg.FRAMEWORK_PROFILES, **_cfg.SCAN_PROFILES}
    resolved = (
        all_profiles.get(benchmark_id)
        or all_profiles.get(f"trustyai_garak::{benchmark_id}")
        or {}
    )
    if resolved and isinstance(resolved, GarakCommandConfig):
        resolved = resolved.to_dict(exclude_none=True)
    return deepcopy(resolved)


def _set_nested_value(target: dict[str, Any], path: tuple[str, str], value: Any) -> None:
    section, field = path
    target.setdefault(section, {})
    target[section][field] = deepcopy(value)


def legacy_overrides_from_benchmark_config(
    benchmark_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate legacy flat benchmark config keys to GarakCommandConfig shape."""
    overrides: dict[str, Any] = {}
    key_to_path: dict[str, tuple[str, str]] = {
        "parallel_attempts": ("system", "parallel_attempts"),
        "parallel_requests": ("system", "parallel_requests"),
        "generations": ("run", "generations"),
        "probe_tags": ("run", "probe_tags"),
        "seed": ("run", "seed"),
        "deprefix": ("run", "deprefix"),
        "eval_threshold": ("run", "eval_threshold"),
        "probes": ("plugins", "probe_spec"),
        "detectors": ("plugins", "detector_spec"),
        "extended_detectors": ("plugins", "extended_detectors"),
        "buffs": ("plugins", "buff_spec"),
        "probe_options": ("plugins", "probes"),
        "detector_options": ("plugins", "detectors"),
        "buff_options": ("plugins", "buffs"),
        "harness_options": ("plugins", "harnesses"),
        "taxonomy": ("reporting", "taxonomy"),
    }

    for key, path in key_to_path.items():
        value = benchmark_config.get(key)
        if value is None:
            continue
        _set_nested_value(overrides, path, value)

    return overrides


def _normalize_plugin_specs(config_dict: dict[str, Any]) -> None:
    plugins = config_dict.get("plugins")
    if not isinstance(plugins, dict):
        return

    for field in ("probe_spec", "detector_spec", "buff_spec"):
        value = plugins.get(field)
        if isinstance(value, list):
            plugins[field] = ",".join(str(item) for item in value)


def build_effective_garak_config(
    benchmark_config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> GarakCommandConfig:
    """Build merged Garak config with explicit precedence rules.

    Precedence (highest to lowest):
    1. benchmark_config.garak_config + legacy flat-key overrides
    2. profile.garak_config
    3. GarakCommandConfig() defaults
    """
    bc = dict(benchmark_config or {})
    profile_garak_cfg = profile.get("garak_config", {})

    explicit_garak_cfg = bc.get("garak_config", {})
    if explicit_garak_cfg is None:
        explicit_garak_cfg = {}
    if explicit_garak_cfg and not isinstance(explicit_garak_cfg, dict):
        raise ValueError("benchmark_config.garak_config must be a dictionary")
    if profile_garak_cfg and not isinstance(profile_garak_cfg, dict):
        raise ValueError("profile.garak_config must be a dictionary")

    merged = deep_merge_dicts(
        GarakCommandConfig().to_dict(exclude_none=True),
        dict(profile_garak_cfg or {}),
    )
    merged = deep_merge_dicts(merged, dict(explicit_garak_cfg or {}))
    merged = deep_merge_dicts(merged, legacy_overrides_from_benchmark_config(bc))
    _normalize_plugin_specs(merged)

    return GarakCommandConfig.from_dict(merged)


def resolve_timeout_seconds(
    benchmark_config: Mapping[str, Any],
    profile: Mapping[str, Any],
    default_timeout: int = 600,
) -> int:
    """Resolve timeout with explicit override, profile fallback, then default."""
    bc = dict(benchmark_config or {})

    timeouts = [bc.get("timeout_seconds"), bc.get("timeout"), profile.get("timeout")]
    for timeout in timeouts:
        if timeout is not None:
            try:
                int_timeout = int(timeout)
                if int_timeout > 0:
                    return int_timeout
            except (TypeError, ValueError):
                pass

    return default_timeout

__all__ = [
    "build_effective_garak_config",
    "deep_merge_dicts",
    "legacy_overrides_from_benchmark_config",
    "resolve_scan_profile",
    "resolve_timeout_seconds",
]
