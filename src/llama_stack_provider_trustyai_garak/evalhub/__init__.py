"""eval-hub adapter for Garak red-teaming framework.

This module provides a FrameworkAdapter implementation that integrates
Garak with the eval-hub evaluation platform.

Usage:
    The adapter is designed to run as a K8s Job with two entrypoints:

    # Simple mode (garak runs in the same pod):
    CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]

    # KFP mode (garak runs in a separate KFP pod):
    CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub.kfp_adapter"]
"""

from .garak_adapter import GarakAdapter


def __getattr__(name: str):
    if name == "GarakKFPAdapter":
        from .kfp_adapter import GarakKFPAdapter

        return GarakKFPAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GarakAdapter", "GarakKFPAdapter"]
