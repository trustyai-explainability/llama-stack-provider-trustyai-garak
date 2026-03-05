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
from .kfp_adapter import GarakKFPAdapter

__all__ = ["GarakAdapter", "GarakKFPAdapter"]
