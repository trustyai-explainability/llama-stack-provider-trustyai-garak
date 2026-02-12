"""eval-hub adapter for Garak red-teaming framework.

This module provides a FrameworkAdapter implementation that integrates
Garak with the eval-hub evaluation platform.

Usage:
    The adapter is designed to run as a K8s Job. The entrypoint reads
    JobSpec from /etc/eval-job/spec.json and communicates with the
    eval-hub sidecar.

    # In Containerfile:
    CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]
"""

from .garak_adapter import GarakAdapter

__all__ = ["GarakAdapter"]
