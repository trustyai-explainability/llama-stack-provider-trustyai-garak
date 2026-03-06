"""KFP-specific Garak adapter for eval-hub.

This module provides a GarakAdapter subclass that always runs in KFP mode,
bypassing the runtime execution_mode resolution. Use this entrypoint when
the adapter pod should act purely as a KFP orchestrator (submit pipeline,
poll, download results from S3).

Usage:
    python -m llama_stack_provider_trustyai_garak.evalhub.kfp_adapter
"""

from .garak_adapter import GarakAdapter, main
from ..constants import EXECUTION_MODE_KFP


class GarakKFPAdapter(GarakAdapter):
    """Garak adapter that always delegates scan execution to KFP."""

    @staticmethod
    def _resolve_execution_mode(benchmark_config: dict) -> str:
        return EXECUTION_MODE_KFP


if __name__ == "__main__":
    main(adapter_cls=GarakKFPAdapter)
