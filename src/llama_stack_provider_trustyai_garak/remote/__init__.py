import logging
from typing import Dict, Optional

from llama_stack.apis.datatypes import Api
from llama_stack.providers.datatypes import ProviderSpec

from ..config import GarakRemoteConfig
from .garak_remote_eval import GarakRemoteEvalAdapter

# Set up logging
logger = logging.getLogger(__name__)

async def get_adapter_impl(
    config: GarakRemoteConfig,
    deps: Optional[Dict[Api, ProviderSpec]] = None,
) -> GarakRemoteEvalAdapter:
    """
    Get a remote Garak implementation from the configuration.
    """
    try:
        if deps is None:
            deps = {}
        impl = GarakRemoteEvalAdapter(config=config, deps=deps)
        await impl.initialize()
        return impl
    except Exception as e:
        raise Exception(
            f"Failed to create remote Garak implementation: {str(e)}"
        ) from e


__all__ = [
     # Factory methods
    "get_adapter_impl",
    # Configurations
    "GarakRemoteEvalAdapter",
]