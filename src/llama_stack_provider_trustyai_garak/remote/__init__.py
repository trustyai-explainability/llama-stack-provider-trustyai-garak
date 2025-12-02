import logging
from typing import Dict, Optional

from ..compat import Api, ProviderSpec

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
        logger.error(f"Failed to create remote Garak implementation: {str(e)}")
        raise


__all__ = [
     # Factory methods
    "get_adapter_impl",
    # Configurations
    "GarakRemoteEvalAdapter",
]