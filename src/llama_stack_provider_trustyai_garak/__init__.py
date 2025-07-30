import logging
from typing import Dict, Optional

from llama_stack.distribution.datatypes import Api, ProviderSpec
from .config import GarakEvalProviderConfig
from .garak_eval import GarakEvalAdapter
from .provider import get_provider_spec

# Set up logging
logger = logging.getLogger(__name__)

async def get_provider_impl(
        config: GarakEvalProviderConfig,
        deps: Optional[Dict[Api, ProviderSpec]] = None,
) -> GarakEvalAdapter:
    """Get an Garak implementation from the configuration.

    Args:
        config: Garak configuration
        deps: Optional dependencies

    Returns:
        Configured Garak implementation

    Raises:
        Exception: If configuration is invalid
    """
    try:
        if deps is None:
            deps = {}

        # Extract base_url from config if available
        base_url = None
        if hasattr(config, "base_url"):
            base_url = config.base_url
            logger.debug(f"Using base_url from config: {base_url}")

        impl = GarakEvalAdapter(config=config, deps=deps)
        await impl.initialize()
        return impl
    except Exception as e:
        raise Exception(
            f"Failed to create Garak implementation: {str(e)}"
        ) from e

__all__ = [
     # Factory methods
    "get_provider_impl",
    # Configurations
    "GarakEvalAdapter",
    "get_provider_spec",

]