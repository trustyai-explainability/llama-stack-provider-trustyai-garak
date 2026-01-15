import logging
from typing import Dict, Optional

from ..compat import Api, ProviderSpec

from ..config import GarakInlineConfig
from .garak_eval import GarakInlineEvalAdapter

# Set up logging
logger = logging.getLogger(__name__)

async def get_provider_impl(
        config: GarakInlineConfig,
        deps: Optional[Dict[Api, ProviderSpec]] = None,
) -> GarakInlineEvalAdapter:
    """Get an inline Garak implementation from the configuration.

    Args:
        config: Garak inline configuration
        deps: Optional dependencies

    Returns:
        Configured Garak inline implementation

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

        impl = GarakInlineEvalAdapter(config=config, deps=deps)
        await impl.initialize()
        return impl
    except Exception as e:
        raise Exception(
            f"Failed to create inline Garak implementation: {str(e)}"
        ) from e

__all__ = [
     # Factory methods
    "get_provider_impl",
    # Configurations
    "GarakInlineEvalAdapter",
]