import logging

logger = logging.getLogger(__name__)


def _has_inline_dependencies() -> bool:
    """Check if inline dependencies are available."""
    try:
        import garak  # noqa: F401
        return True
    except ImportError:
        return False


def _has_remote_dependencies() -> bool:
    """Check if remote dependencies are available."""
    try:
        import kfp  # noqa: F401
        import kubernetes  # noqa: F401
        import boto3  # noqa: F401
        import kfp_server_api  # noqa: F401

        return True
    except ImportError:
        return False


def get_provider_spec():
    providers = []

    # Remote provider loads by default
    if _has_remote_dependencies():
        from .remote.provider import get_provider_spec as get_remote_provider_spec
        providers.append(get_remote_provider_spec())
    else:
        logger.info(
            "Remote provider dependencies not found. "
            "This should not happen. "
            "Reinstall with: pip install llama-stack-provider-trustyai-garak"
        )

    # Inline provider is optional (for local development/testing)
    if _has_inline_dependencies():
        from .inline.provider import get_provider_spec as get_inline_provider_spec
        providers.append(get_inline_provider_spec())
    else:
        logger.info(
            "Inline provider dependencies not found, skipping inline provider. "
            "Enable inline evaluation with 'pip install llama-stack-provider-trustyai-garak[inline]'."
        )
    
    if not providers:
        logger.error(
            "No provider dependencies found. This is likely a broken installation. "
            "Reinstall with: pip install llama-stack-provider-trustyai-garak"
        )

    return providers


__all__ = ["get_provider_spec"]

