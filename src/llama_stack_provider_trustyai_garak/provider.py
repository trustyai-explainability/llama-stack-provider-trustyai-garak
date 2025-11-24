import logging

from .inline.provider import get_provider_spec as get_inline_provider_spec

logger = logging.getLogger(__name__)


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
    providers = [get_inline_provider_spec()]

    if _has_remote_dependencies():
        from .remote.provider import get_provider_spec as get_remote_provider_spec

        providers.append(get_remote_provider_spec())
    else:
        logger.info(
            "Remote provider dependencies not found, returning inline provider only. "
            "Enable remote evaluation with 'pip install llama-stack-provider-trustyai-garak[remote]'."
        )

    return providers


__all__ = ["get_provider_spec"]

