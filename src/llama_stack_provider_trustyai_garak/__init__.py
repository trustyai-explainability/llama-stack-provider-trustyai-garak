from .provider import get_provider_spec
from .version_utils import get_garak_version
from .garak_command_config import GarakSystemConfig, GarakRunConfig, GarakPluginsConfig, GarakReportingConfig, GarakCommandConfig

# Version managed by setuptools-scm
try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development/editable installs
    try:
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)
    except Exception:
        __version__ = "unknown"

__all__ = [
    "get_provider_spec",
    "get_garak_version",
    "GarakSystemConfig",
    "GarakRunConfig",
    "GarakPluginsConfig",
    "GarakReportingConfig",
    "GarakCommandConfig",
    "__version__",
]

