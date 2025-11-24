"""Utilities for managing package version information."""

import logging

logger = logging.getLogger(__name__)


def get_garak_version() -> str:
    """
    Extract garak version from package metadata.
    
    This reads from the installed package metadata to get the garak
    version requirement, ensuring a single source of truth.
    
    Returns:
        str: Garak version string (e.g., "garak==0.12.0")
        Falls back to "garak" if version cannot be determined.
    """
    try:
        from importlib.metadata import distribution
        
        dist = distribution("llama-stack-provider-trustyai-garak")
        
        # parse requirements to find garak
        if dist.requires:
            for req in dist.requires:
                #match "garak==X.Y.Z" or "garak>=X.Y.Z" etc.
                if req.startswith("garak"):
                    garak_req = req.split(";")[0].strip()
                    return garak_req
    except Exception as e:
        logger.error(f"Error getting garak version from importlib.metadata: {e}")
        logger.warning("Error getting garak version from importlib.metadata, trying pyproject.toml")
        # fallback: try to read from pyproject.toml
        try:
            from pathlib import Path
            import tomllib  # built-in for python 3.11+
            
            # find pyproject.toml relative to this file
            pkg_root = Path(__file__).parent.parent.parent
            toml_path = pkg_root / "pyproject.toml"
            
            if toml_path.exists():
                with open(toml_path, "rb") as f:
                    pyproject = tomllib.load(f)
                    deps = pyproject.get("project", {}).get("dependencies", [])
                    for dep in deps:
                        if dep.startswith("garak"):
                            return dep.split(";")[0].strip()
        except Exception as e:
            logger.error(f"Error getting garak version: {e}")
    
    # final fallback
    return "garak"

