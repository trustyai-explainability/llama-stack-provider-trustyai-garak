from typing import Union
import httpx
import os
from pathlib import Path


def _ensure_xdg_vars() -> None:
    """
    Ensure XDG environment variables are set to writable locations.
    
    For production/container environments where the default
    XDG directories might not be writable. Garak use these
    directories for cache, config, and data storage.
    
    This function is idempotent and safe to call multiple times.
    It only sets variables that are not already set.
    """
    # Default to /tmp subdirectories (always writable)
    xdg_defaults = {
        "XDG_CACHE_HOME": "/tmp/.cache",
        "XDG_DATA_HOME": "/tmp/.local/share",
        "XDG_CONFIG_HOME": "/tmp/.config",
    }
    
    for var, default_path in xdg_defaults.items():
        if var not in os.environ:
            os.environ[var] = default_path
            # Create the directory if it doesn't exist
            Path(default_path).mkdir(parents=True, exist_ok=True)


def get_scan_base_dir() -> Path:
    """
    Get the base directory for scan files.
    
    This uses XDG_CACHE_HOME which is automatically set to /tmp/.cache,
    ensuring it's always writable.
    
    Can be overridden with GARAK_SCAN_DIR environment variable.
    
    XDG variables are lazily initialized on first call to avoid
    side effects at module import time.
    
    Returns:
        Path: Base directory for scan operations
    """
    # Lazy initialization of XDG vars (only on first use)
    _ensure_xdg_vars()
    
    if scan_dir := os.environ.get("GARAK_SCAN_DIR"):
        return Path(scan_dir)
    
    # Use XDG_CACHE_HOME/llama_stack_garak_scans
    return Path(os.environ["XDG_CACHE_HOME"]) / "llama_stack_garak_scans"


def get_http_client_with_tls(tls_verify: Union[bool, str] = True) -> httpx.Client:
    """
    Get an HTTP client with TLS verification.

    Args:
        tls_verify: Whether to verify TLS certificates. Can be a boolean or a path to a CA certificate file.

    Returns:
        httpx.Client: An HTTP client with TLS verification.
    """
    if isinstance(tls_verify, str):
        if tls_verify.lower().strip() in ("true", "1", "yes", "on", ""):
            tls_verify = True
        elif tls_verify.lower().strip() in ("false", "0", "no", "off"):
            tls_verify = False
        else:
            tls_verify = tls_verify
    if tls_verify is None:
        tls_verify = True
    return httpx.Client(verify=tls_verify)