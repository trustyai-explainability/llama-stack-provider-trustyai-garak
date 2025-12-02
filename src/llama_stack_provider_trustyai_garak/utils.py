from typing import Union
import httpx


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