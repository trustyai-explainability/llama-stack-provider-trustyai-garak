"""Shared S3 client factory for Data Connection credentials.

All KFP components and the evalhub adapter need to create boto3 S3 clients
from the same set of environment variables injected by a Data Connection secret.  
This module centralises that logic so it isn't
duplicated across every call-site.
"""

import os


def create_s3_client():
    """Create a boto3 S3 client from Data Connection environment variables.

    Expects standard env vars (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_S3_ENDPOINT``, etc.) injected
    via ``envFrom`` or ``kubernetes.use_secret_as_env``.

    When ``AWS_S3_ENDPOINT`` is set (e.g. for MinIO / Ceph), path-style
    addressing is used automatically.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    s3_endpoint = os.environ.get("AWS_S3_ENDPOINT", "")
    client_kwargs: dict = {
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if s3_endpoint:
        client_kwargs["endpoint_url"] = s3_endpoint
        client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

    return boto3.client("s3", **client_kwargs)
