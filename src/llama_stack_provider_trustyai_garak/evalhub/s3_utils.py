"""Shared S3 client factory for Data Connection credentials.

All KFP components and the evalhub adapter need to create boto3 S3 clients
from the same set of environment variables injected by a Data Connection secret.  
This module centralises that logic so it isn't
duplicated across every call-site.
"""

import os


def create_s3_client(
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str | None = None,
):
    """Create a boto3 S3 client from Data Connection environment variables.

    Explicit parameters take precedence over environment variables, which
    allows the adapter pod to supply credentials read from a k8s secret
    when they are not present in its own environment.

    Args:
        endpoint_url: S3-compatible endpoint URL. Overrides ``AWS_S3_ENDPOINT``.
        access_key: AWS access key ID. Overrides ``AWS_ACCESS_KEY_ID``.
        secret_key: AWS secret access key. Overrides ``AWS_SECRET_ACCESS_KEY``.
        region: AWS region. Overrides ``AWS_DEFAULT_REGION``.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    resolved_endpoint = endpoint_url or os.environ.get("AWS_S3_ENDPOINT", "")
    client_kwargs: dict = {
        "aws_access_key_id": access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "region_name": region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }
    if resolved_endpoint:
        client_kwargs["endpoint_url"] = resolved_endpoint
        client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

    return boto3.client("s3", **client_kwargs)
