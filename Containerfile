# syntax=docker/dockerfile:1
# ============================================================================
# Stage 1: Builder - Install dependencies and build Python packages
# ============================================================================
FROM registry.access.redhat.com/ubi9/python-312:latest AS builder

WORKDIR /opt/app-root

# Switch to root for installing build dependencies
USER root

# Install build tools needed for compiling Rust-based Python packages
# Use cache mount to speed up subsequent builds
RUN --mount=type=cache,target=/var/cache/dnf \
    dnf install -y --setopt install_weak_deps=0 --nodocs \
    --disablerepo=rhel-9-for-aarch64-appstream-rpms \
    --disablerepo=rhel-9-for-aarch64-baseos-rpms \
    --disablerepo=rhel-9-for-x86_64-appstream-rpms \
    --disablerepo=rhel-9-for-x86_64-baseos-rpms \
    cargo \
    rust \
    git && \
    dnf clean all

# Install uv for fast Python package installation (10-100x faster than pip)
# Copy from official uv image (minimal security surface - single static binary)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all source code and install everything
COPY pyproject.toml ./
COPY src src

# Install package and all dependencies to the UBI9 Python venv at /opt/app-root
# This venv is automatically created and activated by the base image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python=/opt/app-root/bin/python .[inline]


# ============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ============================================================================
FROM registry.access.redhat.com/ubi9/python-312:latest AS runtime

WORKDIR /opt/app-root

# Switch to root to copy files and set permissions
USER root

# Copy the entire /opt/app-root venv from builder (includes all packages)
# The UBI9 Python image auto-creates and activates a venv at this location
COPY --from=builder /opt/app-root /opt/app-root

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache \
    XDG_DATA_HOME=/tmp/.local/share \
    XDG_CONFIG_HOME=/tmp/.config

# Create XDG directories and set permissions
RUN mkdir -p ${XDG_CACHE_HOME} ${XDG_DATA_HOME} ${XDG_CONFIG_HOME} && \
    chown -R 1001:0 /opt/app-root && \
    chmod -R g=u /opt/app-root && \
    chmod -R 1777 ${XDG_CACHE_HOME} ${XDG_DATA_HOME} ${XDG_CONFIG_HOME} /tmp/.local

# Switch back to non-root user (UBI9 uses 1001)
USER 1001

# Metadata labels
LABEL org.opencontainers.image.title="TrustyAI Garak Provider for Llama Stack" \
      org.opencontainers.image.description="Out-of-tree Llama Stack provider for Garak red-teaming with EvalHub integration" \
      org.opencontainers.image.source="https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak" \
      org.opencontainers.image.vendor="TrustyAI" \
      org.opencontainers.image.version="0.2.0"

# Default CMD for eval-hub (runs as K8s Job)
# Note: KFP components override this via @dsl.component, so this doesn't affect KFP usage
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]
