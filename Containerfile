# syntax=docker/dockerfile:1
# ============================================================================
# Multi-stage build for FIPS compliance and minimal runtime image
#
# Stage 1: Builder - Compile dependencies with build tools
# Stage 2: Runtime - Clean minimal image without build tools
# ============================================================================

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM registry.access.redhat.com/ubi9/python-312:latest AS builder

WORKDIR /opt/app-root
USER root

# Install build-time dependencies required for Rust-based Python packages
# These tools (cargo, rust, git) are excluded from the final runtime image
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

# Install uv for fast dependency installation (10-100x faster than pip)
# Using pip install instead of COPY --from for Konflux hermetic build compatibility
# This avoids external registry dependencies (ghcr.io) while maintaining performance
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/app-root/bin/python -m pip install --no-warn-script-location uv

# Copy source code AND .git directory for setuptools-scm version detection
# .git is only needed in builder stage and will NOT be in final image
COPY pyproject.toml ./
COPY src src
COPY .git .git

# Install to UBI9's native venv at /opt/app-root (auto-created by base image)
# This ensures proper Python path resolution and follows Red Hat's standard pattern
# setuptools-scm will automatically detect version from .git directory
RUN --mount=type=cache,target=/root/.cache/uv \
    /opt/app-root/bin/uv pip install --python=/opt/app-root/bin/python .[inline]

# Remove build artifacts, caches, source files, and .git directory to reduce image size
# Package is already installed in venv's site-packages
# .git directory is REMOVED here - it was only needed for version detection
RUN rm -rf /opt/app-root/src/.cache \
           /opt/app-root/src/.local \
           /opt/app-root/src/llama_stack_provider_trustyai_garak \
           /opt/app-root/src/llama_stack_provider_trustyai_garak.egg-info \
           /opt/app-root/pyproject.toml \
           /opt/app-root/.git && \
    find /opt/app-root -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/app-root -type f -name '*.pyc' -delete 2>/dev/null || true



# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM registry.access.redhat.com/ubi9/python-312-minimal:latest AS runtime

# Optional build arg for version label - can be provided or auto-detected
# If not provided, can be queried from installed package at runtime
ARG VERSION=""

WORKDIR /opt/app-root
USER root

# Copy venv from builder containing all installed packages
# UBI9 Python images use /opt/app-root for the virtual environment
COPY --from=builder /opt/app-root /opt/app-root

# Python optimization: disable buffering and bytecode generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Configure XDG directories to use /tmp for garak's cache and data files
# /tmp is always writable regardless of OpenShift's random UID assignment
ENV XDG_CACHE_HOME=/tmp/.cache \
    XDG_DATA_HOME=/tmp/.local/share \
    XDG_CONFIG_HOME=/tmp/.config

# Set up directories and permissions for OpenShift compatibility
# - Creates XDG directories
# - Sets ownership to user 1001 with group 0 (root group)
# - Allows group write access (OpenShift uses arbitrary UIDs in root group)
RUN mkdir -p ${XDG_CACHE_HOME} ${XDG_DATA_HOME} ${XDG_CONFIG_HOME} && \
    chown -R 1001:0 /opt/app-root && \
    chmod -R g=u /opt/app-root && \
    chmod -R 1777 ${XDG_CACHE_HOME} ${XDG_DATA_HOME} ${XDG_CONFIG_HOME} /tmp/.local

# Run as non-root user for security
USER 1001

# Image metadata
# VERSION can be optionally passed as build arg for the label
# If not provided, version is still available in the installed package
LABEL org.opencontainers.image.title="TrustyAI Garak Provider for Llama Stack" \
      org.opencontainers.image.description="Out-of-tree Llama Stack provider for Garak red-teaming with EvalHub integration" \
      org.opencontainers.image.source="https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak" \
      org.opencontainers.image.vendor="TrustyAI" \
      org.opencontainers.image.version="${VERSION}"

# Default entrypoint for EvalHub K8s Job execution
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]
