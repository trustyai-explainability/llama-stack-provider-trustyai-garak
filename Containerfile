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

# Copy dependency specification and create minimal src structure
COPY pyproject.toml ./
RUN mkdir -p src/llama_stack_provider_trustyai_garak && \
    touch src/llama_stack_provider_trustyai_garak/__init__.py

# Create virtual environment using uv (much faster than python -m venv)
RUN uv venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Install all dependencies from pyproject.toml (installs package too but we'll reinstall in runtime)
# This layer is cached and won't be invalidated by real source code changes
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install .[inline]


# ============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ============================================================================
FROM registry.access.redhat.com/ubi9/python-312:latest AS runtime

WORKDIR /opt/app-root

# Switch to root to copy files and set permissions
USER root

# Copy virtual environment with all dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source code and pyproject.toml (this layer changes frequently but is small and fast to rebuild)
COPY src src
COPY pyproject.toml ./

# Set environment to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONUNBUFFERED=1

# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache \
    XDG_DATA_HOME=/tmp/.local/share \
    XDG_CONFIG_HOME=/tmp/.config

# Install the package without dependencies and set permissions
# setuptools is already in venv from builder stage (from build-system.requires in pyproject.toml)
# Combine operations to reduce layers and remove uv after use
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
RUN uv pip install --no-deps . && \
    rm /usr/local/bin/uv && \
    mkdir -p ${XDG_CACHE_HOME} ${XDG_DATA_HOME} ${XDG_CONFIG_HOME} && \
    chown -R 1001:0 /opt/app-root /opt/venv && \
    chmod -R g=u /opt/app-root /opt/venv && \
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
# Use absolute path to venv Python to ensure the correct Python is used
CMD ["/opt/venv/bin/python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]
