FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /opt/app-root

# Switch to root only for installing packages
USER root

# For Rust-based Python packages
RUN dnf install -y --setopt install_weak_deps=0 --nodocs \
    cargo \
    rust \
    && dnf clean all

COPY . .

# Build argument to specify architecture
ARG TARGETARCH=x86_64

# Install dependencies
RUN if [ "$TARGETARCH" = "amd64" ] || [ "$TARGETARCH" = "x86_64" ]; then \
        echo "Installing x86_64 dependencies ..."; \
        pip install --no-cache-dir -r requirements-x86_64.txt; \
    elif [ "$TARGETARCH" = "arm64" ] || [ "$TARGETARCH" = "aarch64" ]; then \
        echo "Installing ARM64 dependencies ..."; \
        pip install --no-cache-dir -r requirements-aarch64.txt; \
    else \
        echo "ERROR: Unsupported architecture: $TARGETARCH"; \
        exit 1; \
    fi

# Install the package itself (--no-deps since dependencies already installed)
# Use [inline] to get garak dependency
RUN pip install --no-cache-dir --no-deps -e ".[inline]"

# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_DATA_HOME=/tmp/.local/share
ENV XDG_CONFIG_HOME=/tmp/.config

# Switch back to non-root user
# UBI9 uses 1001
USER 1001
