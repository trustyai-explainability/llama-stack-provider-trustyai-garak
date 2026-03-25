FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /opt/app-root

# Switch to root only for installing packages
USER root

COPY pyproject.toml .
COPY src src

# Install cpu torch to reduce image size
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the package + sdg deps (everything except garak, which comes from git)
RUN pip install --no-cache-dir ".[sdg]"
# Install garak from midstream git (tag derived from pyproject.toml)
RUN GARAK_VER=$(grep -oP 'garak==\K[^\s"]+' pyproject.toml) && \
    pip install --no-cache-dir \
    "garak @ git+https://github.com/trustyai-explainability/garak.git@v${GARAK_VER}"

# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_DATA_HOME=/tmp/.local/share
ENV XDG_CONFIG_HOME=/tmp/.config

# Switch back to non-root user
# UBI9 uses 1001
USER 1001
