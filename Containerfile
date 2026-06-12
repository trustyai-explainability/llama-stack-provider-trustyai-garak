FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /opt/app-root

USER root

# Install dependencies
COPY pyproject.toml .
# Stub package so pip can resolve + cache all deps without the real source
RUN mkdir -p src/llama_stack_provider_trustyai_garak && \
    touch src/llama_stack_provider_trustyai_garak/__init__.py
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

# Copy real source and reinstall only our package (deps already cached above)
# Must happen as root so we can remove the egg-info created by the stub install
COPY src src
RUN rm -rf src/*.egg-info && pip install --no-cache-dir --no-deps ".[sdg]"

# Drop to non-root for runtime (UBI9 uses 1001)
USER 1001

# Default CMD for eval-hub (runs as K8s Job, simple mode — garak in same pod)
# For KFP mode (garak in a separate KFP pod), use:
#   CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub.kfp_adapter"]
CMD ["python", "-m", "llama_stack_provider_trustyai_garak.evalhub"]