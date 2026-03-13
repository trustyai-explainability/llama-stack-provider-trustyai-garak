FROM registry.access.redhat.com/ubi9/python-312:latest
WORKDIR /opt/app-root

# Switch to root only for installing packages
USER root

COPY . .

# Install cpu torch to reduce image size
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the package itself
# Use [inline] to get garak dependency
RUN pip install --no-cache-dir ".[inline]"
# Install midstream garak and sdg-hub dependencies (tmp fix till we get release versions)
RUN pip install --no-cache-dir -r requirements-inline-extra.txt
# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_DATA_HOME=/tmp/.local/share
ENV XDG_CONFIG_HOME=/tmp/.config

# Switch back to non-root user
# UBI9 uses 1001
USER 1001
