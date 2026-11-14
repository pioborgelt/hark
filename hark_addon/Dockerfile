# The Supervisor passes its Alpine base image as BUILD_FROM and current versions ignore
# build.yaml, so the Debian base is fixed here (multi-arch: amd64 + aarch64).
FROM python:3.12-slim-bookworm

# Released add-on versions install exactly this tag (keep in sync with config.yaml).
ARG HARK_PIP_SPEC="hark-ww[wyoming] @ git+https://github.com/pioborgelt/hark.git@v0.3.0-alpha"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARK_CACHE_DIR=/data/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        git \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"

# Development builds: scripts/stage_local_addon.sh puts a wheel and the release weights here.
COPY local/ /opt/hark-local/
RUN set -e; \
    wheel="$(ls /opt/hark-local/wheels/hark_ww-*.whl 2>/dev/null | head -n1 || true)"; \
    if [ -n "$wheel" ]; then \
        echo "Installing local development wheel $wheel"; \
        pip install --no-cache-dir "${wheel}[wyoming]"; \
    else \
        pip install --no-cache-dir "$HARK_PIP_SPEC"; \
    fi; \
    rm -rf /opt/hark-local/wheels

COPY run.py /run.py
RUN chmod a+x /run.py

EXPOSE 10400
CMD ["/run.py"]
