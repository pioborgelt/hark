ARG BUILD_FROM=python:3.12-slim-bookworm
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARK_CACHE_DIR=/data/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        git \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3" \
    && pip install --no-cache-dir "hark-ww[wyoming] @ git+https://github.com/pioborgelt/hark.git@main"

COPY run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 10400
CMD ["/run.sh"]
