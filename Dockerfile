FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARK_CACHE_DIR=/data/models \
    HARK_URI=tcp://0.0.0.0:10400 \
    HARK_PHRASES="Okay Haus:de;Hey Jarvis:en" \
    HARK_SENSITIVITY=balanced \
    HARK_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"

COPY pyproject.toml LICENSE README.md ./
COPY hark ./hark

RUN pip install --no-cache-dir ".[wyoming]" && mkdir -p /data/models

VOLUME ["/data"]
EXPOSE 10400

ENTRYPOINT ["wyoming-hark"]
