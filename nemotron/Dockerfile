# Nemotron 3 Diarization (NVIDIA Sortformer) worker. Build from the repository root:
#   docker build -f nemotron/Dockerfile -t lintoai/linto-diarization-nemotron .

ARG PYTHON_IMAGE=python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9@sha256:538e0b39736e7feae937a65983e49d2ab75e1559d35041f9878b7b7e51de91e4

FROM ${UV_IMAGE} AS uv

################################################################################
# Builder: resolve the locked environment (hashes checked by uv) into /usr/local/venv
################################################################################
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/usr/local/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra nemotron --no-install-project --no-cache

################################################################################
# Models: fetched once at build time, checksum-verified. No HuggingFace access at runtime.
################################################################################
FROM builder AS models

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# nvidia/Nemotron-3-Diarization, HuggingFace revision f667ed73aee57d40cc39428eb768b4fd87a0a29e (OpenMDW-1.1)
ARG NEMOTRON_MODEL_URL=https://dl.linto.ai/downloads/model-distribution/speaker-diarization/Nemotron-3-Diarization-f667ed73.nemo
ARG NEMOTRON_MODEL_SHA256=867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d
RUN mkdir -p /opt/models/nemotron-3-diarization && \
    curl -fsSL "$NEMOTRON_MODEL_URL" -o /opt/models/nemotron-3-diarization/Nemotron-3-Diarization.nemo && \
    echo "$NEMOTRON_MODEL_SHA256  /opt/models/nemotron-3-diarization/Nemotron-3-Diarization.nemo" | sha256sum -c -
# OpenMDW-1.1 requires the license text and origin notices to ship with the model
COPY nemotron/model-license/ /opt/models/nemotron-3-diarization/

# ECAPA speaker-embedding model (speaker identification), pinned to MODEL_REVISION in
# identification/spkid_core.py: with that revision in the cache, huggingface_hub never
# touches the network.
ENV HF_HOME=/opt/models/huggingface
RUN /usr/local/venv/bin/python -c "\
from huggingface_hub import hf_hub_download; \
[hf_hub_download('speechbrain/spkrec-ecapa-voxceleb', f, revision='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286') \
 for f in ('hyperparams.yaml', 'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt', 'label_encoder.txt')]"

################################################################################
# Runtime: slim base, no compiler, git, uv or downloader
################################################################################
FROM ${PYTHON_IMAGE} AS runtime

LABEL maintainer="contact@linto.ai" \
      org.opencontainers.image.source="https://github.com/linto-ai/linto-diarization" \
      org.opencontainers.image.title="linto-diarization-nemotron"

# ffmpeg: audio decoding (torchaudio/torchcodec, voiceprint conversion)
# curl: HTTP-mode healthcheck. procps: pgrep in healthcheck. gosu: drop root in the entrypoint.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl procps gosu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/venv /usr/local/venv
COPY --from=models /opt/models /opt/models

ENV PATH="/usr/local/venv/bin:${PATH}" \
    HF_HOME=/opt/models/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    NEMOTRON_MODEL=/opt/models/nemotron-3-diarization/Nemotron-3-Diarization.nemo \
    DIARIZATION_ENGINE=nemotron \
    DIARIZATION_MAX_SPEAKERS=8

# NeMo compiles flex_attention with Triton, which needs a C compiler at runtime. Eager mode:
# same diarization quality, about 5 times slower, no compiler (see runtime-compiled below).
ENV TORCHDYNAMO_DISABLE=1

WORKDIR /usr/src/app
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY identification /usr/src/app/identification
COPY nemotron/diarization /usr/src/app/diarization
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh nemotron/RELEASE.md ./

ENV PYTHONPATH="/usr/src/app:/usr/src/app/diarization"

HEALTHCHECK CMD ./healthcheck.sh
ENTRYPOINT ["./docker-entrypoint.sh"]

################################################################################
# Variant with a C compiler (build with --target runtime-compiled): flex_attention is
# compiled at first use (about 3 s), kernels cached under /opt/cache, which the entrypoint
# hands to the runtime user. A failed compilation falls back to eager. Compile threads = 1:
# the compile worker pool hangs under gevent (HTTP mode).
################################################################################
FROM runtime AS runtime-compiled

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc libc6-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TORCHDYNAMO_DISABLE=0 \
    TORCHDYNAMO_SUPPRESS_ERRORS=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    TRITON_CACHE_DIR=/opt/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/opt/cache/torchinductor

################################################################################
# Default image: eager, no compiler
################################################################################
FROM runtime
