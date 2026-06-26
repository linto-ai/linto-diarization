FROM python:3.10-slim
LABEL maintainer="contact@linto.ai, wghezaiel@linagora.com, jlouradour@linagora.com, dlaine@linagora.com"

RUN apt-get update &&\
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    wget \
    unzip \
    libsndfile1 \
    gosu \
    procps \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Define the main folder
WORKDIR /usr/src/app

# Install python dependencies
COPY pyannote/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The cu126 index pulls the CUDA build of torchcodec, which links NVIDIA NPP
# (libnppicc.so.12) — absent from the slim image, so libtorchcodec fails to load.
# pyannote only uses torchcodec for (CPU) audio decoding, so install the CPU build.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torchcodec==0.7.0 --index-url https://download.pytorch.org/whl/cpu

# Download pyannote community-1 diarization model (pyannote.audio 4.x).
# The archive is expected to extract to /opt/models/speaker-diarization-community-1
# (containing config.yaml and the embedding/, plda/, segmentation/ subfolders).
RUN mkdir -p /opt/models
RUN wget https://dl.linto.ai/downloads/model-distribution/speaker-diarization/pyannote-community-1.zip
RUN unzip pyannote-community-1.zip -d /opt/models
RUN rm pyannote-community-1.zip

# Install sources to serve API
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY pyannote/diarization /usr/src/app/diarization
COPY identification /usr/src/app/identification

# Set VERSION number
COPY pyannote/RELEASE.md ./
RUN export VERSION=$(awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //')

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"
ENV PYANNOTE_CACHE=/opt/models

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
