FROM python:3.10
LABEL maintainer="rbaraglia@linagora.com, wghezaiel@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    nano \
    sox  \
    ffmpeg \
    software-properties-common \
    wget \
    curl \
    unzip \
    lsb-release && \
    apt-get clean

RUN apt-get --yes install libsndfile1

# Install pyBK dependencies
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
RUN ./llvm.sh 11
RUN export LLVM_CONFIG=/usr/bin/llvm-config-10
    
# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Define the main folder
WORKDIR /usr/src/app

COPY diarization /usr/src/app/diarization
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY pyBK/diarizationFunctions.py pyBK/diarizationFunctions.py
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./

# Grep CURRENT VERSION
COPY RELEASE.md ./
RUN export VERSION=$(awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //')

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"

# Limits on OPENBLAS number of thread prevent SEGFAULT on machine with a large number of cpus
ENV OPENBLAS_NUM_THREADS=32
ENV GOTO_NUM_THREADS=32
ENV OMP_NUM_THREADS=32

RUN mkdir -p /root/.cache
RUN wget https://dl.linto.ai/downloads/model-distribution/speaker-diarization/pyannote-2.1.zip
RUN unzip pyannote-2.1.zip -d /root/.cache/
RUN rm pyannote-2.1.zip

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
