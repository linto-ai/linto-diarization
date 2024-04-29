FROM python:3.10
LABEL maintainer="contact@linto.ai, wghezaiel@linagora.com, jlouradour@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    curl \
    wget \
    unzip \
    libsndfile1 \
    && \
    apt-get clean

# Download pyannote models
RUN mkdir -p $HOME/.cache
#RUN wget https://dl.linto.ai/downloads/model-distribution/speaker-diarization/pyannote-2.1.zip
#RUN unzip pyannote-2.1.zip -d $HOME/.cache/
#RUN rm pyannote-2.1.zip

RUN apt-get remove -y \
    wget \
    unzip \
    && \
    apt-get clean
   
# Define the main folder
WORKDIR /usr/src/app

# Install python dependencies
COPY pyannote/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY pyannote/diarization /usr/src/app/diarization

# Grep CURRENT VERSION
COPY pyannote/RELEASE.md ./
RUN export VERSION=$(awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //')

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"

# Limits on OPENBLAS number of thread prevent SEGFAULT on machine with a large number of cpus
ENV OPENBLAS_NUM_THREADS=32
ENV GOTO_NUM_THREADS=32
ENV OMP_NUM_THREADS=32

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
