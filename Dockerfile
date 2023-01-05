FROM python:3.10
LABEL maintainer="wghezaiel@linagora.com, jlouradour@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    curl \
    wget \
    unzip \
    libsndfile1 \
    && \
    apt-get clean



RUN apt-get remove -y \
    wget \
    unzip \
    && \
    apt-get clean
    
# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Define the main folder
WORKDIR /usr/src/app

COPY diarization /usr/src/app/diarization
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY simple_diarizer/cluster.py simple_diarizer/cluster.py
COPY simple_diarizer/diarizer.py simple_diarizer/diarizer.py
COPY simple_diarizer/utils.py simple_diarizer/utils.py
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./

# Grep CURRENT VERSION
COPY RELEASE.md ./
RUN export VERSION=$(awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //')

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"

# Limits on OPENBLAS number of thread prevent SEGFAULT on machine with a large number of cpus
ENV OPENBLAS_NUM_THREADS=32
ENV GOTO_NUM_THREADS=32
ENV OMP_NUM_THREADS=32

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
