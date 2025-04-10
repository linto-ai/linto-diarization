FROM python:3.10
LABEL maintainer="contact@linto.ai, wghezaiel@linagora.com, jlouradour@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    curl \
    ffmpeg \
    wget \
    unzip \
    libsndfile1 \
    gosu \
    && \
    apt-get clean
   
# Define the main folder
WORKDIR /usr/src/app

# Install python dependencies
COPY pyannote/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download pyannote models
ENV HOME=/var/www
RUN mkdir -p $HOME/.cache
RUN wget https://dl.linto.ai/downloads/model-distribution/speaker-diarization/pyannote-3.1.zip
RUN unzip pyannote-3.1.zip -d $HOME/.cache/
RUN chmod -R ug+rwX $HOME
RUN rm pyannote-3.1.zip

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

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
