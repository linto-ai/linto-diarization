FROM python:3.10
LABEL maintainer="contact@linto.ai, wghezaiel@linagora.com, jlouradour@linagora.com"

# Re-Sets work directory (not sure if this is necessary)
WORKDIR /usr/src/app

# Install python and other dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip ffmpeg git sqlite3 libsqlite3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# # Set a common work directory
WORKDIR /usr/src/app

# Install python packages
# Reuse labels and base dependencies from the base stage
COPY simple/requirements.txt /usr/src/app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY simple/diarization /usr/src/app/diarization
COPY simple/requirements.txt /usr/src/app/
COPY simple/RELEASE.md ./
COPY identification /usr/src/app/identification

# Set path
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"

# Extract version information
RUN export VERSION=$(awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //')


# Health check and entrypoint for GPU target
HEALTHCHECK CMD ./healthcheck.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
