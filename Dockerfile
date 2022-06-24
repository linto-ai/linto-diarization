FROM python:3.10.4-slim-bullseye
LABEL maintainer="irebai@linagora.com, rbaraglia@linagora.com, wghezaiel@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    nano \
    sox  \
    ffmpeg \
    software-properties-common \
    wget \
    curl \
    lsb-release && \
    apt-get clean

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

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]