FROM python:3.10
LABEL maintainer="contact@linto.ai, wghezaiel@linagora.com"

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
RUN apt-add-repository "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-18 main" && apt-get update
RUN ./llvm.sh 18
RUN export LLVM_CONFIG=/usr/bin/llvm-config-18
    
# Define the main folder
WORKDIR /usr/src/app

# Install python dependencies
COPY pybk/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY document /usr/src/app/document
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY pybk/diarization /usr/src/app/diarization
COPY pybk/pyBK/diarizationFunctions.py pyBK/diarizationFunctions.py

# Grep CURRENT VERSION
COPY pybk/RELEASE.md ./
RUN export VERSION=$(awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //')

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/diarization"

# Limits on OPENBLAS number of thread prevent SEGFAULT on machine with a large number of cpus
ENV OPENBLAS_NUM_THREADS=32
ENV GOTO_NUM_THREADS=32
ENV OMP_NUM_THREADS=32

HEALTHCHECK CMD ./healthcheck.sh

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
