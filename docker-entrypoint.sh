#!/bin/bash
set -e

echo "Diarization starting..."

check_gpu_availability() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi 2> /dev/null > /dev/null && GPU_AVAILABLE=1 || GPU_AVAILABLE=0
    else
        GPU_AVAILABLE=0
    fi

    if [ $GPU_AVAILABLE -eq 1 ]; then
        echo "GPU detected"
    else
        echo "No GPU detected"
    fi
}

run_http_server() {
    echo "HTTP server Mode"
    python http_server/ingress.py --debug
    # TODO: Service port as an environment variable would be a good idea --> switching from 80 enables to run the service as a non-root user and avoid setcaps
}

run_celery_worker() {
    echo "Worker Mode"
    if [ $GPU_AVAILABLE -eq 1 ]; then
        OPT="--pool=solo"
    else
        OPT=""
    fi

    if [[ -z "$SERVICES_BROKER" ]]; then 
        echo "ERROR: SERVICES_BROKER variable not specified, cannot start celery worker."
        exit 1
    fi
    echo "Running celery worker"
    /usr/src/app/wait-for-it.sh "$(echo $SERVICES_BROKER | cut -d'/' -f 3)" --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up" || exit $?
    # MICRO SERVICE
    ## QUEUE NAME
    QUEUE=$(python -c "from celery_app.register import queue; print(queue())")
    echo "Service set to $QUEUE"
    ## REGISTRATION
    python -c "from celery_app.register import register; register()" || exit $?
    echo "Service registered"
    ## WORKER
    celery --app=celery_app.celeryapp worker $OPT -Ofair -n diarization_worker@%h --queues=$QUEUE -c ${CONCURRENCY:-1} || exit $?
    ## UNREGISTERING
    python -c "from celery_app.register import unregister; unregister()" || exit $?
    echo "Service unregistered"
}

# Main logic
check_gpu_availability

# check for FORCE_CPU environment variable

if [ $GPU_AVAILABLE -eq 1 ]; then
    echo "GPU is available. Diarization will run on GPU."
else
    echo "Diarization will run on CPU."
    export CUDA_VISIBLE_DEVICES=""
fi

if [ -z "$SERVICE_MODE" ]; then
    echo "ERROR: Must specify a serving mode: [ http | task ]"
    exit 1
elif [ "$SERVICE_MODE" = "http" ]; then
    run_http_server
elif [ "$SERVICE_MODE" == "task" ]; then
    run_celery_worker
else
    echo "ERROR: Wrong serving mode: $SERVICE_MODE"
    exit 1
fi

echo "Service stopped"