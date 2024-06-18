#!/usr/bin/env bash

set -eax

if [ "$SERVICE_MODE" = "http" ]
then
    curl --fail http://localhost:80/healthcheck || exit 1
else
    # Update last alive, using Heartbeat(register(True))
    python -c "from celery_app.register import register; register(True)"

    # Check if Celery worker process is running
    # warning : We might rework this when switching to non-root user. Which is required for security reasons.
    # Our docker images are currently running as root and this is bad :)
    if ! pgrep -f "celeryapp worker"; then
        echo "HealtchCheck FAIL : Celery worker process not running"
        exit 1
    fi

    # Check GPU utilization
    if nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | grep -v '^0$'; then
        #"GPU is being utilized, assuming healthy"
        exit 0
    else
        # If GPU is not being utilized, attempt a ping as a secondary check
        if ! celery --app=celery_app.celeryapp inspect ping -d ${SERVICE_NAME}_worker@$HOSTNAME --timeout=20; then
            echo "HealtchCheck FAIL : Celery worker not responding in time and GPU is not being utilized"
            exit 1
        fi
    fi
fi