#!/usr/bin/env bash

set -eax

if [ "$SERVICE_MODE" = "http" ]
then
    curl --fail http://localhost:80/healthcheck || exit 1
else
    # Update last alive
    python -c "from celery_app.register import register; register(False)"

    # Check if Celery worker process is running
    # warning : We might rework this when switching to non-root user. Which is required for security reasons.
    # Our docker images are currently running as root and this is bad :)
    PID=`pgrep -f "celeryapp worker"`
    if [ -z "$PID" ]; then
        echo "HealtchCheck FAIL : Celery worker process not running"
        exit 1
    fi

    # Attempt a ping
    while [ 1 -gt 0 ];do
    if ! celery --app=celery_app.celeryapp inspect ping -d diarization_worker@$HOSTNAME --timeout=20; then
        # Check GPU utilization
        if nvidia-smi --query-compute-apps pid --format=csv,noheader | grep $PID; then
            # GPU is being utilized, assuming healthy
            continue
        fi
        echo "HealtchCheck FAIL : Celery worker not responding in time and GPU is not being utilized"
        exit 1
    fi
    break
    done
fi