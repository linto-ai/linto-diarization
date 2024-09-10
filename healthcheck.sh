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

    # Check if GPU is in used
    has_gpu=$(nvidia-smi --query-compute-apps pid --format=csv,noheader | wc -l)
    # Note: we should add a check on the PID ("| grep $PID")... BUT the PID can be different in the container than on the host

    # Attempt a ping
    if ! celery --app=celery_app.celeryapp inspect ping -d ${SERVICE_NAME}_worker@$HOSTNAME --timeout=20; then
        # Check GPU utilization
        if [ $has_gpu -gt 0 ]; then
            # GPU is being utilized, assuming healthy
            echo "Celery worker not responding in time but GPU is being utilized (trying to ping again)"
            exit 0
        fi
        echo "HealtchCheck FAIL : Celery worker not responding in time and GPU is not being utilized"
        exit 1
    fi
fi