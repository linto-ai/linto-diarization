#!/usr/bin/env bash

set -eax

# Put last healthcheck logs in an file (You might persist this for healthcheck failures analysis as the container is ephemeral)
# Otherwise healthcheck logs are found in docker inspect <container_id>
exec > /tmp/healthcheck.log 2>&1

if [ "$SERVICE_MODE" = "http" ]
then
    # HTTP mode healthcheck
    curl --fail http://localhost:80/healthcheck || exit 1
else
    # Update last alive
    python -c "from celery_app.register import register; register(False)"

    # Check if Celery worker process is running
    PID=`pgrep -f "celeryapp worker"`
    if [ -z "$PID" ]; then
        echo "HealthCheck FAIL: Celery worker process not running"
        exit 1
    fi

    # Check if GPU is in use
    has_gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | grep -v '^0$' | wc -l)
    if [ "$has_gpu" -gt 0 ]; then
        echo "HealthCheck PASS: GPU is being utilized, marking service as healthy."
        exit 0
    fi

    # Attempt to ping Celery worker
    if ! celery --app=celery_app.celeryapp inspect ping -d ${SERVICE_NAME}_worker@$HOSTNAME --timeout=20; then
        echo "HealthCheck FAIL: Celery worker not responding in time and GPU is not being utilized"
        exit 1
    fi

    echo "HealthCheck PASS: Celery worker is responsive but idle, marking service as healthy."
    exit 0
fi