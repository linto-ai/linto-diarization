#!/usr/bin/env bash

set -eax

# Put last healthcheck logs in an file (You might persist this for healthcheck failures analysis as the container is ephemeral)
# Otherwise healthcheck logs are found in docker inspect <container_id>
exec > /tmp/healthcheck.log 2>&1

if [ "$SERVICE_MODE" = "http" ]
then
    # HTTP mode healthcheck
    curl --fail http://localhost:${SERVICE_PORT:-80}/healthcheck || exit 1
else
    # Update last alive
    python -c "from celery_app.register import register; register(False)"

    # Check if Celery worker process is running
    PID=`pgrep -f "celeryapp worker"`
    if [ -z "$PID" ]; then
        echo "HealthCheck FAIL: Celery worker process not running"
        exit 1
    fi

    # Attempt to ping Celery worker (answers even while a task runs, see docker-entrypoint.sh)
    if ! celery --app=celery_app.celeryapp inspect ping -d ${SERVICE_NAME}_worker@$HOSTNAME --timeout=20; then
        echo "HealthCheck FAIL: Celery worker not responding in time"
        exit 1
    fi

    echo "HealthCheck PASS: Celery worker is responsive, marking service as healthy."
    exit 0
fi