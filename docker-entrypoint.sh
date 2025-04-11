#!/bin/bash
set -e

echo "Diarization starting..."

# Set default UID and GID (defaults to www-data: 33:33 if not specified)
USER_ID=${USER_ID:-33}
GROUP_ID=${GROUP_ID:-33}

# Default values for user and group names
USER_NAME="appuser"
GROUP_NAME="appgroup"

# Function to create a user/group if needed and adjust permissions
function setup_user() {
    echo "Configuring runtime user with UID=$USER_ID and GID=$GROUP_ID"

    # Check if a group with the specified GID already exists
    if getent group "$GROUP_ID" >/dev/null 2>&1; then
        GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)
        echo "A group with GID=$GROUP_ID already exists: $GROUP_NAME"
    else
        # Create the group if it does not exist
        echo "Creating group with GID=$GROUP_ID"
        groupadd -g "$GROUP_ID" "$GROUP_NAME"
    fi

    # Check if a user with the specified UID already exists
    if id -u "$USER_ID" >/dev/null 2>&1; then
        USER_NAME=$(getent passwd "$USER_ID" | cut -d: -f1)
        echo "A user with UID=$USER_ID already exists: $USER_NAME"
    else
        # Create the user if it does not exist
        echo "Creating user with UID=$USER_ID and GID=$GROUP_ID"
        useradd -m -u "$USER_ID" -g "$GROUP_NAME" "$USER_NAME"
    fi

    # Adjust ownership of the application directories
    echo "Adjusting ownership of application directories"
    chown -R "$USER_NAME:$GROUP_NAME" /usr/src/app

    # Get the user's home directory from the system
    USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)

    # Ensure the home directory exists
    if [ ! -d "$USER_HOME" ]; then
        echo "Ensure home directory exists: $USER_HOME"
        mkdir -p "$USER_HOME"
        chown -R "$USER_NAME:$GROUP_NAME" "$USER_HOME"
    fi

    # Grant full permissions to the user on their home directory
    echo "Granting full permissions to $USER_NAME on $USER_HOME"
    chmod -R u+rwx "$USER_HOME"

     # Grant full permissions to /opt for user $USER_NAME
    echo "Granting full permissions to $USER_NAME on /opt"
    chown -R "$USER_NAME:$GROUP_NAME" /opt
}

check_gpu_availability() {
    if [ "$DEVICE" == "cpu" ];then
        GPU_AVAILABLE=0
        echo "GPU disabled"
    else
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
    fi

}

# Wait for Qdrant to be available
wait_for_qdrant() {
    # Check if QDRANT_HOST and QDRANT_PORT are set
    if [[ -z "${QDRANT_HOST}" || -z "${QDRANT_PORT}" ]]; then
        echo "Qdrant environment variables are not set. Skipping wait for Qdrant."
        return 0
    fi
    echo "Waiting for Qdrant to be reachable..."
    /usr/src/app/wait-for-it.sh "${QDRANT_HOST}:${QDRANT_PORT}" --timeout=20 --strict -- echo "Qdrant is up"
    if [ $? -ne 0 ]; then
        echo "ERROR: Qdrant service not reachable at ${QDRANT_HOST}:${QDRANT_PORT}"
        exit 1
    fi
}

run_http_server() {
    echo "HTTP server Mode"
    gosu "$USER_NAME" python http_server/ingress.py --debug
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
    gosu "$USER_NAME" celery --app=celery_app.celeryapp worker $OPT -Ofair -n ${SERVICE_NAME}_worker@%h --queues=$QUEUE -c ${CONCURRENCY:-1} || exit $?
    ## UNREGISTERING
    python -c "from celery_app.register import unregister; unregister()" || exit $?
    echo "Service unregistered"
}

# Main logic
setup_user
check_gpu_availability
wait_for_qdrant

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
