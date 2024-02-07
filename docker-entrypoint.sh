#!/bin/bash
set -a

echo "RUNNING Diarization"

# Launch parameters, environement variables and dependencies check
if [ -z "$SERVICE_MODE" ]
then
    echo "ERROR: Must specify a serving mode: [ http | task ]"
    exit -1
else
    if [ "$SERVICE_MODE" = "http" ] 
    then
        echo "RUNNING DIARIZATION HTTP SERVER"
        python http_server/ingress.py --debug
    elif [ "$SERVICE_MODE" == "task" ]
    then
        if [[ -z "$SERVICES_BROKER" ]]
        then 
            echo "ERROR: SERVICES_BROKER variable not specified, cannot start celery worker."
            exit -1
        fi

        nvidia-smi 2> /dev/null > /dev/null
        if [ $? -eq 0 ];then
            echo "GPU detected"
            OPT="--pool=solo"
        else
            echo "No GPU detected"
            OPT=""
        fi

        echo "Running celery worker"
        /usr/src/app/wait-for-it.sh $(echo $SERVICES_BROKER | cut -d'/' -f 3) --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up" || exit $?
        # MICRO SERVICE
        ## QUEUE NAME
        QUEUE=$(python -c "from celery_app.register import queue; exit(queue())" 2>&1)
        echo "Service set to $QUEUE"

        ## REGISTRATION
        python -c "from celery_app.register import register; register()" || exit $?
        echo "Service registered"

        ## WORKER
        celery --app=celery_app.celeryapp worker $OPT -Ofair -n diarization_worker@%h --queues=$QUEUE -c $CONCURRENCY || exit $?

        ## UNREGISTERING
        python -c "from celery_app.register import unregister; unregister()" || exit $?
        echo "Service unregistered"
        
    else
        echo "ERROR: Wrong serving command: $1"
        exit -1
    fi
fi

echo "Service stopped"