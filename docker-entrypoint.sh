#!/bin/bash
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
        echo "Running celery worker"
        /usr/src/app/wait-for-it.sh $(echo $SERVICES_BROKER | cut -d'/' -f 3) --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up"
        # MICRO SERVICE
        ## QUEUE NAME
        QUEUE=$(python -c "from celery_app.register import queue; exit(queue())" 2>&1)
        echo "Service set to $QUEUE"

        ## REGISTRATION
        python -c "from celery_app.register import register; register()"
        echo "Service registered"

        ## WORKER
        celery --app=celery_app.celeryapp worker -Ofair -n diarization_worker@%h --queues=$QUEUE -c $CONCURRENCY

        ## UNREGISTERING
        python -c "from celery_app.register import unregister; unregister()"
        echo "Service unregistered"
        
    else
        echo "ERROR: Wrong serving command: $1"
        exit -1
    fi
fi

echo "Service stopped"