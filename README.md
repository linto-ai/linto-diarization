# LinTO-diarization

LinTO-diarization is an API for Speaker Diarization (segmenting an audio stream into homogeneous segments according to the speaker identity),
with some capabilities for Speaker Identification when audio samples of known speakers are provided.

LinTO-diarization can currently work with several technologies.
The following families of technologies are currently supported (please refer to respective documentation for more details):
* [PyAnnote](pyannote/README.md)
* [simple_diarizer](simple/README.md)
* [PyBK](pybk/README.md) (deprecated)

LinTO-diarization can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

## Quick test

Below are examples of how to test diarization with "simple_diarizer", on Linux OS with docker installed.

"PyAnnote" is the recommended diarization method.
In what follow, you can replace "pyannote" by "simple" or "pybk" to try other methods.

### HTTP Server

1. If you want to use speaker identification, make sure Qdrant is running.
First, create a custom bridge network so the diarization container can communicate with qdrant :

```bash
docker network create diarization_network
```
 You can start Qdrant using the following Docker command:

```bash
docker run 
    --name qdrant \
    --network diarization_network \
    -p 6333:6333 \  # Qdrant default port
    -v ./qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

2. If needed, build docker image 

```bash
docker build . -t linto-diarization-pyannote:latest -f pyannote/Dockerfile
```  

3. Launch docker container (and keep it running)

If you want to enable speaker identification, make sure to mount reference speaker audio samples to `/opt/speaker_samples`.

```bash
docker run -it --rm \
    --name linto-diarization \
    --network diarization_network \
    -p 8080:80 \
    -v ./data/speakers_samples:/opt/speaker_samples \ # Reference speaker samples. Enables speaker identification
    --shm-size=1gb --tmpfs /run/user/0 \
    --env SERVICE_MODE=http \
    --env QDRANT_HOST=qdrant \ # Only specify if enabling speaker identification
    --env QDRANT_PORT=6333 \ # Only specify if enabling speaker identification
    --env QDRANT_COLLECTION_NAME=speaker_embeddings \ # Only specify if enabling speaker identification
    --env QDRANT_RECREATE_COLLECTION=true \ # Only specify if enabling speaker identification
    --env SERVICE_MODE=http \
    linto-diarization-pyannote:latest
```

Alternatively, you can use docker-compose :

```yaml

services:
  qdrant:
    image: qdrant/qdrant
    container_name: qdrant
    ports:
      - "6333:6333"  # Qdrant default port
    volumes:
      - ./qdrant_storage:/qdrant/storage:z

  diarization_app: 
    build: 
      context : .
      dockerfile: pyannote/Dockerfile
    container_name: diarization_app
    shm_size: '1gb'
    stdin_open: true
    tty: true     
    ports :
      - 8080:80
    environment:
      - QDRANT_HOST
      - QDRANT_PORT
      - QDRANT_COLLECTION_NAME
      - QDRANT_RECREATE_COLLECTION
      - SERVICE_MODE
      - SERVICE_NAME
      - SERVICES_BROKER
      - CONCURRENCY
    volumes:
      - ./data/speakers_samples:/opt/speaker_samples # Reference Speaker samples : This enables speaker identification
    depends_on:
      - qdrant  # Ensure Qdrant starts before the app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

```

Run it using this command :
```bash
docker compose up
```  

4. Open the swagger in a browser: [http://localhost:8080/docs](http://localhost:8080/docs)
   Unfold `/diarization` route and click "Try it out". Then
   - Choose a file
   - Specify either `speaker_count` (Fixed number of speaker) or `max_speaker` (Max number of speakers)
   - Click `Execute`

### Celery worker

In the following we assume we want to test on an audio that is in `$HOME/test.wav`

1. If needed, build docker image 

```bash
docker build . -t linto-diarization-pyannote:latest -f pyannote/Dockerfile
```

2. If you want to use speaker identification, make sure Qdrant is running. You can start Qdrant using the following Docker command:

```bash
docker run 
    -p 6333:6333 \  # Qdrant default port
    -v ./qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

3. Run Redis server

```bash
docker run -it --rm \
    -p 6379:6379 \
    redis/redis-stack-server:latest \
    redis-server /etc/redis-stack.conf --protected-mode no --bind 0.0.0.0 --loglevel debug
```

4. Launch docker container, attaching the volume where is the audio file on which you will test

```bash
docker run -it --rm \
    -v $HOME:$HOME \
    --env SERVICE_MODE=task \
    --env SERVICE_NAME=diarization \
    --env SERVICES_BROKER=redis://172.17.0.1:6379 \
    --env BROKER_PASS= \
    --env CONCURRENCY=2 \
    --env QDRANT_HOST=localhost \
    --env QDRANT_PORT=6333 \
    --env QDRANT_COLLECTION_NAME=speaker_embeddings \
    --env QDRANT_RECREATE_COLLECTION=true \
    linto-diarization-pyannote:latest
```

5. Testing with a given audio file can be done using python3 (with packages `celery` and `redis` installed).
   For example with the following command for the file `$HOME/test.wav` with 2 speakers

```bash
pip3 install redis celery # if not installed yet

python3 -c "\
import celery; \
import os; \
worker = celery.Celery(broker='redis://localhost:6379/0', backend='redis://localhost:6379/1'); \
print(worker.send_task('diarization_task', (os.environ['HOME']+'/test.wav', 2, None), queue='diarization').get());\
"
```

## License
This project is developped under the AGPLv3 License (see LICENSE).
