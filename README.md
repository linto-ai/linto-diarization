# LinTO-diarization

LinTO-diarization is an API for Speaker Diarization (segmenting an audio stream into homogeneous segments according to the speaker identity),
with some capabilities for Speaker Identification when audio samples of known speakers are provided.

LinTO-diarization can currently work with several technologies.
The following families of technologies are currently supported (please refer to respective documentation for more details):
* [PyBK](pybk/README.md) 
* [PyAnnote](pyannote/README.md)
* [simple_diarizer](simple/README.md)

LinTO-diarization can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

## Quick test

Below are examples of how to test diarization with "simple_diarizer", on Linux OS with docker installed.

"simple_diarizer" is the recommended diarization method.
In what follow, you can replace "simple" by "pybk" or "pyannote" to try other methods.

### HTTP Server

1. If needed, build docker image 

```bash
docker build . -t linto-diarization-simple:latest -f simple/Dockerfile
```

2. Launch docker container (and keep it running)

```bash
docker run -it --rm \
    -p 8080:80 \
    --shm-size=1gb --tmpfs /run/user/0 \
    --env SERVICE_MODE=http \
    linto-diarization-simple:latest
```

3. Open the swagger in a browser: [http://localhost:8080/docs](http://localhost:8080/docs)
   Unfold `/diarization` route and click "Try it out". Then
   - Choose a file
   - Specify either `speaker_count` (Fixed number of speaker) or `max_speaker` (Max number of speakers)
   - Click `Execute`

### Celery worker

In the following we assume we want to test on an audio that is in `$HOME/test.wav`

1. If needed, build docker image 

```bash
docker build . -t linto-diarization-simple:latest -f simple/Dockerfile
```

2. Run Redis server

```bash
docker run -it --rm \
    -p 6379:6379 \
    redis/redis-stack-server:latest \
    redis-server /etc/redis-stack.conf --protected-mode no --bind 0.0.0.0 --loglevel debug
```

3. Launch docker container, attaching the volume where is the audio file on which you will test

```bash
docker run -it --rm \
    -v $HOME:$HOME \
    --env SERVICE_MODE=task \
    --env SERVICE_NAME=diarization \
    --env SERVICES_BROKER=redis://172.17.0.1:6379 \
    --env BROKER_PASS= \
    --env CONCURRENCY=2 \
    linto-diarization-simple:latest
```

3. Testing with a given audio file can be done using python3 (with packages `celery` and `redis` installed).
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
