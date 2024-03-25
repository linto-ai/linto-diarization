# LinTO-diarization
LinTO-diarization is the [LinTO](https://linto.ai/) service for speaker diarization.

LinTO-diarization can either be used as a standalone diarization service or deployed as a micro-services.

* [Prerequisites](#pre-requisites)
* [Deploy](#deploy)
  * [HTTP](#http)
  * [MicroService](#micro-service)
* [Usage](#usages)
  * [HTTP API](#http-api)
    * [/healthcheck](#healthcheck)
    * [/diarization](#diarization)
    * [/docs](#docs)
  * [Using celery](#using-celery)

* [License](#license)
***

## Pre-requisites

### Docker

**This Docker image is built upon _nvidia/cuda:12.3.2-runtime-ubuntu22.04_**

The transcription service requires [docker](https://www.docker.com/products/docker-desktop/) up and running.

For GPU capabilities, it is also needed to install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

When using GPU, the consumed VRAM should be around 1GB per worker.


### (micro-service) Service broker and shared folder
The diarization only entry point in job mode are tasks posted on a Redis message broker.
Futhermore, to prevent large audio from transiting through the message broker, diarization uses a shared storage folder mounted on /opt/audio.

## Deploy
linto-diarization can be deployed:
* As a standalone diarization service through an HTTP API.
* As a micro-service connected to a message broker.

**1- First step is to build the image:**

```bash
git clone https://github.com/linto-ai/linto-diarization.git
cd linto-diarization
docker build . -t linto-diarization-simple:latest -f simple/Dockerfile
```

### HTTP

**1- Fill the .env**
```bash
cp .envdefault .env
```

Fill the .env with your values.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `SERVING_MODE` | (Required) Specify launch mode | `http` |
| `CONCURRENCY` | Number of worker(s) | `1` \| `2` \| ... |
| `DEVICE` | Device to use for the model (by default, GPU/CUDA is used if it is available, CPU otherwise) | `cpu` \| `cuda` |
| `NUM_THREADS` | Number of threads (maximum) to use for things running on CPU | `1` \| `4` \| ... |
| `CUDA_VISIBLE_DEVICES` | GPU device index to use, when running on GPU/CUDA. We also recommend to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines | `0` \| `1` \| `2` \| ... |


**2- Run the container**

```bash
docker run --rm \
-v <SHARED_FOLDER>:/opt/audio \
-p <HOST_SERVING_PORT>:80 \
--env-file .env \
linto-diarization-simple:latest
```

You may also want to add ```--gpus all``` to enable GPU capabilities
(and maybe set `CUDA_VISIBLE_DEVICES` if there are several available GPU cards).


This will run a container providing an http API binded on the host `<HOST_SERVING_PORT>` port.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `<HOST_SERVING_PORT>` | Host serving port | 80 |

### Using celery
>LinTO-diarization can be deployed as a micro-service using celery. Used this way, the container spawn celery worker waiting for diarization task on a message broker.

You need a message broker up and running at SERVICES_BROKER.

**1- Fill the .env**
```bash
cp .envdefault .env
```

Fill the .env with your values.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `SERVING_MODE` | (Required) Specify launch mode | `task` |
| `CONCURRENCY` | Number of worker(s) | `1` \| `2` \| ... |
| `DEVICE` | Device to use for the model (by default, GPU/CUDA is used if it is available, CPU otherwise) | `cpu` \| `cuda` |
| `NUM_THREADS` | Number of threads (maximum) to use for things running on CPU | `1` \| `4` \| ... |
| `CUDA_VISIBLE_DEVICES` | GPU device index to use, when running on GPU/CUDA. We also recommend to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines | `0` \| `1` \| `2` \| ... |
| `SERVICES_BROKER` | Service broker uri | `redis://my_redis_broker:6379` |
| `BROKER_PASS` | Service broker password (Leave empty if there is no password) | `my_password` |
| `QUEUE_NAME` | Overide the generated queue's name (See Queue name bellow) | `my_queue` |
| `SERVICE_NAME` | Service's name | `diarization-ml` |
| `LANGUAGE` | Language code as a BCP-47 code | `en-US` or * or languages separated by "\|" |
| `MODEL_INFO` | Human readable description of the model | `Multilingual diarization model` | 

**2- Fill the docker-compose.yml**

`#docker-compose.yml`
```yaml
version: '3.7'

services:
  punctuation-service:
    image: linto-diarization-simple:latest
    volumes:
      - /path/to/shared/folder:/opt/audio
    env_file: .env
    deploy:
      replicas: 1
    networks:
      - your-net

networks:
  your-net:
    external: true
```

**3- Run with docker compose**

```bash
docker stack deploy --resolve-image always --compose-file docker-compose.yml your_stack
```

**Queue name:**

By default the service queue name is generated as `SERVICE_NAME`.

The queue name can be overided using the `QUEUE_NAME` env variable. 

**Service discovery:**

As a micro-service, the instance will register itself in the service registry for discovery. The service information are stored as a JSON object in redis's db0 under the id `service:{HOST_NAME}`.

The following information are registered:

```json
{
  "service_name": $SERVICE_NAME,
  "host_name": $HOST_NAME,
  "service_type": "diarization",
  "service_language": $LANGUAGE,
  "queue_name": $QUEUE_NAME,
  "version": "1.2.0", # This repository's version
  "info": $MODEL_INFO,
  "last_alive": 65478213,
  "concurrency": 1
}
```



## Usages

### HTTP API

#### /healthcheck

Returns the state of the API

Method: GET

Returns "1" if healthcheck passes.

#### /diarization

Diarization API

* Method: POST
* Response content: application/json
* File: A Wave file
* spk_number: (integer - optional) Number of speakers. If empty, diarization will clusterize automatically.
* max_speaker: (integer - optional) Max number of speakers if spk_number is unknown. 

Return a json object when using structured as followed:
```json
{
  "speakers": [
      {"spk_id": "spk5", "duration": 2.2, "nbr_seg": 1},
      ...
  ],
  "segments": [
      {"seg_id": 1, "spk_id": "spk5", "seg_begin": 0.0, "seg_end": 2.2},
      ...
  ]
}
```

#### /docs
The /docs route offers a OpenAPI/swagger interface. 

### Through the message broker

Diarization worker accepts requests with the following arguments:

* `file_path`: (str) Is the location of the file within the shared_folder. /.../SHARED_FOLDER/{file_path}
* `speaker_count`: (int default None) Fixed number of speakers.
* `max_speaker`: (int default None) Max number of speaker if speaker_count=None. 

#### Return format
On a successfull transcription the returned object is a json object structured as follow:
```json
{
  "speakers": [
      {"spk_id": "spk5", "duration": 2.0, "nbr_seg": 1},
      ...
  ],
  "segments": [
      {"seg_id": 1, "spk_id": "spk5", "seg_begin": 0.0, "seg_end": 2.0},
      ...
  ]
}
```

* The `speakers` field contains an arraw of speaker with overall duration and number of segments.
* The `segments` field contains each audio segment with the associated speaker id start time and end time.

## Test
### Curl
You can test you http API using curl:
```bash 
curl -X POST "http://YOUR_SERVICE:PORT/diarization" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@YOUR_FILE.wav;type=audio/x-wav" -F "spk_number=NUMBER_OF_SPEAKERS"
```

## License
This project is developped under the AGPLv3 License (see LICENSE).

## Acknowlegment.

* [cvqluu/simple_diarizer](https://github.com/cvqluu/simple_diarizer) Diarization framework (License GPL v3).
* [tango4j/Auto-Tuning-Spectral-Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering) Auto-tuning spectral clustering (License MIT).
* [desh2608/diarizer](https://github.com/desh2608/diarizer) Several diarization methods.