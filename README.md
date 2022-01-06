# LINTO-PLATFORM-DIARIZATION
LinTO-platform-diarization is the speaker diarization service within the [LinTO stack](https://github.com/linto-ai/linto-platform-stack).

LinTO-platform-diarization can either be used as a standalone diarization service or deployed within a micro-services infrastructure using a message broker connector.

## Pre-requisites

### Docker
The transcription service requires docker up and running.

### (micro-service) Service broker and shared folder
The Siarization only entry point in job mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, lp-diarization use a shared storage folder.

## Deploy linto-platform-diarization
linto-platform-stt can be deployed three ways:
* As a standalone diarization service through an HTTP API.
* As a micro-service connected to a message broker.

**1- First step is to build the image:**

```bash
git clone https://github.com/linto-ai/linto-platform-diarization.git
cd linto-platform-diarization
git submodule init
git submodule update
docker build . -t linto-platform-diarization:latest
```

### HTTP API

```bash
docker run --rm \
-p HOST_SERVING_PORT:80 \
--env SERVICE_MODE=http \
linto-platform-diarization:latest
```

This will run a container providing an http API binded on the host HOST_SERVING_PORT port.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| HOST_SERVING_PORT | Host serving port | 80 |

### Micro-service within LinTO-Platform stack
>LinTO-platform-diarization can be deployed within the linto-platform-stack through the use of linto-platform-services-manager. Used this way, the container spawn celery worker waiting for diarization task on a message broker.
>LinTO-platform-diarization in task mode is not intended to be launch manually.
>However, if you intent to connect it to your custom message's broker here are the parameters:

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v AM_PATH:/opt/models/AM \
-v LM_PATH:/opt/models/LM \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env SERVICES_BROKER=MY_SERVICE_BROKER \
--env BROKER_PASS=MY_BROKER_PASS \
--env SERVICE_MODE=task \
--env CONCURRENCY=1 \
linstt:dev
```

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| SERVICES_BROKER | Service broker uri | redis://my_redis_broker:6379 |
| BROKER_PASS | Service broker password (Leave empty if there is no password) | my_password |
| CONCURRENCY | Number of worker (1 worker = 1 cpu) | [ 1 -> numberOfCPU] |

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
* File: An Wave file
* spk_number: (integer - optional) Number of speakers. If empty, diarization will guess.

Return a json object when using structured as followed:
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

#### /docs
The /docs route offers a OpenAPI/swagger interface. 

### Through the message broker

STT-Worker accepts requests with the following arguments:
```file_path: str, with_metadata: bool```

* <ins>file_path</ins>: (str) Is the location of the file within the shared_folder. /.../SHARED_FOLDER/{file_path}
* <ins>speaker_count</ins>: (int default None) Fixed number of speakers. 

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

* The <ins>speakers</ins> field contains an arraw of speaker with overall duration and number of segments.
* The <ins>segments</ins> field contains each audio segment with the associated speaker id start time and end time.

## Test
### Curl
You can test you http API using curl:
```bash 
curl -X POST "http://YOUR_SERVICE:PORT/diarization" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@YOUR_FILE.wav;type=audio/x-wav" -F "spk_number=NUMBER_OF_SPEAKERS"
```

## License
This project is developped under the AGPLv3 License (see LICENSE).

## Acknowlegment.

* [Vosk, speech recognition toolkit](https://alphacephei.com/vosk/).
* [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi)
