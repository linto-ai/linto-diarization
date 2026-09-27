# LinTO-diarization

Speaker diarization service for [LinTO](https://linto.ai/): it splits an audio file into speech turns and tells
who speaks when (`spk1`, `spk2`...). It can also put names on the speakers by comparing their voice with enrolled
voiceprints (speaker identification).

It runs either as a Celery worker behind [linto-transcription-service](https://github.com/linto-ai/linto-transcription-service)
(task mode) or as a standalone HTTP API.

## Engines

One Docker image per engine. All of them share the Celery task, the HTTP API, the result format and speaker
identification.

| Engine | Image | Use it for | Documentation |
|---|---|---|---|
| pyannote | `lintoai/linto-diarization-pyannote` | default, any number of speakers | [pyannote/README.md](pyannote/README.md) |
| Nemotron | `lintoai/linto-diarization-nemotron` (and `-compiled` tags) | meetings up to 8 speakers: more accurate and faster; also live streams over websocket ([nemotron/LIVE.md](nemotron/LIVE.md)) | [nemotron/README.md](nemotron/README.md) |
| simple_diarizer | `lintoai/linto-diarization-simple` | CPU only, lightweight | [simple/README.md](simple/README.md) |
| PyBK | – | deprecated | [pybk/README.md](pybk/README.md) |

Measured on French meetings (DER, collar 0.25 s, overlapped speech scored; time on an RTX 4090 Laptop):

| | SUMM-RE (7 meetings) | Simsamu (26 calls) | 58 min of audio |
|---|---|---|---|
| pyannote community-1 | 21.9 % | 15.5 % | 66.6 s |
| Nemotron | 17.0 % | 13.7 % | 12.4 s (2.2 s with the `-compiled` image) |

Nemotron cannot tell more than 8 speakers apart: above that its DER goes up to 43-57 % (pyannote: 31-32 %). Its
result then carries `"saturated": true`, and linto-transcription-service reruns the job on pyannote when both
are deployed. See also [speaker-diarization-benchmark](https://github.com/linagora-labs/speaker-diarization-benchmark)
for pyannote and simple.

## Modes

**Task mode** (`SERVICE_MODE=task`): a Celery worker reading its queue on a Redis broker. Audio files are not sent
through the broker: they are read from a shared folder mounted on `/opt/audio`. At startup the worker registers
itself in the broker (Redis database 0) so that transcription-service can discover it, with its engine (and
speaker ceiling for Nemotron) in the `info` field. On GPU the worker runs `--pool=threads -c 1`: one task at a time, and it
keeps answering ping and inspect while a task runs.

Tasks:

| Task | Arguments | Result |
|---|---|---|
| `diarization_task` | `file` (path relative to `/opt/audio`), `speaker_count`, `max_speaker`, `speaker_names` (identification, see below) | [result](#result) |
| `voiceprint_compute_task` | `audio_files` (paths relative to `/opt/audio`) | `{vector, model_id, dim, duration_used, files_used}` |
| `speaker_upsert_task` | `collection`, `speaker_id`, `name`, `vector`, `model_id` | `{status, point_id, created_collection}` |
| `speaker_delete_task` | `collection`, `speaker_ids` | `{status, deleted}` |
| `collection_drop_task` | `collection` | `{status, existed}` |

`diarization_task` publishes its progress as Celery state `PROGRESS` with `{"progress": 0..1}`.

**HTTP mode** (`SERVICE_MODE=http`): `POST /diarization` with header `Accept: application/json` and a form with
`file` (16 kHz wav) and the optional fields `speaker_count`, `max_speaker`, `speaker_names`. `GET /healthcheck`,
and a Swagger UI on `/docs`. The upload is written to `/dev/shm`: give the container `--shm-size` larger than
your biggest file.

`speaker_count` and `max_speaker` are ignored by Nemotron, which finds the number of speakers by itself.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SERVICE_MODE` | – | `task` or `http` |
| `SERVICE_NAME` | `diarization` | name registered for discovery, also the default queue |
| `QUEUE_NAME` | `SERVICE_NAME` | Celery queue |
| `SERVICES_BROKER` | – | `redis://host:port` (task mode) |
| `BROKER_PASS` | – | Redis password |
| `CONCURRENCY` | `1` | Celery concurrency (keep 1 on GPU) |
| `SERVICE_PORT` | `80` | HTTP port |
| `LANGUAGE` | `*` | language registered for discovery |
| `MODEL_INFO` | – | label shown by clients, e.g. `{"en": "Yes", "fr": "Oui"}` |
| `DEVICE` | GPU if available | `cpu`, `cuda`, `cuda:1`... |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY` | – | speaker identification is enabled when `QDRANT_HOST` is set |
| `SPEAKER_ID_MIN_SIMILARITY` | `0.66` | default identification threshold |
| `SPEAKER_ID_MIN_ENROLL_DURATION` | `3` | minimum speech (s) to compute a voiceprint |
| `SPEAKER_ID_MAX_ENROLL_DURATION` | `180` | speech (s) kept to compute a voiceprint |
| `CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER` | `0` | `1` = one enrolled speaker can name several diarized speakers |

Engine-specific variables are in each engine README. `.envdefault` gives an example.

## Result

```json
{
  "speakers": [
    {"spk_id": "Alice Martin", "spk_id_score": 0.912, "duration": 812.4, "nbr_seg": 143},
    {"spk_id": "spk2", "duration": 455.1, "nbr_seg": 98}
  ],
  "segments": [
    {"seg_id": 1, "seg_begin": 0.52, "seg_end": 4.1, "spk_id": "Alice Martin"},
    {"seg_id": 2, "seg_begin": 4.3, "seg_end": 9.87, "spk_id": "spk2"}
  ],
  "engine": "nemotron",
  "saturated": false
}
```

Times are in seconds. `spk_id` is `spk1`, `spk2`... or the name of an identified speaker, with its
`spk_id_score`. `engine` and `saturated` are only set by Nemotron.

## Speaker identification

### How it works

1. **Enrollment.** Studio (or any client) sends a few audio files of a person. `voiceprint_compute_task` keeps up to
   180 s of their speech and computes a voiceprint: a 192-dimension vector from
   [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (ECAPA-TDNN,
   pinned revision, baked in the image). `speaker_upsert_task` stores it in a Qdrant collection named
   `spkid_{organizationId}_{collectionId}`, one point per person, identified by a `speaker_id` such as
   `label:65cc…` or `user:64dd…`.
2. **Identification.** After diarization, for each diarized speaker, starting with the one who speaks the most:
   its longest speech turns (up to 3 min) give a voiceprint, compared (cosine similarity) with the enrolled
   voiceprints of the requested collections. The best enrolled speaker wins if its score is at least
   `minSimilarity`; otherwise the diarized speaker keeps its `spkN` tag. An enrolled speaker is given to one
   diarized speaker at most.
3. **Result.** `spk_id` becomes the enrolled name, and `spk_id_score` gives the similarity.

Identification never changes the diarization itself: it only renames the speakers found by the engine. If the
engine merges two people into one speaker, they get one name.

### Request

Pass a JSON object as `speaker_names` (4th argument of `diarization_task`, or form field in HTTP mode):

```json
{
  "collections": ["spkid_64ff…_65aa…", "spkid_64ff…_65bb…"],
  "speakers": "*",
  "minSimilarity": 0.66
}
```

- `collections` (required): Qdrant collections to search. A collection that does not exist, or that was built with
  another embedding model, is skipped with a warning.
- `speakers` (default `"*"`): restrict to a list of `speaker_id`.
- `minSimilarity` (default `SPEAKER_ID_MIN_SIMILARITY`): threshold in [0, 1].

Through linto-transcription-service, the same object is sent as `speakerIdentificationConfig` in the diarization
config; transcription-service checks that every collection belongs to the caller's organization.

### Accuracy, threshold and voiceprints

Measured with Nemotron on the 34 meetings of SUMM-RE dev (French, 2 to 4 people each, many people attend several
meetings). Every one of the 77 participants has a voiceprint in the collection, like an organization
collection. Each voiceprint comes from **another meeting** than the one identified, so people who attend a
single meeting are not enrolled for it and must stay unknown: 76 enrolled speakers and 49 non-enrolled ones.

| Voiceprint | Threshold | Correct names | Wrong names | Missed |
|---|---|---|---|---|
| 10 s | 0.50 | 70 | 34 | 0 |
| 10 s | 0.60 | 70 | 13 | 0 |
| 10 s | **0.66** | 69 | 5 | 3 |
| 10 s | 0.70 | 65 | 3 | 9 |
| 15 s | **0.66** | 69 | 6 | 3 |
| 15 s | 0.70 | 68 | 5 | 4 |

The default threshold is 0.66. Below it, people who are not in the collection (guests, new members) often get
the name of someone who is; above it, short voiceprints are missed. 0.66 to 0.68 give the same results. Four or
five of the wrong names remain at any threshold: they come from diarization errors (a speaker found by the
engine that mixes two people). pyannote gives similar figures.

Two voiceprints of the same person recorded in two meetings score 0.69 to 0.96 (median 0.88); two different
people score 0.17 in median, 0.46 at the 95th percentile, 0.78 at most.

The voiceprint matters more than the threshold: 18 s recorded in a browser, outside any meeting, scored 0.43
against the same person in a meeting and are never recognized. Recommendations:

- enroll at least 10 to 15 s of speech recorded in the conditions of the meetings (same room, same
  microphones); longer voiceprints change little;
- keep the threshold at 0.66; lowering it to rescue a poor voiceprint brings wrong names;
- enrolled names should be unique within the collections of one request.

### Legacy filesystem mode (deprecated)

Reference samples mounted on `/opt/speaker_samples` (`SPEAKER_SAMPLES_FOLDER`), one file or one folder per person,
are loaded into a single collection `QDRANT_COLLECTION_NAME` at startup (`QDRANT_RECREATE_COLLECTION=true` to
rebuild it). `speaker_names` is then a string: `"*"` (everybody), `"name1|name2"` or a JSON list of names.

## Quick start

HTTP mode on GPU, with Nemotron:

```bash
docker run --rm --gpus all --shm-size=1g -p 8080:80 \
    -e SERVICE_MODE=http \
    lintoai/linto-diarization-nemotron:latest

curl -H 'Accept: application/json' -F file=@meeting.wav http://localhost:8080/diarization
```

To enable identification, start Qdrant and pass `QDRANT_HOST`:

```bash
docker network create diarization
docker run -d --name qdrant --network diarization -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
docker run --rm --gpus all --shm-size=1g -p 8080:80 --network diarization \
    -e SERVICE_MODE=http -e QDRANT_HOST=qdrant -e QDRANT_PORT=6333 \
    lintoai/linto-diarization-pyannote:latest
```

Task mode, with a Redis broker and the audio in `$HOME/audio`:

```bash
docker run -d --name redis -p 6379:6379 redis/redis-stack-server:latest
docker run --rm --gpus all -v "$HOME/audio:/opt/audio" \
    -e SERVICE_MODE=task -e SERVICE_NAME=diarization \
    -e SERVICES_BROKER=redis://172.17.0.1:6379 -e BROKER_PASS= \
    lintoai/linto-diarization-pyannote:latest

pip install celery redis
python3 -c "
import celery
app = celery.Celery(broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')
print(app.send_task('diarization_task', ('meeting.wav', None, None), queue='diarization').get())
"
```

## License

This project is distributed under the AGPLv3 license (see [LICENSE](LICENSE)).

The images bundle pretrained models under their own licenses:

- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1):
  CC BY 4.0 (attribution in [pyannote/README.md](pyannote/README.md#acknowlegment));
- [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization): OpenMDW-1.1, license text
  and origin notice shipped next to the model in the image (`nemotron/model-license/`);
- [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (speaker
  identification): Apache-2.0.
