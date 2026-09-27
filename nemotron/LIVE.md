# Live diarization (websocket)

The Nemotron worker can diarize live audio streams, with speaker identification, next to its Celery
tasks, on the same model and GPU. The client is the Transcriber of linto-studio-plugins: it opens one
websocket per channel, sends the audio and receives speech turns and identities.

Enabled when `NEMOTRON_LIVE_PORT` is set (task mode only).

| Variable | Default | Description |
|---|---|---|
| `NEMOTRON_LIVE_PORT` | – | websocket port; live is disabled when unset |
| `NEMOTRON_MAX_LIVE_SESSIONS` | `16` | sessions accepted by one worker (see [Capacity](#capacity)) |
| `NEMOTRON_LIVE_PRESET` | `low_latency` | streaming preset: 1.04 s of algorithmic latency |
| `NEMOTRON_LIVE_MAX_BATCH` | `16` | sessions processed in one GPU step |
| `NEMOTRON_LIVE_MAX_LAG` | `5` | seconds of backlog above which the worker reports itself not ready |
| `SPEAKER_ID_LIVE_MILESTONES` | `10,30,60` | seconds of a speaker's speech at which it is identified |
| `SPEAKER_ID_LIVE_CONFIRM_SECONDS` | `30` | speech needed for a `confirmed` identity |
| `SPEAKER_ID_MIN_SIMILARITY` | `0.66` | identification threshold, as for files |

## Protocol (v1)

Client to server, first message:

```json
{"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le", "session": "<optional id>",
 "identification": {"organizationId": "64ff…", "collections": ["spkid_64ff…_65aa…"],
                    "speakers": "*", "minSimilarity": null}}
```

`identification` is optional. Every collection must belong to `organizationId`
(`spkid_{organizationId}_{collectionId}`); the caller is responsible for the organization (the port is
internal to the cluster). Then binary messages of PCM (16 kHz, mono, signed 16 bit little endian, any
size, 100 ms is fine) and finally `{"type": "stop"}`.

Server to client:

| Message | Content |
|---|---|
| `ready` | `{"latency_ms": 1040, "max_speakers": 8, "frame_ms": 10}` |
| `turns` | `{"until_ms": 125680, "turns": [{"speaker": "S2", "start_ms": 124960, "end_ms": 125680}]}`: speech of each speaker in the audio processed since the previous message. A turn cut at a chunk edge continues in the next message: join runs of the same speaker that touch. Turns of two speakers may overlap. |
| `identity` | `{"speaker": "S2", "status": "provisional", "speaker_id": "label:…", "name": "Alice", "score": 0.81, "until_ms": 41200, "speech_s": 10.3}` |
| `saturated` | sent once when the 8 speaker slots of the model are used: more people may be merged |
| `error` | `{"code": "invalid_start" \| "full" \| "identification_unavailable", "message": …}` |
| `end` | `{"until_ms": …}` after `stop`, once all the audio is processed; the server then closes |

Times are milliseconds of audio received since the start. Speaker labels (`S1` to `S8`) are stable for
the whole session. Close codes: 1008 invalid start message, 1013 worker full (reconnect: the load
balancer picks another worker).

Identity statuses: `provisional` (first match, less than 30 s of the speaker's speech), `confirmed`
(match with 30 s or more), `revoked` (a later attempt no longer supports the name, or another speaker of
the session takes it with a better score). A speaker is identified at 10, 30 and 60 s of its
non-overlapped speech; an enrolled person names one speaker at most per session.

A reference client is in [tools/live_client.py](tools/live_client.py):

```bash
python nemotron/tools/live_client.py ws://localhost:8080 meeting.wav --speed 1
```

## Deployment

Workers are reached through a Kubernetes Service (ClusterIP) in front of the Nemotron pods, on
`NEMOTRON_LIVE_PORT`. A websocket stays on the pod that accepted it for the whole session; new
sessions are spread by the Service. `GET /ready` on the same port answers 503 when the worker has
`NEMOTRON_MAX_LIVE_SESSIONS` sessions or more than `NEMOTRON_LIVE_MAX_LAG` seconds of backlog: use it as
readiness probe, and the Service stops sending new sessions to a full worker. `GET /healthz` answers 200
while the server runs.

If a worker disappears, its sessions reconnect elsewhere and start again: speaker labels are
renumbered and identities come back after 10 s of speech.

## Inside a worker

One process: a websocket thread (asyncio) receives the audio of every session and sends the messages;
one engine thread owns the model. At each tick the engine takes the next chunk (0.72 s of audio, with
0.32 s of right context) of every session that has one ready and runs them in one batched
`forward_streaming_step` (NeMo asynchronous streaming: each session keeps its own speaker cache). The
Celery file tasks and the identification voiceprints use the same model through a GPU arbiter that
serves live chunks first, then voiceprints, then file chunks: a file slows down when many sessions
run, it never delays them by more than one chunk.

## Tests

All measured on an RTX 4090 Laptop GPU, 27/09/2026.

### Unit tests (no GPU)

```bash
uv run pytest test/nemotron/test_live_identify.py test/nemotron/test_live_units.py \
    test/nemotron/test_live_server.py test/nemotron/test_live_session.py
```

| File | What it checks |
|---|---|
| `test_live_identify.py` (16) | identification state machine: attempts at 10, 30, 60 s of speech and not before, `provisional` then `confirmed`, `revoked` when a later attempt disagrees, name change, threshold from the request or `SPEAKER_ID_MIN_SIMILARITY`, one enrolled person per speaker (better score takes the name, equal score keeps it) |
| `test_live_units.py` (18) | start message validation, identification restricted to the organization's collections, turns from predictions, non-overlapped speech used for identification, GPU arbiter order (live, then identification, then files) |
| `test_live_server.py` (6) | websocket flow with a fake engine and the reference client: turns joined across chunks, `end`, invalid start (1008), full worker (1013, `/ready` 503), stop without audio, client disconnection, identity messages with `until_ms` and `speech_s` |
| `test_live_session.py` (4) | audio buffer: 20 min received at once in under 2 s (reconnection backlog), samples intact after compaction, chunk scheduling with right context and at the end of the stream |

### GPU tests

```bash
NEMOTRON_MODEL=/path/Nemotron-3-Diarization.nemo NEMOTRON_TEST_AUDIO=/path/4min.wav uv run pytest -m gpu test/nemotron/test_live_gpu.py
```

| Test | Result |
|---|---|
| a live session through the websocket vs the file engine with the same preset | same speaker activity on more than 99.9 % of the 10 ms frames |
| 6 sessions started 0.5 s apart on different audio, processed in batches, vs each session alone | more than 99.9 % (during development: probabilities within 1e-4, 100 % same activity) |
| a file job (offline preset) running while 3 live sessions stream on the same model | the file result is bit-identical to the file alone, the live results match their references |

The container was also checked end to end: the Celery worker (threads pool) starts the live server on
`worker_ready`, `/ready` answers 12 s after start, a session gives the same speech per speaker as the
file engine, and the worker answers Celery ping meanwhile.

### Capacity

```bash
python nemotron/tools/live_bench.py ws://host:port meeting.wav --sessions 8 16 24 32 --seconds 60
```

N sessions streamed in real time for 60 s. Delay = audio sent minus audio diarized when a `turns`
message arrives (at least 420 ms: 320 ms of right context and 100 ms of feature margin).

| Sessions | default image: p50 / p95 / max | `-compiled` image: p50 / p95 / max |
|---|---|---|
| 8 | 520 / 600 / 860 ms | 460 / 500 / 2440 ms (first compilation) |
| 16 | 560 / 600 / 960 ms | – |
| 24 | 560 / 600 / 960 ms | – |
| 32 | 560 / 860 / 1360 ms | 460 / 500 / 740 ms |
| 48 | – | 460 / 500 / 800 ms |
| 64 | – | 460 / 500 / 880 ms |

The default image keeps up with about 24 sessions per GPU, the compiled one with 64 or more (no
recompilation when the batch size changes). Set `NEMOTRON_MAX_LIVE_SESSIONS` from a measure on the
target GPU (L4, A4000...), with margin, and account for the file jobs sharing the GPU.

### Speaker identification

```bash
python test/nemotron/eval_live_identification.py --data DIR --url ws://host:port --qdrant host:6333 \
    --voiceprint 10 --threshold 0.6 0.66 0.7
```

The 34 meetings of SUMM-RE dev (French, 2 to 4 people each, 77 people, many of them in several
meetings) are streamed through the websocket with identification. For each meeting, the collection
holds a 10 s voiceprint of every person, taken from **another meeting**: people seen in one meeting
only are not enrolled for it and must stay unknown. Speakers with less than 5 s of speech are ignored.
Results at the end of each meeting, over about 125 speakers (about 80 enrolled, 45 not; counts move by
one between two runs, as batches differ):

| Threshold | Correct names | Wrong names | Missed | Wrong names shown during the meeting | Revocations |
|---|---|---|---|---|---|
| 0.60 | 73 | 9 | 4 | 19 | 8 |
| **0.66** | **72** | **4** | **4** | **9** | **4** |
| 0.70 | 68 | 3 | 9 | 7 | 3 |

At 0.66 the live results are as good as the file mode (69 correct, 5 wrong, 3 missed on the same
meetings). Of the 9 wrong names shown, 5 are `provisional` names given at 10 s of speech and corrected at
30 s; the 4 others are the final wrong names, due to diarization (a speaker that mixes two people).
Clients should show a `provisional` name differently from a `confirmed` one.

A first attempt at 15 s instead of 10 s removes one transient error but delays the first name by 30 s
and loses a correct one; the default stays 10, 30, 60 s.

Median time from the start of the meeting to a speaker's first name: 82 s (`provisional`, 10 s of its
own non-overlapped speech in a 4-person meeting) and 250 s (`confirmed`, 30 s of speech).
