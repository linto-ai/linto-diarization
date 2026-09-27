# LinTO-diarization: Nemotron

Speaker diarization with [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(Sortformer, 100M parameters, OpenMDW-1.1 license), with the same Celery task, HTTP API and speaker
identification as the [pyannote](../pyannote/README.md) image.

## When to use it

- Up to 8 speakers: this is a hard limit of the model. Beyond, speakers get merged. The result then
  carries `"saturated": true`, and callers should run pyannote instead (transcription-service does it).
- On French meetings it beats pyannote community-1 (DER 17.0 % vs 21.9 % on SUMM-RE, 13.7 % vs 15.5 % on
  Simsamu, 0.25 s collar) and runs about 6.7 times faster (58 min of audio in 10 s on an RTX 4090 Laptop).
- `speaker_count` / `max_speaker` are accepted but ignored: the model cannot be constrained.
- GPU: Ampere or newer (L4, L40S, A4000, RTX 30/40...). NVIDIA driver >= 570 (CUDA 12.8 wheels).

## Build

From the repository root:

```bash
docker build -f nemotron/Dockerfile -t lintoai/linto-diarization-nemotron .
# variant with a C compiler, published as <version>-compiled
docker build -f nemotron/Dockerfile --target runtime-compiled -t lintoai/linto-diarization-nemotron:compiled .
```

The model is downloaded at build time from `NEMOTRON_MODEL_URL` and checked against
`NEMOTRON_MODEL_SHA256` (build arguments). Nothing is fetched from HuggingFace at runtime.
The model license (OpenMDW-1.1) and its origin notice are copied next to the model file, as the license
requires for redistribution (`nemotron/model-license/`).

Python dependencies are locked with uv (`pyproject.toml`, `uv.lock` at the repository root, extra
`nemotron`). NeMo is pinned to a commit of NVIDIA-NeMo/Speech `main`: release 3.0.0 cannot load the model.

## Run

Same environment variables and modes as the pyannote image (`SERVICE_MODE=task|http`, `SERVICES_BROKER`,
`BROKER_PASS`, `SERVICE_NAME`, `QUEUE_NAME`, `QDRANT_*`...), plus:

| Variable | Default | Description |
|---|---|---|
| `NEMOTRON_MODEL` | `/opt/models/nemotron-3-diarization/Nemotron-3-Diarization.nemo` | Model file (or HuggingFace id, needs network) |
| `NEMOTRON_PRESET` | `offline` | Streaming preset from the model card: `offline` (30.4 s buffer), `low_latency` (1.04 s), `very_low_latency` (0.64 s), `ultra_low_latency` (0.32 s) |
| `NEMOTRON_BLOCK_SECONDS` | `300` | Audio duration whose features are computed at once. Bounds VRAM, does not change the result |

The service registers `{"engine": "nemotron", "max_speakers": 8}` in its info field.

## Inference

The model processes audio chunk by chunk with a speaker cache (`forward_streaming_step`). Mel features are
computed per block of `NEMOTRON_BLOCK_SECONDS` (1 s margin each side) and the streaming state is carried
from one block to the next, so the output is identical to `diarize()` on the whole file, while VRAM stays
around 1.2 GB (`diarize()` on 5 h 50 of audio peaks at 14.6 GB).

## Image variants

NeMo compiles its attention with Triton, which needs a C compiler at runtime.

| | default | `-compiled` |
|---|---|---|
| C compiler | no | gcc, libc6-dev (+200 MB) |
| 58 min of audio (RTX 4090 Laptop) | 12.4 s | 2.2 s |
| first request after start | normal | about 3 s of compilation |
| DER, 33 annotated files (collar 0.25 s) | SUMM-RE 16.95 %, Simsamu 13.70 % | 16.96 %, 13.70 % |
| scanner findings | baseline | + kernel headers and binutils CVE (none fixable) |

The two variants do not give bit-identical output on long files (small numeric differences add up in the
speaker cache), quality is the same. The compiled variant caches kernels under `/opt/cache` and falls back to
eager if a compilation fails. The default image is the one to deliver where compilers are not allowed in
production.

## Tests

```bash
uv sync --extra nemotron
uv run pytest                      # no GPU needed
NEMOTRON_TEST_AUDIO=/path/to/3min.wav uv run pytest -m gpu
```
