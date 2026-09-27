# 1.0.0
- First release: speaker diarization with NVIDIA Nemotron 3 Diarization (Sortformer, up to 8 speakers)
- Block-wise streaming inference: output identical to NeMo `diarize()` on the whole file, VRAM flat (~1.2 GB) whatever the audio duration
- Result carries `engine` and `saturated` (all 8 speaker slots used: the audio may hold more speakers)
- Speaker identification (ECAPA + Qdrant) shared with the pyannote image
- Task progress published as Celery state `PROGRESS`
- Speaker identification: an enrolled speaker is given to one diarized speaker at most (`CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER` defaults to 0)
- Speaker identification: default similarity threshold 0.66 instead of 0.5 (`SPEAKER_ID_MIN_SIMILARITY`). With 10 s voiceprints taken from other meetings (SUMM-RE, 34 meetings), 0.5 gave an enrolled name to most non-enrolled speakers; at 0.66, 5 wrong names and 3 missed out of 76 enrolled speakers
- Image variant `<version>-compiled` (Dockerfile target `runtime-compiled`): C compiler included, NeMo attention compiled with Triton at first use, kernels cached in `/opt/cache`. The default image stays eager, without compiler
