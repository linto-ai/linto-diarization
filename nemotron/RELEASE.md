# 1.0.0
- First release: speaker diarization with NVIDIA Nemotron 3 Diarization (Sortformer, up to 8 speakers)
- Block-wise streaming inference: output identical to NeMo `diarize()` on the whole file, VRAM flat (~1.2 GB) whatever the audio duration
- Result carries `engine` and `saturated` (all 8 speaker slots used: the audio may hold more speakers)
- Speaker identification (ECAPA + Qdrant) shared with the pyannote image
- Task progress published as Celery state `PROGRESS`
- Speaker identification: an enrolled speaker is given to one diarized speaker at most (`CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER` defaults to 0)
