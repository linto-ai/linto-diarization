# 2.2.0
- Multi-collection speaker identification: `diarization_task` accepts a JSON object speaker specification ({collections, speakers, minSimilarity})
- New Celery tasks for runtime enrollment: voiceprint_compute_task, speaker_upsert_task, speaker_delete_task, collection_drop_task
- Service registration exposes speaker identification capability ({speaker_identification, model_id, dim})
- Speaker identification is now enabled iff QDRANT_HOST is set; filesystem enrollment (SPEAKER_SAMPLES_FOLDER + QDRANT_COLLECTION_NAME) is deprecated but still supported
- Pin the HuggingFace revision of the embedding model
- New environment variables: QDRANT_API_KEY, SPEAKER_ID_MIN_SIMILARITY, SPEAKER_ID_MAX_ENROLL_DURATION, SPEAKER_ID_MIN_ENROLL_DURATION

# 2.1.0
- Switch default Docker image to python:3.10-slim for lighter footprint
- Pin huggingface-hub<0.25 for pyannote.audio 3.3.2 compatibility
- Add explicit requests dependency

# 2.0.2
- Update cache folder to /opt/models
- Set default runtime user to www-data

# 2.0.1
- Use Qdrant for efficient speaker identification
- Update pyannote to 3.3.2 (and speechbrain 1.0.0)

# 2.0.0
- Add speaker identification
- Add progress bar

# 1.1.1
- Fix healthcheck on GPU

# 1.1.0
- Integrate pyannote 3.1.1 with models configuration 3.1 https://huggingface.co/pyannote/speaker-diarization-3.1
- add DEVICE and NUM_THREADS environment variables to control the device and number of threads used

# 1.0.0
- First build of linto-diarization-pyannote
- Based on version 2.0.0 of linto-platform-diarization https://github.com/linto-ai/linto-platform-diarization/blob/b29be8189db51cd963f64bdeffb3d3024b999be0/RELEASE.md
  which corresponds to image linto-platform-diarization:2.0.0 https://hub.docker.com/layers/lintoai/linto-platform-diarization/2.0.0/images/sha256-93f0abb20f7c40e80dde484e8240bc4f2df3ce4e48165cc29b95a20b206ba4c9?context=explore
