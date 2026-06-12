# 2.1.0
- Multi-collection speaker identification: `diarization_task` accepts a JSON object speaker specification ({collections, speakers, minSimilarity})
- New Celery tasks for runtime enrollment: voiceprint_compute_task, speaker_upsert_task, speaker_delete_task, collection_drop_task
- Service registration exposes speaker identification capability ({speaker_identification, model_id, dim})
- Speaker identification is now enabled iff QDRANT_HOST is set; filesystem enrollment (SPEAKER_SAMPLES_FOLDER + QDRANT_COLLECTION_NAME) is deprecated but still supported
- Pin the HuggingFace revision of the embedding model
- New environment variables: QDRANT_API_KEY, SPEAKER_ID_MIN_SIMILARITY, SPEAKER_ID_MAX_ENROLL_DURATION, SPEAKER_ID_MIN_ENROLL_DURATION

# 2.0.2
- Set default runtime user to www-data

# 2.0.1
- Use Qdrant for efficient speaker identification
- Specifying max number of speakers is now optional

# 2.0.0
- Add speaker identification

# 1.1.0
- Use NEMO spectral clustering, with DEVICE_CLUSTERING environment variable to enforce using CPU ("cpu") or GPU ("cuda")
- Switch from silero VAD version 4 to version 3.1 (less problems)
- Fix healthcheck on GPU

# 1.0.1
- Add DEVICE environment variable to enforce using CPU ("cpu") or GPU ("cuda")
- Fix multi-threading, and add NUM_THREADS environment variable to control the maximum number of threads
- Upgrade to speechbrain version 1.0.0

# 1.0.0
- First build of linto-diarization-simple
- Based on version 3.0.4 of linto-platform-diarization https://github.com/linto-ai/linto-platform-diarization/blob/039f0a70da11b978e0c8dab535061200da8d9fe7/RELEASE.md
  which corresponds to image linto-platform-diarization:3.0.4 https://hub.docker.com/layers/lintoai/linto-platform-diarization/3.0.4/images/sha256-209748f948db6387466fb1e43ec536b309951551116a990a574b976686c72f29?context=explore
