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
