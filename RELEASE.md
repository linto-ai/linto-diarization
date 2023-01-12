# 3.0.0
- Use simple_diarization (Silero VAD, ECAPA-SC embeddings) for diarization.

# 1.1.2
- Added service registration.
- Updated healthcheck to add heartbeat.
- Added possibility to overide generated queue name.
# 1.1.1
- Fixed: silences (and short occurrences <1 sec between silences) occurring inside a speaker turn were postponed at the end of the speaker turn (and could be arbitrarily assigned to next speaker)
- Fixed: make diarization deterministic (random seed is fixed)
- Tune length of short occurrences to consider as silences (0.3 sec)

# 1.1.0
- Changed: loading audio file by AudioSegment toolbox. 
- Changed: mfcc are extracted by python_speech_features toolbox.
- Fixed windowRate =< maximumKBMWindowRate.
- Likelihood table is only calculated for the top five gaussian, computation time is reduced.
- Similarity matrix is calculated by Binary keys and cumulative vectors
- Removed: unused AHC.
- Code formated to pep8

# 1.0.3
- Fixed: diarization failing on short audio when n_speaker > 1
- Fixed (TBT): diarization returning segfault on machine with a lot of CPU
- Added: Added Debugging logs using env variable DEBUG
- Changed: Code formated to pep8

# 1.0.2
- Changed: Diarization parameters to reduce computation time
- Fixed: Speaker id shoulds now be continuous and numeroted by order of appearance. 
- Removed: Lot of unused pybk code

# 1.0.1
- Added max_speaker field for HTTP and celery requests.
- Fixed max_speaker not being properly considered during clustering.
- Removed PyBK as a submodule in favor of a hard copy.
- Updated README and API documentation.

# 1.0.0
- Diarization service bases on PyBK.
- Celery connectivity
- HTTP API