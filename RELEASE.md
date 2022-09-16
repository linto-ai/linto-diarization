# 1.1
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




