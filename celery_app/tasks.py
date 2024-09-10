import os

from celery_app.celeryapp import celery
from diarization.processing import diarizationworker
from diarization import logger

@celery.task(name="diarization_task")
def diarization_task(
    file: str,
    speaker_count: int = None,
    max_speaker: int = None,
    speaker_names: str = None,
):
    """transcribe_task do a synchronous call to the transcribe worker API"""
    logger.info(f"Received transcription task for {file} ({speaker_count=}, {max_speaker=})")

    file_path = os.path.join("/opt/audio", file)
    if not os.path.isfile(file_path):
        raise Exception("Could not find ressource {}".format(file_path))

    # Check parameters
    speaker_count = None if speaker_count == 0 else speaker_count
    max_speaker = None if max_speaker == 0 else max_speaker

    if speaker_count and max_speaker:
        max_speaker = None

    # Processing
    try:
        result = diarizationworker.run(
            file_path,
            speaker_count=speaker_count,
            max_speaker=max_speaker,
            speaker_names=speaker_names,
        )
    except Exception as e:
        import traceback
        msg = f"{traceback.format_exc()}\nFailed to decode {file_path}"
        logger.error(msg)
        raise Exception(msg)  # from err

    return result
