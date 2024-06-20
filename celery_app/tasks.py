import os

from celery_app.celeryapp import celery
from celery.signals import celeryd_after_setup
from diarization.processing import diarizationworker
from diarization import logger

_context = {}

@celery.task(name="diarization_task")
def diarization_task(
    file_name: str, speaker_count: int = None, max_speaker: int = None
):
    """transcribe_task do a synchronous call to the transcribe worker API"""
    logger.info(f"Received transcription task for {file_name} ({speaker_count=}, {max_speaker=})")

    file_path = os.path.join("/opt/audio", file_name)
    if not os.path.isfile(file_path):
        raise Exception("Could not find ressource {}".format(file_path))

    # Check parameters
    speaker_count = None if speaker_count == 0 else speaker_count
    max_speaker = None if max_speaker == 0 else max_speaker

    if speaker_count and max_speaker:
        max_speaker = None

    try:
        _context['worker'].consumer.connection.heartbeat_check()
    except: # ConnectionForced
        pass

    # Processing
    try:
        result = diarizationworker.run(
            file_path,
            number_speaker=speaker_count,
            max_speaker=max_speaker,
        )
    except Exception as e:
        import traceback
        msg = f"{traceback.format_exc()}\nFailed to decode {file_path}"
        logger.error(msg)
        raise Exception(msg)  # from err

    return result

@celeryd_after_setup.connect
def setup(sender, instance, **kwargs):
    """
    Get a handle to the worker object so that we can send heartbeats
    when running in 'solo' mode
    """
    _context['worker'] = instance