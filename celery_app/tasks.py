import json
import os

from celery_app.celeryapp import celery
from diarization.processing.speakerdiarization import SpeakerDiarization


@celery.task(name="diarization_task")
def diarization_task(
    file_name: str, speaker_count: int = None, max_speaker: int = None
):
    """transcribe_task do a synchronous call to the transcribe worker API"""
    if not os.path.isfile(os.path.join("/opt/audio", file_name)):
        raise Exception("Could not find ressource {}".format(file_name))

    # Check parameters
    speaker_count = None if speaker_count == 0 else speaker_count
    max_speaker = None if max_speaker == 0 else max_speaker

    if speaker_count and max_speaker:
        max_speaker = None

    # Processing
    try:
        diarizationworker = SpeakerDiarization()
        result = diarizationworker.run(
            os.path.join("/opt/audio", file_name),
            number_speaker=speaker_count,
            max_speaker=max_speaker,
        )
        response =  diarizationworker.format_response(result)
    except Exception as e:
        raise Exception("Diarization has failed : {}".format(e))

    return response
