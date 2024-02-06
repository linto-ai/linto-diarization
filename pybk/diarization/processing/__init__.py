USE_GPU = False

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization()

__all__ = ["diarizationworker"]