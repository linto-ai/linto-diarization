import torch
torch.set_num_threads(1) # This is to avoid hanging in a multi-threaded environment

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization()

__all__ = ["diarizationworker"]