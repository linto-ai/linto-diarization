import os

import torch

device = os.environ.get("DEVICE")
if device is None:
    USE_GPU = torch.cuda.is_available()
else:
    USE_GPU = device != "cpu"

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization()

__all__ = ["diarizationworker"]