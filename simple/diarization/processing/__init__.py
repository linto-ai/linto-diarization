import torch
import os

torch.set_num_threads(1) # This is to avoid hanging in a multi-threaded environment

device = os.environ.get("DEVICE")
if device is None:
   USE_GPU = torch.cuda.is_available()
else:
   USE_GPU = (device != "cpu")

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization(device=device)

__all__ = ["diarizationworker"]