import torch
import os


device = os.environ.get("DEVICE")
if device is None:
   USE_GPU = torch.cuda.is_available()
else:
   USE_GPU = (device != "cpu")

NUM_THREADS = os.environ.get("NUM_THREADS", torch.get_num_threads())
NUM_THREADS = int(NUM_THREADS)
torch.set_num_threads(1) # This is to avoid hanging when creating sub-process (see https://github.com/pytorch/pytorch/issues/58962)

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization(device=device, num_threads=NUM_THREADS)

__all__ = ["diarizationworker"]