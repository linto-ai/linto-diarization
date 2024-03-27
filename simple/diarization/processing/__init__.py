import torch
import os


device = os.environ.get("DEVICE")
if device is None:
   USE_GPU = torch.cuda.is_available()
else:
   USE_GPU = (device != "cpu")

# Number of CPU threads
NUM_THREADS = os.environ.get("NUM_THREADS", torch.get_num_threads())
NUM_THREADS = int(NUM_THREADS)
# This set the number of threads for sklearn
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS) # This must be done BEFORE importing packages (sklearn, etc.)
# For Torch, we will set it afterward, because setting that before loading the model can hang the process (see https://github.com/pytorch/pytorch/issues/58962)
torch.set_num_threads(1)

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization(device=device, num_threads=NUM_THREADS)

__all__ = ["diarizationworker"]