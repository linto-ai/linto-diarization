import os

import torch

device = os.environ.get("DEVICE")
device_vad = "cpu" # Not implemented os.environ.get("DEVICE_VAD", "cpu")
device_clustering = os.environ.get("DEVICE_CLUSTERING")
if torch.cuda.is_available():
   USE_GPU = (
      device != "cpu"
      or device_clustering  != "cpu"
      or device_vad != "cpu"
   )
else:
   USE_GPU = False

# Number of CPU threads
NUM_THREADS = os.environ.get("NUM_THREADS", torch.get_num_threads())
NUM_THREADS = int(NUM_THREADS)
# This set the number of threads for sklearn
os.environ["OMP_NUM_THREADS"] = str(
    NUM_THREADS
)  # This must be done BEFORE importing packages (sklearn, etc.)
# For Torch, we will set it afterward, because setting that before loading the model can hang the process (see https://github.com/pytorch/pytorch/issues/58962)
torch.set_num_threads(1)

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization(device=device, device_clustering=device_clustering, device_vad=device_vad, num_threads=NUM_THREADS)

__all__ = ["diarizationworker"]
