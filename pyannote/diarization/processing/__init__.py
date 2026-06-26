import os
import torch


device = os.environ.get("DEVICE")
if device is None:
   device = "cuda" if torch.cuda.is_available() else "cpu"
try:
   torch.device(device)
except Exception as err:
   raise RuntimeError(f"Invalid device '{device}'") from err

USE_GPU = (device != "cpu")

# Number of CPU threads
NUM_THREADS = os.environ.get(
    "NUM_THREADS", os.environ.get("OMP_NUM_THREADS", min(4, torch.get_num_threads()))
)
NUM_THREADS = int(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
# For Torch, we will set it afterward, because setting that before loading the model can hang the process (see https://github.com/pytorch/pytorch/issues/58962)
torch.set_num_threads(1)

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization(device=device, num_threads=NUM_THREADS)

__all__ = ["diarizationworker"]
