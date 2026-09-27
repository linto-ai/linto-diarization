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
NUM_THREADS = int(os.environ.get(
    "NUM_THREADS", os.environ.get("OMP_NUM_THREADS", min(4, torch.get_num_threads()))
))
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization(device=device)

__all__ = ["diarizationworker", "USE_GPU"]
