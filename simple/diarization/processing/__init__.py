import torch
import os

torch.set_num_threads(1) # This is to avoid hanging in a multi-threaded environment

# Check if FORCE_CPU environment variable is set and if it's set to 'true'
force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'

# Use GPU only if CUDA is available and FORCE_CPU is not set to true
USE_GPU = torch.cuda.is_available() and not force_cpu

from .speakerdiarization import SpeakerDiarization

diarizationworker = SpeakerDiarization()

__all__ = ["diarizationworker"]