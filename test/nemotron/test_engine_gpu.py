"""Block-wise inference must match SortformerEncLabelModel.diarize() on the whole file.

Needs a CUDA GPU, the model (NEMOTRON_MODEL, a .nemo path or a HuggingFace id) and a
16 kHz audio file of a few minutes (NEMOTRON_TEST_AUDIO). Run: uv run pytest -m gpu
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

AUDIO = os.environ.get("NEMOTRON_TEST_AUDIO")
MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/Nemotron-3-Diarization")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not AUDIO, reason="NEMOTRON_TEST_AUDIO not set"),
]

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron")]


def test_blocks_match_full_file():
    from diarization.processing.engine import NemotronEngine

    # Small blocks so that a short file still spans several of them
    engine = NemotronEngine(MODEL, device="cuda", block_seconds=60)
    with torch.inference_mode():
        _, full = engine.model.diarize(audio=[AUDIO], batch_size=1, include_tensor_outputs=True, verbose=False)
    full = full[0].cpu().numpy().reshape(-1, engine.max_speakers)
    blocks = engine.diarize_file(AUDIO)
    n = min(len(full), len(blocks))
    assert abs(len(full) - len(blocks)) <= 2
    assert np.abs(full[:n] - blocks[:n]).max() < 1e-3
