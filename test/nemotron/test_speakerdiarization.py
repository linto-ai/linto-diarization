"""SpeakerDiarization.run with a fake engine and identifier (no model, no GPU)."""
import importlib
import logging
import os
import sys
import types

import pytest

pytest.importorskip("nemo")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)

# Import the module without diarization/processing/__init__.py, which loads the model
package = types.ModuleType("nemotron_processing")
package.__path__ = [os.path.join(ROOT, "nemotron", "diarization", "processing")]
sys.modules.setdefault("nemotron_processing", package)
speakerdiarization = importlib.import_module("nemotron_processing.speakerdiarization")


class FakeEngine:
    max_speakers = 8

    def __init__(self, n_speakers):
        self.n_speakers = n_speakers
        self.progress_callback = None

    def diarize_file(self, path, progress_callback=None):
        self.progress_callback = progress_callback
        return "preds"

    def segments(self, preds):
        return [(float(i), float(i) + 0.5, i) for i in range(self.n_speakers)]


class FakeIdentifier:
    def check_speaker_specification(self, speaker_names):
        return speaker_names

    def speaker_identify_given_diarization(self, path, result, speaker_names):
        # Like the real identifier: returns a new dict without the engine fields
        return {"speakers": result["speakers"], "segments": result["segments"]}


def worker(n_speakers):
    sd = speakerdiarization.SpeakerDiarization.__new__(speakerdiarization.SpeakerDiarization)
    sd.log = logging.getLogger("test")
    sd.engine = FakeEngine(n_speakers)
    sd.speaker_identifier = FakeIdentifier()
    sd.tempfile = None
    return sd


@pytest.mark.parametrize("n, saturated", [(3, False), (8, True)])
def test_engine_fields_survive_identification(n, saturated):
    result = worker(n).run("a.wav")
    assert result["engine"] == "nemotron"
    assert result["saturated"] is saturated
    assert len(result["speakers"]) == n


def test_speaker_count_ignored_and_progress_forwarded():
    sd = worker(2)
    callback = lambda p: None  # noqa: E731
    result = sd.run("a.wav", speaker_count=5, max_speaker=10, progress_callback=callback)
    assert len(result["speakers"]) == 2
    assert sd.engine.progress_callback is callback
