import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import identification.speaker_identify as speaker_identify  # noqa: E402


def reload_with(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER", raising=False)
    else:
        monkeypatch.setenv("CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER", value)
    return importlib.reload(speaker_identify).SpeakerIdentifier


def test_one_name_per_speaker_by_default(monkeypatch):
    assert reload_with(monkeypatch, None)._can_identify_twice_the_same_speaker is False


def test_can_be_enabled(monkeypatch):
    assert reload_with(monkeypatch, "1")._can_identify_twice_the_same_speaker is True
    reload_with(monkeypatch, None)
