import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery_app import register  # noqa: E402


def test_engine_and_ceiling_registered(monkeypatch):
    monkeypatch.setenv("DIARIZATION_ENGINE", "nemotron")
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "8")
    monkeypatch.setenv("MODEL_INFO", '{"en": "Yes", "fr": "Oui"}')
    info = json.loads(register.service_info()["info"])
    assert info["engine"] == "nemotron"
    assert info["max_speakers"] == 8
    assert info["fr"] == "Oui"


def test_no_engine_fields_by_default(monkeypatch):
    monkeypatch.delenv("DIARIZATION_ENGINE", raising=False)
    monkeypatch.delenv("DIARIZATION_MAX_SPEAKERS", raising=False)
    info = json.loads(register.service_info()["info"])
    assert "engine" not in info and "max_speakers" not in info
