import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nemotron", "diarization", "processing"))

from result import format_result  # noqa: E402


def test_speakers_named_by_first_appearance():
    segments = [(5.0, 6.0, 3), (0.5, 2.0, 7), (2.5, 4.0, 3)]
    result = format_result(segments, max_speakers=8)
    assert [s["spk_id"] for s in result["segments"]] == ["spk1", "spk2", "spk2"]
    assert [s["seg_begin"] for s in result["segments"]] == [0.5, 2.5, 5.0]
    assert [s["seg_id"] for s in result["segments"]] == [1, 2, 3]


def test_speaker_totals():
    segments = [(0.0, 1.5, 0), (2.0, 2.25, 0), (1.0, 3.0, 1)]
    speakers = {s["spk_id"]: s for s in format_result(segments, max_speakers=8)["speakers"]}
    assert speakers["spk1"] == {"spk_id": "spk1", "duration": 1.75, "nbr_seg": 2}
    assert speakers["spk2"] == {"spk_id": "spk2", "duration": 2.0, "nbr_seg": 1}


def test_saturation_flag():
    few = [(float(i), i + 0.5, i) for i in range(7)]
    full = [(float(i), i + 0.5, i) for i in range(8)]
    assert format_result(few, max_speakers=8)["saturated"] is False
    assert format_result(full, max_speakers=8)["saturated"] is True


def test_engine_and_empty_audio():
    result = format_result([], max_speakers=8)
    assert result == {"speakers": [], "segments": [], "engine": "nemotron", "saturated": False}
