"""Live diarization pieces that need no model: protocol, turns, GPU arbiter."""
import os
import sys
import threading
import time

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron")]

from diarization.live import protocol, turns  # noqa: E402

ORG = "a" * 24
COLLECTION = f"spkid_{ORG}_{'b' * 24}"


class TestStart:
    def test_minimal(self):
        assert protocol.parse_start({"type": "start"}) == {"session": None, "identification": None}

    @pytest.mark.parametrize("message", [
        {"type": "begin"}, [], {"type": "start", "sample_rate": 8000},
        {"type": "start", "encoding": "opus"}, {"type": "start", "session": 3},
    ])
    def test_invalid(self, message):
        with pytest.raises(protocol.ProtocolError):
            protocol.parse_start(message)

    def test_identification(self):
        spec = protocol.parse_start({"type": "start", "identification": {
            "organizationId": ORG, "collections": [COLLECTION], "minSimilarity": 0.7}})["identification"]
        assert spec == {"collections": [COLLECTION], "speakers": "*", "minSimilarity": 0.7, "organizationId": ORG}

    @pytest.mark.parametrize("identification", [
        {"collections": [COLLECTION]},  # no organization
        {"organizationId": "c" * 24, "collections": [COLLECTION]},  # collection of another organization
        {"organizationId": ORG, "collections": ["speakers"]},  # not a spkid collection
        {"organizationId": ORG, "collections": []},
        {"organizationId": ORG, "collections": [COLLECTION], "minSimilarity": 2},
        {"organizationId": ORG, "collections": [COLLECTION], "extra": 1},
    ])
    def test_identification_rejected(self, identification):
        with pytest.raises(protocol.ProtocolError):
            protocol.parse_start({"type": "start", "identification": identification})


class TestTurns:
    def test_runs(self):
        preds = np.zeros((10, 3), dtype=np.float32)
        preds[0:3, 0] = 0.9
        preds[2:6, 1] = 0.8
        preds[8:10, 0] = 0.6
        assert turns.runs(preds, first_frame=100, frame_ms=10) == [
            (0, 1000, 1030), (1, 1020, 1060), (0, 1080, 1100)]

    def test_threshold_is_strict(self):
        assert turns.runs(np.full((4, 1), 0.5, dtype=np.float32), 0, 10) == []

    def test_json_friendly(self):
        import json
        preds = np.zeros((4, 2), dtype=np.float32); preds[1:3, 1] = 1
        json.dumps(protocol.turns(40, turns.runs(preds, 0, 10)))  # numpy ints would fail here

    def test_single_speaker_excludes_overlap_and_erodes_inner_edges(self):
        preds = np.zeros((12, 2), dtype=np.float32)
        preds[0:8, 0] = 1  # speaker 0 alone on 0-4, overlapped on 5-7
        preds[5:12, 1] = 1  # speaker 1 alone on 8-11
        who = turns.single_speaker(preds, erode=1)
        # frame 0 touches the chunk edge (kept), frame 4 is an inner edge (dropped), 5-7 overlap,
        # frame 8 inner edge (dropped), 11 touches the chunk edge (kept)
        assert who.tolist() == [0, 0, 0, 0, -1, -1, -1, -1, -1, 1, 1, 1]


class TestArbiter:
    def test_live_goes_before_waiting_file_work(self):
        from diarization.gpu import FILE, IDENTIFICATION, LIVE, GpuArbiter

        arbiter, order = GpuArbiter(), []
        started = threading.Event()

        def holder():
            with arbiter.hold(FILE):
                started.set()
                time.sleep(0.2)
                order.append("first file chunk")

        def job(priority, name, delay):
            time.sleep(delay)
            with arbiter.hold(priority):
                order.append(name)

        threads = [threading.Thread(target=holder)]
        threads[0].start(); started.wait()
        for priority, name, delay in ((FILE, "file", 0.01), (IDENTIFICATION, "identification", 0.02), (LIVE, "live", 0.03)):
            threads.append(threading.Thread(target=job, args=(priority, name, delay)))
            threads[-1].start()
        for t in threads:
            t.join(5)
        assert order == ["first file chunk", "live", "identification", "file"]
