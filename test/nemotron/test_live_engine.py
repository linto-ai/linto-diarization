"""LiveEngine scheduling without the model: failing batches and failing sessions."""
import os
import sys
import threading

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron")]
pytest.importorskip("nemo")

from diarization.live.engine import LiveEngine  # noqa: E402
from diarization.live.session import LiveSession  # noqa: E402


def engine_without_model():
    e = LiveEngine.__new__(LiveEngine)
    e.sessions, e.lock, e.max_batch, e.steps = {}, threading.Lock(), 16, 0
    return e


def session(messages):
    s = LiveSession({"type": "start"}, emit=messages.append, geometry=(8, 72, 32), max_speakers=8)
    s.feed(np.zeros(16000 * 3, dtype="<i2").tobytes())
    return s


def test_one_faulty_session_does_not_stop_the_others():
    engine = engine_without_model()
    out = {k: [] for k in "abc"}
    sessions = {k: session(out[k]) for k in "abc"}
    for s in sessions.values():
        engine.sessions[s.key] = s
    stepped = []

    def step(key, items):
        if len(items) > 1:
            raise RuntimeError("batch failed")
        if items[0][0] is sessions["b"]:
            raise RuntimeError("bad session")
        stepped.append(items[0][0])
        return np.zeros((1, 72, 8), dtype=np.float32)

    engine._step = step
    assert engine._tick()
    assert sessions["a"].pos == 72 and sessions["c"].pos == 72  # one chunk each, not two
    assert stepped.count(sessions["a"]) == 1
    assert [m["type"] for m in out["b"]] == ["error", "end"]
    assert sessions["b"].key not in engine.sessions


def test_delivery_failure_closes_only_that_session():
    engine = engine_without_model()
    out_a, out_b = [], []
    a, b = session(out_a), session(out_b)
    for s in (a, b):
        engine.sessions[s.key] = s

    def broken_on_preds(start, end, preds):
        raise RuntimeError("cannot deliver")

    b.on_preds = broken_on_preds
    engine._step = lambda key, items: np.zeros((len(items), 72, 8), dtype=np.float32)
    engine._tick()
    assert a.pos == 72 and a.key in engine.sessions
    assert [m["type"] for m in out_b] == ["error", "end"] and b.key not in engine.sessions
