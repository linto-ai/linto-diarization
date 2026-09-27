"""LiveSession audio buffer and chunk scheduling (no model)."""
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron")]

from diarization.live.session import HOP, MARGIN, LiveSession  # noqa: E402

GEOMETRY = (8, 72, 32)


def session():
    return LiveSession({"type": "start"}, emit=lambda m: None, geometry=GEOMETRY, max_speakers=8)


def pcm(values):
    return np.asarray(values, dtype="<i2").tobytes()


def test_backlog_is_cheap():
    """20 min received at once (reconnection backlog): constant cost per 100 ms block."""
    s = session()
    block = pcm(np.zeros(1600))
    t = time.monotonic()
    for _ in range(12000):
        s.feed(block)
    assert time.monotonic() - t < 2.0
    assert s.received == 12000 * 1600


def test_samples_survive_compaction():
    s = session()
    values = np.arange(16000 * 60) % 30000
    for i in range(0, len(values), 1600):
        s.feed(pcm(values[i:i + 1600]))
    s.pos = 5000  # 50 s processed
    s._trim()
    assert s.audio_base > 0
    a = (s.pos - 8) * HOP - MARGIN
    got = s.samples(a, a + 1600)
    assert np.allclose(got * 32768, values[a:a + 1600])


def test_next_chunk_waits_for_right_context_and_margin():
    s = session()
    need = (72 + 32) * HOP + MARGIN
    s.feed(pcm(np.zeros(need - 1)))
    assert s.next_chunk() is None
    s.feed(pcm(np.zeros(1)))
    assert s.next_chunk() == (0, 0, 72, 32, 0, need)


def test_last_chunks_after_stop():
    s = session()
    s.feed(pcm(np.zeros(16000)))  # 1 s: 101 feature frames
    s.finish()
    left, start, end, right, a, b = s.next_chunk()
    assert (left, start, end, right) == (0, 0, 72, 29)
    s.pos = 72
    assert s.next_chunk()[1:4] == (72, 101, 0)
    s.pos = 101
    assert s.next_chunk() is None and s.done()
