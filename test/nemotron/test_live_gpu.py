"""Live diarization on the real model, through the websocket server and the reference client.

Needs a CUDA GPU, NEMOTRON_MODEL (.nemo path) and NEMOTRON_TEST_AUDIO (16 kHz mono wav, 3 min or
more). Run: uv run pytest -m gpu test/nemotron/test_live_gpu.py
"""
import asyncio
import os
import socket
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
AUDIO = os.environ.get("NEMOTRON_TEST_AUDIO")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif(not AUDIO or not os.environ.get("NEMOTRON_MODEL"), reason="NEMOTRON_TEST_AUDIO / NEMOTRON_MODEL not set"),
]

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron"), os.path.join(ROOT, "nemotron", "tools")]


@pytest.fixture(scope="module")
def live():
    port = socket.socket(); port.bind(("localhost", 0)); p = port.getsockname()[1]; port.close()
    os.environ["NEMOTRON_LIVE_PORT"] = str(p)
    from diarization.processing import diarizationworker

    server = diarizationworker.start_live_server()
    return server, diarizationworker.engine, f"ws://localhost:{p}"


def activity(merged, frames):
    """{label: [(start_ms, end_ms)]} -> bool [frames, 8] at 10 ms."""
    a = np.zeros((frames, 8), dtype=bool)
    for label, runs in merged.items():
        k = int(label[1:]) - 1
        for s, e in runs:
            a[s // 10:e // 10, k] = True
    return a


def file_activity(engine, wav_path):
    """File engine with the live preset: the reference for the live output."""
    saved = engine.preset
    engine.preset = "low_latency"
    try:
        return engine.diarize_file(wav_path) > 0.5
    finally:
        engine.preset = saved


def slices(tmp_path, n, seconds=60, step=20):
    import soundfile as sf
    audio, sr = sf.read(AUDIO, dtype="int16")
    paths = []
    for k in range(n):
        p = str(tmp_path / f"slice{k}.wav")
        sf.write(p, audio[k * step * sr:(k * step + seconds) * sr], sr)
        paths.append(p)
    return paths


def agreement(live_act, ref):
    n = min(len(live_act), len(ref))
    return float((live_act[:n] == ref[:n]).mean())


def test_live_matches_file_engine(live, tmp_path):
    server, engine, url = live
    path = slices(tmp_path, 1, seconds=180)[0]
    r = asyncio.run(__import__("live_client").stream(url, path, speed=0))
    ref = file_activity(engine, path)
    assert not r["errors"]
    assert agreement(activity(r["merged"], len(ref)), ref) > 0.999


def test_concurrent_sessions_match_alone(live, tmp_path):
    """6 sessions started 0.5 s apart on different audio: batched output = each session alone."""
    import live_client
    server, engine, url = live
    paths = slices(tmp_path, 6)

    async def all_sessions():
        async def one(i, p):
            await asyncio.sleep(0.5 * i)
            return await live_client.stream(url, p, speed=0)
        return await asyncio.gather(*(one(i, p) for i, p in enumerate(paths)))

    results = asyncio.run(all_sessions())
    assert server.engine.steps > 0
    for p, r in zip(paths, results):
        ref = file_activity(engine, p)
        assert agreement(activity(r["merged"], len(ref)), ref) > 0.999, p


def test_live_and_file_jobs_share_the_model(live, tmp_path):
    """A file job (offline preset) runs while live sessions start and stream (low_latency preset):
    both outputs equal their references, so the shared model's settings never leak between them."""
    import threading

    import live_client
    server, engine, url = live
    paths = slices(tmp_path, 3)
    file_path = slices(tmp_path, 1, seconds=180)[0]
    reference_file = engine.diarize_file(file_path)  # offline preset, alone
    reference_live = [file_activity(engine, p) for p in paths]
    file_result = {}
    worker = threading.Thread(target=lambda: file_result.update(p=engine.diarize_file(file_path)))
    worker.start()

    async def sessions():
        async def one(i, p):
            await asyncio.sleep(0.3 * i)
            return await live_client.stream(url, p, speed=0)
        return await asyncio.gather(*(one(i, p) for i, p in enumerate(paths)))

    results = asyncio.run(sessions())
    worker.join()
    assert np.array_equal(file_result["p"], reference_file)
    for r, ref in zip(results, reference_live):
        assert not r["errors"]
        assert agreement(activity(r["merged"], len(ref)), ref) > 0.999
