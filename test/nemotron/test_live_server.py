"""Websocket server + sessions with a fake engine (synthetic predictions, no GPU) and the
reference client: protocol flow, capacity, readiness, disconnection, identity messages."""
import asyncio
import json
import os
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron"), os.path.join(ROOT, "nemotron", "tools")]
pytest.importorskip("websockets")
sf = pytest.importorskip("soundfile")

from websockets.asyncio.client import connect  # noqa: E402

import live_client  # noqa: E402
from diarization.live.server import LiveServer  # noqa: E402

ORG = "a" * 24
COLLECTION = f"spkid_{ORG}_{'b' * 24}"
ALICE = ("label:" + "1" * 24, "Alice", 0.9)


class FakeEngine:
    """Same interface as LiveEngine; speaker 0 talks on even 5 s windows, speaker 1 on odd ones."""
    geometry = (8, 72, 32)
    latency_ms = 1040
    max_speakers = 8
    frames_per_pred = 1

    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def count(self):
        return len(self.sessions)

    def max_lag(self):
        return 0.0

    paused = False

    def add(self, session):
        with self.lock:
            self.sessions[session.key] = session
        self.wake.set()

    def remove(self, session):
        session.closed = True
        with self.lock:
            self.sessions.pop(session.key, None)

    def notify(self):
        self.wake.set()

    def _run(self):
        while True:
            self.wake.wait(0.05)
            self.wake.clear()
            with self.lock:
                sessions = list(self.sessions.values())
            for s in sessions:
                while not self.paused and not s.closed and (spec := s.next_chunk()) is not None:
                    _, start, end, _, _, _ = spec
                    frames = np.arange(start, end)
                    preds = np.zeros((end - start, 8), dtype=np.float32)
                    preds[np.arange(end - start), (frames // 500) % 2] = 0.9
                    s.on_preds(start, end, preds)
                    if s.done():
                        self.remove(s)


class FakeIdentification:
    enabled = True

    def __init__(self):
        self.jobs = []

    def usable_collections(self, collections):
        return collections

    def submit(self, session, label, seconds, audio, until_ms=None):
        self.jobs.append((label, int(seconds)))
        session.apply_identity(label, seconds, [ALICE] if label == "S1" else [], until_ms)


def make_server(port, **kw):
    srv = LiveServer(FakeEngine(), FakeIdentification(), port, max_sessions=2, **kw)
    srv.start()
    srv.url = f"ws://localhost:{port}"
    srv.http = f"http://localhost:{port}"
    return srv


@pytest.fixture
def server(unused_tcp_port):
    return make_server(unused_tcp_port)


@pytest.fixture
def unused_tcp_port():
    import socket
    s = socket.socket(); s.bind(("localhost", 0)); port = s.getsockname()[1]; s.close()
    return port


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    sf.write(path, (rng.standard_normal(16000 * 70) * 3000).astype(np.int16), 16000)
    return str(path)


def test_full_session(server, wav):
    r = asyncio.run(live_client.stream(server.url, wav, speed=0))
    assert r["ready"] == {"type": "ready", "latency_ms": 1040, "max_speakers": 8, "frame_ms": 10}
    assert r["until_ms"] == 70000 and not r["errors"]
    runs = r["merged"]
    assert set(runs) == {"S1", "S2"}
    assert runs["S1"][0] == (0, 5000) and runs["S2"][0] == (5000, 10000)  # runs cut at chunk edges are joined
    assert server.engine.count == 0


def test_invalid_start(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.send(json.dumps({"type": "start", "sample_rate": 8000}))
            error = json.loads(await ws.recv())
            await ws.wait_closed()
            return error, ws.close_code
    error, code = asyncio.run(go())
    assert error["code"] == "invalid_start" and code == 1008


def test_capacity_and_readiness(server):
    assert urllib.request.urlopen(server.http + "/ready").status == 200

    async def go():
        async with connect(server.url) as a, connect(server.url) as b:
            for ws in (a, b):
                await ws.send(json.dumps({"type": "start"}))
                assert json.loads(await ws.recv())["type"] == "ready"
            await asyncio.sleep(0.2)
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(server.http + "/ready")
            assert err.value.code == 503
            async with connect(server.url) as c:
                await c.send(json.dumps({"type": "start"}))
                assert json.loads(await c.recv())["code"] == "full"
                await c.wait_closed()
                return c.close_code
    assert asyncio.run(go()) == 1013
    assert urllib.request.urlopen(server.http + "/ready").status == 200  # sessions dropped on disconnect


def test_stop_without_audio(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws.recv()
            await ws.send(json.dumps({"type": "stop"}))
            return json.loads(await ws.recv())
    assert asyncio.run(go()) == {"type": "end", "until_ms": 0}


def test_disconnect_drops_the_session(server, wav):
    async def go():
        async with connect(server.url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws.recv()
            await ws.send(b"\0" * 32000)
            await asyncio.sleep(0.2)
            assert server.engine.count == 1
    asyncio.run(go())
    for _ in range(50):
        if server.engine.count == 0:
            break
        asyncio.run(asyncio.sleep(0.02))
    assert server.engine.count == 0


def test_identity_messages(server, wav):
    spec = {"organizationId": ORG, "collections": [COLLECTION]}
    r = asyncio.run(live_client.stream(server.url, wav, speed=0, identification=spec))
    # S1 speaks 35 s over 70 s: attempts at 10 and 30 s of non-overlapped speech; S2 never matches
    assert [(i["speaker"], i["status"], i.get("name")) for i in r["identities"]] == [
        ("S1", "provisional", "Alice"), ("S1", "confirmed", "Alice")]
    assert sorted(server.identification.jobs) == [("S1", 10), ("S1", 30), ("S2", 10), ("S2", 30)]
    # S1 speaks 5 s out of 10: its 10 s of speech are reached about 20 s into the audio
    first = r["identities"][0]
    assert 19000 <= first["until_ms"] <= 22000 and 10 <= first["speech_s"] < 11


def test_token(unused_tcp_port, wav):
    from websockets.exceptions import InvalidStatus

    srv = make_server(unused_tcp_port, token="s3cret")
    with pytest.raises(InvalidStatus) as err:
        asyncio.run(live_client.stream(srv.url, wav, speed=0))
    assert err.value.response.status_code == 401
    with pytest.raises(InvalidStatus):
        asyncio.run(live_client.stream(srv.url, wav, speed=0, token="wrong"))
    assert urllib.request.urlopen(srv.http + "/ready").status == 200  # probes need no token
    assert asyncio.run(live_client.stream(srv.url, wav, speed=0, token="s3cret"))["until_ms"] == 70000


def test_backlog_closes_the_session(server, wav, monkeypatch):
    import diarization.live.server as server_module

    monkeypatch.setattr(server_module, "MAX_BACKLOG_SECONDS", 5)
    server.engine.paused = True  # nothing is processed: the backlog grows

    async def go():
        from websockets.exceptions import ConnectionClosed

        async with connect(server.url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws.recv()
            messages = []
            try:
                for _ in range(60):
                    await ws.send(b"\0" * 3200)
            except ConnectionClosed:
                pass
            try:
                async for m in ws:
                    messages.append(json.loads(m))
            except ConnectionClosed:
                pass
            return messages, ws.close_code
    messages, code = asyncio.run(go())
    assert messages[-1]["code"] == "backlog" and code == 1013


def test_same_client_session_id_twice(server, wav):
    """Client ids are only labels: two sessions with the same id run side by side."""
    async def both():
        return await asyncio.gather(*(live_client.stream(server.url, wav, speed=0, session="channel-12") for _ in range(2)))
    for r in asyncio.run(both()):
        assert r["until_ms"] == 70000 and not r["errors"]


def test_unexpected_text_messages_are_ignored(server):
    async def go():
        async with connect(server.url) as ws:
            await ws.send(json.dumps({"type": "start"}))
            await ws.recv()
            await ws.send("[1, 2]")
            await ws.send(json.dumps({"type": "pause"}))
            await ws.send(json.dumps({"type": "stop"}))
            return json.loads(await ws.recv())
    assert asyncio.run(go()) == {"type": "end", "until_ms": 0}
