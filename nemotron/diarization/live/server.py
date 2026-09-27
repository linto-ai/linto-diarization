"""Websocket server of the live diarization, run in a thread of the worker process.

GET /ready answers 503 when the pod is full or late, so that the Kubernetes Service stops sending
it new sessions; a session opened anyway is closed with code 1013 and the client retries.
When NEMOTRON_LIVE_TOKEN is set, websocket clients must send "Authorization: Bearer <token>".
"""
import asyncio
import hmac
import json
import logging
import os
import threading
from http import HTTPStatus

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from . import protocol
from .session import LiveSession

log = logging.getLogger("__live-server__")

START_TIMEOUT = 10
MAX_LAG_SECONDS = float(os.environ.get("NEMOTRON_LIVE_MAX_LAG", 5))
# Audio received but not processed yet, per session: beyond it the session is closed (memory bound)
MAX_BACKLOG_SECONDS = float(os.environ.get("NEMOTRON_LIVE_MAX_BACKLOG", 300))


class LiveServer:
    def __init__(self, live_engine, identification, port, max_sessions, token=None):
        self.engine = live_engine
        self.identification = identification
        self.port = port
        self.max_sessions = max_sessions
        self.token = token
        self.active = 0  # sessions admitted, counted on the event loop (no race between handshakes)
        self.started = threading.Event()

    def accepting(self):
        return self.active < self.max_sessions and self.engine.max_lag() < MAX_LAG_SECONDS

    def start(self):
        threading.Thread(target=self._thread, name="live-server", daemon=True).start()
        self.started.wait(10)

    def _thread(self):
        asyncio.run(self._serve())

    async def _serve(self):
        async with serve(self.handle, "0.0.0.0", self.port, process_request=self._http, max_size=2**20):
            log.info(f"Live diarization listening on port {self.port} (max {self.max_sessions} sessions)")
            self.started.set()
            await asyncio.Future()

    def _http(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "OK\n")
        if request.path == "/ready":
            if self.accepting():
                return connection.respond(HTTPStatus.OK, "ready\n")
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "full\n")
        if self.token and not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {self.token}"):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None  # websocket handshake

    async def handle(self, websocket):
        outbox = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def emit(message):
            loop.call_soon_threadsafe(outbox.put_nowait, message)

        try:
            raw = await asyncio.wait_for(websocket.recv(), START_TIMEOUT)
            config = protocol.parse_start(json.loads(raw))
        except (asyncio.TimeoutError, ValueError, TypeError) as err:
            await websocket.send(json.dumps(protocol.error("invalid_start", str(err))))
            await websocket.close(protocol.CLOSE_INVALID, "invalid start message")
            return
        except ConnectionClosed:
            return
        if self.active >= self.max_sessions:
            await websocket.send(json.dumps(protocol.error("full", "no capacity left, retry")))
            await websocket.close(protocol.CLOSE_FULL, "server full")
            return
        self.active += 1
        try:
            await self._run_session(websocket, config, emit, outbox)
        finally:
            self.active -= 1

    async def _run_session(self, websocket, config, emit, outbox):
        loop = asyncio.get_running_loop()
        session = await loop.run_in_executor(
            None, lambda: LiveSession(config, emit, self.engine.geometry, self.engine.max_speakers, self.identification,
                                      self.engine.frames_per_pred))
        if config.get("identification") and session.identity is None:
            emit(protocol.error("identification_unavailable", "no usable collection, identification disabled"))
        emit(protocol.ready(self.engine.latency_ms, self.engine.max_speakers, session.frame_ms))
        await loop.run_in_executor(None, self.engine.add, session)
        sender = asyncio.create_task(self._send(websocket, outbox))
        log.info(f"Live session {session.id} started ({self.engine.count} running)")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    if not session.finished:
                        session.feed(message)
                        self.engine.notify()
                        if session.lag_seconds > MAX_BACKLOG_SECONDS:
                            await websocket.send(json.dumps(protocol.error(
                                "backlog", f"more than {MAX_BACKLOG_SECONDS:g} s of audio waiting, retry")))
                            await websocket.close(protocol.CLOSE_FULL, "backlog")
                            break
                    continue
                message = json.loads(message)
                if isinstance(message, dict) and message.get("type") == "stop":
                    session.finish()
                    if session.received == 0:
                        session.ended = True
                        emit(protocol.end(0))
                        self.engine.remove(session)
                    self.engine.notify()
        except (ConnectionClosed, ValueError):
            pass
        finally:
            if not session.ended:
                # client gone before the end: drop the session
                self.engine.remove(session)
                sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
        log.info(f"Live session {session.id} closed")

    async def _send(self, websocket, outbox):
        try:
            while True:
                message = await outbox.get()
                await websocket.send(json.dumps(message))
                if message["type"] == "end":
                    await websocket.close()
                    return
        except ConnectionClosed:
            return
        except Exception:
            log.exception("Live server: cannot send a message, closing the session")
            await websocket.close(1011, "internal error")
