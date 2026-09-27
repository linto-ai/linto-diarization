"""Live (streaming) diarization over websocket, served by the Nemotron worker next to its Celery tasks.

Enabled when NEMOTRON_LIVE_PORT is set. See nemotron/LIVE.md for the protocol.
"""
import logging
import os

log = logging.getLogger("__live__")


def start_live_server(speaker_diarization):
    """Start the live engine and websocket server on the worker's model. Returns the server or None."""
    port = os.environ.get("NEMOTRON_LIVE_PORT")
    if not port:
        return None
    from diarization.processing.engine import IDENTIFICATION

    from .engine import LiveEngine
    from .identify import IdentificationWorker
    from .server import LiveServer

    engine = speaker_diarization.engine
    live = LiveEngine(
        engine,
        preset=os.environ.get("NEMOTRON_LIVE_PRESET", "low_latency"),
        max_batch=int(os.environ.get("NEMOTRON_LIVE_MAX_BATCH", 16)),
    )
    identifier = speaker_diarization.speaker_identifier
    identifier.embedding.gpu_guard = lambda: engine.gpu.hold(IDENTIFICATION)
    server = LiveServer(
        live,
        IdentificationWorker(identifier),
        int(port),
        int(os.environ.get("NEMOTRON_MAX_LIVE_SESSIONS", 16)),
    )
    server.start()
    return server
