"""GPU arbiter shared by the file engine and the live engine of the Nemotron worker."""
import contextlib
import threading

# GPU priorities: live chunks first, then speaker embeddings, then file chunks
LIVE, IDENTIFICATION, FILE = 0, 1, 2


class GpuArbiter:
    """Lets one thread at a time use the model; a waiting thread with a lower priority value goes first."""

    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._waiting = [0, 0, 0]

    @contextlib.contextmanager
    def hold(self, priority):
        with self._cond:
            self._waiting[priority] += 1
            while self._busy or any(self._waiting[p] for p in range(priority)):
                self._cond.wait()
            self._waiting[priority] -= 1
            self._busy = True
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()
