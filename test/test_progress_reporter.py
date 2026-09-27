import logging
import os
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# celery_app.tasks loads the diarization model at import: stub the engine package
os.environ.setdefault("SERVICES_BROKER", "redis://localhost:6379")
if "diarization" not in sys.modules:
    diarization = types.ModuleType("diarization")
    diarization.logger = logging.getLogger("test")
    processing = types.ModuleType("diarization.processing")
    processing.diarizationworker = None
    sys.modules["diarization"] = diarization
    sys.modules["diarization.processing"] = processing

from celery_app.tasks import _progress_reporter  # noqa: E402


class FakeTask:
    def __init__(self):
        self.published = []

    def update_state(self, state, meta):
        assert state == "PROGRESS"
        self.published.append(meta["progress"])


def test_throttled_and_final_always_sent():
    task = FakeTask()
    report = _progress_reporter(task, min_step=0.1)
    for p in (0.0, 0.05, 0.1, 0.15, 0.25, 0.99, 1.0):
        report(p)
    assert task.published == [0.0, 0.1, 0.25, 0.99, 1.0]


def test_rounded():
    task = FakeTask()
    _progress_reporter(task)(1 / 3)
    assert task.published == [0.333]
