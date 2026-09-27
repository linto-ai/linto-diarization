"""Live engine: one thread runs the chunks of all live sessions in batches on the shared model.

Sessions whose next chunk has the same geometry are stacked into one forward_streaming_step
(NeMo async streaming: each row keeps its own speaker cache and FIFO lengths). Batched and
single-session outputs agree to 1e-4 (float rounding), with the same speaker activity.
"""
import logging
import threading
import time

import torch

from diarization.gpu import LIVE

from . import protocol
from .session import HOP

log = logging.getLogger("__live-engine__")

STATE_KEYS = ("spkcache", "spkcache_preds", "spkcache_lengths", "spkcache_compressed", "fifo", "fifo_lengths",
              "mean_sil_emb", "n_sil_frames")


class LiveEngine:
    def __init__(self, engine, preset="low_latency", max_batch=16):
        self.engine = engine  # NemotronEngine (model, GPU arbiter)
        self.preset = preset
        self.max_batch = max_batch
        self.sessions = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.geometry = engine.preset_geometry(preset)  # left, chunk, right in 10 ms frames
        self.latency_ms = (self.geometry[1] + self.geometry[2]) * 10
        self.max_speakers = engine.max_speakers
        # Nemotron 3 predicts every 10 ms (high resolution): one prediction per feature frame
        self.frames_per_pred = int(engine.model.output_subsampling_factor)
        self.steps = 0
        self.thread = threading.Thread(target=self._run, name="live-engine", daemon=True)
        self.thread.start()

    # --- sessions ---

    def add(self, session):
        with self.engine.gpu.hold(LIVE):
            self.engine.use_preset(self.preset, async_streaming=True)  # state sizes depend on the preset
            session.state = self.engine.model.sortformer_modules.init_streaming_state(
                batch_size=1, async_streaming=True, device=self.engine.device)
        with self.lock:
            self.sessions[session.key] = session
        self.wake.set()

    def remove(self, session):
        session.closed = True
        with self.lock:
            self.sessions.pop(session.key, None)

    def notify(self):
        self.wake.set()

    @property
    def count(self):
        return len(self.sessions)

    def max_lag(self):
        with self.lock:
            sessions = list(self.sessions.values())
        return max((s.lag_seconds for s in sessions), default=0.0)

    # --- loop ---

    def _run(self):
        while True:
            self.wake.wait(timeout=0.05)
            self.wake.clear()
            try:
                while self._tick():
                    pass
            except Exception:
                log.exception("Live engine tick failed")
                time.sleep(0.5)

    def _tick(self):
        """Run one chunk for every session that has one ready. Return True if work was done."""
        with self.lock:
            sessions = list(self.sessions.values())
        groups = {}
        for session in sessions:
            if session.closed:
                continue
            spec = session.next_chunk()
            if spec is None:
                continue
            left, start, end, right, s, e = spec
            # Same geometry = same tensor shapes and offsets in the batch
            key = (left, right, end - start, e - s, (start - left) - s // HOP)
            groups.setdefault(key, []).append((session, spec))
        if not groups:
            return False
        for key, items in groups.items():
            for i in range(0, len(items), self.max_batch):
                batch = items[i:i + self.max_batch]
                try:
                    preds = self._step(key, batch)
                except Exception:
                    # nothing was updated: run the sessions one by one to isolate the faulty one
                    log.exception(f"Live step failed for {len(batch)} sessions, retrying them one by one")
                    for item in batch:
                        try:
                            self._deliver([item], self._step(key, [item]))
                        except Exception as err:
                            self._fail(item[0], err)
                    continue
                self._deliver(batch, preds)
        return True

    def _deliver(self, items, preds):
        for j, (session, (_, start, end, _, _, _)) in enumerate(items):
            try:
                session.on_preds(start, end, preds[j])
            except Exception as err:
                self._fail(session, err)
                continue
            if session.done():
                self.remove(session)

    def _fail(self, session, err):
        log.exception(f"Live session {session.id} failed, closing it")
        self.remove(session)
        session.ended = True
        session.emit(protocol.error("internal", f"diarization failed: {err}"))
        session.emit(protocol.end(session.pos * 10))

    @torch.inference_mode()
    def _step(self, key, items):
        """One batched streaming step; returns the predictions. Session states are replaced only
        once the whole step succeeded."""
        left, right, width, n_samples, rel = key
        engine, model = self.engine, self.engine.model
        sm = model.sortformer_modules
        audio = torch.stack([torch.from_numpy(session.samples(s, e)) for session, (_, _, _, _, s, e) in items])
        with engine.gpu.hold(LIVE):
            engine.use_preset(self.preset, async_streaming=True)
            x = audio.to(engine.device)
            lengths = torch.full((len(items),), n_samples, device=engine.device)
            feats, _ = model.preprocessor(input_signal=x, length=lengths)
            chunks = feats[:, :, rel:rel + left + width + right]
            state = sm.init_streaming_state(batch_size=len(items), async_streaming=True, device=engine.device)
            for k in STATE_KEYS:
                setattr(state, k, torch.cat([getattr(session.state, k) for session, _ in items]))
            state, preds = model.forward_streaming_step(
                processed_signal=chunks.transpose(1, 2),
                processed_signal_length=torch.full((len(items),), chunks.shape[2], device=engine.device),
                streaming_state=state,
                total_preds=torch.zeros((len(items), 0, sm.n_spk), device=engine.device),
                left_offset=left,
                right_offset=right,
            )
            preds = preds.float().cpu().numpy()
            for j, (session, _) in enumerate(items):
                for k in STATE_KEYS:
                    setattr(session.state, k, getattr(state, k)[j:j + 1].clone())
        self.steps += 1
        return preds
