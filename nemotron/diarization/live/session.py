"""One live diarization session: audio buffer, chunk scheduling, streaming state and outputs."""
import threading
import uuid

import numpy as np

from . import protocol, turns
from .identify import SpeakerIdentityTracker

HOP = 160  # samples per 10 ms feature frame
MARGIN = 1600  # audio around a chunk for its features (800 samples are enough for identical features)


class LiveSession:
    def __init__(self, config, emit, geometry, max_speakers, identification=None, frames_per_pred=1):
        """emit(message) sends a message to the client (thread-safe).
        geometry: (left, chunk, right) in 10 ms feature frames; frames_per_pred: feature frames per
        prediction frame (1 for Nemotron 3, which predicts every 10 ms).
        identification: IdentificationWorker or None."""
        self.id = config.get("session") or uuid.uuid4().hex
        self.emit = emit
        self.left, self.chunk, self.right = geometry
        self.frames_per_pred = frames_per_pred
        self.frame_ms = 10 * frames_per_pred
        self.max_speakers = max_speakers
        self.lock = threading.Lock()
        self._buffer = np.zeros(16000 * 30, dtype=np.float32)  # grows by doubling
        self._length = 0  # samples held in self._buffer
        self.audio_base = 0  # absolute index of self._buffer[0]
        self.received = 0  # samples received
        self.pos = 0  # next chunk start, in 10 ms frames
        self.finished = False  # stop received
        self.closed = False  # removed from the engine
        self.ended = False  # "end" sent
        self.state = None  # streaming state (async, batch of 1), created by the engine
        self.seen = set()
        self.saturated_sent = False
        self.identity = None
        self.collections = []
        self.identification = None
        spec = config.get("identification")
        if spec and identification is not None and identification.enabled:
            self.collections = identification.usable_collections(spec["collections"])
            if self.collections:
                self.identity = SpeakerIdentityTracker(spec)
                self.identification = identification

    # --- audio input (websocket thread) ---

    def feed(self, pcm):
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        with self.lock:
            need = self._length + len(samples)
            if need > len(self._buffer):
                grown = np.zeros(max(need, 2 * len(self._buffer)), dtype=np.float32)
                grown[:self._length] = self._buffer[:self._length]
                self._buffer = grown
            self._buffer[self._length:need] = samples
            self._length = need
            self.received += len(samples)

    def finish(self):
        with self.lock:
            self.finished = True

    @property
    def lag_seconds(self):
        return max(0.0, (self.received - self.pos * HOP) / protocol.SAMPLE_RATE)

    # --- scheduling (engine thread) ---

    def total_frames(self):
        """Feature frames of the whole audio once the session is finished (center-padded STFT)."""
        return self.received // HOP + 1

    def next_chunk(self):
        """(left, start, end, right, s, e) for the next chunk if its audio is there, else None.
        [s, e) is the absolute sample range to compute features on."""
        with self.lock:
            finished, received = self.finished, self.received
        if finished:
            total = self.total_frames()
            if self.pos >= total:
                return None
        start = self.pos
        left = min(self.left, start)
        end = start + self.chunk
        right = self.right
        if finished:
            end = min(end, total)
            right = min(right, total - end)
        elif (end + right) * HOP + MARGIN > received:
            return None
        s = max(0, (start - left) * HOP - MARGIN)
        e = min(received, (end + right) * HOP + MARGIN)
        return left, start, end, right, s, e

    def samples(self, s, e):
        with self.lock:
            return self._buffer[s - self.audio_base:e - self.audio_base].copy()

    def done(self):
        return self.finished and self.pos >= self.total_frames()

    # --- results (engine thread) ---

    def on_preds(self, start, end, preds):
        """preds: [80 ms frames, speakers] for feature frames [start, end)."""
        first = start // self.frames_per_pred
        preds = preds[: max(0, -(-(end - start) // self.frames_per_pred))]
        found = turns.runs(preds, first, self.frame_ms)
        self.pos = end
        # the last feature frame covers the STFT padding: never report beyond the audio received
        until_ms = min(end * 10, self.received * 1000 // protocol.SAMPLE_RATE)
        if found:
            self.emit(protocol.turns(until_ms, found))
            self.seen.update(r[0] for r in found)
            if not self.saturated_sent and len(self.seen) >= self.max_speakers:
                self.saturated_sent = True
                self.emit(protocol.saturated(self.max_speakers))
        if self.identity is not None:
            self._collect_speech(first, preds)
        self._trim()
        if self.done():
            self.ended = True
            self.emit(protocol.end(until_ms))

    def _collect_speech(self, first, preds):
        who = turns.single_speaker(preds, erode=max(1, 50 // self.frame_ms))
        step = self.frames_per_pred * HOP
        base = first * step
        audio = self.samples(base, base + len(who) * step)  # the audio of this chunk only
        for speaker in set(who[who >= 0].tolist()):
            frames = np.flatnonzero(who == speaker)
            parts = [audio[f * step:(f + 1) * step] for f in frames]
            parts = [p for p in parts if len(p)]
            if not parts:
                continue
            label = protocol.speaker_label(speaker)
            for job in self.identity.add_speech(label, np.concatenate(parts)):
                self.identification.submit(self, *job, until_ms=self.pos * 10)

    def _trim(self):
        keep_from = max(0, (self.pos - self.left) * HOP - MARGIN)
        with self.lock:
            drop = keep_from - self.audio_base
            # compact once half of the buffer is behind us: one copy of what is left
            if drop > 0 and drop * 2 >= self._length:
                rest = self._length - drop
                self._buffer[:rest] = self._buffer[drop:self._length]
                self._length = rest
                self.audio_base = keep_from

    def apply_identity(self, label, seconds, ranked, until_ms=None):
        """Called by the identification worker. until_ms: audio position when the attempt started."""
        with self.lock:
            messages = self.identity.apply_result(label, seconds, ranked)
        for message in messages:
            message["until_ms"] = until_ms
            message["speech_s"] = round(seconds, 1)
            self.emit(message)
