"""Speaker identification during a live session.

For each diarized speaker the session keeps its non-overlapped speech. When that speech
reaches a milestone (10, 30, 60 s by default) a voiceprint is computed and searched in the
requested collections. Statuses sent to the client:
  provisional  first match, before CONFIRM_SECONDS of speech
  confirmed    match with at least CONFIRM_SECONDS of speech
  revoked      a later attempt no longer supports the name (score below the threshold, or
               the enrolled speaker is given to another diarized speaker with a better score)
An enrolled speaker names one diarized speaker at most per session.
"""
import logging
import os
import queue
import threading

import numpy as np
import torch

from identification.spkid_core import MODEL_ID, rank_speaker_votes, resolve_collections, resolve_min_similarity

from . import protocol

log = logging.getLogger("__live-identification__")

SAMPLE_RATE = protocol.SAMPLE_RATE


def _milestones():
    raw = os.environ.get("SPEAKER_ID_LIVE_MILESTONES", "10,30,60")
    return tuple(sorted(float(x) for x in raw.split(",") if x.strip()))


CONFIRM_SECONDS = float(os.environ.get("SPEAKER_ID_LIVE_CONFIRM_SECONDS", 30))


class SpeakerIdentityTracker:
    """Identity state of the diarized speakers of one session. Pure logic, no I/O.

    add_speech() returns the attempts to run; apply_result() returns the identity messages."""

    def __init__(self, spec, milestones=None, confirm_seconds=CONFIRM_SECONDS):
        self.min_similarity = resolve_min_similarity(spec.get("minSimilarity"))
        self.allowed = None if spec.get("speakers", "*") == "*" else set(spec["speakers"])
        self.milestones = milestones or _milestones()
        self.confirm_seconds = confirm_seconds
        self.max_samples = int(self.milestones[-1] * SAMPLE_RATE)
        self.speech = {}  # label -> list of float32 arrays
        self.samples = {}  # label -> number of samples kept
        self.next_milestone = {}  # label -> index in milestones
        self.current = {}  # label -> (speaker_id, name, score, status)

    def add_speech(self, label, audio):
        """Append non-overlapped speech of a diarized speaker; return [(label, seconds, audio)] to identify."""
        have = self.samples.get(label, 0)
        if have >= self.max_samples or len(audio) == 0:
            return []
        audio = audio[: self.max_samples - have]
        self.speech.setdefault(label, []).append(audio)
        self.samples[label] = have + len(audio)
        index = self.next_milestone.get(label, 0)
        if index < len(self.milestones) and self.samples[label] >= self.milestones[index] * SAMPLE_RATE:
            self.next_milestone[label] = index + 1
            seconds = self.samples[label] / SAMPLE_RATE
            return [(label, seconds, np.concatenate(self.speech[label]))]
        return []

    def held_by_others(self, label):
        """Enrolled speaker id -> score, for the names held by other diarized speakers."""
        return {v[0]: v[2] for k, v in self.current.items() if k != label}

    def apply_result(self, label, seconds, ranked):
        """ranked: [(speaker_id, name, score)] best first, already above the threshold."""
        messages = []
        held = self.held_by_others(label)
        choice = None
        for speaker_id, name, score in ranked:
            if speaker_id in held and held[speaker_id] >= score:
                continue  # another diarized speaker holds this name with a better score
            choice = (speaker_id, name, score)
            break
        previous = self.current.get(label)
        if choice is None:
            if previous is not None:
                del self.current[label]
                messages.append(protocol.identity(label, "revoked"))
            return messages
        speaker_id, name, score = choice
        # Take the name from the diarized speaker that held it with a lower score
        for other, value in list(self.current.items()):
            if other != label and value[0] == speaker_id:
                del self.current[other]
                messages.append(protocol.identity(other, "revoked"))
        status = "confirmed" if seconds >= self.confirm_seconds else "provisional"
        if previous is not None and previous[0] != speaker_id:
            messages.append(protocol.identity(label, "revoked"))
        if previous is None or previous[0] != speaker_id or previous[3] != status:
            messages.append(protocol.identity(label, status, speaker_id, name, score))
        self.current[label] = (speaker_id, name, score, status)
        return messages


class IdentificationWorker:
    """Thread computing voiceprints (under the GPU arbiter) and searching Qdrant for live sessions."""

    def __init__(self, speaker_identifier):
        self.identifier = speaker_identifier
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="live-identification", daemon=True)
        self.thread.start()

    @property
    def enabled(self):
        return self.identifier is not None and self.identifier.is_speaker_identification_enabled()

    def usable_collections(self, collections):
        store = self.identifier.store
        return resolve_collections(collections, store.collection_exists, store.get_collection_model_id, MODEL_ID, log=log)

    def submit(self, session, label, seconds, audio, until_ms=None):
        self.jobs.put((session, label, seconds, audio, until_ms))

    def search(self, audio, collections, tracker):
        tensor = torch.from_numpy(audio).unsqueeze(0)
        vector = self.identifier.embedding.compute_embedding(tensor)[0].flatten()
        hits = []
        for collection in collections:
            try:
                results = self.identifier.store.search(collection, vector, limit=10)
            except Exception as err:
                log.warning(f"Live identification: search failed on {collection}: {err}")
                continue
            hits.extend((r.score, r.payload or {}) for r in results)
        return rank_speaker_votes(hits, tracker.min_similarity, allowed_speaker_ids=tracker.allowed)

    def _run(self):
        while True:
            session, label, seconds, audio, until_ms = self.jobs.get()
            if session.closed:
                continue
            try:
                ranked = self.search(audio, session.collections, session.identity)
                session.apply_identity(label, seconds, ranked, until_ms)
            except Exception as err:
                log.exception(f"Live identification failed for {session.id} {label}: {err}")
