"""SpeakerIdentityTracker: milestones, statuses, exclusivity (pure logic, no model, no Qdrant)."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron")]

from diarization.live.identify import SpeakerIdentityTracker  # noqa: E402

SR = 16000
SPEC = {"collections": ["spkid_" + "a" * 24 + "_" + "b" * 24], "speakers": "*", "minSimilarity": None}
ALICE = ("label:" + "1" * 24, "Alice", 0.8)
BOB = ("label:" + "2" * 24, "Bob", 0.75)


def seconds(n):
    return np.zeros(int(n * SR), dtype=np.float32)


def tracker(**kw):
    return SpeakerIdentityTracker(SPEC, milestones=(10, 30, 60), confirm_seconds=30, **kw)


class TestMilestones:
    def test_no_attempt_before_10s(self):
        t = tracker()
        assert t.add_speech("S1", seconds(9.9)) == []

    def test_attempts_at_10_30_60_then_stop(self):
        t = tracker()
        jobs = []
        for _ in range(80):
            jobs += t.add_speech("S1", seconds(1))
        assert [round(j[1]) for j in jobs] == [10, 30, 60]
        assert len(jobs[-1][2]) == 60 * SR  # speech capped at the last milestone

    def test_one_attempt_per_milestone_even_with_big_blocks(self):
        t = tracker()
        jobs = t.add_speech("S1", seconds(45))
        assert [round(j[1]) for j in jobs] == [45]  # the 10 s milestone fires with 45 s of speech
        assert [round(j[1]) for j in t.add_speech("S1", seconds(1))] == [46]  # then the 30 s one

    def test_speakers_are_independent(self):
        t = tracker()
        assert t.add_speech("S1", seconds(10)) and t.add_speech("S2", seconds(10))

    def test_threshold_from_env(self, monkeypatch):
        monkeypatch.setenv("SPEAKER_ID_MIN_SIMILARITY", "0.7")
        assert tracker().min_similarity == 0.7

    def test_threshold_from_spec(self):
        assert SpeakerIdentityTracker(dict(SPEC, minSimilarity=0.8)).min_similarity == 0.8


class TestStatuses:
    def test_provisional_then_confirmed(self):
        t = tracker()
        assert [m["status"] for m in t.apply_result("S1", 10, [ALICE])] == ["provisional"]
        m = t.apply_result("S1", 30, [ALICE])
        assert [(x["status"], x["name"]) for x in m] == [("confirmed", "Alice")]
        assert t.apply_result("S1", 60, [ALICE]) == []  # nothing new

    def test_confirmed_directly_with_enough_speech(self):
        assert tracker().apply_result("S1", 30, [ALICE])[0]["status"] == "confirmed"

    def test_no_match_no_message(self):
        assert tracker().apply_result("S1", 10, []) == []

    def test_revoked_when_no_longer_matching(self):
        t = tracker()
        t.apply_result("S1", 10, [ALICE])
        assert [m["status"] for m in t.apply_result("S1", 30, [])] == ["revoked"]
        assert "S1" not in t.current

    def test_name_change_is_revoke_then_new_name(self):
        t = tracker()
        t.apply_result("S1", 10, [ALICE])
        m = t.apply_result("S1", 30, [BOB])
        assert [(x["status"], x.get("name")) for x in m] == [("revoked", None), ("confirmed", "Bob")]

    def test_score_rounded_and_fields(self):
        m = tracker().apply_result("S1", 10, [("label:x", "Alice", 0.123456)])[0]
        assert m == {"type": "identity", "speaker": "S1", "status": "provisional", "speaker_id": "label:x",
                     "name": "Alice", "score": 0.123}


class TestExclusivity:
    def test_name_held_with_better_score_goes_to_next_candidate(self):
        t = tracker()
        t.apply_result("S1", 10, [ALICE])  # 0.8
        m = t.apply_result("S2", 10, [(ALICE[0], "Alice", 0.7), BOB])
        assert [(x["speaker"], x["name"]) for x in m] == [("S2", "Bob")]

    def test_name_held_with_better_score_and_no_other_candidate(self):
        t = tracker()
        t.apply_result("S1", 10, [ALICE])
        assert t.apply_result("S2", 10, [(ALICE[0], "Alice", 0.7)]) == []

    def test_better_score_takes_the_name(self):
        t = tracker()
        t.apply_result("S1", 10, [ALICE])  # 0.8
        m = t.apply_result("S2", 30, [(ALICE[0], "Alice", 0.9)])
        assert [(x["speaker"], x["status"]) for x in m] == [("S1", "revoked"), ("S2", "confirmed")]
        assert set(t.current) == {"S2"}

    def test_equal_score_keeps_the_first_holder(self):
        t = tracker()
        t.apply_result("S1", 10, [ALICE])
        assert t.apply_result("S2", 10, [ALICE]) == []
