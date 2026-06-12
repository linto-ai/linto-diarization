"""Unit tests for identification/spkid_core.py (pure, no torch/speechbrain/Qdrant needed)

Run with: pytest test/test_spkid_core.py
"""

import logging
import os
import sys
import uuid

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identification.spkid_core import (
    MODEL_DIM,
    MODEL_ID,
    NAMESPACE_SPKID,
    aggregate_speaker_votes,
    build_point_payload,
    check_speaker_spec_dict,
    parse_collection_name,
    payload_name,
    resolve_collections,
    resolve_min_similarity,
    speaker_point_id,
)

ORG_ID = "64ff00000000000000000001"
COLL_ID = "65aa00000000000000000002"
COLLECTION = f"spkid_{ORG_ID}_{COLL_ID}"
COLLECTION_2 = f"spkid_{ORG_ID}_65bb00000000000000000003"


class TestPointIds:
    def test_point_id_is_deterministic(self):
        assert speaker_point_id("label:65cc00000000000000000004") == speaker_point_id("label:65cc00000000000000000004")

    def test_point_id_is_uuid5_of_namespace(self):
        speaker_id = "user:64dd00000000000000000005"
        assert speaker_point_id(speaker_id) == str(uuid.uuid5(NAMESPACE_SPKID, speaker_id))

    def test_point_id_differs_per_speaker(self):
        assert speaker_point_id("label:65cc00000000000000000004") != speaker_point_id("user:65cc00000000000000000004")


class TestCollectionName:
    def test_parse_valid(self):
        assert parse_collection_name(COLLECTION) == (ORG_ID, COLL_ID)

    @pytest.mark.parametrize("name", [
        "",
        None,
        "spkid_123_456",
        f"spkid_{ORG_ID}_{COLL_ID}_extra",
        f"other_{ORG_ID}_{COLL_ID}",
        f"spkid_{ORG_ID.upper()}_{COLL_ID}",
        f"spkid_{ORG_ID}",
    ])
    def test_parse_invalid(self, name):
        with pytest.raises(ValueError):
            parse_collection_name(name)


class TestPayload:
    def test_payload_name_new_format(self):
        assert payload_name({"name": "Griogy", "person": "ignored"}) == "Griogy"

    def test_payload_name_legacy_format(self):
        assert payload_name({"person": "Yoann"}) == "Yoann"

    def test_payload_name_empty(self):
        assert payload_name({}) is None

    def test_build_point_payload(self):
        payload = build_point_payload(COLLECTION, "label:65cc00000000000000000004", "Griogy", MODEL_ID)
        assert payload["speaker_id"] == "label:65cc00000000000000000004"
        assert payload["name"] == "Griogy"
        assert payload["organization_id"] == ORG_ID
        assert payload["studio_collection_id"] == COLL_ID
        assert payload["model_id"] == MODEL_ID
        assert payload["updated_at"]

    def test_build_point_payload_invalid_collection(self):
        with pytest.raises(ValueError):
            build_point_payload("bad_name", "label:65cc00000000000000000004", "Griogy", MODEL_ID)


class TestCheckSpeakerSpecDict:
    def test_minimal_spec(self):
        spec = check_speaker_spec_dict({"collections": [COLLECTION]})
        assert spec == {"collections": [COLLECTION], "speakers": "*", "minSimilarity": None}

    def test_full_spec(self):
        spec = check_speaker_spec_dict({
            "collections": [COLLECTION, COLLECTION_2],
            "speakers": ["label:65cc00000000000000000004"],
            "minSimilarity": 0.7,
        })
        assert spec["collections"] == [COLLECTION, COLLECTION_2]
        assert spec["speakers"] == ["label:65cc00000000000000000004"]
        assert spec["minSimilarity"] == 0.7

    def test_speakers_none_means_wildcard(self):
        spec = check_speaker_spec_dict({"collections": [COLLECTION], "speakers": None})
        assert spec["speakers"] == "*"

    @pytest.mark.parametrize("bad_spec", [
        {},
        {"collections": []},
        {"collections": "not_a_list"},
        {"collections": [COLLECTION], "unknown_key": 1},
        {"collections": ["bad_collection_name"]},
        {"collections": [COLLECTION, 42]},
        {"collections": [COLLECTION], "speakers": "all"},
        {"collections": [COLLECTION], "speakers": [42]},
        {"collections": [COLLECTION], "speakers": [""]},
        {"collections": [COLLECTION], "minSimilarity": "high"},
        {"collections": [COLLECTION], "minSimilarity": 1.5},
        {"collections": [COLLECTION], "minSimilarity": -0.1},
        {"collections": [COLLECTION], "minSimilarity": True},
        "not_a_dict",
    ])
    def test_invalid_specs(self, bad_spec):
        with pytest.raises(ValueError):
            check_speaker_spec_dict(bad_spec)

    def test_is_idempotent(self):
        spec = check_speaker_spec_dict({"collections": [COLLECTION], "minSimilarity": 0.6})
        assert check_speaker_spec_dict(spec) == spec


class TestResolveMinSimilarity:
    def test_spec_value_wins(self):
        assert resolve_min_similarity(0.8, env={"SPEAKER_ID_MIN_SIMILARITY": "0.3"}) == 0.8

    def test_env_fallback(self):
        assert resolve_min_similarity(None, env={"SPEAKER_ID_MIN_SIMILARITY": "0.3"}) == 0.3

    def test_default(self):
        assert resolve_min_similarity(None, env={}) == 0.5

    def test_empty_env_value(self):
        assert resolve_min_similarity(None, env={"SPEAKER_ID_MIN_SIMILARITY": ""}) == 0.5


class TestResolveCollections:
    def test_missing_collection_is_skipped(self):
        log = logging.getLogger("test")
        usable = resolve_collections(
            [COLLECTION, COLLECTION_2],
            exists_fn=lambda c: c == COLLECTION,
            model_id_fn=lambda c: MODEL_ID,
            expected_model_id=MODEL_ID,
            log=log,
        )
        assert usable == [COLLECTION]

    def test_model_mismatch_is_skipped(self):
        usable = resolve_collections(
            [COLLECTION, COLLECTION_2],
            exists_fn=lambda c: True,
            model_id_fn=lambda c: "other/model" if c == COLLECTION_2 else MODEL_ID,
            expected_model_id=MODEL_ID,
        )
        assert usable == [COLLECTION]

    def test_unknown_model_is_kept(self):
        # Empty collection (no readable model_id): kept, cannot yield mismatched hits
        usable = resolve_collections(
            [COLLECTION],
            exists_fn=lambda c: True,
            model_id_fn=lambda c: None,
            expected_model_id=MODEL_ID,
        )
        assert usable == [COLLECTION]

    def test_all_skipped(self):
        usable = resolve_collections(
            [COLLECTION],
            exists_fn=lambda c: False,
            model_id_fn=lambda c: None,
            expected_model_id=MODEL_ID,
        )
        assert usable == []


def _hit(score, speaker_id, name=None, person=None):
    payload = {"speaker_id": speaker_id} if speaker_id else {}
    if name:
        payload["name"] = name
    if person:
        payload["person"] = person
    return (score, payload)


class TestAggregateSpeakerVotes:
    def test_simple_winner(self):
        speaker_id, name, score = aggregate_speaker_votes(
            [_hit(0.9, "label:a" + "0" * 23, name="Alice"), _hit(0.6, "label:b" + "0" * 23, name="Bob")],
            min_similarity=0.5,
        )
        assert name == "Alice"
        assert score == pytest.approx(0.9)

    def test_votes_are_summed_per_speaker_id(self):
        hits = [
            _hit(0.6, "label:a", name="Alice"),
            _hit(0.55, "label:a", name="Alice"),
            _hit(0.9, "label:b", name="Bob"),
        ]
        speaker_id, name, score = aggregate_speaker_votes(hits, min_similarity=0.5)
        assert speaker_id == "label:a"
        assert score == pytest.approx(1.15)

    def test_min_similarity_filters_hits(self):
        hits = [_hit(0.4, "label:a", name="Alice"), _hit(0.6, "label:b", name="Bob")]
        speaker_id, name, score = aggregate_speaker_votes(hits, min_similarity=0.5)
        assert name == "Bob"

    def test_no_match(self):
        assert aggregate_speaker_votes([_hit(0.3, "label:a", name="Alice")], min_similarity=0.5) == (None, None, None)

    def test_allowed_speakers_restriction(self):
        hits = [_hit(0.9, "label:a", name="Alice"), _hit(0.6, "label:b", name="Bob")]
        speaker_id, name, score = aggregate_speaker_votes(
            hits, min_similarity=0.5, allowed_speaker_ids={"label:b"}
        )
        assert name == "Bob"

    def test_exclude_speakers(self):
        hits = [_hit(0.9, "label:a", name="Alice"), _hit(0.6, "label:b", name="Bob")]
        speaker_id, name, score = aggregate_speaker_votes(
            hits, min_similarity=0.5, exclude_speaker_ids=["label:a"]
        )
        assert name == "Bob"

    def test_namesakes_across_collections_are_distinct(self):
        # Two "Alice" with different speaker_ids must not be merged
        hits = [
            _hit(0.6, "label:a1", name="Alice"),
            _hit(0.55, "label:a2", name="Alice"),
            _hit(0.9, "label:b", name="Bob"),
        ]
        speaker_id, name, score = aggregate_speaker_votes(hits, min_similarity=0.5)
        assert speaker_id == "label:b"
        assert name == "Bob"

    def test_winner_name_from_best_hit(self):
        # Name is taken from the highest-scored hit of the winning speaker_id
        hits = [
            _hit(0.6, "label:a", name="Alice (old)"),
            _hit(0.8, "label:a", name="Alice"),
        ]
        speaker_id, name, score = aggregate_speaker_votes(hits, min_similarity=0.5)
        assert name == "Alice"
        assert score == pytest.approx(1.4)

    def test_legacy_person_payload(self):
        hits = [(0.8, {"speaker_id": "label:a", "person": "Alice"})]
        speaker_id, name, score = aggregate_speaker_votes(hits, min_similarity=0.5)
        assert name == "Alice"

    def test_hits_without_speaker_id_are_ignored(self):
        hits = [(0.9, {"name": "Anonymous"}), _hit(0.6, "label:b", name="Bob")]
        speaker_id, name, score = aggregate_speaker_votes(hits, min_similarity=0.5)
        assert name == "Bob"

    def test_empty_hits(self):
        assert aggregate_speaker_votes([], min_similarity=0.5) == (None, None, None)
