"""Integration tests for identification/qdrant_store.py

These tests require a reachable Qdrant instance and are skipped otherwise.
Configure with QDRANT_TEST_HOST/QDRANT_TEST_PORT (default localhost:6333).

Run with: pytest test/test_qdrant_store.py
"""

import os
import random
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

qdrant_client = pytest.importorskip("qdrant_client")

from identification.qdrant_store import QdrantStore
from identification.spkid_core import (
    MODEL_DIM,
    MODEL_ID,
    build_point_payload,
    speaker_point_id,
)

TEST_HOST = os.environ.get("QDRANT_TEST_HOST", "localhost")
TEST_PORT = os.environ.get("QDRANT_TEST_PORT", "6333")
TEST_COLLECTION = "spkid_" + "deadbeef" * 3 + "_" + "cafebabe" * 3


def _qdrant_reachable():
    try:
        client = qdrant_client.QdrantClient(url=f"http://{TEST_HOST}:{TEST_PORT}", timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_reachable(), reason=f"No Qdrant instance reachable at {TEST_HOST}:{TEST_PORT}"
)


@pytest.fixture
def store():
    store = QdrantStore(host=TEST_HOST, port=TEST_PORT)
    yield store
    store.drop(TEST_COLLECTION)


def _random_vector():
    return [random.uniform(-1, 1) for _ in range(MODEL_DIM)]


def test_store_disabled_without_host():
    assert QdrantStore(host=None).enabled is False


def test_ensure_collection_is_idempotent(store):
    assert store.ensure_collection(TEST_COLLECTION) is True
    assert store.ensure_collection(TEST_COLLECTION) is False
    assert store.collection_exists(TEST_COLLECTION) is True


def test_upsert_search_roundtrip(store):
    store.ensure_collection(TEST_COLLECTION)
    speaker_id = "label:65cc00000000000000000004"
    point_id = speaker_point_id(speaker_id)
    vector = _random_vector()
    payload = build_point_payload(TEST_COLLECTION, speaker_id, "Griogy", MODEL_ID)
    store.upsert_point(TEST_COLLECTION, point_id, vector, payload)

    results = store.search(TEST_COLLECTION, vector, limit=5)
    assert len(results) == 1
    assert results[0].payload["speaker_id"] == speaker_id
    assert results[0].payload["name"] == "Griogy"
    assert results[0].score == pytest.approx(1.0, abs=1e-3)

    assert store.get_collection_model_id(TEST_COLLECTION) == MODEL_ID

    # Upsert with the same speaker_id replaces the point (deterministic id)
    store.upsert_point(TEST_COLLECTION, point_id, vector, payload | {"name": "Griogy 2"})
    results = store.search(TEST_COLLECTION, vector, limit=5)
    assert len(results) == 1
    assert results[0].payload["name"] == "Griogy 2"


def test_delete_points_is_idempotent(store):
    store.ensure_collection(TEST_COLLECTION)
    speaker_id = "label:65cc00000000000000000004"
    point_id = speaker_point_id(speaker_id)
    store.upsert_point(
        TEST_COLLECTION, point_id, _random_vector(),
        build_point_payload(TEST_COLLECTION, speaker_id, "Griogy", MODEL_ID),
    )

    assert store.delete_points(TEST_COLLECTION, [point_id]) == 1
    assert store.delete_points(TEST_COLLECTION, [point_id]) == 0
    assert store.delete_points("spkid_" + "0" * 24 + "_" + "0" * 24, [point_id]) == 0


def test_drop_is_idempotent(store):
    store.ensure_collection(TEST_COLLECTION)
    assert store.drop(TEST_COLLECTION) is True
    assert store.drop(TEST_COLLECTION) is False


def test_get_collection_model_id_empty(store):
    store.ensure_collection(TEST_COLLECTION)
    assert store.get_collection_model_id(TEST_COLLECTION) is None
