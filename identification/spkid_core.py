"""
Pure helpers and constants for speaker identification.

This module MUST stay importable without torch/speechbrain/qdrant-client
(stdlib only): it is used by celery_app/register.py at service registration
time and by the unit test suite.
"""

import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone

# Embedding model identity. MODEL_ID also tags every Qdrant point (model_id
# payload) so that two builds of the worker can never silently mix incompatible
# embeddings. MODEL_REVISION optionally pins a HuggingFace revision of the model.
MODEL_ID = "pyannote/embedding"
MODEL_REVISION = os.environ.get("SPEAKER_ID_MODEL_REVISION") or None
MODEL_DIM = 512

# Fixed namespace used to derive Qdrant point IDs from speaker ids
# (point_id = uuid5(NAMESPACE_SPKID, speaker_id)). DO NOT change this value:
# it conditions every point ID ever written.
NAMESPACE_SPKID = uuid.UUID("b8a9cf6e-0000-4000-8000-5370656b4944")

# Qdrant collection naming convention: spkid_{organizationId}_{studioCollectionId}
QDRANT_COLLECTION_PATTERN = re.compile(r"^spkid_([0-9a-f]{24})_([0-9a-f]{24})$")

DEFAULT_MIN_SIMILARITY = 0.5


def speaker_point_id(speaker_id):
    """Derive the (deterministic) Qdrant point ID for a speaker id ("label:..." or "user:...")"""
    return str(uuid.uuid5(NAMESPACE_SPKID, speaker_id))


def parse_collection_name(collection):
    """Extract (organization_id, studio_collection_id) from a Qdrant collection name

    Raises ValueError if the collection name does not follow the
    spkid_{orgId}_{collectionId} convention.
    """
    match = QDRANT_COLLECTION_PATTERN.match(collection or "")
    if not match:
        raise ValueError(
            f"Invalid speaker identification collection name: '{collection}' "
            "(expected spkid_{24 hex}_{24 hex})"
        )
    return match.group(1), match.group(2)


def payload_name(payload):
    """Get the display name from a Qdrant point payload

    Supports both the current payload format ("name") and the legacy
    filesystem-bootstrap format ("person").
    """
    return payload.get("name") or payload.get("person")


def build_point_payload(collection, speaker_id, name, model_id, updated_at=None):
    """Build the full Qdrant point payload for a speaker"""
    organization_id, studio_collection_id = parse_collection_name(collection)
    return {
        "speaker_id": speaker_id,
        "name": name,
        "organization_id": organization_id,
        "studio_collection_id": studio_collection_id,
        "model_id": model_id,
        "updated_at": updated_at or datetime.now(timezone.utc).isoformat(),
    }


def check_speaker_spec_dict(spec):
    """Validate and normalize a dict speaker identification specification

    Expected format:
        {"collections": ["spkid_..._...", ...],     # required, 1..N
         "speakers": "*" | ["label:...", ...],      # optional, default "*"
         "minSimilarity": float | None}             # optional

    Returns the normalized dict. Raises ValueError on invalid specification.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported speaker specification of type {type(spec)}: {spec}")

    unknown_keys = set(spec.keys()) - {"collections", "speakers", "minSimilarity"}
    if unknown_keys:
        raise ValueError(f"Unsupported keys in speaker specification: {sorted(unknown_keys)}")

    collections = spec.get("collections")
    if not isinstance(collections, list) or not collections:
        raise ValueError("Speaker specification must provide a non-empty 'collections' list")
    for collection in collections:
        if not isinstance(collection, str):
            raise ValueError(f"Invalid collection name of type {type(collection)}: {collection}")
        # Defense in depth: the transcription service already validated this
        parse_collection_name(collection)

    speakers = spec.get("speakers", "*")
    if speakers is None:
        speakers = "*"
    if speakers != "*":
        if not isinstance(speakers, list) or not all(isinstance(s, str) and s for s in speakers):
            raise ValueError(
                f"'speakers' must be \"*\" or a list of speaker ids, got: {speakers}"
            )

    min_similarity = spec.get("minSimilarity")
    if min_similarity is not None:
        if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)):
            raise ValueError(f"'minSimilarity' must be a number, got: {min_similarity}")
        min_similarity = float(min_similarity)
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError(f"'minSimilarity' must be in [0, 1], got: {min_similarity}")

    return {
        "collections": list(collections),
        "speakers": speakers if speakers == "*" else list(speakers),
        "minSimilarity": min_similarity,
    }


def resolve_min_similarity(spec_value=None, env=None):
    """Resolve the similarity threshold: spec value > SPEAKER_ID_MIN_SIMILARITY env > 0.5"""
    if spec_value is not None:
        return float(spec_value)
    env = os.environ if env is None else env
    env_value = env.get("SPEAKER_ID_MIN_SIMILARITY")
    if env_value:
        return float(env_value)
    return DEFAULT_MIN_SIMILARITY


def resolve_collections(collections, exists_fn, model_id_fn, expected_model_id, log=None):
    """Filter the requested collections down to the usable ones

    A collection is skipped (with a warning) when it does not exist or when the
    model_id of its points does not match the worker model. A collection with no
    readable model_id (e.g. empty) is kept: it cannot yield mismatched hits.
    """
    usable = []
    for collection in collections:
        if not exists_fn(collection):
            if log:
                log.warning(f"Speaker identification: collection '{collection}' does not exist, skipping it")
            continue
        model_id = model_id_fn(collection)
        if model_id is not None and model_id != expected_model_id:
            if log:
                log.warning(
                    f"Speaker identification: collection '{collection}' was built with model "
                    f"'{model_id}' (worker model is '{expected_model_id}'), skipping it"
                )
            continue
        usable.append(collection)
    return usable


def aggregate_speaker_votes(hits, min_similarity, allowed_speaker_ids=None, exclude_speaker_ids=None):
    """Aggregate search hits into a winning speaker

    Args:
        hits: iterable of (score, payload) tuples (merged from all collections)
        min_similarity (float): minimum similarity for a hit to count
        allowed_speaker_ids: optional restriction (set/list of payload speaker_id)
        exclude_speaker_ids: optional exclusion (already identified speakers)

    Votes are summed per payload speaker_id (two namesakes across collections
    remain two distinct candidates). The winner name is taken from its best hit.

    Returns:
        (speaker_id, name, score) of the winner, or (None, None, None)
    """
    votes = defaultdict(float)
    best_hit = {}
    for score, payload in hits:
        speaker_id = payload.get("speaker_id")
        if not speaker_id:
            continue
        if score < min_similarity:
            continue
        if allowed_speaker_ids is not None and speaker_id not in allowed_speaker_ids:
            continue
        if exclude_speaker_ids and speaker_id in exclude_speaker_ids:
            continue
        votes[speaker_id] += score
        if speaker_id not in best_hit or score > best_hit[speaker_id][0]:
            best_hit[speaker_id] = (score, payload_name(payload))

    if not votes:
        return None, None, None
    winner = max(votes, key=votes.get)
    return winner, best_hit[winner][1], votes[winner]
