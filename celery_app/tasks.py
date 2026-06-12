import os

from celery_app.celeryapp import celery
from diarization.processing import diarizationworker
from diarization import logger

from identification.spkid_core import (
    MODEL_DIM,
    MODEL_ID,
    build_point_payload,
    parse_collection_name,
    speaker_point_id,
)

AUDIO_FOLDER = "/opt/audio"


def _get_speaker_identifier():
    """Return the worker speaker identifier, or raise if speaker identification is disabled"""
    speaker_identifier = diarizationworker.speaker_identifier
    if not speaker_identifier.is_speaker_identification_enabled():
        raise RuntimeError("Speaker identification is disabled (QDRANT_HOST is not set)")
    return speaker_identifier


@celery.task(name="diarization_task")
def diarization_task(
    file: str,
    speaker_count: int = None,
    max_speaker: int = None,
    speaker_names = None,
):
    """transcribe_task do a synchronous call to the transcribe worker API

    speaker_names can be:
    - a string (legacy speaker identification: "*", "name1|name2|...", JSON list of names)
    - a dict (multi-collection speaker identification):
      {"collections": [...], "speakers": "*" | [speaker_ids], "minSimilarity": float | None}
    """
    logger.info(f"Received transcription task for {file} ({speaker_count=}, {max_speaker=})")

    file_path = os.path.join(AUDIO_FOLDER, file)
    if not os.path.isfile(file_path):
        raise Exception("Could not find ressource {}".format(file_path))

    # Check parameters
    speaker_count = None if speaker_count == 0 else speaker_count
    max_speaker = None if max_speaker == 0 else max_speaker

    if speaker_count and max_speaker:
        max_speaker = None

    # Processing
    try:
        result = diarizationworker.run(
            file_path,
            speaker_count=speaker_count,
            max_speaker=max_speaker,
            speaker_names=speaker_names,
        )
    except Exception as e:
        import traceback
        msg = f"{traceback.format_exc()}\nFailed to decode {file_path}"
        logger.error(msg)
        raise Exception(msg)  # from err

    return result


@celery.task(name="voiceprint_compute_task")
def voiceprint_compute_task(audio_files: list):
    """Compute a speaker voiceprint (embedding) from audio files

    Args:
        audio_files (list): paths relative to /opt/audio (shared volume)

    Returns:
        {"vector": [floats], "model_id": str, "dim": int,
         "duration_used": float, "files_used": int}
    """
    logger.info(f"Received voiceprint compute task for {len(audio_files)} audio file(s)")

    speaker_identifier = _get_speaker_identifier()

    max_duration = float(os.environ.get("SPEAKER_ID_MAX_ENROLL_DURATION", 180))
    min_duration = float(os.environ.get("SPEAKER_ID_MIN_ENROLL_DURATION", 3))

    file_paths = []
    for file in audio_files:
        file_path = os.path.join(AUDIO_FOLDER, file)
        if os.path.isfile(file_path):
            file_paths.append(file_path)
        else:
            logger.warning(f"Could not find ressource {file_path}")
    if not file_paths:
        raise ValueError("no_valid_audio: none of the provided audio files were found")

    speaker_identifier.embedding.load()
    audio, duration_used, files_used = speaker_identifier.embedding.load_audio_concat(
        file_paths, max_duration=max_duration, on_error="skip"
    )
    if audio is None:
        raise ValueError("no_valid_audio: none of the provided audio files could be loaded")
    if duration_used < min_duration:
        raise ValueError(
            f"audio_too_short: {duration_used:.2f}s of audio (minimum is {min_duration:.0f}s)"
        )

    embedding = speaker_identifier.embedding.compute_embedding(audio)
    vector = embedding[0].flatten().cpu().tolist()

    logger.info(
        f"Voiceprint computed on {duration_used:.2f}s of audio from {files_used} file(s)"
    )
    return {
        "vector": vector,
        "model_id": MODEL_ID,
        "dim": MODEL_DIM,
        "duration_used": round(duration_used, 3),
        "files_used": files_used,
    }


@celery.task(name="speaker_upsert_task")
def speaker_upsert_task(collection: str, speaker_id: str, name: str, vector: list, model_id: str):
    """Upsert a speaker voiceprint into a Qdrant collection (created if absent)

    Returns:
        {"status": "ok", "point_id": "<uuid5>", "created_collection": bool}
    """
    logger.info(f"Received speaker upsert task for collection {collection} ({speaker_id=})")

    speaker_identifier = _get_speaker_identifier()

    # Validates the collection name (spkid_{orgId}_{collectionId})
    parse_collection_name(collection)

    if model_id != MODEL_ID:
        raise ValueError(f"model_mismatch: expected '{MODEL_ID}', got '{model_id}'")
    if not isinstance(vector, list) or len(vector) != MODEL_DIM:
        raise ValueError(
            f"invalid_vector: expected a list of {MODEL_DIM} floats, got "
            f"{len(vector) if isinstance(vector, list) else type(vector)}"
        )

    created_collection = speaker_identifier.store.ensure_collection(collection, dim=MODEL_DIM)
    point_id = speaker_point_id(speaker_id)
    payload = build_point_payload(collection, speaker_id, name, MODEL_ID)
    speaker_identifier.store.upsert_point(collection, point_id, vector, payload)

    logger.info(f"Speaker {speaker_id} upserted in collection {collection}")
    return {"status": "ok", "point_id": point_id, "created_collection": created_collection}


@celery.task(name="speaker_delete_task")
def speaker_delete_task(collection: str, speaker_ids: list):
    """Delete speakers from a Qdrant collection (idempotent)

    Returns:
        {"status": "ok", "deleted": int}
    """
    logger.info(f"Received speaker delete task for collection {collection} ({len(speaker_ids)} speaker(s))")

    speaker_identifier = _get_speaker_identifier()

    point_ids = [speaker_point_id(speaker_id) for speaker_id in speaker_ids]
    deleted = speaker_identifier.store.delete_points(collection, point_ids)

    logger.info(f"Deleted {deleted} speaker(s) from collection {collection}")
    return {"status": "ok", "deleted": deleted}


@celery.task(name="collection_drop_task")
def collection_drop_task(collection: str):
    """Drop a Qdrant collection (idempotent)

    Returns:
        {"status": "ok", "existed": bool}
    """
    logger.info(f"Received collection drop task for collection {collection}")

    speaker_identifier = _get_speaker_identifier()

    existed = speaker_identifier.store.drop(collection)

    logger.info(f"Collection {collection} dropped (existed={existed})")
    return {"status": "ok", "existed": existed}
