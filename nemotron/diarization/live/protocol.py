"""Live diarization websocket protocol (v1): message validation and construction.

Client -> server: a JSON "start" message, then binary PCM (s16le, 16 kHz, mono), then {"type": "stop"}.
Server -> client: "ready", then "turns" and "identity" as audio is processed, "saturated" once, "end".
Times are in milliseconds of audio received since the start of the session.
"""
import re

from identification.spkid_core import check_speaker_spec_dict, parse_collection_name

SAMPLE_RATE = 16000
ORGANIZATION_ID = re.compile(r"^[0-9a-f]{24}$")

# Close codes
CLOSE_INVALID = 1008  # invalid start message
CLOSE_FULL = 1013  # no capacity left on this pod: retry (the load balancer picks another pod)


class ProtocolError(ValueError):
    pass


def parse_start(message):
    """Validate a start message; return {"session", "identification"} (identification may be None)."""
    if not isinstance(message, dict) or message.get("type") != "start":
        raise ProtocolError('the first message must be {"type": "start", ...}')
    sample_rate = message.get("sample_rate", SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        raise ProtocolError(f"sample_rate must be {SAMPLE_RATE}, got {sample_rate}")
    encoding = message.get("encoding", "pcm_s16le")
    if encoding != "pcm_s16le":
        raise ProtocolError(f"encoding must be pcm_s16le, got {encoding}")
    session = message.get("session")
    if session is not None and not isinstance(session, str):
        raise ProtocolError("session must be a string")
    identification = message.get("identification")
    if identification is not None:
        identification = parse_identification(identification)
    return {"session": session, "identification": identification}


def parse_identification(spec):
    """{"organizationId", "collections", "speakers"?, "minSimilarity"?}: every collection must belong
    to organizationId (spkid_{organizationId}_...)."""
    if not isinstance(spec, dict):
        raise ProtocolError("identification must be an object")
    spec = dict(spec)
    organization_id = spec.pop("organizationId", None)
    if not isinstance(organization_id, str) or not ORGANIZATION_ID.match(organization_id):
        raise ProtocolError("identification.organizationId must be 24 hexadecimal characters")
    try:
        spec = check_speaker_spec_dict(spec)
        for collection in spec["collections"]:
            if parse_collection_name(collection)[0] != organization_id:
                raise ProtocolError(f"collection {collection} does not belong to organization {organization_id}")
    except ProtocolError:
        raise
    except ValueError as err:
        raise ProtocolError(str(err)) from err
    spec["organizationId"] = organization_id
    return spec


def speaker_label(index):
    return f"S{index + 1}"


def ready(latency_ms, max_speakers, frame_ms):
    return {"type": "ready", "latency_ms": latency_ms, "max_speakers": max_speakers, "frame_ms": frame_ms}


def turns(until_ms, runs):
    """runs: list of (speaker_index, start_ms, end_ms) found in the audio processed up to until_ms."""
    return {
        "type": "turns",
        "until_ms": until_ms,
        "turns": [{"speaker": speaker_label(i), "start_ms": s, "end_ms": e} for i, s, e in runs],
    }


def identity(label, status, speaker_id=None, name=None, score=None):
    message = {"type": "identity", "speaker": label, "status": status}
    if status != "revoked":
        message.update(speaker_id=speaker_id, name=name, score=round(float(score), 3))
    return message


def saturated(max_speakers):
    return {"type": "saturated", "max_speakers": max_speakers}


def error(code, message):
    return {"type": "error", "code": code, "message": message}


def end(until_ms):
    return {"type": "end", "until_ms": until_ms}
