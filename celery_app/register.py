"""The register Module allow registering and unregistering operations within the service stack for service discovery purposes"""
import json
import os
import sys
import uuid
from socket import gethostname
from time import time
from xmlrpc.client import ResponseError

import redis

from identification.spkid_core import MODEL_DIM, MODEL_ID
from redis.commands.json.path import Path
from redis.commands.search.field import NumericField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

SERVICE_DISCOVERY_DB = 0
SERVICE_TYPE = "diarization"

service_name = os.environ.get("SERVICE_NAME", SERVICE_TYPE)
service_lang = os.environ.get("LANGUAGE", "*")
host_name = gethostname()


def register(is_heartbeat: bool = False) -> bool:
    """Registers the service and act as heartbeat.

    Returns:
        bool: registering status
    """
    host, port = os.environ.get("SERVICES_BROKER").split("//")[1].split(":")
    password = os.environ.get("BROKER_PASS", None)
    r = redis.Redis(
        host=host, port=int(port), db=SERVICE_DISCOVERY_DB, password=password
    )

    res = r.json()
    res = res.set(f"service:{host_name}", Path.root_path(), service_info())
    if is_heartbeat:
        return res
    else:
        print(f"Service registered as service:{host_name}")
    schema = (
        TextField("$.service_name", as_name="service_name"),
        TextField("$.service_type", as_name="service_type"),
        TextField("$.service_language", as_name="service_language"),
        TextField("$.queue_name", as_name="queue_name"),
        TextField("$.version", as_name="version"),
        TextField("$.info", as_name="info"),
        NumericField("$.last_alive", as_name="last_alive"),
        NumericField("$.concurrency", as_name="concurrency"),
    )
    try:
        r.ft().create_index(
            schema,
            definition=IndexDefinition(prefix=["service:"], index_type=IndexType.JSON),
        )
    except Exception as error:
        pass
    return res


def unregister() -> None:
    """Un-register the service"""
    try:
        password = os.environ.get("BROKER_PASS", None)
        host, port = os.environ.get("SERVICES_BROKER").split("//")[1].split(":")
        r = redis.Redis(
            host=host, port=int(port), db=SERVICE_DISCOVERY_DB, password=password
        )
        r.json().delete(f"service:{host_name}")
    except Exception as error:
        print(f"Failed to unregister: {repr(error)}")


def queue() -> str:
    return os.environ.get("QUEUE_NAME", service_name)


def service_info() -> dict:
    # The info field keeps its historical role (the MODEL_INFO locale label
    # shown in the UI, e.g. {"en": "Yes", "fr": "Oui"}) and is enriched with the
    # speaker identification capability so service discovery can route to it.
    # Merge (do not replace) so the diarization option label is preserved.
    info_obj = {}
    model_info_raw = os.environ.get("MODEL_INFO")
    if model_info_raw:
        try:
            parsed = json.loads(model_info_raw)
            if isinstance(parsed, dict):
                info_obj = parsed
        except (ValueError, TypeError):
            pass
    # Speaker identification is enabled iff Qdrant is configured
    # (docker-entrypoint.sh wait_for_qdrant guarantees reachability at boot)
    info_obj.update(
        {
            "speaker_identification": bool(os.environ.get("QDRANT_HOST")),
            "model_id": MODEL_ID,
            "dim": MODEL_DIM,
        }
    )
    info = json.dumps(info_obj)
    return {
        "service_name": service_name,
        "host_name": host_name,
        "service_type": SERVICE_TYPE,
        "service_language": service_lang,
        "queue_name": queue(),
        "version": os.environ.get("VERSION", "unknown"),
        "info": info,
        "last_alive": int(time()),
        "concurrency": int(os.environ.get("CONCURRENCY", 2)),
    }


if __name__ == "__main__":
    sys.exit(register())
