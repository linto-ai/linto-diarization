"""
End-to-end integration test of the multi-collection speaker identification
core, on real speaker data, using an in-memory Qdrant (no Qdrant server).

It exercises the real worker code path: EmbeddingBackend (ECAPA),
build_point_payload / speaker_point_id (as in speaker_upsert_task), and
SpeakerIdentifier.speaker_identify_given_diarization with a multi-collection
dict spec.

Heavy (downloads the ECAPA model, needs torch/speechbrain) and data-dependent,
so it is SKIPPED unless a LibriSpeech-style corpus is provided:

    LIBRISPEECH_PATH=/path/to/LibriSpeech/dev-clean \
        python -m pytest test/test_identification_integration.py -v -s

Get the corpus (open, direct download):
    https://www.openslr.org/resources/12/dev-clean.tar.gz

Layout expected: <LIBRISPEECH_PATH>/<speaker_id>/<chapter>/<utt>.flac
"""

import glob
import os

import pytest

DATA = os.environ.get("LIBRISPEECH_PATH")

pytestmark = pytest.mark.skipif(
    not DATA or not os.path.isdir(DATA),
    reason="set LIBRISPEECH_PATH to a LibriSpeech dev-clean directory to run",
)

ORG_ID = "a1a1a1a1a1a1a1a1a1a1a1a1"
COLL_ID = "b2b2b2b2b2b2b2b2b2b2b2b2"
COLLECTION = f"spkid_{ORG_ID}_{COLL_ID}"
SR = 16000
N_ENROLLED = 8
N_ENROLL_UTT = 3
N_TEST_UTT = 3
N_IMPOSTORS = 3
THRESHOLD = 0.5  # production default


def _utterances(spk):
    return sorted(glob.glob(os.path.join(DATA, spk, "*", "*.flac")))


@pytest.fixture(scope="module")
def identifier():
    os.environ["QDRANT_HOST"] = "inmemory"
    from qdrant_client import QdrantClient
    from identification.qdrant_store import QdrantStore
    from identification.speaker_identify import SpeakerIdentifier

    si = SpeakerIdentifier(device="cpu")
    store = QdrantStore(host="inmemory")
    store.client = QdrantClient(":memory:")
    si.store = store
    si.embedding.load()
    return si


@pytest.fixture(scope="module")
def enrolled(identifier):
    import random
    from identification import spkid_core

    random.seed(1234)
    speakers = sorted(os.listdir(DATA))
    random.shuffle(speakers)
    enrolled = speakers[:N_ENROLLED]
    impostors = speakers[N_ENROLLED : N_ENROLLED + N_IMPOSTORS]

    identifier.store.ensure_collection(COLLECTION, dim=spkid_core.MODEL_DIM)
    for idx, spk in enumerate(enrolled):
        files = _utterances(spk)[:N_ENROLL_UTT]
        audio, _, _ = identifier.embedding.load_audio_concat(files, max_duration=180)
        vector = identifier.embedding.compute_embedding(audio)[0].flatten().cpu().tolist()
        ref = f"label:{idx:024d}"
        payload = spkid_core.build_point_payload(COLLECTION, ref, spk, spkid_core.MODEL_ID)
        identifier.store.upsert_point(COLLECTION, spkid_core.speaker_point_id(ref), vector, payload)
    return {"enrolled": enrolled, "impostors": impostors}


def _identify(identifier, audio_file):
    import torchaudio

    info = torchaudio.info(audio_file)
    duration = info.num_frames / info.sample_rate
    diar = {"segments": [{"seg_begin": 0.0, "seg_end": float(duration), "spk_id": "spk1"}]}
    spec = {"collections": [COLLECTION], "speakers": "*", "minSimilarity": THRESHOLD}
    result = identifier.speaker_identify_given_diarization(audio_file, diar, spec)
    return result["segments"][0]["spk_id"]


def test_identification_accuracy(identifier, enrolled):
    correct = total = 0
    for spk in enrolled["enrolled"]:
        for f in _utterances(spk)[N_ENROLL_UTT : N_ENROLL_UTT + N_TEST_UTT]:
            if _identify(identifier, f) == spk:
                correct += 1
            total += 1
    acc = correct / total
    print(f"\naccuracy: {correct}/{total} = {acc:.1%}")
    assert acc >= 0.9


def test_two_speaker_segment_routing(identifier, enrolled, tmp_path):
    import torch
    import torchaudio

    a, b = enrolled["enrolled"][0], enrolled["enrolled"][1]
    wa, _ = torchaudio.load(_utterances(a)[N_ENROLL_UTT])
    wb, _ = torchaudio.load(_utterances(b)[N_ENROLL_UTT])
    boundary = wa.shape[1] / SR
    total = (wa.shape[1] + wb.shape[1]) / SR
    out = str(tmp_path / "two.wav")
    torchaudio.save(out, torch.cat([wa, wb], dim=1), SR)

    spec = {"collections": [COLLECTION], "speakers": "*", "minSimilarity": THRESHOLD}
    diar = {"segments": [
        {"seg_begin": 0.0, "seg_end": boundary, "spk_id": "spk1"},
        {"seg_begin": boundary, "seg_end": total, "spk_id": "spk2"},
    ]}
    res = identifier.speaker_identify_given_diarization(out, diar, spec)
    assert res["segments"][0]["spk_id"] == a
    assert res["segments"][1]["spk_id"] == b


def test_impostor_rejection(identifier, enrolled):
    rejected = total = 0
    for spk in enrolled["impostors"]:
        for f in _utterances(spk)[:N_TEST_UTT]:
            pred = _identify(identifier, f)
            # an un-enrolled speaker must keep a raw diarization tag (spkN)
            if pred.startswith("spk"):
                rejected += 1
            total += 1
    rate = rejected / total
    print(f"\nimpostor rejection: {rejected}/{total} = {rate:.1%}")
    assert rate >= 0.9
