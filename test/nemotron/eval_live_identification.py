"""Evaluation of live speaker identification on SUMM-RE (not a pytest test: needs the dataset, a GPU
worker serving the live websocket and a Qdrant instance). See nemotron/LIVE.md for the results.

    python test/nemotron/eval_live_identification.py --data DIR --url ws://localhost:18090 \
        --qdrant localhost:16334 --voiceprint 10 --threshold 0.66

DIR holds wav/<meeting>.wav (16 kHz mono) and ref/<meeting>.rttm. For each meeting a collection gets
one voiceprint per participant of the corpus, taken from ANOTHER meeting than the one streamed
(people seen in a single meeting are therefore not enrolled for it and must stay unknown). Every
meeting is streamed through the live websocket with identification; the identity messages are
compared with the reference.
"""
import argparse
import asyncio
import collections
import glob
import json
import os
import sys
import uuid

import numpy as np
import soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "nemotron", "tools")]

import live_client  # noqa: E402

ORG = "e" * 24


def rttm(path):
    return [(float(p[3]), float(p[3]) + float(p[4]), p[7]) for p in (l.split() for l in open(path)) if p and p[0] == "SPEAKER"]


def solo_turns(ref, person):
    for s, e, k in ref:
        if k == person and e - s > 1.0 and all(not (x < e and y > s) for x, y, o in ref if o != person):
            yield s, e


def voiceprint_audio(audio, sr, ref, person, seconds):
    clips, total = [], 0.0
    for s, e in solo_turns(ref, person):
        take = min(e - s, seconds - total)
        clips.append(audio[int(s * sr):int((s + take) * sr)]); total += take
        if total >= seconds:
            break
    return np.concatenate(clips) if total >= min(seconds, 4) else None


def enroll(data, meetings, seconds, qdrant):
    """One collection per meeting: every participant, voiceprint from another meeting."""
    import torch
    from qdrant_client import QdrantClient, models

    from identification.embedding import EmbeddingBackend
    from identification.spkid_core import MODEL_DIM, MODEL_ID, build_point_payload, speaker_point_id

    host, port = qdrant.split(":")
    client = QdrantClient(host=host, port=int(port))
    emb = EmbeddingBackend(device="cuda"); emb.load()
    refs = {m: rttm(f"{data}/ref/{m}.rttm") for m in meetings}
    where = collections.defaultdict(list)
    for m, ref in refs.items():
        for k in {k for _, _, k in ref}:
            where[k].append(m)
    cache = {}

    def vp(person, m):
        if (person, m) not in cache:
            audio, sr = sf.read(f"{data}/wav/{m}.wav", dtype="float32")
            clip = voiceprint_audio(audio, sr, refs[m], person, seconds)
            cache[person, m] = None if clip is None else emb.compute_embedding(torch.from_numpy(clip).unsqueeze(0))[0].flatten().cpu().tolist()
        return cache[person, m]

    solo = lambda m, p: sum(e - s for s, e in solo_turns(refs[m], p))
    colls, enrolled = {}, {}
    for m in meetings:
        coll = f"spkid_{ORG}_{uuid.uuid5(uuid.NAMESPACE_URL, m).hex[:24]}"
        if client.collection_exists(coll):
            client.delete_collection(coll)
        client.create_collection(coll, vectors_config=models.VectorParams(size=MODEL_DIM, distance=models.Distance.COSINE))
        points, names = [], set()
        for person, ms in where.items():
            others = [x for x in ms if x != m]
            if not others:
                continue
            v = vp(person, max(others, key=lambda x: solo(x, person)))
            if v is None:
                continue
            sid = "label:" + uuid.uuid5(uuid.NAMESPACE_URL, person).hex[:24]
            points.append(models.PointStruct(id=speaker_point_id(sid), vector=v,
                                             payload=build_point_payload(coll, sid, person, MODEL_ID)))
            names.add(person)
        client.upsert(coll, points)
        colls[m], enrolled[m] = coll, names
    return colls, enrolled, refs


def label_truth(merged, ref):
    """Live label -> reference speaker with the most overlapping speech, and the speech duration."""
    out = {}
    for label, runs in merged.items():
        over = collections.Counter()
        for s, e in runs:
            for x, y, k in ref:
                o = min(e / 1000, y) - max(s / 1000, x)
                if o > 0:
                    over[k] += o
        if over:
            out[label] = (over.most_common(1)[0][0], sum(e - s for s, e in runs) / 1000)
    return out


def score(result, ref, enrolled):
    truth = label_truth(result["merged"], ref)
    final, first_name, transient_wrong, revocations = {}, {}, 0, 0
    wrong_at = collections.Counter()  # wrong names shown, by seconds of speech used
    for i in result["identities"]:
        label = i["speaker"]
        if i["status"] == "revoked":
            revocations += 1
            final.pop(label, None)
            continue
        final[label] = i
        t = truth.get(label, (None,))[0]
        if i["name"] != t:
            transient_wrong += 1
            wrong_at[round(i["speech_s"] / 5) * 5] += 1
        first_name.setdefault((label, i["status"]), i["until_ms"] / 1000)
    counts = collections.Counter()
    delays = collections.defaultdict(list)
    for label, (person, speech) in truth.items():
        if speech < 5:
            continue
        got = final.get(label)
        if got is None:
            counts["missed" if person in enrolled else "unknown_ok"] += 1
        elif got["name"] == person:
            counts["correct"] += 1
            for status in ("provisional", "confirmed"):
                if (label, status) in first_name:
                    delays[status].append(first_name[label, status])
        else:
            counts["wrong_unenrolled" if person not in enrolled else "wrong_swap"] += 1
    counts["transient_wrong"] = transient_wrong
    for k, v in wrong_at.items():
        counts[f"wrong_shown_at_{k}s"] = v
    counts["revocations"] = revocations
    return counts, delays


async def stream_all(url, data, meetings, colls, threshold, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def one(m):
        async with sem:
            spec = {"organizationId": ORG, "collections": [colls[m]], "minSimilarity": threshold}
            return m, await live_client.stream(url, f"{data}/wav/{m}.wav", speed=0, identification=spec)
    return dict(await asyncio.gather(*(one(m) for m in meetings)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--qdrant", required=True)
    parser.add_argument("--voiceprint", type=float, default=10)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.66])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--meetings", type=int, default=0, help="limit to the first N meetings")
    args = parser.parse_args()
    meetings = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{args.data}/wav/*_merged.wav"))
    if args.meetings:
        meetings = meetings[:args.meetings]
    colls, enrolled, refs = enroll(args.data, meetings, args.voiceprint, args.qdrant)
    print(f"{len(meetings)} meetings, voiceprints of {args.voiceprint:g} s from other meetings")
    for threshold in args.threshold:
        results = asyncio.run(stream_all(args.url, args.data, meetings, colls, threshold, args.concurrency))
        total, delays = collections.Counter(), collections.defaultdict(list)
        for m, r in results.items():
            c, d = score(r, refs[m], enrolled[m])
            total.update(c)
            for k, v in d.items():
                delays[k] += v
        med = {k: round(float(np.median(v)), 1) for k, v in delays.items() if v}
        print(json.dumps({"threshold": threshold, **total, "median_meeting_time_to_name_s": med}))


if __name__ == "__main__":
    main()
