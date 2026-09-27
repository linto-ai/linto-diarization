"""Reference client of the live diarization websocket, for tests and for integrators.

    python nemotron/tools/live_client.py ws://localhost:8080 meeting.wav [--speed 1] [--identification spec.json]

Streams a 16 kHz mono wav as 100 ms binary frames (speed 1 = real time, 0 = as fast as possible),
prints the messages, and returns the merged speech turns and the identity timeline.
"""
import argparse
import asyncio
import json
import time

import soundfile as sf
from websockets.asyncio.client import connect

FRAME_SAMPLES = 1600  # 100 ms


def merge(turns, gap_ms=0):
    """Join the runs of a speaker cut at chunk edges: {speaker: [(start_ms, end_ms)]}."""
    by = {}
    for t in sorted(turns, key=lambda t: (t["speaker"], t["start_ms"])):
        runs = by.setdefault(t["speaker"], [])
        if runs and t["start_ms"] <= runs[-1][1] + gap_ms:
            runs[-1] = (runs[-1][0], max(runs[-1][1], t["end_ms"]))
        else:
            runs.append((t["start_ms"], t["end_ms"]))
    return by


async def stream(url, wav_path, speed=1.0, identification=None, session=None, verbose=False, max_seconds=None,
                 token=None):
    audio, sr = sf.read(wav_path, dtype="int16")
    assert sr == 16000 and audio.ndim == 1, "16 kHz mono wav expected"
    if max_seconds:
        audio = audio[: int(max_seconds * sr)]
    start = {"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le"}
    if session:
        start["session"] = session
    if identification:
        start["identification"] = identification
    result = {"turns": [], "identities": [], "errors": [], "saturated": False, "latencies": []}
    sent_ms = [0]
    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with connect(url, max_size=2**22, additional_headers=headers) as ws:
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            result["errors"].append(ready)
            return result
        result["ready"] = ready
        t0 = time.monotonic()

        async def sender():
            for i in range(0, len(audio), FRAME_SAMPLES):
                await ws.send(audio[i:i + FRAME_SAMPLES].tobytes())
                sent_ms[0] = (i + FRAME_SAMPLES) * 1000 // sr
                if speed > 0:
                    delay = t0 + sent_ms[0] / 1000 / speed - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                elif i % (FRAME_SAMPLES * 50) == 0:
                    await asyncio.sleep(0)
            await ws.send(json.dumps({"type": "stop"}))

        task = asyncio.create_task(sender())
        async for raw in ws:
            m = json.loads(raw)
            if verbose:
                print(m)
            kind = m["type"]
            if kind == "turns":
                result["turns"].extend(m["turns"])
                if speed > 0:
                    # audio sent minus audio diarized, at the time the message arrives
                    result["latencies"].append(sent_ms[0] - m["until_ms"])
            elif kind == "identity":
                result["identities"].append(dict(m, audio_ms=sent_ms[0]))
            elif kind == "saturated":
                result["saturated"] = True
            elif kind == "error":
                result["errors"].append(m)
            elif kind == "end":
                result["until_ms"] = m["until_ms"]
                break
        await task
    result["merged"] = merge(result["turns"])
    result["wall_seconds"] = time.monotonic() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--identification", help="JSON file with the identification object")
    parser.add_argument("--token", help="value of NEMOTRON_LIVE_TOKEN on the server")
    args = parser.parse_args()
    spec = json.load(open(args.identification)) if args.identification else None
    r = asyncio.run(stream(args.url, args.wav, args.speed, spec, verbose=True, token=args.token))
    print(json.dumps({k: v for k, v in r.items() if k != "turns"}, indent=1, default=str))


if __name__ == "__main__":
    main()
