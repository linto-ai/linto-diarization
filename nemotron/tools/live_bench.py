"""Capacity of the live diarization: N sessions streamed in real time, delay of the results.

    python nemotron/tools/live_bench.py ws://host:port audio.wav --sessions 8 16 24 --seconds 60

Each session streams a different 60 s slice of audio.wav at real time. The delay is the audio sent
minus the audio diarized when a "turns" message arrives: the 1.04 s of the low_latency preset plus
queueing and processing. A pod keeps up with N sessions while the delay stays flat.
"""
import argparse
import asyncio
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_client  # noqa: E402


async def run(url, paths, stagger):
    async def one(i, p):
        await asyncio.sleep(stagger * i)
        return await live_client.stream(url, p, speed=1.0)
    return await asyncio.gather(*(one(i, p) for i, p in enumerate(paths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("wav")
    parser.add_argument("--sessions", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    audio, sr = sf.read(args.wav, dtype="int16")
    tmp = tempfile.mkdtemp()
    print("sessions  delay_p50_ms  delay_p95_ms  delay_max_ms  errors")
    for n in args.sessions:
        step = max(1, (len(audio) // sr - args.seconds) // max(1, n))
        paths = []
        for k in range(n):
            p = os.path.join(tmp, f"{k}.wav")
            s = (k * step) % max(1, len(audio) // sr - args.seconds)
            sf.write(p, audio[s * sr:(s + args.seconds) * sr], sr)
            paths.append(p)
        results = asyncio.run(run(args.url, paths, stagger=1.0 / n))
        # skip the first 5 s of each session (connection, first chunk)
        delays = np.array([d for r in results for d in r["latencies"][5:]])
        errors = sum(len(r["errors"]) for r in results)
        print(f"{n:8}  {np.percentile(delays, 50):12.0f}  {np.percentile(delays, 95):12.0f}  {delays.max():12.0f}  {errors:6}")


if __name__ == "__main__":
    main()
