"""From speaker activity probabilities (one row per prediction frame) to speech turns and single-speaker frames."""
import numpy as np

THRESHOLD = 0.5


def runs(preds, first_frame, frame_ms, threshold=THRESHOLD):
    """Runs of activity per speaker in preds [frames, speakers], as (speaker, start_ms, end_ms),
    sorted by start then speaker. Runs cut at the chunk edges are joined by the client."""
    active = preds > threshold
    out = []
    for speaker in range(preds.shape[1]):
        column = active[:, speaker]
        if not column.any():
            continue
        padded = np.concatenate(([False], column, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(edges[::2], edges[1::2]):
            out.append((speaker, int(first_frame + start) * frame_ms, int(first_frame + end) * frame_ms))
    out.sort(key=lambda r: (r[1], r[0]))
    return out


def single_speaker(preds, threshold=THRESHOLD, erode=1):
    """Speaker index per frame when exactly one speaker is active, else -1. `erode` frames are
    dropped at both ends of each single-speaker run inside the chunk (boundaries are unreliable);
    runs touching the chunk edges are only eroded on their inner side."""
    active = preds > threshold
    count = active.sum(axis=1)
    who = np.where(count == 1, active.argmax(axis=1), -1)
    kept = who.copy()
    n = len(who)
    i = 0
    while i < n:
        j = i
        while j < n and who[j] == who[i]:
            j += 1
        if who[i] >= 0:
            if i > 0:
                kept[i:min(j, i + erode)] = -1
            if j < n:
                kept[max(i, j - erode):j] = -1
        i = j
    return kept
