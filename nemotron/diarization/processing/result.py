"""Result formatting, kept free of heavy imports (testable without the model)."""


def format_result(segments, max_speakers):
    """Build the linto-diarization result from (start, end, speaker_index) tuples.

    Same layout as the pyannote engine: speakers renamed spk1..N by order of first
    appearance, segments sorted by start time. `saturated` is true when every speaker
    slot of the model is used: the audio may hold more speakers than the model can tell apart.
    """
    segments = sorted(segments, key=lambda s: (s[0], s[1], s[2]))
    names = {}
    speakers = {}
    result_segments = []
    for i, (start, end, index) in enumerate(segments):
        if index not in names:
            names[index] = f"spk{len(names) + 1}"
        name = names[index]
        duration = round(end - start, 3)
        if name not in speakers:
            speakers[name] = {"spk_id": name, "duration": duration, "nbr_seg": 1}
        else:
            speakers[name]["duration"] = round(speakers[name]["duration"] + duration, 3)
            speakers[name]["nbr_seg"] += 1
        result_segments.append({
            "seg_id": i + 1,
            "seg_begin": round(start, 3),
            "seg_end": round(end, 3),
            "spk_id": name,
        })
    return {
        "speakers": list(speakers.values()),
        "segments": result_segments,
        "engine": "nemotron",
        "saturated": len(names) >= max_speakers,
    }
