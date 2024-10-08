import argparse
import json
import os
import pprint
import sys
import time

from simple_diarizer.diarizer import Diarizer

parser = argparse.ArgumentParser(
    description="Speaker diarization",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(dest="audio_name", type=str, help="Input audio file")
parser.add_argument(
    dest="outputfile", nargs="?", default=None, help="Optional output file"
)
parser.add_argument(
    "--embed_model",
    dest="embed_model",
    default="ecapa",
    type=str,
    help="Name of embedding",
)
parser.add_argument(
    "--number_of_speakers",
    dest="number_of_speaker",
    default=None,
    type=int,
    help="Number of speakers (if known)",
)
parser.add_argument(
    "--max_speakers",
    dest="max_speakers",
    default=25,
    type=int,
    help="Maximum number of speakers (if number of speaker is unknown)",
)
parser.add_argument(
    "--cluster_method",
    dest="cluster_method",
    default="nme-sc",
    type=str,
    help="Clustering method",
)
parser.add_argument(
    "--cand_speaker_names",
    dest="cand_speaker_names",
    type=str,
    help="List of speaker ID",
)
parser.add_argument(
    "--device", dest="device", default=None, type=str, help="choise of cpu or cuda"
)
args = parser.parse_args()

diar = Diarizer(
    embed_model=args.embed_model,  # 'xvec' and 'ecapa' supported
    cluster_method=args.cluster_method,  # 'ahc' 'sc' and 'nme-sc' supported
    device=args.device,
)

WAV_FILE = args.audio_name
num_speakers = args.number_of_speaker if args.number_of_speaker != "None" else None
max_spk = args.max_speakers
output_file = args.outputfile
names_speaker = args.cand_speaker_names
names_speaker = json.loads(names_speaker)

speakers = []
for item in names_speaker:
    if type(item) == int:
        speakers.append(item)
    elif type(item) == dict:
        start = item["start"]
        end = item["end"]
        for x in range(start, end + 1):
            speakers.append(x)

import sqlite3

# TODO: FIXME (or remove)
conn = sqlite3.connect("speakers_database")
c = conn.cursor()
speakers_list = []
for i in speakers:
    item = c.execute("SELECT Name FROM speaker_names WHERE id = '%s'" % i)
    speakers_list.append(item.fetchone()[0])
# Closing the connection
conn.close()
print(speakers_list)


t0 = time.time()

segments = diar.diarize(
    WAV_FILE,
    num_speakers=num_speakers,
    max_speakers=max_spk,
    speaker_names=speakers_list,
    outfile=output_file,
)

print("Time used for processing:", time.time() - t0)

if not output_file:

    json = {}
    _segments = []
    _speakers = {}
    seg_id = 1
    spk_i = 1
    spk_i_dict = {}

    for seg in segments:

        segment = {}
        segment["seg_id"] = seg_id

        if seg["label"] not in spk_i_dict.keys():
            spk_i_dict[seg["label"]] = spk_i
            spk_i += 1

        spk_id = "spk" + str(spk_i_dict[seg["label"]])
        segment["spk_id"] = spk_id
        segment["seg_begin"] = round(seg["start"])
        segment["seg_end"] = round(seg["end"])

        if spk_id not in _speakers:
            _speakers[spk_id] = {}
            _speakers[spk_id]["spk_id"] = spk_id
            _speakers[spk_id]["duration"] = seg["end"] - seg["start"]
            _speakers[spk_id]["nbr_seg"] = 1
        else:
            _speakers[spk_id]["duration"] += seg["end"] - seg["start"]
            _speakers[spk_id]["nbr_seg"] += 1

        _segments.append(segment)
        seg_id += 1

    for spkstat in _speakers.values():
        spkstat["duration"] = round(spkstat["duration"])

    json["speakers"] = list(_speakers.values())
    json["segments"] = _segments

    # pprint.pprint(json)
