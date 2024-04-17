import os,sys,time
import argparse
from simple_diarizer.diarizer import Diarizer
import pprint

parser = argparse.ArgumentParser(
   description="Speaker diarization",
   formatter_class=argparse.ArgumentDefaultsHelpFormatter,

)
parser.add_argument(dest='audio_name', type=str, help="Input audio file")
parser.add_argument(dest='outputfile', nargs="?", default=None, help="Optional output file")
parser.add_argument("--embed_model", dest='embed_model', default="ecapa", type=str, help="Name of embedding")
parser.add_argument("--cluster_method", dest='cluster_method', default="nme-sc", type=str, help="Clustering method")
parser.add_argument("--cand_speaker_names", dest='cand_speaker_names', nargs='+', type=str, help="Names of speaker")
parser.add_argument("--device", dest='device', default=None, type=str, help="choise of cpu or cuda")
args = parser.parse_args() 

diar = Diarizer(
   embed_model=args.embed_model,  # 'xvec' and 'ecapa' supported
   cluster_method=args.cluster_method,  # 'ahc' 'sc' and 'nme-sc' supported
   device=args.device
)

WAV_FILE=args.audio_name
num_speakers=len(args.cand_speaker_names)
max_spk= None
output_file=args.outputfile
names_speaker=args.cand_speaker_names

print(num_speakers)




t0 = time.time() 

segments = diar.diarize(WAV_FILE, num_speakers=num_speakers,max_speakers=max_spk,spk_names=names_speaker,outfile=output_file)

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
                  
      if seg['label'] not in spk_i_dict.keys():
         spk_i_dict[seg['label']] = spk_i
         spk_i += 1

      spk_id = "spk" + str(spk_i_dict[seg['label']])
      segment["spk_id"] = spk_id
      segment["seg_begin"] = round(seg['start'])
      segment["seg_end"] = round(seg['end'])

      if spk_id not in _speakers:
         _speakers[spk_id] = {}
         _speakers[spk_id]["spk_id"] = spk_id
         _speakers[spk_id]["duration"] = seg['end']-seg['start']
         _speakers[spk_id]["nbr_seg"] = 1
      else:
         _speakers[spk_id]["duration"] += seg['end']-seg['start']
         _speakers[spk_id]["nbr_seg"] += 1

      _segments.append(segment)
      seg_id += 1

   for spkstat in _speakers.values():
      spkstat["duration"] = round(spkstat["duration"])

   json["speakers"] = list(_speakers.values())
   json["segments"] = _segments

   #pprint.pprint(json)

