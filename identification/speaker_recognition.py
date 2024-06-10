import speechbrain
if speechbrain.__version__ >= "1.0.0":
   from speechbrain.inference.speaker import SpeakerRecognition, EncoderClassifier
else:
   from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
import os
from collections import defaultdict
import torch
import torchaudio
import time

import memory_tempfile
import werkzeug
import pickle as pkl
if torch.cuda.is_available():
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
    embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
else:
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cpu"})
    embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cpu"})

#Calculte embeddings in voices_ref
def speaker_ref_embedding(voice_ref,embedding_dir):
    for root, dirs, files in os.walk(voice_ref):
        for file in files:
            if file.endswith(".wav"):
                name = os.path.splitext(os.path.basename(file))[0]                
                voice_file_audio,_ = torchaudio.load(os.path.join(root, file))
                spk_embed = embed_model.encode_batch(voice_file_audio)                      
                embeddings_file = embedding_dir+'/' + name + '.pkl'                
                pkl.dump(spk_embed, open(embeddings_file, 'wb'))
    

# recognize speaker name
def speaker_recognition(audio, voices_folder, cand_speakers, segments, wildcards):
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    if len(cand_speakers) > 0:
        speakers = cand_speakers
    else:
        speakers = os.listdir(voices_folder)

    id_count = defaultdict(int)
    i = 0
    
    '''
    iterate over segments and check speaker for increased accuracy.
    assign speaker name to arbitrary speaker tag 'SPEAKER_XX'
    '''

    limit = 60 * 16000   #maximum duration of speech in samples to try speaker ID
    duration = 0
    
    for segment in segments:
        start = int(segment[0] * 16000 )  # start time in stamps
        end = int(segment[1] * 16000 )    # end time in stamps        
        clip = audio[:, start:end]        
        if (end-start) < 600:    # ECAPA-TDNN embedding are only extracted for speech of duration > 0.15s      
          clip=torch.cat((clip, clip,clip, clip), 1)
          
        i = i + 1         
        max_score = 0
        person = "unknown"      # if no match to any voice, then return unknown        
              
                          
        for speaker in speakers:                
            embed_file = voices_folder + "/" + "embeddings" + "/" + speaker + ".pkl"             
            # compare voice file with audio fil                
            with (open(embed_file, "rb")) as openfile:                    
                    emb1=pkl.load(openfile)
                    
            emb2 = embed_model.encode_batch(clip)                
            score = similarity(emb1, emb2)  
            score=score[0]                           
            if score > 0.25:                
                if score >= max_score:
                    max_score = score
                    speakerId = speaker.split(".")[0]                                                
                    if speakerId not in wildcards:        # speaker_00 cannot be speaker_01
                        person = speakerId
                        

        id_count[person] += 1        
        current_pred = max(id_count, key=id_count.get)
        duration += (end - start)
        
        if duration >= limit and current_pred != "unknown":
            break
    
    most_common_Id = max(id_count, key=id_count.get)
    
    return most_common_Id


def run_speaker_identification(audioFile, diarization, spk_names, log=None):
    
    if isinstance(audioFile, werkzeug.datastructures.file_storage.FileStorage):
        tempfile = memory_tempfile.MemoryTempfile(filesystem_types=['tmpfs', 'shm'], fallback=True)
        if log:
            log.info(f"Using temporary folder {tempfile.gettempdir()}")

        with tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
            audioFile.save(ntf.name)
            return run_speaker_identification(ntf.name, diarization, spk_names)
        
    audio, fs = torchaudio.load(audioFile)
    
    if spk_names is not None and len(spk_names) > 0:

        voices_box = "voices_ref"        
        speaker_tags = []
        speakers = {}
        common = []
        speaker_map = {}
        speaker_surnames = {}

        for segment in diarization["segments"]:

            start = segment["seg_begin"]
            end = segment["seg_end"]
            speaker = segment["spk_id"]
            common.append([start, end, speaker])

            # find different speakers
            if speaker not in speaker_tags:
                speaker_tags.append(speaker)
                speaker_map[speaker] = speaker
                speakers[speaker] = []

            speakers[speaker].append([start, end, speaker])

        if voices_box != None and voices_box != "":
            identified = []
            if log:
                log.info("running speaker recognition...")
            tic = time.time()

            for spk_tag, spk_segments in speakers.items():
                spk_name = speaker_recognition(
                    audio, voices_box, spk_names, spk_segments, identified
                )                
                identified.append(spk_name)
                if spk_name != "unknown":
                    speaker_map[spk_tag] = spk_name
                else:
                    speaker_map[spk_tag] = spk_tag

            if log:
                log.info(
                    f"Speaker recognition done in {time.time() - tic:.3f} seconds"
                )

        json = {}
        _segments = []
        _speakers = {}
        speaker_surnames = {}
        for iseg, segment in enumerate(diarization["segments"]):
            start = segment["seg_begin"]
            end = segment["seg_end"]
            speaker = segment["spk_id"]

            # Convert speaker names to spk1, spk2, etc.
            if speaker not in speaker_surnames:
                speaker_surnames[speaker] = (
                    speaker  # "spk"+str(len(speaker_surnames)+1)
                )
            speaker = speaker_surnames[speaker]
            speaker_name = speaker_map[speaker]
            if speaker_name == "unknown":
                speaker_name = speaker

            segment["spk_id"] = speaker_name

            _segments.append(segment)

            if speaker_name not in _speakers:
                _speakers[speaker_name] = {"spk_id": speaker_name}
                _speakers[speaker_name]["duration"] = round(end - start)
                _speakers[speaker_name]["nbr_seg"] = 1
            else:
                _speakers[speaker_name]["duration"] += round(end - start)
                _speakers[speaker_name]["nbr_seg"] += 1

        json["speakers"] = list(_speakers.values())
        json["segments"] = _segments

        return json

def round(number):
    # Return number with precision 0.001
    return float("{:.3f}".format(number))
