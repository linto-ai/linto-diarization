from speechbrain.pretrained import SpeakerRecognition
import os
#from pydub import AudioSegment
import torchaudio
import tempfile
from collections import defaultdict
import torch

if torch.cuda.is_available():
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cuda"})
else:
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",run_opts={"device":"cpu"})

# recognize speaker name
def speaker_recognition(file_name, voices_folder, cand_speakers, segments, wildcards):
    
    if len(cand_speakers) > 0:
        speakers = cand_speakers
    else:
        speakers = os.listdir(voices_folder)

    id_count = defaultdict(int)
    # Load the WAV file    
    audio, fs = torchaudio.load(file_name)  
    
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
        if (end-start) < 600:          
          clip=torch.cat((clip, clip,clip, clip), 1)
          
        i = i + 1         
        max_score = 0
        person = "unknown"      # if no match to any voice, then return unknown
        
        for speaker in speakers:            
            voices = os.listdir(voices_folder + "/" + speaker)
           
            for voice in voices:
                voice_file = voices_folder + "/" + speaker + "/" + voice                
                
                # compare voice file with audio fil
                voice_file_audio,_ = torchaudio.load(voice_file)
                score, prediction = verification.verify_batch(voice_file_audio, clip)                            
                prediction = prediction[0].item()
                score = score[0].item()                                
                if prediction == True:
                    if score >= max_score:
                        max_score = score
                        speakerId = speaker.split(".")[0]                         
                        if speakerId not in wildcards:        # speaker_00 cannot be speaker_01
                            person = speakerId
                        

        id_count[person] += 1

        # Delete the WAV file after processing
        #os.remove(file)
        
        current_pred = max(id_count, key=id_count.get)

        duration += (end - start)
        
        if duration >= limit and current_pred != "unknown":
            break
    
    most_common_Id = max(id_count, key=id_count.get)
    
    return most_common_Id

