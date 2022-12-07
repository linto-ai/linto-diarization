#!/usr/bin/env python3
import logging
import os
import time
import uuid
import torchaudio
from pyannote.audio import Pipeline, Audio
import io

class SpeakerDiarization:
    def __init__(self):
        self.log = logging.getLogger("__speaker-diarization__" + __name__)

        if os.environ.get("DEBUG", False) in ["1", 1, "true", "True"]:
            self.log.setLevel(logging.DEBUG)
            self.log.info("Debug logs enabled")
        else:
            self.log.setLevel(logging.INFO)

        self.log.debug("Instanciating SpeakerDiarization")
                
        self.pipeline = Pipeline.from_pretrained(
                "/root/.cache/torch/pyannote/models--pyannote--speaker-diarization/snapshots/25bcc7e3631933a02af5ee39379797d704aee3f8/config.yaml",
                cache_dir = "/root/.cache")
    
    def run_pyannote(self, audioFile, number_speaker, max_speaker):
        try:
            start_time = time.time()

            if isinstance(audioFile, io.IOBase):
                # Workaround for https://github.com/pyannote/pyannote-audio/issues/1179
                waveform, sample_rate = torchaudio.load(audioFile)
                audioFile = {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                }

            if number_speaker!= None:
                diarization = self.pipeline(audioFile, num_speakers=number_speaker)
            else:
                diarization = self.pipeline(audioFile, min_speakers=2, max_speakers=max_speaker)
            
            diarization=diarization.support(collar= 3)
            json = {}
            _segments=[]
            _speakers={}
            seg_id = 1
            spk_i = 1
            spk_i_dict = {}
            for segment, track, speaker in diarization.itertracks(yield_label=True):
            
                
                formats={}
                formats["seg_id"] =track
                formats["seg_begin"]=float("{:.2f}".format(segment.start))
                formats["seg_end"]=float("{:.2f}".format(segment.end))
                formats["spk_id"]=speaker
                if formats["spk_id"] not in _speakers:
                    _speakers[formats["spk_id"]] = {}
                    _speakers[formats["spk_id"]]["spk_id"] = formats["spk_id"]
                    _speakers[formats["spk_id"]]["duration"] =float("{:.2f}".format(formats["seg_end"]-formats["seg_begin"]))

                    _speakers[formats["spk_id"]]["nbr_seg"] = 1
                else:
                    _speakers[formats["spk_id"]]["duration"] += float("{:.2f}".format(formats["seg_end"]-formats["seg_begin"]))

                    _speakers[formats["spk_id"]]["nbr_seg"] += 1
                    

                _segments.append(formats)
            
            json["speakers"] = list(_speakers.values())
            json["segments"] = _segments
                            
            if len(_segments) > 0:
                duration = _segments[-1]["seg_end"]
                self.log.info(
                    "Speaker Diarization took %d[s] with a speed %0.2f[xRT]"
                    % (
                        int(time.time() - start_time),
                        float(int(time.time() - start_time)/ duration),
                    )
                )

        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )
    
        return json


    def run(self, file_path, number_speaker: int = None, max_speaker: int = None):
        self.log.debug(f"Starting diarization on file {file_path}")
        
        return self.run_pyannote(file_path, number_speaker = number_speaker, max_speaker = max_speaker)
            
                
        
            

