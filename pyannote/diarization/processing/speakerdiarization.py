#!/usr/bin/env python3
import logging
import os
import time
import uuid
import torchaudio
from pyannote.audio import Pipeline, Audio
import io
import werkzeug


class SpeakerDiarization:
    def __init__(self):
        self.log = logging.getLogger("__speaker-diarization__" + __name__)

        if os.environ.get("DEBUG", False) in ["1", 1, "true", "True"]:
            self.log.setLevel(logging.DEBUG)
            self.log.info("Debug logs enabled")
        else:
            self.log.setLevel(logging.INFO)

        self.log.info("Instanciating SpeakerDiarization")
        self.tolerated_silence = 3   #tolerated_silence=3s: silence duration tolerated to merge same speaker segments####
        home = os.path.expanduser('~')

        self.pipeline = Pipeline.from_pretrained(
                home + "/.cache/torch/pyannote/models--pyannote--speaker-diarization/snapshots/25bcc7e3631933a02af5ee39379797d704aee3f8/config.yaml",
                cache_dir = home + "/.cache"
        )
    
    def run_pyannote(self, audioFile, number_speaker, max_speaker):
        
        start_time = time.time()

        if isinstance(audioFile, io.IOBase):
            # Workaround for https://github.com/pyannote/pyannote-audio/issues/1179
            waveform, sample_rate = torchaudio.load(audioFile)
            audioFile = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }

        elif isinstance(audioFile, werkzeug.datastructures.file_storage.FileStorage):
            audioFile = io.BytesIO(audioFile.read())

        elif isinstance(audioFile, str):
            audioFile = {"audio": audioFile, "channel": 0}

        else:
            raise ValueError(f"Unsupported audio file type {type(audioFile  )}")

        if number_speaker!= None:
            diarization = self.pipeline(audioFile, num_speakers=number_speaker)
        else:
            diarization = self.pipeline(audioFile, min_speakers=2, max_speakers=max_speaker)
        
        diarization=diarization.support(collar= self.tolerated_silence)
        json = {}
        _segments=[]
        _speakers={}
        speaker_surnames = {}
        for iseg, (segment, track, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        
            # Convert speaker names to spk1, spk2, etc.
            if speaker not in speaker_surnames:
                speaker_surnames[speaker] = "spk"+str(len(speaker_surnames)+1)
            speaker = speaker_surnames[speaker]
            
            formats={}
            formats["seg_id"] = iseg + 1 # Note: we could use track, which is a string
            formats["seg_begin"] = self.round(segment.start)
            formats["seg_end"] = self.round(segment.end)
            formats["spk_id"] = speaker
            
            if formats["spk_id"] not in _speakers:
                _speakers[speaker] = {"spk_id" : speaker}
                _speakers[speaker]["duration"] = self.round(segment.end-segment.start)
                _speakers[speaker]["nbr_seg"] = 1
            else:
                _speakers[speaker]["duration"] += self.round(segment.end-segment.start)
                _speakers[speaker]["nbr_seg"] += 1

            _segments.append(formats)
        
        json["speakers"] = list(_speakers.values())
        json["segments"] = _segments
                        
        if len(_segments) > 0:
            duration = _segments[-1]["seg_end"]
            self.log.info(
                "Speaker Diarization took %.3f[s] with a speed %0.2f[xRT]"
                % (
                    time.time() - start_time,
                    (time.time() - start_time)/ duration,
                )
            )
    
        return json

    def round(self, number):
        # Return number with precision 0.01
        return float("{:.2f}".format(number))


    def run(self, file_path, number_speaker: int = None, max_speaker: int = None):
        self.log.info(f"Starting diarization on file {file_path}")
        try:
            return self.run_pyannote(file_path, number_speaker = number_speaker, max_speaker = max_speaker)
        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )    
                
        
            

