#!/usr/bin/env python3
import logging
import os
import time
from simple_diarizer.diarizer import Diarizer
import memory_tempfile

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
        self.diar = Diarizer(
                  embed_model='ecapa', # 'xvec' and 'ecapa' supported
                  cluster_method='sc' # 'ahc' and 'sc' supported
               )

        self.tempfile = None
    
    def run_simple_diarizer(self, file_path, number_speaker, max_speaker):
        
        start_time = time.time()

        
        if type(file_path) is not str:

            if self.tempfile is None:

                self.tempfile = memory_tempfile.MemoryTempfile(
                    preferred_paths=['/dev/shm'], filesystem_types=['tmpfs', 'shm'], fallback=False
                )

                self.log.info(f"Using temporary folder {self.tempfile.gettempdir()}")

            with self.tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
                file_path.save(ntf.name)
                return self.run_simple_diarizer(ntf.name, number_speaker, max_speaker)
        
        if number_speaker!= None:
            diarization = self.diar.diarize(file_path, num_speakers=number_speaker,silence_tolerance=self.tolerated_silence)
        else:
            diarization = self.diar.diarize(file_path, num_speakers=None, max_speakers=max_speaker,threshold=3e-1, silence_tolerance=self.tolerated_silence)

        # Approximate estimation of duration for RTF
        duration = diarization[-1]["end"] if len(diarization) > 0 else 1
        # info = torchaudio.info(file_path)
        # duration = info.num_frames / info.sample_rate
        self.log.info(
            "Speaker Diarization took %.3f[s] with a speed %0.2f[xRT]"
            % (
                time.time() - start_time,
                (time.time() - start_time)/ duration,
            )
        )
            
        return diarization

    def format_response(self, segments: list) -> dict:
        #########################
        # Response format is
        #
        # {
        #   "speakers":[
        #       {
        #           "id":"spk1",
        #           "tot_dur":10.5,
        #           "nbr_segs":4
        #       },
        #       {
        #           "id":"spk2",
        #           "tot_dur":6.1,
        #           "nbr_segs":2
        #       }
        #   ],
        #   "segments":[
        #       {
        #           "seg_id":1,
        #           "spk_id":"spk1",
        #           "seg_begin":0,
        #           "seg_end":3.3,
        #       },
        #       {
        #           "seg_id":2,
        #           "spk_id":"spk2",
        #           "seg_begin":3.6,
        #           "seg_end":6.2,
        #       },
        #   ]
        # }
        #########################

        json = {}
        _segments = []
        _speakers = {}
        seg_id = 1
        spk_i = 1
        spk_i_dict = {}
        
        for seg in segments:
        
            segment = {}
            segment["seg_id"] = seg_id

            # Ensure speaker id continuity and numbers speaker by order of appearance.
            if seg['label'] not in spk_i_dict.keys():
                spk_i_dict[seg['label']] = spk_i
                spk_i += 1

            spk_id = "spk" + str(spk_i_dict[seg['label']])
            segment["spk_id"] = spk_id
            segment["seg_begin"] = self.round(seg['start'])
            segment["seg_end"] = self.round(seg['end'])

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
            spkstat["duration"] = self.round(spkstat["duration"])

        json["speakers"] = list(_speakers.values())
        json["segments"] = _segments
        return json

    
    def round(self, x):
        return round(x, 2)

    def run(self, file_path, number_speaker: int = None, max_speaker: int = None):
        self.log.debug(f"Starting diarization on file {file_path}")
        try:
            return self.run_simple_diarizer(file_path, number_speaker = number_speaker, max_speaker = max_speaker)
        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )    
                
        
            

