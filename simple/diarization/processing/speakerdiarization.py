#!/usr/bin/env python3
import logging
import os
import sys
import time
import json

import memory_tempfile
import torch
import werkzeug
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "simple_diarizer"))
import simple_diarizer
import simple_diarizer.diarizer

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "identification"
    )
)
from identification.speaker_identify import SpeakerIdentifier

class SpeakerDiarization:
    def __init__(self, device=None, device_vad=None, device_clustering=None, num_threads=None):
        self.log = logging.getLogger("__speaker-diarization__" + __name__)
        if os.environ.get("DEBUG", False) in ["1", 1, "true", "True"]:
            self.log.setLevel(logging.DEBUG)
            self.log.info("Debug logs enabled")
        else:
            self.log.setLevel(logging.INFO)

        self.log.info("Instanciating SpeakerDiarization")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.device_vad = device_vad
        self.device_clustering = device_clustering
        self.num_threads = num_threads


        self.log.info(f"Simple diarization version {simple_diarizer.__version__}")
        self.tolerated_silence = 3  # tolerated_silence=3s: silence duration tolerated to merge same speaker segments####

        self.diar = simple_diarizer.diarizer.Diarizer(
                  embed_model='ecapa', # 'xvec' and 'ecapa' supported
                  cluster_method='nme-sc', # 'ahc' 'sc' and 'nme-sc' supported
                  device= self.device,
                  device_vad=self.device_vad,
                  device_clustering=self.device_clustering,
                  num_threads=num_threads,
               )

        self.tempfile = None
        self.speaker_identifier = SpeakerIdentifier(device=device, log=self.log)

        self.speaker_identifier.initialize_speaker_identification()

    def run_simple_diarizer(self, file_path, speaker_count, max_speaker):

        cache_file = None
        if os.environ.get("CACHE_DIARIZATION_RESULTS", False) in ["1", 1, "true", "True"]:
            cache_dir = "/opt/cache_diarization"
            os.makedirs(cache_dir, exist_ok=True)
            # Get the md5sum of the file

            import subprocess
            import hashlib, pickle
            p = subprocess.Popen(["md5sum", file_path], stdout = subprocess.PIPE)
            (stdout, stderr) = p.communicate()
            assert p.returncode == 0, f"Error running md5sum: {stderr}"
            file_md5sum = stdout.decode("utf-8").split()[0]
            def hashmd5(obj):
                return hashlib.md5(pickle.dumps(obj)).hexdigest()

            cache_file = os.path.join(cache_dir, hashmd5((file_md5sum, speaker_count, max_speaker if not speaker_count else None)) + ".json")
            if os.path.isfile(cache_file):
                self.log.info(f"Using cached diarization result from {cache_file}")
                with open(cache_file, "r") as f:
                    return json.load(f)
            self.log.info(f"Cache file {cache_file} will be used")

        start_time = time.time()

        diarization = self.diar.diarize(
            file_path,
            num_speakers=speaker_count,
            max_speakers=max_speaker,
            silence_tolerance=self.tolerated_silence,
            threshold=3e-1,
        )

        # Approximate estimation of duration for RTF
        duration = diarization[-1]["end"] if len(diarization) > 0 else 1
        self.log.info(
            "Speaker Diarization took %.3f[s] with a speed %0.2f[xRT]"
            % (
                time.time() - start_time,
                (time.time() - start_time) / duration,
            )
        )

        result = self.format_response(diarization)

        if cache_file:
            with open(cache_file, "w") as f:
                json.dump(result, f)

        return result

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
            if seg["label"] not in spk_i_dict.keys():
                spk_i_dict[seg["label"]] = spk_i
                spk_i += 1

            spk_id = "spk" + str(spk_i_dict[seg["label"]])
            segment["spk_id"] = spk_id
            segment["seg_begin"] = self.round(seg["start"])
            segment["seg_end"] = self.round(seg["end"])

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
            spkstat["duration"] = self.round(spkstat["duration"])

        json["speakers"] = list(_speakers.values())
        json["segments"] = _segments
        return json

    def round(self, x):
        return round(x, 2)

    def run(
        self,
        file_path,
        speaker_count: int = None,
        max_speaker: int = None,
        speaker_names = None,
    ):
        # Early check on speaker names
        speaker_names = self.speaker_identifier.check_speaker_specification(speaker_names)

        if isinstance(file_path, werkzeug.datastructures.file_storage.FileStorage):
            if self.tempfile is None:
                self.tempfile = memory_tempfile.MemoryTempfile(
                    filesystem_types=["tmpfs", "shm"], fallback=True
                )
                self.log.info(f"Using temporary folder {self.tempfile.gettempdir()}")

            with self.tempfile.NamedTemporaryFile(suffix=".wav") as ntf:
                file_path.save(ntf.name)
                return self.run(ntf.name, speaker_count, max_speaker, speaker_names=speaker_names)

        self.log.info(f"Starting diarization on file {file_path}")

        if speaker_count is None and max_speaker is None:
            max_speaker = 50 # default value
            warnings.warn(f"No speaker count nor maximum specified, using default value {max_speaker=}")

        try:                       
            result = self.run_simple_diarizer(
                file_path, speaker_count=speaker_count, max_speaker=max_speaker
            )
            result = self.speaker_identifier.speaker_identify_given_diarization(file_path, result, speaker_names)
            return result
        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )
