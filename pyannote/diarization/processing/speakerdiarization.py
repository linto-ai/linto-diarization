#!/usr/bin/env python3
import io
import logging
import os
import sys
import time

import memory_tempfile
import torch
import torchaudio
import werkzeug
from pyannote.audio import Audio, Pipeline

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "identification"
    )
)
import identification
from identification.speaker_recognition import run_speaker_identification, initialize_speaker_identification

class SpeakerDiarization:
    def __init__(
        self,
        device=None,
        num_threads=4,
        tolerated_silence=0,
    ):
        """
        Speaker Diarization class

        Args:
            device (str): device to use (cpu or cuda)
            num_threads (int): number of threads to use
            tolerated_silence (int): tolerated silence duration to merge same speaker segments (it was previously set to 3s)
        """
        self.log = logging.getLogger("__speaker-diarization__" + __name__)
        if os.environ.get("DEBUG", False) in ["1", 1, "true", "True"]:
            self.log.setLevel(logging.DEBUG)
            self.log.info("Debug logs enabled")
        else:
            self.log.setLevel(logging.INFO)

        if device == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log.info(
            f"Instanciating SpeakerDiarization with device={device}"
            + (f" ({num_threads} threads)" if device == "cpu" else "")
        )
        self.tolerated_silence = tolerated_silence
        home = os.path.expanduser('~')

        model_configuration = "pyannote/speaker-diarization-3.1"
        local_cache_yaml = {
            "pyannote/speaker-diarization-2.1" : "torch/pyannote/models--pyannote--speaker-diarization/snapshots/25bcc7e3631933a02af5ee39379797d704aee3f8/config.yaml",
            "pyannote/speaker-diarization-3.1" : "models--pyannote--speaker-diarization-3.1/snapshots/19c7c42a5047c3e982102ee1eb687ed866b4d193/config.yaml",
        }
        cache_parent_folder = os.path.join(home, ".cache")
        model_configuration = os.path.join(cache_parent_folder, local_cache_yaml[model_configuration])

        self.pipeline = Pipeline.from_pretrained(
            model_configuration,
            cache_dir=cache_parent_folder
        )

        self.pipeline = self.pipeline.to(torch.device(device))
        self.num_threads = num_threads
        self.tempfile = None

        initialize_speaker_identification(self.log)


    def run_pyannote(self, audioFile, number_speaker, max_speaker):

        torch.set_num_threads(self.num_threads)
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
            diarization = self.pipeline(audioFile, min_speakers=1, max_speakers=max_speaker)

        # Remove small silences inside speaker turns
        if self.tolerated_silence:
            diarization = diarization.support(collar= self.tolerated_silence)

        json = {}
        _segments=[]
        _speakers={}
        speaker_surnames = {}
        for iseg, (segment, track, speaker) in enumerate(diarization.itertracks(yield_label=True)):

            # Convert speaker names to spk1, spk2, etc.
            if speaker not in speaker_surnames:
                speaker_surnames[speaker] = "spk"+str(len(speaker_surnames)+1)
            speaker = speaker_surnames[speaker]

            formats = {}
            formats["seg_id"] = iseg + 1  # Note: we could use track, which is a string
            formats["seg_begin"] = self.round(segment.start)
            formats["seg_end"] = self.round(segment.end)
            formats["spk_id"] = speaker

            if formats["spk_id"] not in _speakers:
                _speakers[speaker] = {"spk_id": speaker}
                _speakers[speaker]["duration"] = self.round(segment.end - segment.start)
                _speakers[speaker]["nbr_seg"] = 1
            else:
                _speakers[speaker]["duration"] += self.round(
                    segment.end - segment.start
                )
                _speakers[speaker]["nbr_seg"] += 1

            _segments.append(formats)

        json["speakers"] = list(_speakers.values())
        json["segments"] = _segments

        return json

    def round(self, number):
        # Return number with precision 0.001
        return float("{:.3f}".format(number))

    def run(
        self,
        file_path,
        number_speaker: int = None,
        max_speaker: int = None,
        speaker_names = None,
    ):
        self.log.info(f"Starting diarization on file {file_path}")

        # If we run both speaker diarization and speaker identification, we need to save the file
        if speaker_names and isinstance(file_path, werkzeug.datastructures.file_storage.FileStorage):

            if self.tempfile is None:
                self.tempfile = memory_tempfile.MemoryTempfile(
                    filesystem_types=["tmpfs", "shm"], fallback=True
                )
                self.log.info(f"Using temporary folder {self.tempfile.gettempdir()}")

            with self.tempfile.NamedTemporaryFile(suffix=".wav") as ntf:
                file_path.save(ntf.name)
                return self.run(ntf.name, number_speaker, max_speaker, speaker_names=speaker_names)

        try:
            result = self.run_pyannote(
                file_path, number_speaker=number_speaker, max_speaker=max_speaker
            )
            result = run_speaker_identification(file_path, result, speaker_names, log=self.log)
            return result
        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )
