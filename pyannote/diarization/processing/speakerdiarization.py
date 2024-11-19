#!/usr/bin/env python3
import io
import logging
import os
import sys
import json

import memory_tempfile
import tqdm
import torch
import torchaudio
import werkzeug
from pyannote.audio import Audio, Pipeline

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "identification"
    )
)

from identification.speaker_identify import SpeakerIdentifier

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
        self.speaker_identifier = SpeakerIdentifier(device=device, log=self.log)

        self.speaker_identifier.initialize_speaker_identification()


    def run_pyannote(self, audioFile, speaker_count, max_speaker):

        cache_file = None
        if os.environ.get("CACHE_DIARIZATION_RESULTS", False) in ["1", 1, "true", "True"]:
            cache_dir = "/opt/cache_diarization"
            os.makedirs(cache_dir, exist_ok=True)
            # Get the md5sum of the file

            import subprocess
            import hashlib, pickle
            p = subprocess.Popen(["md5sum", audioFile], stdout = subprocess.PIPE)
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

        class ProgressBarHook:
            def __init__(self):
                self.pbar = None
                self.step_name = None

            def __call__(
                self,
                step_name,
                step_artifact,
                file = None,
                total = None,
                completed = None,
            ):
                if step_name != self.step_name:
                    self.step_name = step_name
                    self.pbar = tqdm.tqdm(total=total)
                elif total:
                    self.pbar.total = total
                self.pbar.set_description(step_name)
                self.pbar.update(1)

        if speaker_count!= None:
            diarization = self.pipeline(audioFile, num_speakers=speaker_count, hook=ProgressBarHook())
        else:
            diarization = self.pipeline(audioFile, min_speakers=1, max_speakers=max_speaker, hook=ProgressBarHook())

        # Remove small silences inside speaker turns
        if self.tolerated_silence:
            diarization = diarization.support(collar= self.tolerated_silence)

        result = {}
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

        result["speakers"] = list(_speakers.values())
        result["segments"] = _segments

        if cache_file:
            with open(cache_file, "w") as f:
                json.dump(result, f)

        return result

    def round(self, number):
        # Return number with precision 0.001
        return float("{:.3f}".format(number))

    def run(
        self,
        file_path,
        speaker_count: int = None,
        max_speaker: int = None,
        speaker_names = None,
    ):
        # Early check on speaker names
        speaker_names = self.speaker_identifier.check_speaker_specification(speaker_names)

        # If we run both speaker diarization and speaker identification, we need to save the file
        if speaker_names and isinstance(file_path, werkzeug.datastructures.file_storage.FileStorage):

            if self.tempfile is None:
                self.tempfile = memory_tempfile.MemoryTempfile(
                    filesystem_types=["tmpfs", "shm"], fallback=True
                )
                self.log.info(f"Using temporary folder {self.tempfile.gettempdir()}")

            with self.tempfile.NamedTemporaryFile(suffix=".wav") as ntf:
                file_path.save(ntf.name)
                return self.run(ntf.name, speaker_count, max_speaker, speaker_names=speaker_names)

        self.log.info(f"Starting diarization on file {file_path}")

        try:
            result = self.run_pyannote(
                file_path, speaker_count=speaker_count, max_speaker=max_speaker
            )
            result = self.speaker_identifier.speaker_identify_given_diarization(file_path, result, speaker_names)
            return result
        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )
