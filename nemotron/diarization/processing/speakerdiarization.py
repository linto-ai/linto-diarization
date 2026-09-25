import logging
import os

import memory_tempfile
import werkzeug

from identification.speaker_identify import SpeakerIdentifier

from .engine import NemotronEngine
from .result import format_result

MODEL_PATH = os.environ.get("NEMOTRON_MODEL", "/opt/models/nemotron-3-diarization/Nemotron-3-Diarization.nemo")
PRESET = os.environ.get("NEMOTRON_PRESET", "offline")
BLOCK_SECONDS = float(os.environ.get("NEMOTRON_BLOCK_SECONDS", 300))


class SpeakerDiarization:
    def __init__(self, device=None):
        self.log = logging.getLogger("__speaker-diarization__" + __name__)
        if os.environ.get("DEBUG", False) in ["1", 1, "true", "True"]:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)
        self.log.info(f"Instanciating SpeakerDiarization (Nemotron) with device={device}")

        self.engine = NemotronEngine(MODEL_PATH, device=device, preset=PRESET, block_seconds=BLOCK_SECONDS)
        self.tempfile = None
        self.speaker_identifier = SpeakerIdentifier(device=device, log=self.log)
        self.speaker_identifier.initialize_speaker_identification()

    def run(
        self,
        file_path,
        speaker_count: int = None,
        max_speaker: int = None,
        speaker_names=None,
        progress_callback=None,
    ):
        # Early check on speaker names
        speaker_names = self.speaker_identifier.check_speaker_specification(speaker_names)

        # HTTP mode hands over an uploaded file: the engine works on a path
        if isinstance(file_path, werkzeug.datastructures.file_storage.FileStorage):
            if self.tempfile is None:
                self.tempfile = memory_tempfile.MemoryTempfile(filesystem_types=["tmpfs", "shm"], fallback=True)
            with self.tempfile.NamedTemporaryFile(suffix=".wav") as ntf:
                file_path.save(ntf.name)
                return self.run(ntf.name, speaker_count, max_speaker, speaker_names, progress_callback)

        if speaker_count or max_speaker:
            # The model cannot be constrained to a number of speakers
            self.log.info(f"Ignoring speaker_count={speaker_count} / max_speaker={max_speaker} (not supported by Nemotron)")

        self.log.info(f"Starting diarization on file {file_path}")
        try:
            preds = self.engine.diarize_file(file_path, progress_callback)
            result = format_result(self.engine.segments(preds), self.engine.max_speakers)
            extra = {k: result[k] for k in ("engine", "saturated")}
            # Identification rebuilds the result dict: carry the extra fields over
            result = self.speaker_identifier.speaker_identify_given_diarization(file_path, result, speaker_names)
            result.update(extra)
            return result
        except Exception as e:
            self.log.error(e)
            raise Exception("Speaker diarization failed during processing the speech signal")
