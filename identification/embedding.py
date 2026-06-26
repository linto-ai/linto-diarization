"""
Embedding backend for speaker identification: pyannote embedding model loading
and audio -> embedding computation. This module imports torch/pyannote.audio and
must only be imported by the worker runtime (not by the unit test suite).
"""

import os
import subprocess
import time

import torch
import torchaudio
from pyannote.audio import Inference, Model

from identification.spkid_core import MODEL_ID, MODEL_REVISION


class EmbeddingBackend:
    """Speaker embedding model (loaded once per worker process)"""

    def __init__(self, device=None, log=None):
        self.device = device or self._get_device()
        self.log = log
        self._embedding_model = None
        self._inference = None

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def is_loaded(self):
        return self._embedding_model is not None

    def load(self):
        """Load the embedding model (no-op if already loaded)"""
        if self._embedding_model is not None:
            return
        tic = time.time()
        # The pyannote embedding model can be fetched from the HuggingFace hub
        # (gated: needs an access token) or loaded from a local snapshot. The
        # source and optional token/revision are configurable via env.
        model_source = os.environ.get("SPEAKER_ID_EMBEDDING_MODEL", MODEL_ID)
        auth_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        load_kwargs = {}
        if auth_token:
            load_kwargs["use_auth_token"] = auth_token
        if MODEL_REVISION:
            load_kwargs["revision"] = MODEL_REVISION
        self._embedding_model = Model.from_pretrained(model_source, **load_kwargs)
        # window="whole" -> a single embedding for the whole input waveform
        self._inference = Inference(self._embedding_model, window="whole")
        self._inference.to(torch.device(self.device))
        if self.log:
            self.log.info(
                f"Speaker identification model loaded in {time.time() - tic:.3f} seconds on {self.device}"
            )

    def compute_embedding(self, audio, min_len=640):
        """
        Compute speaker embedding from audio

        Args:
            audio (torch.Tensor): audio waveform
        """
        assert self._embedding_model is not None, "Speaker identification model not initialized"
        # The following is to avoid a failure on too short audio (less than 640 samples = 40ms at 16kHz)
        if audio.shape[-1] < min_len:
            audio = torch.cat([audio, torch.zeros(audio.shape[0], min_len - audio.shape[-1])], dim=-1)
        # pyannote returns a 1D numpy vector for window="whole"; wrap it as a
        # [1, dim] tensor so callers can use embedding[0].flatten() and .cpu()
        embedding = self._inference({"waveform": audio, "sample_rate": 16000})
        return torch.from_numpy(embedding).unsqueeze(0)

    @staticmethod
    def convert_wavfile(wavfile, outfile):
        """
        Converts file to 16khz single channel mono wav
        """
        cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(
            wavfile, outfile
        )
        subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE).wait()
        if not os.path.isfile(outfile):
            raise RuntimeError(f"Failed to run conversion: {cmd}")
        return outfile

    def check_wav_16khz_mono(self, wavfile):
        """
        Returns the audio signal if a wav file is 16khz and single channel
        """
        try:
            signal, fs = torchaudio.load(wavfile)
        except:
            if self.log: self.log.info(f"Could not load {wavfile}")
            return None
        assert len(signal.shape) == 2
        mono = (signal.shape[0] == 1)
        freq = (fs == 16000)
        if mono and freq:
            return signal

        reason = ""
        if not mono:
            reason += " is not mono"
        if not freq:
            if reason:
                reason += " and"
            reason += f" is in {freq/1000} kHz"
        if self.log: self.log.info(f"File {wavfile} {reason}")

    def load_audio_concat(self, audio_files, max_duration, sample_rate=16_000, on_error="raise"):
        """
        Load and concatenate audio files (16 kHz mono, converting with ffmpeg when
        needed), up to max_duration seconds.

        Args:
            audio_files (list): paths to audio files
            max_duration (float): maximum total duration (in seconds) to keep
            sample_rate (int): expected sample rate
            on_error (str): "raise" to fail on an unreadable file, "skip" to ignore it

        Returns:
            (audio, duration_used, files_used):
                audio (torch.Tensor | None): concatenated waveform (None if nothing was loaded)
                duration_used (float): seconds of audio kept
                files_used (int): number of files that contributed audio
        """
        assert on_error in ["raise", "skip"]
        audio = None
        files_used = 0
        max_samples = int(max_duration * sample_rate)
        for audio_file in audio_files:
            try:
                clip_audio = self.check_wav_16khz_mono(audio_file)
                if clip_audio is not None:
                    clip_sample_rate = 16000
                else:
                    if self.log: self.log.info(f"Converting audio file {audio_file} to single channel 16kHz WAV using ffmpeg...")
                    converted_wavfile = os.path.join(
                        os.path.dirname(audio_file), "___{}.wav".format(os.path.splitext(os.path.basename(audio_file))[0])
                    )
                    self.convert_wavfile(audio_file, converted_wavfile)
                    try:
                        clip_audio, clip_sample_rate = torchaudio.load(converted_wavfile)
                    finally:
                        os.remove(converted_wavfile)

                assert clip_sample_rate == sample_rate, f"Unsupported sample rate {clip_sample_rate} (only {sample_rate} is supported)"
            except Exception:
                if on_error == "raise":
                    raise
                if self.log:
                    self.log.warning(f"Skipping audio file {audio_file} (could not be loaded)")
                continue

            if clip_audio.shape[1] > max_samples:
                clip_audio = clip_audio[:, :max_samples]
            if audio is None:
                audio = clip_audio
            else:
                audio = torch.cat((audio, clip_audio), 1)
            files_used += 1
            # Update maximum number of remaining samples
            max_samples -= clip_audio.shape[1]
            if max_samples <= 0:
                break

        duration_used = audio.shape[1] / sample_rate if audio is not None else 0.0
        return audio, duration_used, files_used
