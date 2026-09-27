"""Nemotron diarization engine: Sortformer streaming inference over a file, block by block.

Kept independent from Celery/HTTP so that the live (websocket) engine can share the model.
"""
import contextlib
import logging
import os
import threading

import numpy as np
import soundfile as sf
import torch

from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import (
    load_postprocessing_from_yaml,
    predlist_to_timestamps,
)

log = logging.getLogger("__speaker-diarization__" + __name__)

SAMPLE_RATE = 16000
HOP = 160  # mel hop length in samples (10 ms)

# Streaming presets from the model card, in 80 ms frames:
# (chunk_len, chunk_right_context, fifo_len, spkcache_update_period, spkcache_len)
PRESETS = {
    "offline": (340, 40, 40, 300, 264),  # 30.4 s input buffer
    "low_latency": (9, 4, 264, 222, 264),  # 1.04 s
    "very_low_latency": (6, 2, 264, 222, 264),  # 0.64 s
    "ultra_low_latency": (3, 1, 264, 222, 264),  # 0.32 s
}


# GPU priorities: live chunks first, then speaker embeddings, then file chunks
LIVE, IDENTIFICATION, FILE = 0, 1, 2


class GpuArbiter:
    """Lets one thread at a time use the model; a waiting thread with a lower priority value goes first."""

    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._waiting = [0, 0, 0]

    @contextlib.contextmanager
    def hold(self, priority):
        with self._cond:
            self._waiting[priority] += 1
            while self._busy or any(self._waiting[p] for p in range(priority)):
                self._cond.wait()
            self._waiting[priority] -= 1
            self._busy = True
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()


class NemotronEngine:
    """Loads the model once and runs files through the streaming loop.

    Mel features are computed on GPU per audio block (BLOCK_SECONDS, with a margin on
    each side) and the same streaming state is carried from one block to the next.
    Output is bit-identical to SortformerEncLabelModel.diarize() on the whole file, with
    a VRAM footprint that no longer grows with the audio duration.
    """

    def __init__(self, model_path, device="cuda", preset="offline", block_seconds=300.0, margin_seconds=1.0):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}', expected one of {sorted(PRESETS)}")
        self.device = torch.device(device)
        self.preset = preset
        self.gpu = GpuArbiter()
        if os.path.isfile(model_path):
            self.model = SortformerEncLabelModel.restore_from(model_path, map_location=self.device)
        else:
            self.model = SortformerEncLabelModel.from_pretrained(model_path, map_location=self.device)
        self.model.eval()
        sm = self.model.sortformer_modules
        self.use_preset(preset, async_streaming=False)
        self.max_speakers = sm.n_spk
        self.block_frames = int(block_seconds * SAMPLE_RATE / HOP)
        self.margin = int(margin_seconds * SAMPLE_RATE) // HOP * HOP
        self.postprocessing = load_postprocessing_from_yaml(None)
        log.info(f"Nemotron engine ready on {self.device} (preset={preset}, max_speakers={self.max_speakers})")

    def use_preset(self, preset, async_streaming):
        """Set the streaming parameters of the shared model. Call while holding self.gpu."""
        sm = self.model.sortformer_modules
        (sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_update_period, sm.spkcache_len) = PRESETS[preset]
        sm.chunk_left_context = 1
        self.model.async_streaming = async_streaming

    def num_frames(self, num_samples):
        """Number of 10 ms feature frames the preprocessor yields for num_samples."""
        length = torch.tensor([float(num_samples)], device=self.device)
        return int(self.model.preprocessor.featurizer.get_seq_len(length)[0])

    def diarize_file(self, path, progress_callback=None):
        """Return speaker activity probabilities, float32 array [frames, max_speakers] at 10 ms."""
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
        except sf.LibsndfileError:
            # Formats libsndfile cannot read (mp3 in some builds, m4a, ...): decode with FFmpeg
            import torchaudio
            audio, sr = torchaudio.load(path)
            wav = audio.mean(dim=0).numpy()
        if sr != SAMPLE_RATE:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
        return self.diarize_waveform(np.ascontiguousarray(wav, dtype=np.float32), progress_callback)

    @torch.inference_mode()
    def diarize_waveform(self, wav, progress_callback=None):
        num_samples = len(wav)
        total = self.num_frames(num_samples)
        cache = {"start": 0, "end": 0, "feats": None}

        def features(a, b):
            if not (cache["start"] <= a and b <= cache["end"]):
                start, end = a, max(b, min(total, a + self.block_frames))
                s = max(0, start * HOP - self.margin)
                e = min(num_samples, end * HOP + self.margin)
                x = torch.from_numpy(wav[s:e]).unsqueeze(0).to(self.device)
                f, _ = self.model.preprocessor(input_signal=x, length=torch.tensor([e - s], device=self.device))
                offset = s // HOP
                cache.update(start=start, end=end, feats=f[:, :, start - offset:end - offset])
                if progress_callback:
                    progress_callback(min(1.0, start / max(1, total)))
            return cache["feats"][:, :, a - cache["start"]:b - cache["start"]]

        preds = self._run_chunks(features, total)
        if progress_callback:
            progress_callback(1.0)
        return preds

    def _run_chunks(self, features, total):
        """Same chunking as SortformerModules.streaming_feat_loader, on global frame indices.
        Each chunk takes the GPU with the FILE priority, so live sessions are served in between."""
        m, sm = self.model, self.model.sortformer_modules
        sub = sm.subsampling_factor
        with self.gpu.hold(FILE):
            self.use_preset(self.preset, async_streaming=False)
            state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=self.device)
        empty = torch.zeros((1, 0, sm.n_spk), device=self.device)
        out = []
        start = end = 0
        while end < total:
            with self.gpu.hold(FILE):
                self.use_preset(self.preset, async_streaming=False)
                left = min(sm.chunk_left_context * sub, start)
                end = min(start + sm.chunk_len * sub, total)
                right = min(sm.chunk_right_context * sub, total - end)
                chunk = features(start - left, end + right)
                length = torch.tensor([chunk.shape[2]], device=self.device)
                state, preds = m.forward_streaming_step(
                    processed_signal=chunk.transpose(1, 2),
                    processed_signal_length=length,
                    streaming_state=state,
                    total_preds=empty,
                    left_offset=left,
                    right_offset=right,
                )
                out.append(preds[0].float().cpu())
            start = end
        return torch.cat(out).numpy() if out else np.zeros((0, sm.n_spk), dtype=np.float32)

    def segments(self, preds):
        """NeMo post-processing (same as diarize()): list of (start, end, speaker_index)."""
        timestamps = predlist_to_timestamps(
            batch_preds_list=[torch.from_numpy(preds).unsqueeze(0)],
            audio_rttm_map_dict={"audio": {"offset": 0.0}},
            cfg_vad_params=self.postprocessing,
            unit_10ms_frame_count=self.model.output_subsampling_factor,
        )[0]
        lines = generate_diarization_output_lines(speaker_timestamps=timestamps, model_spk_num=len(timestamps))
        result = []
        for line in lines:
            start, end, speaker = line.split()
            result.append((float(start), float(end), int(speaker.rsplit("_", 1)[1])))
        return result
