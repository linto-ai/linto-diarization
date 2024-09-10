#!/usr/bin/env python3
import logging
import os
import time
import uuid

import librosa
import numpy as np

import pyBK.diarizationFunctions as pybk

# from spafe.features.mfcc import mfcc, imfcc
from pydub import AudioSegment
from python_speech_features import mfcc

import pyBK.diarizationFunctions as pybk

import memory_tempfile
import werkzeug


class SpeakerDiarization:
    def __init__(self):
        self.log = logging.getLogger("__speaker-diarization__" + __name__)

        if os.environ.get("DEBUG", False) in ["1", 1, "true", "True"]:
            self.log.setLevel(logging.DEBUG)
            self.log.info("Debug logs enabled")
        else:
            self.log.setLevel(logging.INFO)

        self.log.debug("Instanciating SpeakerDiarization")
        # MFCC FEATURES PARAMETERS
        self.frame_length_s = 0.03
        self.frame_shift_s = 0.01
        self.num_bins = 30
        self.num_ceps = 30

        # Segment
        self.seg_length = 100  # Window size in frames
        self.seg_increment = 100  # Window increment after and before window in frames
        self.seg_rate = 100  # Window shifting in frames

        # KBM
        # Minimum number of Gaussians in the initial pool
        self.minimumNumberOfInitialGaussians = 1024
        self.maximumKBMWindowRate = 50  # Maximum window rate for Gaussian computation
        self.windowLength = 200  # Window length for computing Gaussians
        self.kbmSize = 320  # Number of final Gaussian components in the KBM
        # If set to 1, the KBM size is set as a proportion, given by "relKBMsize", of the pool size
        self.useRelativeKBMsize = True
        # Relative KBM size if "useRelativeKBMsize = 1" (value between 0 and 1).
        self.relKBMsize = 0.1

        # BINARY_KEY
        self.topGaussiansPerFrame = 5  # Number of top selected components per frame
        self.bitsPerSegmentFactor = (
            0.2  # Percentage of bits set to 1 in the binary keys
        )

        # CLUSTERING
        self.N_init = 25  # Number of initial clusters

        # Linkage criterion used if linkage==1 ('average', 'single', 'complete')
        self.linkageCriterion = "average"
        # Similarity metric: 'cosine' for cumulative vectors, and 'jaccard' for binary keys
        self.metric = "cosine"

        # CLUSTERING_SELECTION
        # Distance metric used in the selection of the output clustering solution ('jaccard','cosine')
        self.metric_clusteringSelection = "cosine"
        # Method employed for number of clusters selection. Can be either 'elbow' for an elbow criterion based on within-class sum of squares (WCSS) or 'spectral' for spectral clustering
        self.bestClusteringCriterion = "spectral"
        self.sigma = 1  # Spectral clustering parameters, employed if bestClusteringCriterion == spectral
        self.percentile = 80
        self.maxNrSpeakers = 20  # If known, max nr of speakers in a sesssion in the database. This is to limit the effect of changes in very small meaningless eigenvalues values generating huge eigengaps

        # RESEGMENTATION
        self.resegmentation = 1  # Set to 1 to perform re-segmentation
        self.modelSize = 16  # Number of GMM components
        self.nbIter = 5  # Number of expectation-maximization (EM) iterations
        self.smoothWin = 100  # Size of the likelihood smoothing window in nb of frames

        # Pseudo-randomness
        self.seed = 0

        # Short segments to ignore
        self.min_duration = 0.3

        self.tempfile = None

    def compute_feat_Librosa(self, file_path):
        try:
            if isinstance(file_path, werkzeug.datastructures.file_storage.FileStorage):

                if self.tempfile is None:
                    self.tempfile = memory_tempfile.MemoryTempfile(filesystem_types=['tmpfs', 'shm'], fallback=True)
                    self.log.info(f"Using temporary folder {self.tempfile.gettempdir()}")

                with self.tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
                    file_path.save(ntf.name)
                    return self.compute_feat_Librosa(ntf.name)

            self.sr = 16000
            audio = AudioSegment.from_wav(file_path)
            audio = audio.set_frame_rate(self.sr)
            audio = audio.set_channels(1)
            self.data = np.array(audio.get_array_of_samples())

            # frame_length_inSample = self.frame_length_s * self.sr
            # hop = int(self.frame_shift_s * self.sr)
            # NFFT = int(2 ** np.ceil(np.log2(frame_length_inSample)))

            framelength_in_samples = self.frame_length_s * self.sr
            n_fft = int(2 ** np.ceil(np.log2(framelength_in_samples)))

            additional_kwargs = {}
            if self.sr >= 16000:
                additional_kwargs.update({"lowfreq": 20, "highfreq": 7600})

            mfcc_coef = mfcc(
                signal=self.data,
                samplerate=self.sr,
                numcep=30,
                nfilt=30,
                nfft=n_fft,
                winlen=0.03,
                winstep=0.01,
                **additional_kwargs,
            )

        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed when extracting features!!!")
        else:
            return mfcc_coef

    def computeVAD_WEBRTC(self, data, sr, nFeatures):
        try:
            if sr not in [8000, 16000, 32000, 48000]:
                data = librosa.resample(data, sr, 16000)
                sr = 16000

            va_framed = pybk.py_webrtcvad(
                data, fs=sr, fs_vad=sr, hoplength=30, vad_mode=0
            )
            segments = pybk.get_py_webrtcvad_segments(va_framed, sr)
            maskSAD = np.zeros([1, nFeatures])
            for seg in segments:
                start = int(np.round(seg[0] / self.frame_shift_s))
                end = int(np.round(seg[1] / self.frame_shift_s))
                maskSAD[0][start:end] = 1
        except Exception as e:
            self.log.error(e)
            raise ValueError(
                "Speaker diarization failed while voice activity detection!!!"
            )
        else:

            return maskSAD

    def getSegments(self, frameshift, finalSegmentTable, finalClusteringTable, dur):
        numberOfSpeechFeatures = finalSegmentTable[-1, 2].astype(int) + 1
        solutionVector = np.zeros([1, numberOfSpeechFeatures])
        for i in range(np.size(finalSegmentTable, 0)):
            solutionVector[
                0,
                np.arange(finalSegmentTable[i, 1], finalSegmentTable[i, 2] + 1).astype(
                    int
                ),
            ] = finalClusteringTable[i]
        seg = np.empty([0, 3])
        solutionDiff = np.diff(solutionVector)[0]
        first = 0
        duration_silence = 0 # Silence that can be included inside a speaker turn
        for i in range(0, np.size(solutionDiff, 0)):
            if solutionDiff[i]:
                last = i + 1
                start = (first) * frameshift
                duration = (last - first) * frameshift
                spklabel = solutionVector[0, last - 1]
                silence = not spklabel or duration <= self.min_duration
                if seg.shape[0] != 0 and spklabel == seg[-1][2]: # Same speaker as before
                    seg[-1][1] += duration + duration_silence
                    duration_silence = 0
                elif not silence: # New speaker
                    seg = np.vstack((seg, [start, duration, spklabel]))
                    duration_silence = 0
                else: # Silence (within speaker turn or between speaker turns... we do not know yet)
                    duration_silence += duration
                first = i + 1
        last = np.size(solutionVector, 1)
        start = (first - 1) * frameshift
        duration = (last - first + 1) * frameshift
        spklabel = solutionVector[0, last - 1]
        silence = not spklabel or duration <= self.min_duration
        if spklabel == seg[-1][2]:
            seg[-1][1] += duration + duration_silence
        elif not silence:
            seg = np.vstack((seg, [start, duration, spklabel]))
        return seg

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

        # Remove the last line of the segments.
        # It indicates the end of the file and segments.
        if segments[len(segments) - 1][2] == -1:
            segments = segments[: len(segments) - 1]

        for seg in segments:
            segment = {}
            segment["seg_id"] = seg_id

            # Ensure speaker id continuity and numbers speaker by order of appearance.
            if seg[2] not in spk_i_dict.keys():
                spk_i_dict[seg[2]] = spk_i
                spk_i += 1

            segment["spk_id"] = "spk" + str(spk_i_dict[seg[2]])
            segment["seg_begin"] = float("{:.2f}".format(seg[0]))
            segment["seg_end"] = float("{:.2f}".format(seg[0] + seg[1]))

            if segment["spk_id"] not in _speakers:
                _speakers[segment["spk_id"]] = {}
                _speakers[segment["spk_id"]]["spk_id"] = segment["spk_id"]
                _speakers[segment["spk_id"]]["duration"] = float(
                    "{:.2f}".format(seg[1])
                )
                _speakers[segment["spk_id"]]["nbr_seg"] = 1
            else:
                _speakers[segment["spk_id"]]["duration"] += seg[1]
                _speakers[segment["spk_id"]]["nbr_seg"] += 1
                _speakers[segment["spk_id"]]["duration"] = float(
                    "{:.2f}".format(_speakers[segment["spk_id"]]["duration"])
                )

            _segments.append(segment)
            seg_id += 1

        json["speakers"] = list(_speakers.values())
        json["segments"] = _segments
        return json

    def run(self, audioFile, speaker_count: int = None, max_speaker: int = None):
        self.log.debug(f"Starting diarization on file {audioFile}")
        try:
            start_time = time.time()
            self.log.debug(
                "Extracting features ... (t={:.2f}s)".format(time.time() - start_time)
            )
            feats = self.compute_feat_Librosa(audioFile)
            nFeatures = feats.shape[0]
            duration = nFeatures * self.frame_shift_s
            self.log.debug(
                "Computing SAD Mask ... (t={:.2f}s)".format(time.time() - start_time)
            )
            maskSAD = self.computeVAD_WEBRTC(self.data, self.sr, nFeatures)
            maskUEM = np.ones([1, nFeatures])

            mask = np.logical_and(maskUEM, maskSAD)
            mask = mask[0][0:nFeatures]
            nSpeechFeatures = np.sum(mask)
            speechMapping = np.zeros(nFeatures)
            # you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
            # so that we don't lose features on the way
            speechMapping[np.nonzero(mask)] = np.arange(1, nSpeechFeatures + 1)
            data = feats[np.where(mask == 1)]
            del feats

            self.log.debug(
                "Computing segment table ... (t={:.2f}s)".format(
                    time.time() - start_time
                )
            )
            segmentTable = pybk.getSegmentTable(
                mask, speechMapping, self.seg_length, self.seg_increment, self.seg_rate
            )
            numberOfSegments = np.size(segmentTable, 0)
            self.log.debug(f"Number of segment: {numberOfSegments}")

            if numberOfSegments == 1:
                self.log.debug(f"Single segment: returning")
                return [[0, duration, 1], [duration, -1, -1]]

            # create the KBM
            # set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
            windowRate = np.floor(
                (nSpeechFeatures - self.windowLength)
                / self.minimumNumberOfInitialGaussians
            )
            if windowRate > self.maximumKBMWindowRate:
                windowRate = self.maximumKBMWindowRate
            elif windowRate == 0:
                windowRate = 1

            poolSize = np.floor((nSpeechFeatures - self.windowLength) / windowRate)
            if self.useRelativeKBMsize:
                kbmSize = int(np.floor(poolSize * self.relKBMsize))
            else:
                kbmSize = int(self.kbmSize)

            # Training pool of',int(poolSize),'gaussians with a rate of',int(windowRate),'frames'
            self.log.debug(
                "Training KBM ... (t={:.2f}s)".format(time.time() - start_time)
            )
            kbm, gmPool = pybk.trainKBM(data, self.windowLength, windowRate, kbmSize)

            #'Selected',kbmSize,'gaussians from the pool'
            Vg = pybk.getVgMatrix(data, gmPool, kbm, self.topGaussiansPerFrame)

            #'Computing binary keys for all segments... '
            self.log.debug(
                "Computing binary keys ... (t={:.2f}s)".format(time.time() - start_time)
            )
            segmentBKTable, segmentCVTable = pybk.getSegmentBKs(
                segmentTable, kbmSize, Vg, self.bitsPerSegmentFactor, speechMapping
            )

            #'Performing initial clustering... '
            self.log.debug(
                "Performing initial clustering ... (t={:.2f}s)".format(
                    time.time() - start_time
                )
            )

            #'Selecting best clustering...'
            # self.bestClusteringCriterion == 'spectral':
            self.log.debug(
                "Selecting best clustering ... (t={:.2f}s)".format(
                    time.time() - start_time
                )
            )
            bestClusteringID = (
                pybk.getSpectralClustering(
                    self.metric_clusteringSelection,
                    self.N_init,
                    segmentBKTable,
                    segmentCVTable,
                    speaker_count,
                    self.sigma,
                    self.percentile,
                    max_speaker if max_speaker is not None else self.maxNrSpeakers,
                    random_state=self.seed,
                )
                + 1
            )

            if self.resegmentation and np.size(np.unique(bestClusteringID), 0) > 1:
                self.log.debug(
                    "Performing resegmentation ... (t={:.2f}s)".format(
                        time.time() - start_time
                    )
                )
                (
                    finalClusteringTableResegmentation,
                    finalSegmentTable,
                ) = pybk.performResegmentation(
                    data,
                    speechMapping,
                    mask,
                    bestClusteringID,
                    segmentTable,
                    self.modelSize,
                    self.nbIter,
                    self.smoothWin,
                    nSpeechFeatures,
                )
                self.log.debug(
                    "Get segments ... (t={:.2f}s)".format(time.time() - start_time)
                )
                segments = self.getSegments(
                    self.frame_shift_s,
                    finalSegmentTable,
                    np.squeeze(finalClusteringTableResegmentation),
                    duration,
                )
            else:
                return [[0, duration, 1], [duration, -1, -1]]

            self.log.info(
                "Speaker Diarization took %d[s] with a speed %0.2f[xRT]"
                % (
                    int(time.time() - start_time),
                    float(int(time.time() - start_time) / duration),
                )
            )
        except ValueError as v:
            self.log.error(v)
            raise ValueError(
                "Speaker diarization failed during processing the speech signal"
            )
        except Exception as e:
            self.log.error(e)
            raise Exception(
                "Speaker diarization failed during processing the speech signal"
            )
        segments = self.format_response(segments)
        self.log.debug(segments)
        return segments
