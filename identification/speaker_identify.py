import os
import torch
import torchaudio
import time
import glob
import subprocess
import json
import werkzeug
import memory_tempfile
from collections import defaultdict
from tqdm import tqdm
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client import models, QdrantClient
import speechbrain
if speechbrain.__version__ >= "1.0.0":
   from speechbrain.inference.speaker import EncoderClassifier
else:
   from speechbrain.pretrained import EncoderClassifier


class SpeakerIdentifier:
    # Define class-level constants
    _FOLDER_WAV = os.environ.get("SPEAKER_SAMPLES_FOLDER", "/opt/speaker_samples")
    _can_identify_twice_the_same_speaker = os.environ.get("CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER", "1").lower() in ["true", "1", "yes"]
    _UNKNOWN = "<<UNKNOWN>>"
    _RECREATE_COLLECTION = os.getenv("QDRANT_RECREATE_COLLECTION", "False").lower() in ["true", "1", "yes"]


    def __init__(self, device=None, log=None):
        self.device = device or self._get_device()
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.qdrant_port = os.getenv("QDRANT_PORT")
        self.qdrant_client = QdrantClient(url=f"http://{self.qdrant_host}:{self.qdrant_port}") if (self.qdrant_host and self.qdrant_port) else None
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME")
        self._embedding_model = None
        self.log = log

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def initialize_speaker_identification(
        self,
        max_duration=60 * 3,
        sample_rate=16_000,
    ):
        # Check if speaker identification is enabled
        if not self.is_speaker_identification_enabled() :
            if self.log: self.log.info(f"Speaker identification is disabled")
            return
        
        # Raise error if Qdrant client is not set
        elif self.qdrant_client is None:
            raise EnvironmentError(
                "Qdrant client is not set. Please ensure that the environment variables 'QDRANT_HOST' "
                "and 'QDRANT_PORT' are set to enable speaker identification."
            )
                
        if self._embedding_model is None:
            tic = time.time()
            self._embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device":self.device}
            )
            if self.log: self.log.info(f"Speaker identification model loaded in {time.time() - tic:.3f} seconds on {self.device}")
        
        if self.log: self.log.info(f"Speaker identification is enabled")
        
        # Check if the collection exists
        if self.qdrant_client.collection_exists(collection_name=self.qdrant_collection):
            if self._RECREATE_COLLECTION:
                if self.log:
                    self.log.info(f"Deleting existing collection: {self.qdrant_collection}")
                self.qdrant_client.delete_collection(collection_name=self.qdrant_collection)
            else:
                if self.log:
                    self.log.info(f"Using existing collection: {self.qdrant_collection}")
                speakers = self._get_db_speaker_names()
                if self.log:
                    self.log.info(f"Speaker identification initialized with {len(speakers)} speakers")
                return

        # Create collection
        if self.log:
            self.log.info(f"Creating collection: {self.qdrant_collection}")
        self.qdrant_client.create_collection(
            collection_name=self.qdrant_collection,
            vectors_config=VectorParams(
                size=192,
                distance=Distance.COSINE
            ),
        )

        speakers = list(self._get_speaker_names())
        points = []  # List to store points for Qdrant upsert
        for speaker_idx,speaker_name in enumerate(tqdm(speakers, desc="Compute ref. speaker embeddings")):
            audio_files = self._get_speaker_sample_files(speaker_name)
            assert len(audio_files) > 0, f"No audio files found for speaker {speaker_name}"
            
            audio = None
            max_samples = max_duration * sample_rate
            for audio_file in audio_files:
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
                if clip_audio.shape[1] > max_samples:
                    clip_audio = clip_audio[:, :max_samples]
                if audio is None:
                    audio = clip_audio
                else:
                    audio = torch.cat((audio, clip_audio), 1)
                # Update maximum number of remaining samples
                max_samples -= clip_audio.shape[1]
                if max_samples <= 0:
                    break

            spk_embed = self.compute_embedding(audio)
            # Note: it is important to save the embeddings on the CPU (to be able to load them on the CPU later on)
            spk_embed = spk_embed.cpu()
            # Prepare point for Qdrant
            point = PointStruct(
                id=speaker_idx+1,
                vector=spk_embed[0].flatten(),  # Convert to 1D list for Qdrant [[[1, 2, 3, ...]]] -> [1, 2, 3, ...]
                payload={"person": speaker_name.strip()}
            )

            points.append(point)
        
        # Upsert all points to Qdrant in one go
        if points:
                self.qdrant_client.upsert(
                collection_name=self.qdrant_collection,
                wait=True,
                points=points
            )

        if self.log: self.log.info(f"Speaker identification initialized with {len(speakers)} speakers")

    # Create a method to check if speaker identification is enabled
    def is_speaker_identification_enabled(self):
        return os.path.isdir(self._FOLDER_WAV)
    
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

    def check_wav_16khz_mono(self,wavfile):
        """
        Returns True if a wav file is 16khz and single channel
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


    def compute_embedding(self,audio, min_len = 640):
        """
        Compute speaker embedding from audio

        Args:
            audio (torch.Tensor): audio waveform
        """
        assert self._embedding_model is not None, "Speaker identification model not initialized"
        # The following is to avoid a failure on too short audio (less than 640 samples = 40ms at 16kHz)
        if audio.shape[-1] < min_len:
            audio = torch.cat([audio, torch.zeros(audio.shape[0], min_len - audio.shape[-1])], dim=-1)
        return self._embedding_model.encode_batch(audio)


    def _get_db_speaker_names(self, batch_size=100):
        all_points = []
        offset = None  # Start without any offset
        
        while True:
            # Scroll request with batch_size
            response, next_offset = self.qdrant_client.scroll(
                collection_name=self.qdrant_collection,
                offset=offset,  # Use offset to get the next batch
                limit=batch_size,
                with_payload=True,
            )
            
            all_points.extend(response)  # Collect the points

            # Break the loop if no more points are available
            if next_offset is None:
                break

            # Update the offset for the next iteration
            offset = next_offset

        return [point.payload.get("person") for point in all_points]


    def _get_db_speaker_name(self,speaker_id):
        # Retrieve the point from Qdrant
        response = self.qdrant_client.retrieve(
            collection_name=self.qdrant_collection,
            ids=[speaker_id],
        )
        # Extract the 'person' payload from the response
        if response :
            return response[0].payload.get('person')
    
    
    def _get_db_speaker_id(self,speaker_name):
        # Get qdrant id corresponding to speaker_name
        response = self.qdrant_client.scroll(
            collection_name=self.qdrant_collection,
            scroll_filter = models.Filter(
            must=[
                    models.FieldCondition(
                        key="person",
                        match=models.MatchValue(value=speaker_name),
                    )
                ])
        )
        # Extract the id
        points = response[0] if response else []

        if len(points) == 0:
            raise ValueError(f"Person with name '{speaker_name}' not found in the Qdrant collection.")
        if len(points) > 1:
            raise ValueError(f"Multiple persons with the name '{speaker_name}' found. Ensure uniqueness.")
        return points[0].id
 
    def _get_speaker_sample_files(self,speaker_name):
        if os.path.isdir(os.path.join(self._FOLDER_WAV, speaker_name)):
            audio_files = sorted(glob.glob(os.path.join(self._FOLDER_WAV, speaker_name, '*')))
        else:
            prefix = os.path.join(self._FOLDER_WAV, speaker_name)
            audio_files = glob.glob(prefix + '.*')
            audio_files = [file for file in audio_files if os.path.splitext(file)[0] == prefix]
            assert len(audio_files) == 1
        return audio_files
    
    def _get_speaker_names(self):
        assert os.path.isdir(self._FOLDER_WAV)
        for root, dirs, files in os.walk(self._FOLDER_WAV):
            if root == self._FOLDER_WAV:
                for file in files:
                    yield os.path.splitext(file)[0]
            else:
                yield os.path.basename(root.rstrip("/"))

    def speaker_identify(
        self,
        audio,
        speaker_names,
        segments,
        exclude_speakers,
        min_similarity=0.5,
        sample_rate=16_000,
        limit_duration=3 * 60,
        spk_tag = None,
        ):
        """
        Run speaker identification on given segments of an audio

        Args:
            audio (torch.Tensor): audio waveform
            speaker_names (list): list of reference speaker names
            segments (list): list of segments to analyze (tuples of start and end times in seconds)
            exclude_speakers (list): list of speaker names to exclude
            min_similarity (float): minimum similarity to consider a speaker match
                The default value 0.25 was taken from https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/inference/speaker.py#L61
            sample_rate (int): audio sample rate
            limit_duration (int): maximum duration (in seconds) of speech to identify a speaker (the first seconds of speech will be used, the other will be ignored)
            spk_tag: information for the logger

        Returns:
            str: identified speaker name
            float: similarity score
        """
        tic = time.time()

        assert len(speaker_names) > 0

        votes = defaultdict(int)

        # Sort segments by duration (longest first)
        segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)
        assert len(segments)

        total_duration = sum([end - start for (start, end) in segments])

        # Glue all the speaker segments up to a certain length
        audio_selection = None
        limit_samples = limit_duration * sample_rate
        for start, end in segments:
            start = int(start * sample_rate)
            end = int(end * sample_rate)
            if end - start > limit_samples:
                end = start + limit_samples
            
            clip = audio[:, start:end]
            if audio_selection is None:
                audio_selection = clip
            else:
                audio_selection = torch.cat((audio_selection, clip), 1)
            limit_samples -= (end - start)
            if limit_samples <= 0:
                break

        embedding_audio = self.compute_embedding(audio_selection)

        # Search for similar embeddings in Qdrant
        results = self.qdrant_client.search(self.qdrant_collection, embedding_audio[0].flatten())
        
        for result in results:
            speaker_name = result.payload["person"]
            
            # Check if the speaker is in the exclude list
            if speaker_name in exclude_speakers:
                continue
            
            # Use the similarity score returned by Qdrant
            score = result.score  # Directly get the similarity score from the result
            if (score >= min_similarity) and (speaker_name in speaker_names):
                votes[speaker_name] += score


        score = None
        if not votes:
            argmax_speaker = self._UNKNOWN
        else:
            argmax_speaker = max(votes, key=votes.get)    
            score = votes[argmax_speaker]

        if self.log:
            self.log.info(
                f"Speaker recognition {spk_tag} -> {argmax_speaker} (done in {time.time() - tic:.3f} seconds, on {audio_selection.shape[1] / sample_rate:.3f} seconds of audio out of {total_duration:.3f})"
            )

        return argmax_speaker, score
    
    def check_speaker_specification(
        self,
        speakers_spec,     
        ):
        """
        Check and convert speaker specification to list of speaker names

        Args:
            speakers_spec (str, list): speaker specification
        """

        if speakers_spec and not self.is_speaker_identification_enabled():
            raise RuntimeError("Speaker identification is disabled (no reference speakers)")

        # Read list / dictionary
        if isinstance(speakers_spec, str):
            speakers_spec = speakers_spec.strip()
            if speakers_spec:
                if speakers_spec == "*":
                    # Wildcard: all speakers
                    speakers_spec = self._get_db_speaker_names()
                elif speakers_spec[0] in "[{":
                    try:
                        speakers_spec = json.loads(speakers_spec)
                    except Exception as err:
                        if "|" in speakers_spec:
                            speakers_spec = speakers_spec.split("|")
                        else:
                            raise ValueError(f"Unsupported reference speaker specification: {speakers_spec} (except empty string, \"*\", or \"speaker1|speaker2|...|speakerN\", or \"[\"speaker1\", \"speaker2\", ..., \"speakerN\"]\")") from err
                    if isinstance(speakers_spec, dict):
                        speakers_spec = [speakers_spec]
                else:
                    speakers_spec = speakers_spec.split("|")

        # Convert to list of speaker names
        if not speakers_spec:
            return []

        if not isinstance(speakers_spec, list):
            raise ValueError(f"Unsupported reference speaker specification of type {type(speakers_spec)}: {speakers_spec}")

        speakers_spec = [s for s in speakers_spec if s]
        all_speaker_names = None
        speaker_names = []
        for item in speakers_spec:
            if isinstance(item, int):
                speaker_names.append(self._get_db_speaker_name(item))

            elif isinstance(item, dict):
                # Should we really keep this format ?
                start = item['start']
                end = item['end']
                items=[]
                for id in range(start,end+1):
                    speaker_names.append(self._get_db_speaker_name(id))
                
            
            elif isinstance(item, str):
                if all_speaker_names is None:
                    all_speaker_names = self._get_db_speaker_names()
                if item not in all_speaker_names:
                    raise ValueError(f"Unknown speaker name '{item}'")
                speaker_names.append(item)
            
            else:
                raise ValueError(f"Unsupported reference speaker specification of type {type(item)} (in list): {speakers_spec}")

        return speaker_names
        

    def speaker_identify_given_diarization(
        self,
        audioFile, 
        diarization, 
        speakers_spec="*",
        options={}):
        """
        Run speaker identification on given diarized audio file

        Args:
            audioFile (str): path to audio file
            diarization (dict): diarization result
            speakers_spec (list): list of reference speaker ids or ranges (e.g. [1, 2, {"start": 3, "end": 5}])
            options (dict): optional options (e.g. {"min_similarity": 0.25, "limit_duration": 60})
        """

        speaker_names = self.check_speaker_specification(speakers_spec)

        if not speaker_names:
            return diarization

        if self.log:
            full_tic = time.time()
            self.log.info(f"Running speaker identification with {len(speaker_names)} reference speakers")
        
        if isinstance(audioFile, werkzeug.datastructures.file_storage.FileStorage):
            tempfile = memory_tempfile.MemoryTempfile(filesystem_types=['tmpfs', 'shm'], fallback=True)
            if self.log:
                self.log.info(f"Using temporary folder {tempfile.gettempdir()}")

            with tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
                audioFile.save(ntf.name)
                return self.speaker_identify_given_diarization(ntf.name, diarization, speaker_names)

        speaker_tags = []
        speaker_segments = {}
        common = []
        speaker_map = {}
        speaker_surnames = {}

        for segment in diarization["segments"]:

            start = segment["seg_begin"]
            end = segment["seg_end"]
            speaker = segment["spk_id"]
            common.append([start, end, speaker])

            # find different speakers
            if speaker not in speaker_tags:
                speaker_tags.append(speaker)
                speaker_map[speaker] = speaker
                speaker_segments[speaker] = []

            speaker_segments[speaker].append([start, end])
        
        audio, sample_rate = torchaudio.load(audioFile)
        # This should be OK, since this is enforced by the diarization API
        assert sample_rate == 16_000, f"Unsupported sample rate {sample_rate} (only 16kHz is supported)"

        # Process the speakers with the longest speech turns first
        def speech_duration(spk):
            return sum([end - start for (start, end) in speaker_segments[spk]])
        already_identified = []
        speaker_id_scores = {}
        for spk_tag in sorted(speaker_segments.keys(), key=speech_duration, reverse=True):
            spk_segments = speaker_segments[spk_tag]
            
            spk_name, spk_id_score = self.speaker_identify(
                audio, speaker_names, spk_segments,
                # TODO : do we really want to avoid that 2 speakers are the same ?
                #        and if we do, not that it's not invariant to the order in which segments are taken (so we should choose a somewhat optimal order)
                exclude_speakers=([] if self._can_identify_twice_the_same_speaker else already_identified),
                spk_tag=spk_tag,
                **options
            )
            if spk_name == self._UNKNOWN:
                speaker_map[spk_tag] = spk_tag
            else:
                already_identified.append(spk_name)
                speaker_map[spk_tag] = spk_name
                speaker_id_scores[spk_name] = spk_id_score

        result = {}
        _segments = []
        _speakers = {}
        speaker_surnames = {}
        for iseg, segment in enumerate(diarization["segments"]):
            start = segment["seg_begin"]
            end = segment["seg_end"]
            speaker = segment["spk_id"]

            # Convert speaker names to spk1, spk2, etc.
            if speaker not in speaker_surnames:
                speaker_surnames[speaker] = (
                    speaker  # "spk"+str(len(speaker_surnames)+1)
                )
            speaker = speaker_surnames[speaker]
            speaker_name = speaker_map[speaker]
            if speaker_name == self._UNKNOWN:
                speaker_name = speaker

            segment["spk_id"] = speaker_name

            _segments.append(segment)

            if speaker_name not in _speakers:
                _speakers[speaker_name] = {"spk_id": speaker_name}
                if speaker_name in speaker_id_scores:
                    _speakers[speaker_name]["spk_id_score"] = round(speaker_id_scores[speaker_name], 3)
                _speakers[speaker_name]["duration"] = round(end - start, 3)
                _speakers[speaker_name]["nbr_seg"] = 1

            else:
                _speakers[speaker_name]["duration"] += round(end - start, 3)
                _speakers[speaker_name]["nbr_seg"] += 1

        result["speakers"] = list(_speakers.values())
        result["segments"] = _segments

        if self.log:
            self.log.info(f"Speaker identification done in {time.time() - full_tic:.3f} seconds")

        return result