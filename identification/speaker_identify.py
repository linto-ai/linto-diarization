import glob
import json
import os
import time

import memory_tempfile
import torch
import torchaudio
import werkzeug
from tqdm import tqdm

from identification.embedding import EmbeddingBackend
from identification.qdrant_store import PointStruct, QdrantStore
from identification.spkid_core import (
    MODEL_DIM,
    MODEL_ID,
    aggregate_speaker_votes,
    check_speaker_spec_dict,
    resolve_collections,
    resolve_min_similarity,
)


class SpeakerIdentifier:
    # Define class-level constants
    _FOLDER_WAV = os.environ.get("SPEAKER_SAMPLES_FOLDER", "/opt/speaker_samples")
    _can_identify_twice_the_same_speaker = os.environ.get("CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER", "1").lower() in ["true", "1", "yes"]
    _UNKNOWN = "<<UNKNOWN>>"
    _RECREATE_COLLECTION = os.getenv("QDRANT_RECREATE_COLLECTION", "False").lower() in ["true", "1", "yes"]


    def __init__(self, device=None, log=None):
        self.log = log
        self.embedding = EmbeddingBackend(device=device, log=log)
        self.store = QdrantStore.from_env(log=log)
        # Legacy single-collection mode (filesystem bootstrap)
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME")

    @property
    def device(self):
        return self.embedding.device

    # Create a method to check if speaker identification is enabled
    def is_speaker_identification_enabled(self):
        return self.store.enabled

    def initialize_speaker_identification(
        self,
        max_duration=60 * 3,
        sample_rate=16_000,
    ):
        # Check if speaker identification is enabled
        if not self.is_speaker_identification_enabled():
            if self.log: self.log.info(f"Speaker identification is disabled (QDRANT_HOST is not set)")
            return

        self.embedding.load()

        if self.log: self.log.info(f"Speaker identification is enabled")

        # Legacy filesystem enrollment (deprecated): populate a single collection
        # from reference speaker samples at startup
        if os.path.isdir(self._FOLDER_WAV):
            if self.qdrant_collection:
                self._bootstrap_legacy_collection(max_duration=max_duration, sample_rate=sample_rate)
            elif self.log:
                self.log.warning(
                    f"Speaker samples folder {self._FOLDER_WAV} exists but QDRANT_COLLECTION_NAME is not set: "
                    "skipping legacy filesystem enrollment"
                )

    def _bootstrap_legacy_collection(
        self,
        max_duration=60 * 3,
        sample_rate=16_000,
    ):
        # Check if the collection exists
        if self.store.collection_exists(self.qdrant_collection):
            if self._RECREATE_COLLECTION:
                if self.log:
                    self.log.info(f"Deleting existing collection: {self.qdrant_collection}")
                self.store.drop(self.qdrant_collection)
            else:
                if self.log:
                    self.log.info(f"Using existing collection: {self.qdrant_collection}")
                speakers = self._get_db_speaker_names()
                if self.log:
                    self.log.info(f"Speaker identification initialized with {len(speakers)} speakers")
                return

        # Create collection
        self.store.ensure_collection(self.qdrant_collection, dim=MODEL_DIM)

        speakers = list(self._get_speaker_names())
        points = []  # List to store points for Qdrant upsert
        for speaker_idx, speaker_name in enumerate(tqdm(speakers, desc="Compute ref. speaker embeddings")):
            audio_files = self._get_speaker_sample_files(speaker_name)
            assert len(audio_files) > 0, f"No audio files found for speaker {speaker_name}"

            audio, _, _ = self.embedding.load_audio_concat(
                audio_files, max_duration=max_duration, sample_rate=sample_rate
            )

            spk_embed = self.embedding.compute_embedding(audio)
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
            self.store.upsert_points(self.qdrant_collection, points)

        if self.log: self.log.info(f"Speaker identification initialized with {len(speakers)} speakers")

    def compute_embedding(self, audio, min_len=640):
        """
        Compute speaker embedding from audio

        Args:
            audio (torch.Tensor): audio waveform
        """
        return self.embedding.compute_embedding(audio, min_len=min_len)

    def _get_db_speaker_names(self, batch_size=100):
        all_points = []
        offset = None  # Start without any offset

        while True:
            # Scroll request with batch_size
            response, next_offset = self.store.client.scroll(
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


    def _get_db_speaker_name(self, speaker_id):
        # Retrieve the point from Qdrant
        response = self.store.client.retrieve(
            collection_name=self.qdrant_collection,
            ids=[speaker_id],
        )
        # Extract the 'person' payload from the response
        if response:
            return response[0].payload.get('person')


    def _get_db_speaker_id(self, speaker_name):
        from qdrant_client import models

        # Get qdrant id corresponding to speaker_name
        response = self.store.client.scroll(
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

    def _get_speaker_sample_files(self, speaker_name):
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

    @staticmethod
    def _select_speaker_audio(audio, segments, sample_rate=16_000, limit_duration=3 * 60):
        """
        Glue the segments of a diarized speaker up to a certain length

        Args:
            audio (torch.Tensor): full audio waveform
            segments (list): list of segments (tuples of start and end times in seconds)
            sample_rate (int): audio sample rate
            limit_duration (int): maximum duration (in seconds) of speech to keep

        Returns:
            (audio_selection, total_duration)
        """
        # Sort segments by duration (longest first)
        segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)
        assert len(segments)

        total_duration = sum([end - start for (start, end) in segments])

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

        return audio_selection, total_duration

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
        Run speaker identification on given segments of an audio (legacy single-collection mode)

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

        audio_selection, total_duration = self._select_speaker_audio(
            audio, segments, sample_rate=sample_rate, limit_duration=limit_duration
        )

        embedding_audio = self.embedding.compute_embedding(audio_selection)

        # Search for similar embeddings in Qdrant
        results = self.store.search(self.qdrant_collection, embedding_audio[0].flatten())

        votes = {}
        for result in results:
            speaker_name = result.payload["person"]

            # Check if the speaker is in the exclude list
            if speaker_name in exclude_speakers:
                continue

            # Use the similarity score returned by Qdrant
            score = result.score  # Directly get the similarity score from the result
            if (score >= min_similarity) and (speaker_name in speaker_names):
                votes[speaker_name] = votes.get(speaker_name, 0) + score


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

    def _speaker_identify_multi(
        self,
        audio,
        segments,
        collections,
        allowed_speaker_ids,
        exclude_speaker_ids,
        min_similarity,
        sample_rate=16_000,
        limit_duration=3 * 60,
        search_limit=10,
        spk_tag=None,
        ):
        """
        Run speaker identification on given segments of an audio, against several
        Qdrant collections (new multi-collection mode)

        Args:
            audio (torch.Tensor): audio waveform
            segments (list): list of segments to analyze (tuples of start and end times in seconds)
            collections (list): list of (usable) Qdrant collection names
            allowed_speaker_ids (set | None): restriction on payload speaker_id (None = no restriction)
            exclude_speaker_ids (list): speaker ids to exclude (already identified)
            min_similarity (float): minimum similarity to consider a speaker match
            sample_rate (int): audio sample rate
            limit_duration (int): maximum duration (in seconds) of speech to identify a speaker
            search_limit (int): maximum number of hits per collection
            spk_tag: information for the logger

        Returns:
            (speaker_id, name, score) of the winner, or (None, None, None)
        """
        tic = time.time()

        audio_selection, total_duration = self._select_speaker_audio(
            audio, segments, sample_rate=sample_rate, limit_duration=limit_duration
        )

        embedding_audio = self.embedding.compute_embedding(audio_selection)
        vector = embedding_audio[0].flatten()

        # Search for similar embeddings in each collection and merge the hits
        hits = []
        for collection in collections:
            try:
                results = self.store.search(collection, vector, limit=search_limit)
            except Exception as err:
                if self.log:
                    self.log.warning(f"Speaker identification: search failed on collection '{collection}': {err}")
                continue
            hits.extend([(result.score, result.payload or {}) for result in results])

        speaker_id, name, score = aggregate_speaker_votes(
            hits,
            min_similarity,
            allowed_speaker_ids=allowed_speaker_ids,
            exclude_speaker_ids=exclude_speaker_ids,
        )

        if self.log:
            self.log.info(
                f"Speaker recognition {spk_tag} -> {name if speaker_id else self._UNKNOWN} (done in {time.time() - tic:.3f} seconds, on {audio_selection.shape[1] / sample_rate:.3f} seconds of audio out of {total_duration:.3f})"
            )

        return speaker_id, name, score

    def check_speaker_specification(
        self,
        speakers_spec,
        ):
        """
        Check and normalize a speaker specification

        Args:
            speakers_spec (str, list, dict): speaker specification
                str / list: legacy mode (collection set by QDRANT_COLLECTION_NAME,
                    wildcard "*", "speaker1|speaker2|...", or JSON list of names)
                dict: multi-collection mode
                    {"collections": [...], "speakers": "*" | [speaker_ids], "minSimilarity": float | None}

        Returns:
            list of speaker names (legacy mode), normalized dict (multi-collection mode),
            or an empty value if identification is not requested
        """

        if speakers_spec and not self.is_speaker_identification_enabled():
            raise RuntimeError("Speaker identification is disabled (QDRANT_HOST is not set)")

        # Multi-collection specification
        if isinstance(speakers_spec, dict):
            if "collections" in speakers_spec:
                return check_speaker_spec_dict(speakers_spec)
            # Legacy range dict ({"start": ..., "end": ...})
            speakers_spec = [speakers_spec]

        # Read list / dictionary
        if isinstance(speakers_spec, str):
            speakers_spec = speakers_spec.strip()
            if speakers_spec:
                if speakers_spec == "*":
                    # Wildcard: all speakers
                    self._check_legacy_collection_configured()
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
                        if "collections" in speakers_spec:
                            # Multi-collection specification passed as a JSON string
                            return check_speaker_spec_dict(speakers_spec)
                        speakers_spec = [speakers_spec]
                else:
                    speakers_spec = speakers_spec.split("|")

        # Convert to list of speaker names
        if not speakers_spec:
            return []

        if not isinstance(speakers_spec, list):
            raise ValueError(f"Unsupported reference speaker specification of type {type(speakers_spec)}: {speakers_spec}")

        self._check_legacy_collection_configured()

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

    def _check_legacy_collection_configured(self):
        if not self.qdrant_collection:
            raise RuntimeError(
                "Legacy speaker identification requires QDRANT_COLLECTION_NAME to be set "
                "(or use the multi-collection specification)"
            )

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
            speakers_spec (str, list, dict): speaker specification (see check_speaker_specification)
            options (dict): optional options (e.g. {"limit_duration": 60})
        """

        speakers_spec = self.check_speaker_specification(speakers_spec)

        if not speakers_spec:
            return diarization

        multi_mode = isinstance(speakers_spec, dict)
        if multi_mode:
            collections = resolve_collections(
                speakers_spec["collections"],
                self.store.collection_exists,
                self.store.get_collection_model_id,
                MODEL_ID,
                log=self.log,
            )
            if not collections:
                if self.log:
                    self.log.warning("Speaker identification: no usable collection, returning diarization unchanged")
                return diarization
            min_similarity = resolve_min_similarity(speakers_spec.get("minSimilarity"))
            allowed_speaker_ids = None if speakers_spec["speakers"] == "*" else set(speakers_spec["speakers"])
        else:
            speaker_names = speakers_spec

        if self.log:
            full_tic = time.time()
            if multi_mode:
                self.log.info(f"Running speaker identification on {len(collections)} collection(s) (min_similarity={min_similarity})")
            else:
                self.log.info(f"Running speaker identification with {len(speaker_names)} reference speakers")

        if isinstance(audioFile, werkzeug.datastructures.file_storage.FileStorage):
            tempfile = memory_tempfile.MemoryTempfile(filesystem_types=['tmpfs', 'shm'], fallback=True)
            if self.log:
                self.log.info(f"Using temporary folder {tempfile.gettempdir()}")

            with tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
                audioFile.save(ntf.name)
                return self.speaker_identify_given_diarization(ntf.name, diarization, speakers_spec)

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

            if multi_mode:
                # In multi-collection mode the threshold comes from the specification (or env)
                multi_options = {k: v for k, v in options.items() if k != "min_similarity"}
                spk_id, spk_name, spk_id_score = self._speaker_identify_multi(
                    audio, spk_segments,
                    collections=collections,
                    allowed_speaker_ids=allowed_speaker_ids,
                    exclude_speaker_ids=([] if self._can_identify_twice_the_same_speaker else already_identified),
                    min_similarity=min_similarity,
                    spk_tag=spk_tag,
                    **multi_options
                )
                if spk_id is None:
                    spk_name = self._UNKNOWN
                else:
                    already_identified.append(spk_id)
            else:
                spk_name, spk_id_score = self.speaker_identify(
                    audio, speaker_names, spk_segments,
                    # TODO : do we really want to avoid that 2 speakers are the same ?
                    #        and if we do, not that it's not invariant to the order in which segments are taken (so we should choose a somewhat optimal order)
                    exclude_speakers=([] if self._can_identify_twice_the_same_speaker else already_identified),
                    spk_tag=spk_tag,
                    **options
                )
                if spk_name != self._UNKNOWN:
                    already_identified.append(spk_name)

            if spk_name == self._UNKNOWN:
                speaker_map[spk_tag] = spk_tag
            else:
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
