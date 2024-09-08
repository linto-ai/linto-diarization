import speechbrain
if speechbrain.__version__ >= "1.0.0":
   from speechbrain.inference.speaker import EncoderClassifier
else:
   from speechbrain.pretrained import EncoderClassifier
import os
from collections import defaultdict
import torch
import torchaudio
import time
import subprocess
import memory_tempfile
import werkzeug
import pickle as pkl
import hashlib
import sqlite3
import glob
import json
from tqdm import tqdm

device = os.environ.get("DEVICE_IDENTIFICATION", os.environ.get("DEVICE", None))
if device is None:
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

_can_identify_twice_the_same_speaker = os.environ.get("CAN_IDENTIFY_TWICE_THE_SAME_SPEAKER", "1").lower() in ["true", "1", "yes"]

_embedding_model = None

# Constants (that could be env variables)
_FOLDER_WAV = os.environ.get("SPEAKER_SAMPLES_FOLDER", "/opt/speaker_samples")
_FOLDER_INTERNAL = os.environ.get("SPEAKER_PRECOMPUTED_FOLDER", "/opt/speaker_precomputed")
_FOLDER_EMBEDDINGS = f"{_FOLDER_INTERNAL}/embeddings"
_FILE_DATABASE = f"{_FOLDER_INTERNAL}/speakers_database"

_UNKNOWN = "<<UNKNOWN>>"


def initialize_speaker_identification(log):
    initialize_db(log)
    initialize_embeddings(log)


def is_speaker_identification_enabled():
    return os.path.isdir(_FOLDER_WAV)

# Create / update / check database
def initialize_db(log):
    if not is_speaker_identification_enabled():
        if log: log.info(f"Speaker identification is disabled")
        return
    if log: log.info(f"Speaker identification is enabled")
    os.makedirs(os.path.dirname(_FILE_DATABASE), exist_ok=True)
    # Create connection
    conn = sqlite3.connect(_FILE_DATABASE)
    cursor = conn.cursor()
    # Creating and inserting into table
    cursor.execute("""CREATE TABLE IF NOT EXISTS speaker_names (id integer UNIQUE, name TEXT UNIQUE)""")
    all_ids = list(_get_db_speaker_ids(cursor))
    all_names = _get_db_speaker_names(cursor)
    assert all_ids == list(range(1, len(all_ids)+1)), f"Speaker ids are not continuous"
    assert len(all_names) == len(all_ids), f"Speaker names are not unique"
    new_id = len(all_ids) + 1
    for speaker_name in _get_speaker_names():
        if speaker_name not in all_names:
            cursor.execute("INSERT OR IGNORE INTO speaker_names (id, name) VALUES (?, ?)", (
                new_id,
                speaker_name,
            ))
            new_id += 1
    conn.commit()
    conn.close()

def check_wav_16khz_mono(wavfile, log=None):
    """
    Returns True if a wav file is 16khz and single channel
    """
    try:
        signal, fs = torchaudio.load(wavfile)
    except:
        if log: log.info(f"Could not load {wavfile}")
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
    if log: log.info(f"File {wavfile} {reason}")


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

def initialize_embeddings(
    log = None,
    max_duration = 60 * 3,
    sample_rate = 16_000,
    ):
    """
    Pre-compute and store reference speaker embeddings

    Args:
        log (logging.Logger): optional logger
        max_duration (int): maximum duration (in seconds) of speech to use for speaker embeddings
        sample_rate (int): sample rate (of the embedding model)
    """
    if not is_speaker_identification_enabled():
        return
    
    global _embedding_model
    if _embedding_model is None:
        tic = time.time()
        _embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            # savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device":device}
        )
        if log: log.info(f"Speaker identification model loaded in {time.time() - tic:.3f} seconds on {device}")

    os.makedirs(_FOLDER_EMBEDDINGS, exist_ok=True)
    speakers = list(_get_speaker_names())
    for speaker_name in tqdm(speakers, desc="Compute ref. speaker embeddings"):
        embedding_file = _get_speaker_embedding_file(speaker_name)
        if os.path.isfile(embedding_file):
            try:
                with open(embedding_file, "rb") as f:
                    pkl.load(f)
                if log: log.info(f"Speaker {speaker_name} embedding already computed")
                continue
            except Exception as e:
                os.remove(embedding_file)
        audio_files = _get_speaker_sample_files(speaker_name)
        assert len(audio_files) > 0, f"No audio files found for speaker {speaker_name}"
        audio = None
        max_samples = max_duration * sample_rate
        for audio_file in audio_files:
            clip_audio = check_wav_16khz_mono(audio_file, log=log)
            if clip_audio is not None:
                clip_sample_rate = 16000
            else:
                if log: log.info(f"Converting audio file {audio_file} to single channel 16kHz WAV using ffmpeg...")
                converted_wavfile = os.path.join(
                    os.path.dirname(audio_file), "___{}.wav".format(os.path.splitext(os.path.basename(audio_file))[0])
                )
                convert_wavfile(audio_file, converted_wavfile)
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

        spk_embed = compute_embedding(audio)
        # Note: it is important to save the embeddings on the CPU (to be able to load them on the CPU later on)
        spk_embed = spk_embed.cpu()
        with open(embedding_file, "wb") as f:
            pkl.dump(spk_embed, f)
    if log: log.info(f"Speaker identification initialized with {len(speakers)} speakers")

def compute_embedding(audio, min_len = 640):
    """
    Compute speaker embedding from audio

    Args:
        audio (torch.Tensor): audio waveform
    """
    assert _embedding_model is not None, "Speaker identification model not initialized"
    # The following is to avoid a failure on too short audio (less than 640 samples = 40ms at 16kHz)
    if audio.shape[-1] < min_len:
        audio = torch.cat([audio, torch.zeros(audio.shape[0], min_len - audio.shape[-1])], dim=-1)
    return _embedding_model.encode_batch(audio)

def _get_db_speaker_ids(cursor=None):
    return _get_db_possible_values("id", cursor)

def _get_db_speaker_names(cursor=None):
    return _get_db_possible_values("name", cursor)

def _get_db_possible_values(name, cursor, check_unique=True):
    create_connection = (cursor is None)
    if create_connection:
        conn = sqlite3.connect(_FILE_DATABASE)
        cursor = conn.cursor()
    cursor.execute(f"SELECT {name} FROM speaker_names")
    values = cursor.fetchall()
    values = [value[0] for value in values]
    if check_unique:
        assert len(values) == len(set(values)), f"Values are not unique"
    else:
        values = list(set(values))
    if create_connection:
        conn.close()
    return values

def _get_db_speaker_name(speaker_id, cursor=None):
    return _get_db_speaker_attribute(speaker_id, "id", "name", cursor)

def _get_db_speaker_id(speaker_name, cursor=None):
    return _get_db_speaker_attribute(speaker_name, "name", "id", cursor)

def _get_db_speaker_attribute(value, orig, dest, cursor):
    create_connection = (cursor is None)
    if create_connection:
        conn = sqlite3.connect(_FILE_DATABASE)
        cursor = conn.cursor()
    item = cursor.execute(f"SELECT {dest} FROM speaker_names WHERE {orig} = '{value}'")
    item = item.fetchone()
    assert item, f"Speaker {orig} {value} not found"
    assert len(item) == 1, f"Speaker {orig} {value} not unique"
    value = item[0]
    if create_connection:
        conn.close()
    return value


def _get_speaker_embedding_file(speaker_name):
    hash = _get_speaker_hash(speaker_name)
    return os.path.join(_FOLDER_EMBEDDINGS, hash + '.pkl')

def _get_speaker_sample_files(speaker_name):
    if os.path.isdir(os.path.join(_FOLDER_WAV, speaker_name)):
        audio_files = sorted(glob.glob(os.path.join(_FOLDER_WAV, speaker_name, '*')))
    else:
        prefix = os.path.join(_FOLDER_WAV, speaker_name)
        audio_files = glob.glob(prefix + '.*')
        audio_files = [file for file in audio_files if os.path.splitext(file)[0] == prefix]
        assert len(audio_files) == 1
    return audio_files

_cached_speaker_hashes = {}
def _get_speaker_hash(speaker_name):
    """
    Return a hash depending on the speaker audio filenames
    """
    if speaker_name in _cached_speaker_hashes:
        return _cached_speaker_hashes[speaker_name]
    files = _get_speaker_sample_files(speaker_name)
    hashes = md5sum_files(files)
    hash = md5sum_object(hashes)
    prefix = speaker_name.replace(" ", "-").replace("/", "--").lower().strip("_-") + "_"
    hash = prefix + hash
    _cached_speaker_hashes[speaker_name] = hash
    return hash

def md5sum_files(filenames):
    """Compute the md5 hash of a file or a list of files"""
    single = False
    if not isinstance(filenames, list):
        single = True
        filenames = [filenames]
    p = subprocess.Popen(["md5sum", *filenames], stdout = subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    assert p.returncode == 0, f"Error running md5sum: {stderr}"
    md5_string = stdout.decode("utf-8").strip()
    md5_list = [f.split()[0] for f in md5_string.split("\n")]
    return md5_list[0] if single else md5_list


def md5sum_object(obj):
    return hashlib.md5(pkl.dumps(obj)).hexdigest()

def _get_speaker_names():
    assert os.path.isdir(_FOLDER_WAV)
    for root, dirs, files in os.walk(_FOLDER_WAV):
        if root == _FOLDER_WAV:
            for file in files:
                yield os.path.splitext(file)[0]
        else:
            yield os.path.basename(root.rstrip("/"))

def speaker_identify(
    audio,
    speaker_names,
    segments,
    exclude_speakers,
    min_similarity=0.5,
    sample_rate=16_000,
    limit_duration=3 * 60,
    log = None,
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
        log: logger
        spk_tag: information for the logger

    Returns:
        str: identified speaker name
        float: similarity score
    """
    tic = time.time()

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
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

    embedding_audio = compute_embedding(audio_selection)

    # Loop on the target speakers
    for speaker_name in speaker_names:
        if speaker_name in exclude_speakers:
            continue

        # Get speaker embedding
        with open(_get_speaker_embedding_file(speaker_name), "rb") as f:
            embedding_speaker = pkl.load(f)
        embedding_speaker = embedding_speaker.to(_embedding_model.device)

        # Compute score similarity
        score = similarity(embedding_speaker, embedding_audio)
        score = score.item()
        if score >= min_similarity:
            votes[speaker_name] += score

    score = None
    if not votes:
        argmax_speaker = _UNKNOWN
    else:
        argmax_speaker = max(votes, key=votes.get)    
        score = votes[argmax_speaker]

    if log:
        log.info(
            f"Speaker recognition {spk_tag} -> {argmax_speaker} (done in {time.time() - tic:.3f} seconds, on {audio_selection.shape[1] / sample_rate:.3f} seconds of audio out of {total_duration:.3f})"
        )

    return argmax_speaker, score

def check_speaker_specification(speakers_spec, cursor=None):
    """
    Check and convert speaker specification to list of speaker names

    Args:
        speakers_spec (str, list): speaker specification
        cursor (sqlite3.Cursor): optional database cursor
    """

    if speakers_spec and not is_speaker_identification_enabled():
        raise RuntimeError("Speaker identification is disabled (no reference speakers)")

    # Read list / dictionary
    if isinstance(speakers_spec, str):
        speakers_spec = speakers_spec.strip()
        print("NOCOMMIT", speakers_spec, speakers_spec and (speakers_spec == "*"), _get_db_speaker_names())
        if speakers_spec:
            if speakers_spec == "*":
                # Wildcard: all speakers
                speakers_spec = list(_get_db_speaker_names())
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
            items = [_get_db_speaker_name(item, cursor)]
        
        elif isinstance(item, dict):
            # Should we really keep this format ?
            start = item['start']
            end = item['end']
            for id in range(start,end+1):
                items.append(_get_db_speaker_id(id))
        
        elif isinstance(item, str):
            if all_speaker_names is None:
                all_speaker_names = _get_db_speaker_names(cursor)
            if item not in all_speaker_names:
                raise ValueError(f"Unknown speaker name '{item}'")
            items = [item]
        
        else:
            raise ValueError(f"Unsupported reference speaker specification of type {type(item)} (in list): {speakers_spec}")
        
        for item in items:
            if item not in speaker_names:
                speaker_names.append(item)

    return speaker_names
    

def speaker_identify_given_diarization(audioFile, diarization, speakers_spec="*", log=None, options={}):
    """
    Run speaker identification on given diarized audio file

    Args:
        audioFile (str): path to audio file
        diarization (dict): diarization result
        speakers_spec (list): list of reference speaker ids or ranges (e.g. [1, 2, {"start": 3, "end": 5}])
        log (logging.Logger): optional logger
        options (dict): optional options (e.g. {"min_similarity": 0.25, "limit_duration": 60})
    """

    speaker_names = check_speaker_specification(speakers_spec)

    if not speaker_names:
        return diarization

    if log:
        full_tic = time.time()
        log.info(f"Running speaker identification with {len(speaker_names)} reference speakers")
    
    if isinstance(audioFile, werkzeug.datastructures.file_storage.FileStorage):
        tempfile = memory_tempfile.MemoryTempfile(filesystem_types=['tmpfs', 'shm'], fallback=True)
        if log:
            log.info(f"Using temporary folder {tempfile.gettempdir()}")

        with tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
            audioFile.save(ntf.name)
            return speaker_identify_given_diarization(ntf.name, diarization, speaker_names)

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
        
        spk_name, spk_id_score = speaker_identify(
            audio, speaker_names, spk_segments,
            # TODO : do we really want to avoid that 2 speakers are the same ?
            #        and if we do, not that it's not invariant to the order in which segments are taken (so we should choose a somewhat optimal order)
            exclude_speakers=([] if _can_identify_twice_the_same_speaker else already_identified),
            log=log,
            spk_tag=spk_tag,
            **options
        )
        if spk_name == _UNKNOWN:
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
        if speaker_name == _UNKNOWN:
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

    if log:
        log.info(f"Speaker identification done in {time.time() - full_tic:.3f} seconds")

    return result
