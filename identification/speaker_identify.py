import speechbrain
if speechbrain.__version__ >= "1.0.0":
   from speechbrain.inference.speaker import SpeakerRecognition, EncoderClassifier
else:
   from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
import os
from collections import defaultdict
import torch
import torchaudio
import time
import subprocess
import memory_tempfile
import werkzeug
import pickle as pkl
import sqlite3
import glob
import json
from tqdm import tqdm

# TODO : use an environment variable to set the device
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

# verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":device})
embed_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":device})

# Constants (that could be env variables)
_FOLDER_WAV = "/opt/speaker_samples"
_FOLDER_INTERNAL = "/opt/speaker_precomputed"
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

def check_wav_16khz_mono(wavfile):
    """
    Returns True if a wav file is 16khz and single channel
    """
    try:
        signal, fs = torchaudio.load(wavfile)

        mono = signal.shape[0] == 1
        freq = fs == 16000
        if mono and freq:
            return True
        else:
            return False
    except:
        return False


def convert_wavfile(wavfile, outfile):
    """
    Converts file to 16khz single channel mono wav
    """
    cmd = "ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}".format(
        wavfile, outfile
    )
    subprocess.Popen(cmd, shell=True).wait()
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
    os.makedirs(_FOLDER_EMBEDDINGS, exist_ok=True)
    speakers = list(_get_speaker_names())
    for speaker_name in tqdm(speakers, desc="Compute ref. speaker embeddings"):
        audio_files = _get_speaker_sample_files(speaker_name)
        assert len(audio_files) > 0, f"No audio files found for speaker {speaker_name}"
        audio = None
        max_samples = max_duration * sample_rate
        for audio_file in audio_files:
            if check_wav_16khz_mono(audio_file):
                clip_audio, clip_sample_rate = torchaudio.load(audio_file)
            else:
                log.info("Converting audio file to single channel WAV using ffmpeg...")
                converted_wavfile = os.path.join(
                    os.path.dirname(audio_file), "{}.wav".format(os.path.splitext(os.path.basename(audio_file))[0])
                )
                convert_wavfile(audio_file, converted_wavfile)
                assert os.path.isfile(
                    converted_wavfile
                ), "Couldn't find converted wav file, failed for some reason"
                clip_audio, clip_sample_rate = torchaudio.load(converted_wavfile)
                os.remove(audio_file)

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

        spk_embed = embed_model.encode_batch(audio)
        # Note: it is important to save the embeddings on the CPU (to be able to load them on the CPU later on)
        spk_embed = spk_embed.cpu()
        embeddings_file = _get_speaker_embedding_file(speaker_name)
        pkl.dump(spk_embed, open(embeddings_file, 'wb'))
    if log: log.info(f"Speaker identification initialized with {len(speakers)} speakers")
    

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
    return os.path.join(_FOLDER_EMBEDDINGS, speaker_name + '.pkl')

def _get_speaker_sample_files(speaker_name):
    if os.path.isdir(os.path.join(_FOLDER_WAV, speaker_name)):
        return sorted(glob.glob(os.path.join(_FOLDER_WAV, speaker_name, '*')))
    prefix = os.path.join(_FOLDER_WAV, speaker_name)
    audio_files = glob.glob(prefix + '.*')
    audio_files = [file for file in audio_files if os.path.splitext(file)[0] == prefix]
    assert len(audio_files) == 1
    return audio_files

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
    min_similarity=0.25,
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

    embedding_audio = embed_model.encode_batch(audio_selection)

    # Loop on the target speakers
    for speaker_name in speaker_names:
        if speaker_name in exclude_speakers:
            continue

        # Get speaker embedding
        embed_file = _get_speaker_embedding_file(speaker_name)
        with open(embed_file, "rb") as openfile:
            embedding_speaker = pkl.load(openfile)
        embedding_speaker = embedding_speaker.to(embed_model.device)

        # Compute score similarity
        score = similarity(embedding_speaker, embedding_audio)
        score = score.item()
        if score >= min_similarity:
            votes[speaker_name] += score

    if not votes:
        argmax_speaker = _UNKNOWN
    else:
        argmax_speaker = max(votes, key=votes.get)    

    if log:
        log.info(
            f"Speaker recognition {spk_tag} -> {argmax_speaker} (done in {time.time() - tic:.3f} seconds, on {audio_selection.shape[1] / sample_rate:.3f} seconds of audio out of {total_duration:.3f})"
        )

    return argmax_speaker

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
    if isinstance(speakers_spec, str) and speakers_spec != "*":
        try:
            speakers_spec = json.loads(speakers_spec)
        except Exception as err:
            raise ValueError(f"Unsupported reference speaker specification: {speakers_spec}") from err
        if isinstance(speakers_spec, dict):
            speakers_spec = [speakers_spec]

    # Convert to list of speaker names
    if not speakers_spec:
        return []

    elif isinstance(speakers_spec, list):
        all_speaker_names = None
        speaker_names = []
        for item in speakers_spec:
            if isinstance(item, int):
                items = [_get_db_speaker_name(item, cursor)]
            
            elif isinstance(item, dict):
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

    elif speakers_spec == "*":
        return list(_get_db_speaker_names())

    raise ValueError(f"Unsupported reference speaker specification of type {type(speakers_spec)}: {speakers_spec}")
    

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
    for spk_tag in sorted(speaker_segments.keys(), key=speech_duration, reverse=True):
        spk_segments = speaker_segments[spk_tag]
        
        spk_name = speaker_identify(
            audio, speaker_names, spk_segments,
            # TODO : do we really want to avoid that 2 speakers are the same ?
            #        and if we do, not that it's not invariant to the order in which segments are taken (so we should choose a somewhat optimal order)
            exclude_speakers=already_identified,
            log=log,
            spk_tag=spk_tag,
            **options
        )
        if spk_name != _UNKNOWN:
            already_identified.append(spk_name)
            speaker_map[spk_tag] = spk_name
        else:
            speaker_map[spk_tag] = spk_tag

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
