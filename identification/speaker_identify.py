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
    cur = conn.cursor()
    # Creating and inserting into table
    cur.execute("""CREATE TABLE IF NOT EXISTS speaker_names (id integer UNIQUE, Name TEXT UNIQUE)""")
    for id, speaker_name in enumerate(_get_speakers()):
        cur.execute("INSERT OR IGNORE INTO speaker_names (id, Name) VALUES (?, ?)", (id+1,speaker_name))
    conn.commit()

def get_all_ids():
    conn=sqlite3.connect(_FILE_DATABASE)
    c = conn.cursor()
    c.execute("SELECT id FROM speaker_names")
    ids = c.fetchall()
    conn.close()
    return ids


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
    speakers = list(_get_speakers())
    for speaker_name in tqdm(speakers, desc="Compute ref. speaker embeddings"):
        audio_files = _get_speaker_sample_files(speaker_name)
        assert len(audio_files) > 0, f"No audio files found for speaker {speaker_name}"
        audio = None
        max_samples = max_duration * sample_rate
        for audio_file in audio_files:
            # TODO: convert to 16kHz if needed ! (or fail if not 16kHz...)
            clip_audio, clip_sample_rate = torchaudio.load(audio_file)
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
        embeddings_file = _get_speaker_embeddings_file(speaker_name)
        pkl.dump(spk_embed, open(embeddings_file, 'wb'))
    if log: log.info(f"Speaker identification initialized with {len(speakers)} speakers")
    

def _get_speaker_embeddings_file(speaker_name):
    return os.path.join(_FOLDER_EMBEDDINGS, speaker_name + '.pkl')

def _get_speaker_sample_files(speaker_name):
    if os.path.isdir(os.path.join(_FOLDER_WAV, speaker_name)):
        return sorted(glob.glob(os.path.join(_FOLDER_WAV, speaker_name, '*')))
    prefix = os.path.join(_FOLDER_WAV, speaker_name)
    audio_files = glob.glob(prefix + '.*')
    audio_files = [file for file in audio_files if os.path.splitext(file)[0] == prefix]
    assert len(audio_files) == 1
    return audio_files

def _get_speakers():
    assert os.path.isdir(_FOLDER_WAV)
    for root, dirs, files in os.walk(_FOLDER_WAV):
        for file in files:
            if root == _FOLDER_WAV:
                speaker_name = os.path.splitext(file)[0]
            else:
                speaker_name = os.path.basename(root.rstrip("/"))
            yield speaker_name

def speaker_identify(
    audio,
    speaker_names,
    segments,
    exclude_speakers,
    min_similarity=0.25,
    sample_rate=16_000,
    limit_duration=60,
    ):
    """
    Run speaker identification on given segments of an audio

    Args:
        audio (torch.Tensor): audio waveform
        speaker_names (list): list of reference speaker names
        segments (list): list of segments to analyze (tuples of start and end times in seconds)
        exclude_speakers (list): list of speaker names to exclude
        min_similarity (float): minimum similarity to consider a speaker match
        sample_rate (int): audio sample rate
        limit_duration (int): maximum duration (in seconds) of speech to identify a speaker (the first seconds of speech will be used, the other will be ignored)
    """

    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    assert len(speaker_names) > 0

    id_count = defaultdict(int)
    limit = limit_duration * sample_rate
    duration = 0
    
    for start, end in segments:
        start = int(start * sample_rate)
        end = int(end * sample_rate)
        clip = audio[:, start:end]
        # ECAPA-TDNN embedding are only extracted for speech of duration > 0.15s
        # TODO: make this more explicit, and stress tests on the limit
        if (end-start) < 600:
            clip=torch.cat((clip, clip, clip, clip), 1)
          
        max_score = min_similarity
        person = _UNKNOWN # if no match to any voice, then return unknown

        # TODO: we probably want to "transpose the for loops" to avoid loading the embeddings at each iteration
        #       (if segments are all super short, this could be a bottleneck)
        for speaker_name in speaker_names:
            embed_file = _get_speaker_embeddings_file(speaker_name)
            # compare voice file with audio file
            with (open(embed_file, "rb")) as openfile:
                emb1 = pkl.load(openfile)
            emb1 = emb1.to(embed_model.device)

            emb2 = embed_model.encode_batch(clip)
            score = similarity(emb1, emb2)  
            score = score[0]
            if score >= max_score and speaker_name not in exclude_speakers:
                max_score = score
                person = speaker_name

        # TODO: use the duration to vote for the speaker (otherwise, tiny segments have the same weight as big ones)
        #       JL: I suggest to use 'duration' instead of 1 here
        id_count[person] += 1
        current_pred = max(id_count, key=id_count.get)
        duration += (end - start)
        
        # TODO: do we really want to continue forever for unknown speakers ?
        if duration >= limit and current_pred != _UNKNOWN:
            break
    
    most_common_Id = max(id_count, key=id_count.get)
    
    return most_common_Id


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

    if speakers_spec and not is_speaker_identification_enabled():
        raise RuntimeError("Speaker identification is disabled (no reference speakers)")

    if isinstance(speakers_spec, str) and speakers_spec != "*":
        try:
            speakers_spec = json.loads(speakers_spec)
        except Exception as err:
            raise ValueError(f"Unsupported reference speaker specification: {speakers_spec}") from err

    if not speakers_spec:
        speaker_ids = []
    elif isinstance(speakers_spec, list):
        speaker_ids=[]
        for item in speakers_spec:
            if type(item)==int:
                speaker_ids.append(item)
            elif type(item)==dict:
                start=item['start']
                end=item['end']
                for x in range(start,end+1):
                    speaker_ids.append(x)
            else:
                raise ValueError(f"Unsupported reference speaker specification of type {type(item)} (in list)")
    elif speakers_spec == "*":
        speaker_ids = get_all_ids()
    else:
        raise ValueError(f"Unsupported reference speaker specification of type {type(speakers_spec)}")
    
    if log:
        full_tic = time.time()
        log.info(f"Running speaker identification with {len(speaker_ids)} reference speakers")

    if not speaker_ids:
        return diarization
    
    if isinstance(audioFile, werkzeug.datastructures.file_storage.FileStorage):
        tempfile = memory_tempfile.MemoryTempfile(filesystem_types=['tmpfs', 'shm'], fallback=True)
        if log:
            log.info(f"Using temporary folder {tempfile.gettempdir()}")

        with tempfile.NamedTemporaryFile(suffix = ".wav") as ntf:
            audioFile.save(ntf.name)
            return speaker_identify_given_diarization(ntf.name, diarization, speaker_names)
        
    
    # Conversion ids -> names
    speaker_names=[]
    conn=sqlite3.connect(_FILE_DATABASE)
    c = conn.cursor()
    for id in speaker_ids:
        item=c.execute("SELECT Name FROM speaker_names WHERE id = '%s'" % id)
        speaker_names.append(item.fetchone()[0])

    # Closing the connection 
    conn.close() 

    speaker_tags = []
    speakers = {}
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
            speakers[speaker] = []

        speakers[speaker].append([start, end])

    audio, sample_rate = torchaudio.load(audioFile)
    # This should be OK, since this is enforced by the diarization API
    assert sample_rate == 16_000, f"Unsupported sample rate {sample_rate} (only 16kHz is supported)"

    already_identified = []
    for spk_tag, spk_segments in speakers.items():
        tic = time.time()
        spk_name = speaker_identify(
            audio, speaker_names, spk_segments,
            # TODO : do we really want to avoid that 2 speakers are the same ?
            #        and if we do, not that it's not invariant to the order in which segments are taken (so we should choose a somewhat optimal order)
            exclude_speakers=already_identified,
            **options
        )
        if log:
            log.info(
                f"Speaker recognition {spk_tag} -> {spk_name} (done in {time.time() - tic:.3f} seconds)"
            )
        if spk_name != _UNKNOWN:
            already_identified.append(spk_name)
            speaker_map[spk_tag] = spk_name
        else:
            speaker_map[spk_tag] = spk_tag

    json = {}
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

    json["speakers"] = list(_speakers.values())
    json["segments"] = _segments

    if log:
        log.info(f"Speaker identification done in {time.time() - full_tic:.3f} seconds")

    return json
