import sys
import argparse
from google.protobuf.json_format import MessageToDict
from soniox.speech_service import SpeechClient
from soniox.speech_service_pb2 import (
    AddSpeakerRequest,
    GetSpeakerRequest,
    RemoveSpeakerRequest,
    ListSpeakersRequest,
    AddSpeakerAudioRequest,
    GetSpeakerAudioRequest,
    RemoveSpeakerAudioRequest,
)


def pb_to_dict(pb):
    return MessageToDict(pb, including_default_value_fields=True, preserving_proto_field_name=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List speakers and audios.")
    parser.add_argument(
        "--add_speaker", action="store_true", help="Add speaker (specify --speaker_name)."
    )
    parser.add_argument(
        "--add_audio",
        action="store_true",
        help="Add audio (specify --speaker_name, --audio_name, --audio_fn).",
    )
    parser.add_argument(
        "--get_audio",
        action="store_true",
        help="Get audio (specify --speaker_name, --audio_name, optionally --out_audio_fn).",
    )
    parser.add_argument(
        "--remove_audio",
        action="store_true",
        help="Remove audio (specify --speaker_name, --audio_name).",
    )
    parser.add_argument(
        "--remove_speaker", action="store_true", help="Remove speaker (specify --speaker_name)."
    )
    parser.add_argument("--speaker_name", type=str)
    parser.add_argument("--audio_name", type=str)
    parser.add_argument("--audio_fn", type=str)
    parser.add_argument("--out_audio_fn", type=str)
    args = parser.parse_args()

    if not (
        args.list
        or args.add_speaker
        or args.add_audio
        or args.get_audio
        or args.remove_audio
        or args.remove_speaker
    ):
        parser.print_help()
        sys.exit(0)

    with SpeechClient() as client:
        if args.list:
            print("Listing speakers and audios.")

            response = client.service_stub.ListSpeakers(
                ListSpeakersRequest(
                    api_key=client.api_key,
                )
            )

            if len(response.speakers) == 0:
                print("(no speakers)")
            else:
                for speaker in response.speakers:
                    print(f"  {pb_to_dict(speaker)}")

                    audios = []
                    if speaker.num_audios > 0:
                        audios = client.service_stub.GetSpeaker(
                            GetSpeakerRequest(
                                api_key=client.api_key,
                                name=speaker.name,
                            )
                        ).audios

                    if len(audios) != 0:
                        for audio in audios:
                            print(f"    {pb_to_dict(audio)}")

        if args.add_speaker:
            if args.speaker_name is None:
                print("--add_speaker requires --speaker_name.", file=sys.stderr)
                sys.exit(1)

            print(f'Adding speaker "{args.speaker_name}".')

            client.service_stub.AddSpeaker(
                AddSpeakerRequest(
                    api_key=client.api_key,
                    name=args.speaker_name,
                )
            )

        if args.add_audio:
            if args.speaker_name is None:
                print("--add_audio requires --speaker_name.", file=sys.stderr)
                sys.exit(1)

            if args.audio_name is None:
                print("--add_audio requires --audio_name.", file=sys.stderr)
                sys.exit(1)

            if args.audio_fn is None:
                print("--add_audio requires --audio_fn.", file=sys.stderr)
                sys.exit(1)

            print(f'Reading audio file "{args.audio_fn}".')

            with open(args.audio_fn, "rb") as fh:
                audio = fh.read()

            print(f'Adding audio "{args.audio_name}" for speaker "{args.speaker_name}".')

            client.service_stub.AddSpeakerAudio(
                AddSpeakerAudioRequest(
                    api_key=client.api_key,
                    speaker_name=args.speaker_name,
                    audio_name=args.audio_name,
                    audio=audio,
                )
            )

        if args.get_audio:
            if args.speaker_name is None:
                print("--get_audio requires --speaker_name.", file=sys.stderr)
                sys.exit(1)

            if args.audio_name is None:
                print("--get_audio requires --audio_name.", file=sys.stderr)
                sys.exit(1)

            print(f'Getting audio "{args.audio_name}" for speaker "{args.speaker_name}".')

            response = client.service_stub.GetSpeakerAudio(
                GetSpeakerAudioRequest(
                    api_key=client.api_key,
                    speaker_name=args.speaker_name,
                    audio_name=args.audio_name,
                )
            )

            audio_info = pb_to_dict(response)
            del audio_info["audio"]

            print("Audio:")
            print(f"  {audio_info}")

            if args.out_audio_fn is not None:
                print(f'Writing audio file "{args.out_audio_fn}".')

                with open(args.out_audio_fn, "wb") as fh:
                    fh.write(response.audio)

        if args.remove_audio:
            if args.speaker_name is None:
                print("--remove_audio requires --speaker_name.", file=sys.stderr)
                sys.exit(1)

            if args.audio_name is None:
                print("--remove_audio requires --audio_name.", file=sys.stderr)
                sys.exit(1)

            print(f'Removing audio "{args.audio_name}" for speaker "{args.speaker_name}".')

            client.service_stub.RemoveSpeakerAudio(
                RemoveSpeakerAudioRequest(
                    api_key=client.api_key,
                    speaker_name=args.speaker_name,
                    audio_name=args.audio_name,
                )
            )

        if args.remove_speaker:
            if args.speaker_name is None:
                print("--remove_speaker requires --speaker_name.", file=sys.stderr)
                sys.exit(1)

            print(f'Removing speaker "{args.speaker_name}".')

            client.service_stub.RemoveSpeaker(
                RemoveSpeakerRequest(
                    api_key=client.api_key,
                    name=args.speaker_name,
                )
            )


if __name__ == "__main__":
    main()