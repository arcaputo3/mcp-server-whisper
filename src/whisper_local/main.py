import sys
import os

from dotenv import load_dotenv
from .cli import parse_args
from .transcribe import do_whisper_transcription
from .gpt4o import do_gpt4o_audio


def main():
    """
    Entry point for the CLI tool. Parses arguments, loads environment variables,
    and runs either Whisper transcription or GPT-4o direct audio.
    """
    load_dotenv()
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Could not find audio file '{args.input}'")
        sys.exit(1)

    if args.use_gpt4o:
        do_gpt4o_audio(args)
    else:
        do_whisper_transcription(args)


if __name__ == "__main__":
    main()
