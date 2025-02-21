import sys
import os
import openai

from .convert import convert_to_supported_format


def do_whisper_transcription(args):
    """
    Transcribe audio using the Whisper API via openai.Audio.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
           - args.input: Path to input audio file
           - args.output: Path to output text file (optional)
           - args.model: Whisper model to use (default: whisper-1)
    """
    if not os.path.isfile(args.input):
        print(f"Error: Could not find audio file '{args.input}'")
        sys.exit(1)

    # Check extension and convert if necessary
    _, ext = os.path.splitext(args.input)
    ext = ext.lower().replace(".", "")
    if ext not in ("wav", "mp3"):
        print(
            f"Converting {args.input} from {ext} to a supported format (wav) for Whisper transcription..."
        )
        try:
            new_path = convert_to_supported_format(args.input, target_format="mp3")
            args.input = new_path
        except Exception as e:
            print(f"Error converting file: {e}")
            sys.exit(1)

    # Double-check file size if desired (Whisper also has a 25MB max limit).
    max_size = 25 * 1024 * 1024  # 25 MB
    file_size = os.path.getsize(args.input)
    if file_size > max_size:
        print(f"Error: Audio file '{args.input}' exceeds 25MB size limit.")
        sys.exit(1)

    try:
        with open(args.input, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model=args.model, file=audio_file, response_format="text"
            )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(transcript)
        else:
            print(transcript)

    except FileNotFoundError:
        print(f"Error: Could not find audio file '{args.input}' after conversion.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
