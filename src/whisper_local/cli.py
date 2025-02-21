import argparse


def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments for Whisper or GPT-4o usage.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper or GPT-4o"
    )

    # Positional argument: Path to audio file (common for both Whisper and GPT-4o)
    parser.add_argument(
        "input", help="Path to input audio file"
    )  # /Users/rcaputo3/Movies/Omi Screen Recorder/Audio-2025-02-20-224204.m4a

    # Common output text file option (used if the user wants to save text transcription/response)
    parser.add_argument("-o", "--output", help="Path to output text file (optional)")

    # Whisper-specific argument
    parser.add_argument(
        "-m",
        "--model",
        default="whisper-1",
        help="Whisper model to use (default: whisper-1)",
    )

    # Flag to switch between Whisper or GPT-4o
    parser.add_argument(
        "--use-gpt4o",
        action="store_true",
        help="Use GPT-4o direct audio input instead of Whisper transcription",
    )

    # GPT-4o model argument (if user wants a different GPT-4o model)
    parser.add_argument(
        "--gpt4o-model",
        default="gpt-4o-audio-preview-2024-10-01",
        help="GPT-4o audio model name (default: gpt-4o-audio-preview-2024-10-01)",
    )

    # Comma-separated modalities for GPT-4o call, e.g. "text" or "text,audio"
    parser.add_argument(
        "--modalities",
        default="text",
        help='Comma-separated modalities for GPT-4o (e.g. "text" or "text,audio")',
    )

    # If user wants to incorporate a small text message with the audio
    parser.add_argument(
        "--text-prompt",
        default="",
        help="Optional text prompt to include in the user message along with audio",
    )

    # If user wants GPT-4o to return an audio response
    parser.add_argument(
        "--audio-output",
        action="store_true",
        help='Request an audio response from GPT-4o (must include "audio" in modalities)',
    )

    # For conversation continuity across multiple runs, store the conversation in a JSON file
    parser.add_argument(
        "--conversation-file",
        help="Path to a JSON file for saving/loading conversation context",
    )

    return parser.parse_args()
