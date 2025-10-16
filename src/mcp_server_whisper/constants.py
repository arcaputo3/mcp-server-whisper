"""Constants and enumerations for the MCP Server Whisper."""

from enum import Enum
from typing import Literal

# Type Aliases
SupportedChatWithAudioFormat = Literal["mp3", "wav"]
AudioChatModel = Literal[
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17",
]
EnhancementType = Literal["detailed", "storytelling", "professional", "analytical"]
TTSVoice = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]

# Supported Audio Formats
TRANSCRIBE_AUDIO_FORMATS = {
    ".flac",  # Added FLAC support
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",  # Added OGG support
    ".wav",
    ".webm",
}

CHAT_WITH_AUDIO_FORMATS = {".mp3", ".wav"}

# Enhancement Prompts
ENHANCEMENT_PROMPTS: dict[EnhancementType, str] = {
    "detailed": "The following is a detailed transcript that includes all verbal and non-verbal elements. "
    "Background noises are noted in [brackets]. Speech characteristics like [pause], [laughs], and [sighs] "
    "are preserved. Filler words like 'um', 'uh', 'like', and 'you know' are included. "
    "Hello... [deep breath] Let me explain what I mean by that. [background noise] You know, it's like...",
    "storytelling": "The following is a natural conversation with proper punctuation and flow. "
    "Each speaker's words are captured in a new paragraph with emotional context preserved. "
    "Hello! I'm excited to share this story with you. It began on a warm summer morning...",
    "professional": "The following is a clear, professional transcript with proper capitalization and punctuation. "
    "Each sentence is complete and properly structured. Technical terms and acronyms are preserved exactly. "
    "Welcome to today's presentation on the Q4 financial results. Our KPIs show significant growth.",
    "analytical": "The following is a precise technical transcript that preserves speech patterns and terminology. "
    "Note changes in speaking pace, emphasis, and technical terms exactly as spoken. "
    "Preserve specialized vocabulary, acronyms, and technical jargon with high fidelity. "
    "Example: The API endpoint /v1/completions [spoken slowly] accepts JSON payloads "
    "with a maximum token count of 4096 [emphasis on numbers].",
}


class SortBy(str, Enum):
    """Sorting options for audio files."""

    NAME = "name"
    SIZE = "size"
    DURATION = "duration"
    MODIFIED_TIME = "modified_time"
    FORMAT = "format"


# Default Values
DEFAULT_MAX_FILE_SIZE_MB = 25
DEFAULT_TTS_MAX_LENGTH = 4000
DEFAULT_TTS_SAMPLE_RATE = 11025
