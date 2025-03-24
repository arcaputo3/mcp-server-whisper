"""Whisper MCP server core code."""

import asyncio
import base64
import os
import re
import time
from enum import Enum
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, cast

import aiofiles
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from openai.types import AudioModel, AudioResponseFormat
from openai.types.audio.speech_model import SpeechModel
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import BaseModel, Field
from pydub import AudioSegment  # type: ignore

# Literals for transcription
SupportedChatWithAudioFormat = Literal["mp3", "wav"]
AudioChatModel = Literal[
    "gpt-4o-audio-preview-2024-10-01", "gpt-4o-audio-preview-2024-12-17", "gpt-4o-mini-audio-preview-2024-12-17"
]
EnhancementType = Literal["detailed", "storytelling", "professional", "analytical"]
TTSVoice = Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

# Constants for checks
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

# Enhancement prompts
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


class BaseInputPath(BaseModel):
    """Base file path input."""

    input_file_path: Path = Field(description="Path to the input audio file to process")

    model_config = {"arbitrary_types_allowed": True}


class BaseAudioInputParams(BaseInputPath):
    """Base params for converting audio to mp3."""

    output_file_path: Optional[Path] = Field(
        default=None,
        description="Optional custom path for the output file. "
        "If not provided, defaults to input_file_path with appropriate extension",
    )


class ConvertAudioInputParams(BaseAudioInputParams):
    """Params for converting audio to mp3."""

    target_format: SupportedChatWithAudioFormat = Field(
        default="mp3", description="Target audio format to convert to (mp3 or wav)"
    )


class CompressAudioInputParams(BaseAudioInputParams):
    """Params for compressing audio."""

    max_mb: int = Field(
        default=25, gt=0, description="Maximum file size in MB. Files larger than this will be compressed"
    )


class TranscribeAudioInputParamsBase(BaseInputPath):
    """Params for transcribing audio with audio-to-text model."""

    model: AudioModel = Field(
        default="gpt-4o-mini-transcribe",
        description="The transcription model to use (e.g., 'whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe')",
    )
    response_format: AudioResponseFormat = Field(
        "text",
        description="The response format of the transcription model. "
        'Use `verbose_json` with `model="whisper-1"` for timestamps. '
        "`gpt-4o-transcribe` and `gpt-4o-mini-transcribe` only support `text` and `json`.",
    )
    timestamp_granularities: list[Literal["word", "segment"]] | None = Field(
        None,
        description="""The timestamp granularities to populate for this transcription.
`response_format` must be set `verbose_json` to use timestamp granularities.
Either or both of these options are supported: `word`, or `segment`.
Note: There is no additional latency for segment timestamps, but generating word timestamp incurs additional latency.
""",
    )


class TranscribeAudioInputParams(TranscribeAudioInputParamsBase):
    """Params for transcribing audio with audio-to-text model."""

    prompt: str | None = Field(
        None,
        description="""An optional prompt to guide the transcription model's output. Effective prompts can:

        1. Correct specific words/acronyms: Include technical terms or names that might be misrecognized
           Example: "The transcript discusses OpenAI's DALL·E and GPT-4 technology"

        2. Maintain context from previous segments: Include the last part of previous transcript
           Note: Model only considers final 224 tokens of the prompt

        3. Enforce punctuation: Include properly punctuated example text
           Example: "Hello, welcome to my lecture. Today, we'll discuss..."

        4. Preserve filler words: Include example with verbal hesitations
           Example: "Umm, let me think like, hmm... Okay, here's what I'm thinking"

        5. Set writing style: Use examples in desired format (simplified/traditional, formal/casual)

        The model will try to match the style and formatting of your prompt.""",
    )


class ChatWithAudioInputParams(BaseInputPath):
    """Params for transcribing audio with LLM using custom prompt."""

    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt to use.")
    user_prompt: Optional[str] = Field(default=None, description="Custom user prompt to use.")
    model: AudioChatModel = Field(
        default="gpt-4o-audio-preview-2024-12-17", description="The audio LLM model to use for transcription"
    )


class TranscribeWithEnhancementInputParams(TranscribeAudioInputParamsBase):
    """Params for transcribing audio with LLM using template prompt."""

    enhancement_type: EnhancementType = Field(
        default="detailed",
        description="Type of enhancement to apply to the transcription: "
        "detailed, storytelling, professional, or analytical.",
    )

    def to_transcribe_audio_input_params(self) -> TranscribeAudioInputParams:
        """Transfer audio with LLM using custom prompt."""
        return TranscribeAudioInputParams(
            input_file_path=self.input_file_path,
            prompt=ENHANCEMENT_PROMPTS[self.enhancement_type],
            model=self.model,
            timestamp_granularities=self.timestamp_granularities,
            response_format=self.response_format,
        )


class CreateClaudecastInputParams(BaseModel):
    """Params for text-to-speech using OpenAI's API."""

    text_prompt: str = Field(description="Text to convert to speech")
    output_file_path: Optional[Path] = Field(
        default=None, description="Output file path (defaults to speech.mp3 in current directory)"
    )
    model: SpeechModel = Field(
        default="gpt-4o-mini-tts", description="TTS model to use. gpt-4o-mini-tts is always preferred."
    )
    voice: TTSVoice = Field(
        default="nova",
        description="Voice for the TTS (options: alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer)",
    )
    instructions: str | None = Field(
        default=None,
        description="Optional instructions for the speech conversion, such as tonality, accent, style, etc.",
    )
    speed: float = Field(
        default=1.0,
        gt=0.25,
        lt=4.0,
        description="Speed of the speech conversion. Use if the user prompts slow or fast speech.",
    )

    model_config = {"arbitrary_types_allowed": True}


class FilePathSupportParams(BaseModel):
    """Params for checking if a file at a path supports transcription."""

    file_path: Path = Field(description="Path to the audio file")
    transcription_support: Optional[list[AudioModel]] = Field(
        default=None, description="List of transcription models that support this file format"
    )
    chat_support: Optional[list[AudioChatModel]] = Field(
        default=None, description="List of audio LLM models that support this file format"
    )
    modified_time: float = Field(description="Last modified timestamp of the file (Unix time)")
    size_bytes: int = Field(description="Size of the file in bytes")
    format: str = Field(description="Audio format of the file (e.g., 'mp3', 'wav')")
    duration_seconds: Optional[float] = Field(
        default=None, description="Duration of the audio file in seconds, if available"
    )

    model_config = {"arbitrary_types_allowed": True}


mcp = FastMCP("whisper", dependencies=["openai", "pydub", "aiofiles"])


def check_and_get_audio_path() -> Path:
    """Check if the audio path environment variable is set and exists."""
    audio_path_str = os.getenv("AUDIO_FILES_PATH")
    if not audio_path_str:
        raise ValueError("AUDIO_FILES_PATH environment variable not set")

    audio_path = Path(audio_path_str).resolve()
    if not audio_path.exists():
        raise ValueError(f"Audio path does not exist: {audio_path}")
    return audio_path


async def get_audio_file_support(file_path: Path) -> FilePathSupportParams:
    """Determine audio transcription file format support and metadata.

    Includes file size, format, and duration information where available.
    """
    file_ext = file_path.suffix.lower()

    transcription_support: list[AudioModel] | None = (
        ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] if file_ext in TRANSCRIBE_AUDIO_FORMATS else None
    )
    chat_support: list[AudioChatModel] | None = (
        ["gpt-4o-audio-preview-2024-10-01", "gpt-4o-audio-preview-2024-12-17", "gpt-4o-mini-audio-preview-2024-12-17"]
        if file_ext in CHAT_WITH_AUDIO_FORMATS
        else None
    )

    # Get file stats
    file_stats = file_path.stat()

    # Get file size using aiofiles
    async with aiofiles.open(file_path, "rb") as f:
        file_content = await f.read()
    size_bytes = len(file_content)

    # Get audio format (remove the dot from extension)
    audio_format = file_ext[1:] if file_ext.startswith(".") else file_ext

    # Get duration if possible (could be expensive for large files)
    duration_seconds = None
    try:
        # Load just the metadata to get duration
        audio = await asyncio.to_thread(AudioSegment.from_file, str(file_path), format=audio_format)
        # Convert from milliseconds to seconds
        duration_seconds = len(audio) / 1000.0
    except Exception:
        # If we can't get duration, just continue without it
        pass

    return FilePathSupportParams(
        file_path=file_path,
        transcription_support=transcription_support,
        chat_support=chat_support,
        modified_time=file_stats.st_mtime,
        size_bytes=size_bytes,
        format=audio_format,
        duration_seconds=duration_seconds,
    )


@mcp.tool(
    description="Get the most recent audio file from the audio path. "
    "ONLY USE THIS IF THE USER ASKS FOR THE LATEST FILE."
)
async def get_latest_audio() -> FilePathSupportParams:
    """Get the most recently modified audio file and returns its path with model support info.

    Supported formats:
    - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - GPT-4o: mp3, wav

    Returns detailed file information including size, format, and duration.
    """
    audio_path = check_and_get_audio_path()

    try:
        files = []
        for file_path in audio_path.iterdir():
            if not file_path.is_file():
                continue

            file_ext = file_path.suffix.lower()
            if file_ext in TRANSCRIBE_AUDIO_FORMATS or file_ext in CHAT_WITH_AUDIO_FORMATS:
                files.append((file_path, file_path.stat().st_mtime))

        if not files:
            raise RuntimeError("No supported audio files found")

        latest_file = max(files, key=lambda x: x[1])[0]
        return await get_audio_file_support(latest_file)

    except Exception as e:
        raise RuntimeError(f"Failed to get latest audio file: {e}") from e


@lru_cache(maxsize=32)
async def _get_cached_audio_file_support(file_path: str, _mtime: float) -> FilePathSupportParams:
    """Cache audio file support information using path and mtime as key.

    Uses the file path and modified time as cache key.
    """
    return await get_audio_file_support(Path(file_path))


class SortBy(str, Enum):
    """Sorting options for audio files."""

    NAME = "name"
    SIZE = "size"
    DURATION = "duration"
    MODIFIED_TIME = "modified_time"
    FORMAT = "format"


class ListAudioFilesInputParams(BaseModel):
    """Input parameters for the list_audio_files tool."""

    pattern: Optional[str] = Field(default=None, description="Optional regex pattern to filter audio files by name")
    min_size_bytes: Optional[int] = Field(default=None, description="Minimum file size in bytes")
    max_size_bytes: Optional[int] = Field(default=None, description="Maximum file size in bytes")
    min_duration_seconds: Optional[float] = Field(default=None, description="Minimum audio duration in seconds")
    max_duration_seconds: Optional[float] = Field(default=None, description="Maximum audio duration in seconds")
    min_modified_time: Optional[float] = Field(
        default=None, description="Minimum file modification time (Unix timestamp)"
    )
    max_modified_time: Optional[float] = Field(
        default=None, description="Maximum file modification time (Unix timestamp)"
    )
    format: Optional[str] = Field(default=None, description="Specific audio format to filter by (e.g., 'mp3', 'wav')")
    sort_by: SortBy = Field(
        default=SortBy.NAME, description="Field to sort results by (name, size, duration, modified_time, format)"
    )
    reverse: bool = Field(default=False, description="Sort in reverse order if True")

    model_config = {"arbitrary_types_allowed": True}


@mcp.tool(
    description="List, filter, and sort audio files from the audio path. Supports regex pattern matching, "
    "filtering by metadata (size, duration, date, format), and sorting."
)
async def list_audio_files(inputs: list[ListAudioFilesInputParams]) -> list[list[FilePathSupportParams]]:
    """List, filter, and sort audio files in the AUDIO_FILES_PATH directory with comprehensive options.

    Supported formats:
    - Transcribe: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
    - Chat: mp3, wav

    Filtering options:
    - pattern: Regex pattern for file name/path matching
    - min/max_size_bytes: File size range in bytes
    - min/max_duration_seconds: Audio duration range in seconds
    - min/max_modified_time: File modification time range (Unix timestamps)
    - format: Specific audio format (e.g., 'mp3', 'wav')

    Sorting options:
    - sort_by: Field to sort by (name, size, duration, modified_time, format)
    - reverse: Set to true for descending order

    Returns detailed file information including size, format, duration, and transcription capabilities.
    """

    async def process_single(input_data: ListAudioFilesInputParams) -> list[FilePathSupportParams]:
        audio_path = check_and_get_audio_path()

        try:
            # Store file paths that match our criteria
            file_paths = []

            # First, collect all valid file paths
            for file_path in audio_path.iterdir():
                if not file_path.is_file():
                    continue

                file_ext = file_path.suffix.lower()
                if file_ext in TRANSCRIBE_AUDIO_FORMATS or file_ext in CHAT_WITH_AUDIO_FORMATS:
                    # Apply regex pattern filtering if provided
                    if input_data.pattern and not re.search(input_data.pattern, str(file_path)):
                        continue

                    # Apply format filtering if provided
                    if input_data.format and file_ext[1:].lower() != input_data.format.lower():
                        continue

                    # For other filters, we need file metadata, so add to initial list
                    file_paths.append(file_path)

            # Process all files in parallel with async gather
            # We pass both the path and modification time to the cache function
            cache_tasks = []
            for path in file_paths:
                # Convert Path to string for caching purposes
                path_str = str(path)
                mtime = path.stat().st_mtime
                cache_tasks.append(_get_cached_audio_file_support(path_str, mtime))

            # Gather all the results
            file_support_results = await asyncio.gather(*cache_tasks)

            # Apply post-metadata filters
            filtered_results = []
            for file_info in file_support_results:
                # Apply size filters
                if input_data.min_size_bytes is not None and file_info.size_bytes < input_data.min_size_bytes:
                    continue
                if input_data.max_size_bytes is not None and file_info.size_bytes > input_data.max_size_bytes:
                    continue

                # Apply duration filters if duration is available
                if file_info.duration_seconds is not None:
                    if (
                        input_data.min_duration_seconds is not None
                        and file_info.duration_seconds < input_data.min_duration_seconds
                    ):
                        continue
                    if (
                        input_data.max_duration_seconds is not None
                        and file_info.duration_seconds > input_data.max_duration_seconds
                    ):
                        continue
                # Skip duration filtering if duration info isn't available

                # Apply modification time filters
                if input_data.min_modified_time is not None and file_info.modified_time < input_data.min_modified_time:
                    continue
                if input_data.max_modified_time is not None and file_info.modified_time > input_data.max_modified_time:
                    continue

                # If it passed all filters, add to results
                filtered_results.append(file_info)

            # Sort files according to the requested sort field
            if input_data.sort_by == SortBy.NAME:
                return sorted(filtered_results, key=lambda x: str(x.file_path), reverse=input_data.reverse)
            elif input_data.sort_by == SortBy.SIZE:
                return sorted(filtered_results, key=lambda x: x.size_bytes, reverse=input_data.reverse)
            elif input_data.sort_by == SortBy.DURATION:
                # Use 0 for files with no duration to keep them at the beginning
                return sorted(
                    filtered_results,
                    key=lambda x: x.duration_seconds if x.duration_seconds is not None else 0,
                    reverse=input_data.reverse,
                )
            elif input_data.sort_by == SortBy.MODIFIED_TIME:
                return sorted(filtered_results, key=lambda x: x.modified_time, reverse=input_data.reverse)
            elif input_data.sort_by == SortBy.FORMAT:
                return sorted(filtered_results, key=lambda x: x.format, reverse=input_data.reverse)
            else:
                # Default to sorting by name
                return sorted(filtered_results, key=lambda x: str(x.file_path), reverse=input_data.reverse)

        except Exception as e:
            raise RuntimeError(f"Failed to list audio files: {e}") from e

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


async def convert_to_supported_format(
    input_file: Path,
    output_path: Path | None = None,
    target_format: SupportedChatWithAudioFormat = "mp3",
) -> Path:
    """Async version of audio file conversion using pydub.

    Ensures the output filename is base + .{target_format} if no output_path provided.
    """
    if output_path is None:
        output_path = input_file.with_suffix(f".{target_format}")

    try:
        # Load audio file directly from path instead of reading bytes first
        audio = await asyncio.to_thread(
            AudioSegment.from_file,
            str(input_file),  # pydub expects a string path
            format=input_file.suffix[1:],  # remove the leading dot
        )

        await asyncio.to_thread(
            audio.export,
            str(output_path),  # pydub expects a string path
            format=target_format,
            parameters=["-ac", "2"],
        )
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {str(e)}")


async def compress_mp3_file(mp3_file_path: Path, output_path: Path | None = None, out_sample_rate: int = 11025) -> Path:
    """Downsample an existing mp3.

    If no output_path provided, returns a file named 'compressed_{original_stem}.mp3'.
    """
    if mp3_file_path.suffix.lower() != ".mp3":
        raise ValueError("compress_mp3_file() called on a file that is not .mp3")

    if output_path is None:
        output_path = mp3_file_path.parent / f"compressed_{mp3_file_path.stem}.mp3"

    print(f"\n[Compression] Original file: {mp3_file_path}")
    print(f"[Compression] Output file:   {output_path}")

    try:
        # Load audio file directly from path instead of reading bytes first
        audio_file = await asyncio.to_thread(AudioSegment.from_file, str(mp3_file_path), format="mp3")
        original_frame_rate = audio_file.frame_rate
        print(f"[Compression] Original frame rate: {original_frame_rate}, converting to {out_sample_rate}.")
        await asyncio.to_thread(
            audio_file.export,
            str(output_path),
            format="mp3",
            parameters=["-ar", str(out_sample_rate)],
        )
        return output_path
    except Exception as e:
        raise RuntimeError(f"Error compressing mp3 file: {str(e)}")


async def maybe_compress_file(input_file: Path, output_path: Path | None = None, max_mb: int = 25) -> Path:
    """Compress file if is above {max_mb} and convert to mp3 if needed.

    If no output_path provided, returns the compressed_{stem}.mp3 path if compression happens,
    otherwise returns the original path.
    """
    # Use aiofiles to read file size asynchronously
    async with aiofiles.open(input_file, "rb") as f:
        file_size = len(await f.read())
    threshold_bytes = max_mb * 1024 * 1024

    if file_size <= threshold_bytes:
        return input_file  # No compression needed

    print(f"\n[maybe_compress_file] File '{input_file}' size > {max_mb}MB. Attempting compression...")

    # If not mp3, convert
    if input_file.suffix.lower() != ".mp3":
        try:
            input_file = await convert_to_supported_format(input_file, None, "mp3")
        except Exception as e:
            raise RuntimeError(f"[maybe_compress_file] Error converting to MP3: {str(e)}")

    # now downsample
    try:
        compressed_path = await compress_mp3_file(input_file, output_path, 11025)
    except Exception as e:
        raise RuntimeError(f"[maybe_compress_file] Error compressing MP3 file: {str(e)}")

    # Use aiofiles to read compressed file size asynchronously
    async with aiofiles.open(compressed_path, "rb") as f:
        new_size = len(await f.read())
    print(f"[maybe_compress_file] Compressed file size: {new_size} bytes")
    return compressed_path


@mcp.tool(description="A tool used to convert audio files to mp3 or wav which are gpt-4o compatible.")
async def convert_audio(inputs: list[ConvertAudioInputParams]) -> list[dict[str, Path]]:
    """Convert multiple audio files to supported formats (mp3 or wav) in parallel."""

    async def process_single(input_data: ConvertAudioInputParams) -> dict[str, Path]:
        try:
            output_file = await convert_to_supported_format(
                input_data.input_file_path, input_data.output_file_path, input_data.target_format
            )
            return {"output_path": output_file}
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed for {input_data.input_file_path}: {str(e)}")

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool(
    description="A tool used to compress audio files which are >25mb. "
    "ONLY USE THIS IF THE USER REQUESTS COMPRESSION OR IF OTHER TOOLS FAIL DUE TO FILES BEING TOO LARGE."
)
async def compress_audio(inputs: list[CompressAudioInputParams]) -> list[dict[str, Path]]:
    """Compress multiple audio files in parallel if they're larger than max_mb."""

    async def process_single(input_data: CompressAudioInputParams) -> dict[str, Path]:
        try:
            output_file = await maybe_compress_file(
                input_data.input_file_path, input_data.output_file_path, input_data.max_mb
            )
            return {"output_path": output_file}
        except Exception as e:
            raise RuntimeError(f"Audio compression failed for {input_data.input_file_path}: {str(e)}")

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool(
    description="A tool used to transcribe audio files. It is recommended to use `gpt-4o-mini-transcribe` by default. "
    "If the user wants maximum performance, use `gpt-4o-transcribe`. "
    "Rarely should you use `whisper-1` as it is least performant, but it is available if needed. "
    "You can use prompts to guide the transcription process based on the users preference."
)
async def transcribe_audio(inputs: list[TranscribeAudioInputParams]) -> list[dict[str, Any]]:
    """Transcribe audio using OpenAI's transcribe API for multiple files in parallel.

    Raises an exception on failure, so MCP returns a proper JSON error.
    """

    async def process_single(input_data: TranscribeAudioInputParams) -> dict[str, Any]:
        file_path = input_data.input_file_path
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        client = AsyncOpenAI()

        try:
            # Use aiofiles to read the audio file asynchronously
            async with aiofiles.open(file_path, "rb") as audio_file:
                file_content = await audio_file.read()

            # Create a file-like object from bytes for OpenAI API

            file_obj = BytesIO(file_content)
            file_obj.name = file_path.name  # OpenAI API needs a filename
            transcriptions_create_input = {
                k: v for k, v in input_data.model_dump(exclude_none=True).items() if k != "input_file_path"
            }

            transcript = await client.audio.transcriptions.create(file=file_obj, **transcriptions_create_input)
            if isinstance(transcript, BaseModel):
                return transcript.model_dump()
            return {"text": transcript}

        except Exception as e:
            raise RuntimeError(f"Whisper processing failed for {file_path}: {e}") from e

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool(description="A tool used to chat with audio files. The response will be a response to the audio file sent.")
async def chat_with_audio(
    inputs: list[ChatWithAudioInputParams],
) -> list[dict[str, Any]]:
    """Transcribe multiple audio files using GPT-4 with optional text prompts in parallel."""

    async def process_single(input_data: ChatWithAudioInputParams) -> dict[str, Any]:
        file_path = input_data.input_file_path
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower().replace(".", "")
        assert ext in ["mp3", "wav"], f"Expected mp3 or wav extension, but got {ext}"

        try:
            # Use aiofiles to read the audio file asynchronously
            async with aiofiles.open(file_path, "rb") as audio_file:
                audio_bytes = await audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed reading audio file '{file_path}': {e}") from e

        client = AsyncOpenAI()
        messages: list[ChatCompletionMessageParam] = []
        if input_data.system_prompt:
            messages.append({"role": "system", "content": input_data.system_prompt})

        user_content: list[ChatCompletionContentPartParam] = []
        if input_data.user_prompt:
            user_content.append({"type": "text", "text": input_data.user_prompt})
        user_content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": cast(Literal["wav", "mp3"], ext)},
            }
        )
        messages.append({"role": "user", "content": user_content})

        try:
            completion = await client.chat.completions.create(
                model=input_data.model,
                messages=messages,
                modalities=["text"],
            )
            return {"text": completion.choices[0].message.content}
        except Exception as e:
            raise RuntimeError(f"GPT-4 processing failed for {input_data.input_file_path}: {e}") from e

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool()
async def transcribe_with_enhancement(
    inputs: list[TranscribeWithEnhancementInputParams],
) -> list[dict[str, Any]]:
    """Transcribe multiple audio files with GPT-4 using specific enhancement prompts in parallel.

    Enhancement types:
    - detailed: Provides detailed description including tone, emotion, and background
    - storytelling: Transforms the transcription into a narrative
    - professional: Formats the transcription in a formal, business-appropriate way
    - analytical: Includes analysis of speech patterns, key points, and structure
    """
    converted_inputs = [input_.to_transcribe_audio_input_params() for input_ in inputs]
    result: list[dict[str, Any]] = await transcribe_audio(converted_inputs)
    return result


def split_text_for_tts(text: str, max_length: int = 4000) -> list[str]:
    """Split text into chunks that don't exceed the TTS API limit.

    The function splits text at sentence boundaries (periods, question marks, exclamation points)
    to create natural-sounding chunks. If a sentence is too long, it falls back to
    splitting at commas, then spaces.

    Args:
    ----
        text: The text to split
        max_length: Maximum character length for each chunk (default 4000 to provide buffer)

    Returns:
    -------
        List of text chunks, each below the maximum length

    """
    # If text is already under the limit, return it as a single chunk
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining_text = text

    # Define boundary markers in order of preference
    sentence_boundaries = [". ", "? ", "! ", ".\n", "?\n", "!\n"]
    secondary_boundaries = [", ", ";\n", ";\n", ":\n", "\n", " "]

    while len(remaining_text) > max_length:
        # Try to find the best split point starting from max_length and working backward
        split_index = -1

        # First try sentence boundaries (most preferred)
        for boundary in sentence_boundaries:
            last_boundary = remaining_text[:max_length].rfind(boundary)
            if last_boundary != -1:
                split_index = last_boundary + len(boundary)
                break

        # If no sentence boundary found, try secondary boundaries
        if split_index == -1:
            for boundary in secondary_boundaries:
                last_boundary = remaining_text[:max_length].rfind(boundary)
                if last_boundary != -1:
                    split_index = last_boundary + len(boundary)
                    break

        # If still no boundary found, just cut at max_length (least preferred)
        if split_index == -1 or split_index == 0:
            split_index = max_length

        # Add the chunk and update remaining text
        chunks.append(remaining_text[:split_index])
        remaining_text = remaining_text[split_index:]

    # Add any remaining text as the final chunk
    if remaining_text:
        chunks.append(remaining_text)

    return chunks


@mcp.tool(description="Create text-to-speech audio using OpenAI's TTS API with model and voice selection.")
async def create_claudecast(
    inputs: list[CreateClaudecastInputParams],
) -> list[dict[str, Path]]:
    """Generate text-to-speech audio files from text prompts with customizable voices.

    Options:
    - model: Choose between tts-1 (faster, lower quality) or tts-1-hd (higher quality)
    - voice: Select from multiple voice options (alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer)
    - text_prompt: The text content to convert to speech (supports any length; automatically splits long text)
    - output_file_path: Optional custom path for the output file (defaults to speech.mp3)

    Returns the path to the generated audio file.

    Note: Handles texts of any length by splitting into chunks at natural boundaries and
    concatenating the audio. OpenAI's TTS API has a limit of 4096 characters per request.
    """

    async def process_single(input_data: CreateClaudecastInputParams) -> dict[str, Path]:
        try:
            # Set default output path if not provided
            output_path = input_data.output_file_path
            if output_path is None:
                # Create output directory if it doesn't exist
                audio_path = check_and_get_audio_path()
                output_path = audio_path / f"speech_{time.time_ns()}.mp3"

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            client = AsyncOpenAI()

            # Split text if it exceeds the API limit (with buffer)
            text_chunks = split_text_for_tts(input_data.text_prompt)

            speech_create_input = {
                k: v
                for k, v in input_data.model_dump(exclude_none=True).items()
                if k not in ["output_file_path", "text_prompt"]
            }

            if len(text_chunks) == 1:
                # For single chunk, process directly
                response = await client.audio.speech.create(input=text_chunks[0], **speech_create_input)

                # Stream to file using aiofiles for async IO
                audio_bytes = await response.aread()
                async with aiofiles.open(output_path, "wb") as file:
                    await file.write(audio_bytes)

            else:
                # For multiple chunks, process in parallel and concatenate
                print(f"Text exceeds TTS API limit, splitting into {len(text_chunks)} chunks")

                # Create temporary directory for chunk files
                import tempfile

                temp_dir = Path(tempfile.mkdtemp())

                # Process each chunk in parallel
                async def process_chunk(chunk_text: str, chunk_index: int) -> Path:
                    chunk_path = temp_dir / f"chunk_{chunk_index}.mp3"
                    response = await client.audio.speech.create(input=chunk_text, **speech_create_input)

                    audio_bytes = await response.aread()
                    async with aiofiles.open(chunk_path, "wb") as file:
                        await file.write(audio_bytes)

                    return chunk_path

                # Process all chunks concurrently
                chunk_paths = await asyncio.gather(*[process_chunk(chunk, i) for i, chunk in enumerate(text_chunks)])

                # Concatenate audio files using pydub
                combined = AudioSegment.empty()
                for chunk_path in chunk_paths:
                    segment = await asyncio.to_thread(AudioSegment.from_mp3, str(chunk_path))
                    combined += segment

                # Export the final combined audio
                await asyncio.to_thread(combined.export, str(output_path), format="mp3")

                # Clean up temporary files
                for chunk_path in chunk_paths:
                    chunk_path.unlink(missing_ok=True)
                temp_dir.rmdir()

            return {"output_path": output_path}

        except Exception as e:
            raise RuntimeError(f"Text-to-speech generation failed: {str(e)}") from e

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


def main() -> None:
    """Run main entrypoint."""
    mcp.run()


if __name__ == "__main__":
    main()
