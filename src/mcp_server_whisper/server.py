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
from openai.types import AudioModel
from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field
from pydub import AudioSegment  # type: ignore

# Literals for transcription
SupportedAudioFormat = Literal["mp3", "wav"]
AudioLLM = Literal["gpt-4o-audio-preview-2024-10-01"]
EnhancementType = Literal["detailed", "storytelling", "professional", "analytical"]
TTSModel = Literal["tts-1", "tts-1-hd"]
TTSVoice = Literal["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]

# Constants for checks
WHISPER_AUDIO_FORMATS = {".mp3", ".wav", ".mp4", ".mpeg", ".mpga", ".m4a", ".webm"}
GPT_4O_AUDIO_FORMATS = {".mp3", ".wav"}

# Enhancement prompts
ENHANCEMENT_PROMPTS: dict[EnhancementType, str] = {
    "detailed": "Please transcribe this audio and include details about tone of voice, emotional undertones, "
    "and any background elements you notice. Make it rich and descriptive.",
    "storytelling": "Transform this audio into an engaging narrative. "
    "Maintain the core message but present it as a story.",
    "professional": "Transcribe this audio and format it in a professional, business-appropriate manner. "
    "Clean up any verbal fillers and structure it clearly.",
    "analytical": "Transcribe this audio and analyze the speech patterns, key discussion points, "
    "and overall structure. Include observations about delivery and organization.",
}


class BaseInputPath(BaseModel):
    """Base file path input."""

    input_file_path: Path

    model_config = {"arbitrary_types_allowed": True}


class BaseAudioInputParams(BaseInputPath):
    """Base params for converting audio to mp3."""

    output_file_path: Optional[Path] = None


class ConvertAudioInputParams(BaseAudioInputParams):
    """Params for converting audio to mp3."""

    target_format: SupportedAudioFormat = "mp3"


class CompressAudioInputParams(BaseAudioInputParams):
    """Params for compressing audio."""

    max_mb: int = Field(default=25, gt=0)


class TranscribeAudioInputParams(BaseInputPath):
    """Params for transcribing audio with audio-to-text model."""

    model: AudioModel = "whisper-1"


class TranscribeWithLLMInputParams(BaseInputPath):
    """Params for transcribing audio with LLM using custom prompt."""

    text_prompt: Optional[str] = None
    model: AudioLLM = "gpt-4o-audio-preview-2024-10-01"


class TranscribeWithEnhancementInputParams(BaseInputPath):
    """Params for transcribing audio with LLM using template prompt."""

    enhancement_type: EnhancementType = "detailed"
    model: AudioLLM = "gpt-4o-audio-preview-2024-10-01"

    def to_transcribe_with_llm_input_params(self) -> TranscribeWithLLMInputParams:
        """Transfer audio with LLM using custom prompt."""
        return TranscribeWithLLMInputParams(
            input_file_path=self.input_file_path,
            text_prompt=ENHANCEMENT_PROMPTS[self.enhancement_type],
            model=self.model,
        )


class CreateClaudecastInputParams(BaseModel):
    """Params for text-to-speech using OpenAI's API."""

    text_prompt: str
    """Text to convert to speech"""

    output_file_path: Optional[Path] = None
    """Output file path (defaults to speech.mp3 in current directory)"""

    model: TTSModel = "tts-1-hd"
    """TTS model to use"""

    voice: TTSVoice = "nova"
    """Voice for the TTS"""

    model_config = {"arbitrary_types_allowed": True}


class FilePathSupportParams(BaseModel):
    """Params for checking if a file at a path supports transcription."""

    file_path: Path
    transcription_support: Optional[list[AudioModel]] = None
    llm_support: Optional[list[AudioLLM]] = None
    modified_time: float
    size_bytes: int
    format: str
    duration_seconds: Optional[float] = None

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

    transcription_support: list[AudioModel] | None = ["whisper-1"] if file_ext in WHISPER_AUDIO_FORMATS else None
    llm_support: list[Literal["gpt-4o-audio-preview-2024-10-01"]] | None = (
        ["gpt-4o-audio-preview-2024-10-01"] if file_ext in GPT_4O_AUDIO_FORMATS else None
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
        llm_support=llm_support,
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
            if file_ext in WHISPER_AUDIO_FORMATS or file_ext in GPT_4O_AUDIO_FORMATS:
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

    pattern: Optional[str] = None
    """Optional regex pattern to filter audio files by name"""

    min_size_bytes: Optional[int] = None
    """Minimum file size in bytes"""

    max_size_bytes: Optional[int] = None
    """Maximum file size in bytes"""

    min_duration_seconds: Optional[float] = None
    """Minimum audio duration in seconds"""

    max_duration_seconds: Optional[float] = None
    """Maximum audio duration in seconds"""

    min_modified_time: Optional[float] = None
    """Minimum file modification time (Unix timestamp)"""

    max_modified_time: Optional[float] = None
    """Maximum file modification time (Unix timestamp)"""

    format: Optional[str] = None
    """Specific audio format to filter by (e.g., 'mp3', 'wav')"""

    sort_by: SortBy = SortBy.NAME
    """Field to sort results by"""

    reverse: bool = False
    """Sort in reverse order if True"""

    model_config = {"arbitrary_types_allowed": True}


@mcp.tool(
    description="List, filter, and sort audio files from the audio path. Supports regex pattern matching, "
    "filtering by metadata (size, duration, date, format), and sorting."
)
async def list_audio_files(inputs: list[ListAudioFilesInputParams]) -> list[list[FilePathSupportParams]]:
    """List, filter, and sort audio files in the AUDIO_FILES_PATH directory with comprehensive options.

    Supported formats:
    - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - GPT-4o: mp3, wav

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
                if file_ext in WHISPER_AUDIO_FORMATS or file_ext in GPT_4O_AUDIO_FORMATS:
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
    target_format: SupportedAudioFormat = "mp3",
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


@mcp.tool()
async def transcribe_audio(inputs: list[TranscribeAudioInputParams]) -> list[dict[str, Any]]:
    """Transcribe audio using Whisper API for multiple files in parallel.

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

            transcript = await client.audio.transcriptions.create(
                model=input_data.model, file=file_obj, response_format="text"
            )
            return {"text": transcript}
        except Exception as e:
            raise RuntimeError(f"Whisper processing failed for {file_path}: {e}") from e

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool()
async def transcribe_with_llm(
    inputs: list[TranscribeWithLLMInputParams],
) -> list[dict[str, Any]]:
    """Transcribe multiple audio files using GPT-4 with optional text prompts in parallel."""

    async def process_single(input_data: TranscribeWithLLMInputParams) -> dict[str, Any]:
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
        user_content: list[ChatCompletionContentPartParam] = []
        if input_data.text_prompt:
            user_content.append({"type": "text", "text": input_data.text_prompt})
        user_content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": cast(Literal["wav", "mp3"], ext)},
            }
        )

        try:
            completion = await client.chat.completions.create(
                model=input_data.model,
                messages=[{"role": "user", "content": user_content}],
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
    converted_inputs = [input_.to_transcribe_with_llm_input_params() for input_ in inputs]
    result: list[dict[str, Any]] = await transcribe_with_llm(converted_inputs)
    return result


def split_text_for_tts(text: str, max_length: int = 4000) -> list[str]:
    """Split text into chunks that don't exceed the TTS API limit.

    The function splits text at sentence boundaries (periods, question marks, exclamation points)
    to create natural-sounding chunks. If a sentence is too long, it falls back to
    splitting at commas, then spaces.

    Args:
        text: The text to split
        max_length: Maximum character length for each chunk (default 4000 to provide buffer)

    Returns:
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

            if len(text_chunks) == 1:
                # For single chunk, process directly
                response = await client.audio.speech.create(
                    model=input_data.model,
                    voice=input_data.voice,
                    input=text_chunks[0],
                )

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

                    response = await client.audio.speech.create(
                        model=input_data.model,
                        voice=input_data.voice,
                        input=chunk_text,
                    )

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
