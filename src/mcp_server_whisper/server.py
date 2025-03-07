"""Whisper MCP server core code."""

import asyncio
import base64
import os
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


class FilePathSupportParams(BaseModel):
    """Params for checking if a file at a path supports transcription."""

    file_path: Path
    transcription_support: Optional[list[AudioModel]] = None
    llm_support: Optional[list[AudioLLM]] = None
    modified_time: float

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


def get_audio_file_support(file_path: Path) -> FilePathSupportParams:
    """Determine audio transcription file format support."""
    file_ext = file_path.suffix.lower()

    transcription_support: list[AudioModel] | None = ["whisper-1"] if file_ext in WHISPER_AUDIO_FORMATS else None
    llm_support: list[Literal["gpt-4o-audio-preview-2024-10-01"]] | None = (
        ["gpt-4o-audio-preview-2024-10-01"] if file_ext in GPT_4O_AUDIO_FORMATS else None
    )

    return FilePathSupportParams(
        file_path=file_path,
        transcription_support=transcription_support,
        llm_support=llm_support,
        modified_time=file_path.stat().st_mtime,
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
        return get_audio_file_support(latest_file)

    except Exception as e:
        raise RuntimeError(f"Failed to get latest audio file: {e}") from e


@mcp.resource("dir://audio", description="List audio files from the audio path.")
def list_audio_files() -> list[FilePathSupportParams]:
    """List all audio files in the AUDIO_FILES_PATH directory with format support info.

    Supported formats:
    - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - GPT-4o: mp3, wav
    """
    audio_path = check_and_get_audio_path()

    try:
        files = []
        for file_path in audio_path.iterdir():
            if not file_path.is_file():
                continue

            file_ext = file_path.suffix.lower()
            if file_ext in WHISPER_AUDIO_FORMATS or file_ext in WHISPER_AUDIO_FORMATS:
                files.append(get_audio_file_support(file_path))

        return sorted(files, key=lambda x: str(x.file_path))

    except Exception as e:
        raise RuntimeError(f"Failed to list audio files: {e}") from e


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
    return await transcribe_with_llm([input_.to_transcribe_with_llm_input_params() for input_ in inputs])


def main() -> None:
    """Run main entrypoint."""
    mcp.run()


if __name__ == "__main__":
    main()
