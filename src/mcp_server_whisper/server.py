"""Whisper MCP server core code."""

import asyncio
import base64
import os
from typing import Any, Literal, Optional, cast

import aiofiles
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from openai.types import AudioModel
from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field
from pydub import AudioSegment  # type: ignore

SupportedAudioFormat = Literal["mp3", "wav"]
AudioLLM = Literal["gpt-4o-audio-preview-2024-10-01"]
EnhancementType = Literal["detailed", "storytelling", "professional", "analytical"]


class ConvertAudioInputParams(BaseModel):
    """Params for converting audio to mp3."""

    input_file: str
    output_path: Optional[str] = None
    target_format: SupportedAudioFormat = "mp3"


class CompressAudioInputParams(BaseModel):
    """Params for compressing audio."""

    input_file: str
    output_path: Optional[str] = None
    max_mb: int = Field(default=25, gt=0)


class TranscribeAudioInputParams(BaseModel):
    """Params for transcribing audio with audio-to-text model."""

    file_path: str
    model: AudioModel = "whisper-1"


class TranscribeWithLLMInputParams(BaseModel):
    """Params for transcribing audio with LLM using custom prompt."""

    file_path: str
    text_prompt: Optional[str] = None
    model: AudioLLM = "gpt-4o-audio-preview-2024-10-01"


class TranscribeWithEnhancementInputParams(BaseModel):
    """Params for transcribing audio with LLM using template prompt."""

    file_path: str
    enhancement_type: EnhancementType = "detailed"
    model: AudioLLM = "gpt-4o-audio-preview-2024-10-01"


class FilePathSupportParams(BaseModel):
    """Params for checking if a file at a path supports transcription."""

    path: str
    transcription_support: Optional[list[AudioModel]] = None
    llm_support: Optional[list[AudioLLM]] = None
    modified_time: float


mcp = FastMCP("whisper", dependencies=["openai", "pydub", "aiofiles"])


def get_audio_file_support(file_path: str) -> FilePathSupportParams:
    """Determine audio transcription file format support."""
    whisper_formats = {".mp3", ".wav", ".mp4", ".mpeg", ".mpga", ".m4a", ".webm"}
    gpt4o_formats = {".mp3", ".wav"}

    file_ext = os.path.splitext(file_path.lower())[1]

    transcription_support: list[AudioModel] | None = ["whisper-1"] if file_ext in whisper_formats else None
    llm_support: list[Literal["gpt-4o-audio-preview-2024-10-01"]] | None = (
        ["gpt-4o-audio-preview-2024-10-01"] if file_ext in gpt4o_formats else None
    )

    return FilePathSupportParams(
        path=file_path,
        transcription_support=transcription_support,
        llm_support=llm_support,
        modified_time=os.path.getmtime(file_path),
    )


@mcp.tool()
async def get_latest_audio() -> FilePathSupportParams:
    """Get the most recently modified audio file and returns its path with model support info.

    Supported formats:
    - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - GPT-4o: mp3, wav
    """
    audio_path = os.getenv("AUDIO_FILES_PATH")
    if not audio_path:
        raise ValueError("AUDIO_FILES_PATH environment variable not set")

    whisper_formats = {".mp3", ".wav", ".mp4", ".mpeg", ".mpga", ".m4a", ".webm"}
    gpt4o_formats = {".mp3", ".wav"}

    try:
        files = []
        for file in os.listdir(audio_path):
            file_ext = os.path.splitext(file.lower())[1]
            if file_ext in whisper_formats or file_ext in gpt4o_formats:
                full_path = os.path.abspath(os.path.join(audio_path, file))
                files.append((full_path, os.path.getmtime(full_path)))

        if not files:
            raise RuntimeError("No supported audio files found")

        latest_file = max(files, key=lambda x: x[1])[0]
        return get_audio_file_support(latest_file)

    except Exception as e:
        raise RuntimeError(f"Failed to get latest audio file: {e}") from e


@mcp.resource("dir://audio")
def list_audio_files() -> list[FilePathSupportParams]:
    """List all audio files in the AUDIO_FILES_PATH directory with format support info.

    Supported formats:
    - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
    - GPT-4o: mp3, wav
    """
    audio_path = os.getenv("AUDIO_FILES_PATH")
    if not audio_path:
        raise ValueError("AUDIO_FILES_PATH environment variable not set")

    whisper_formats = {".mp3", ".wav", ".mp4", ".mpeg", ".mpga", ".m4a", ".webm"}
    gpt4o_formats = {".mp3", ".wav"}

    try:
        files = []
        for file in os.listdir(audio_path):
            file_ext = os.path.splitext(file.lower())[1]
            if file_ext in whisper_formats or file_ext in gpt4o_formats:
                abs_path = os.path.abspath(os.path.join(audio_path, file))
                files.append(get_audio_file_support(abs_path))

        return sorted(files, key=lambda x: x.path)

    except Exception as e:
        raise RuntimeError(f"Failed to list audio files: {e}") from e


async def convert_to_supported_format(
    input_file: str,
    output_path: str | None = None,
    target_format: SupportedAudioFormat = "mp3",
) -> str:
    """Async version of audio file conversion using pydub.

    Ensures the output filename is base + .{target_format} if no output_path provided.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_file)
        output_path = base + f".{target_format}"

    try:
        # Load audio file directly from path instead of reading bytes first
        audio = await asyncio.to_thread(
            AudioSegment.from_file,
            input_file,
            format=os.path.splitext(input_file)[1][1:],
        )

        await asyncio.to_thread(audio.export, output_path, format=target_format, parameters=["-ac", "2"])
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {str(e)}")


async def compress_mp3_file(mp3_file_path: str, output_path: str | None = None, out_sample_rate: int = 11025) -> str:
    """Downsample an existing mp3.

    If no output_path provided, returns a file named 'compressed_{original_stem}.mp3'.
    """
    if not mp3_file_path.lower().endswith(".mp3"):
        raise Exception("compress_mp3_file() called on a file that is not .mp3")

    if output_path is None:
        basename = os.path.basename(mp3_file_path)
        name_no_ext, _ = os.path.splitext(basename)
        output_path = f"compressed_{name_no_ext}.mp3"

    print(f"\n[Compression] Original file: {mp3_file_path}")
    print(f"[Compression] Output file:   {output_path}")

    try:
        # Load audio file directly from path instead of reading bytes first
        audio_file = await asyncio.to_thread(AudioSegment.from_file, mp3_file_path, format="mp3")
        original_frame_rate = audio_file.frame_rate
        print(f"[Compression] Original frame rate: {original_frame_rate}, converting to {out_sample_rate}.")
        await asyncio.to_thread(
            audio_file.export,
            output_path,
            format="mp3",
            parameters=["-ar", str(out_sample_rate)],
        )
        return output_path
    except Exception as e:
        raise Exception(f"Error compressing mp3 file: {str(e)}")


async def maybe_compress_file(input_file: str, output_path: str | None = None, max_mb: int = 25) -> str:
    """Compress file if is above {max_mb} and convert to mp3 if needed.

    If no output_path provided, returns the compressed_{stem}.mp3 path if compression happens,
    otherwise returns the original path.
    """
    async with aiofiles.open(input_file, "rb") as f:
        file_size = len(await f.read())
    threshold_bytes = max_mb * 1024 * 1024

    if file_size <= threshold_bytes:
        return input_file  # No compression needed

    print(f"\n[maybe_compress_file] File '{input_file}' size > {max_mb}MB. Attempting compression...")

    # If not mp3, convert
    if not input_file.lower().endswith(".mp3"):
        try:
            input_file = await convert_to_supported_format(input_file, None, "mp3")
        except Exception as e:
            raise Exception(f"[maybe_compress_file] Error converting to MP3: {str(e)}")

    # now downsample
    try:
        compressed_path = await compress_mp3_file(input_file, output_path, 11025)
    except Exception as e:
        raise Exception(f"[maybe_compress_file] Error compressing MP3 file: {str(e)}")

    async with aiofiles.open(compressed_path, "rb") as f:
        new_size = len(await f.read())
    print(f"[maybe_compress_file] Compressed file size: {new_size} bytes")
    return compressed_path


@mcp.tool()
async def convert_audio(inputs: list[ConvertAudioInputParams]) -> list[dict[str, str]]:
    """Convert multiple audio files to supported formats (mp3 or wav) in parallel."""

    async def process_single(input_data: ConvertAudioInputParams) -> dict[str, str]:
        try:
            output_file = await convert_to_supported_format(
                input_data.input_file, input_data.output_path, input_data.target_format
            )
            return {"output_path": output_file}
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed for {input_data.input_file}: {str(e)}")

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool()
async def compress_audio(inputs: list[CompressAudioInputParams]) -> list[dict[str, str]]:
    """Compress multiple audio files in parallel if they're larger than max_mb."""

    async def process_single(input_data: CompressAudioInputParams) -> dict[str, str]:
        try:
            output_file = await maybe_compress_file(input_data.input_file, input_data.output_path, input_data.max_mb)
            return {"output_path": output_file}
        except Exception as e:
            raise RuntimeError(f"Audio compression failed for {input_data.input_file}: {str(e)}")

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool()
async def transcribe_audio(inputs: list[TranscribeAudioInputParams]) -> list[dict[str, Any]]:
    """Transcribe audio using Whisper API for multiple files in parallel.

    Raises an exception on failure, so MCP returns a proper JSON error.
    """

    async def process_single(input_data: TranscribeAudioInputParams) -> dict[str, Any]:
        if not os.path.isfile(input_data.file_path):
            raise FileNotFoundError(f"File not found: {input_data.file_path}")

        client = AsyncOpenAI()

        try:
            with open(input_data.file_path, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model=input_data.model, file=audio_file, response_format="text"
                )
            return {"text": transcript}
        except Exception as e:
            raise RuntimeError(f"Whisper processing failed for {input_data.file_path}: {e}") from e

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


@mcp.tool()
async def transcribe_with_llm(
    inputs: list[TranscribeWithLLMInputParams],
) -> list[dict[str, Any]]:
    """Transcribe multiple audio files using GPT-4 with optional text prompts in parallel."""

    async def process_single(input_data: TranscribeWithLLMInputParams) -> dict[str, Any]:
        if not os.path.isfile(input_data.file_path):
            raise FileNotFoundError(f"File not found: {input_data.file_path}")

        _, ext = os.path.splitext(input_data.file_path)
        ext = ext.lower().replace(".", "")
        assert ext in ["mp3", "wav"], f"Expected mp3 or wav extension, but got {ext}"

        try:
            with open(input_data.file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed reading audio file '{input_data.file_path}': {e}") from e

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
            raise RuntimeError(f"GPT-4 processing failed for {input_data.file_path}: {e}") from e

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
    enhancement_prompts = {
        "detailed": "Please transcribe this audio and include details about tone of voice, emotional undertones, "
        "and any background elements you notice. Make it rich and descriptive.",
        "storytelling": "Transform this audio into an engaging narrative. "
        "Maintain the core message but present it as a story.",
        "professional": "Transcribe this audio and format it in a professional, business-appropriate manner. "
        "Clean up any verbal fillers and structure it clearly.",
        "analytical": "Transcribe this audio and analyze the speech patterns, key discussion points, "
        "and overall structure. Include observations about delivery and organization.",
    }

    async def process_single(
        input_data: TranscribeWithEnhancementInputParams,
    ) -> dict[str, Any]:
        if input_data.enhancement_type not in enhancement_prompts:
            raise ValueError(f"Invalid enhancement_type. Must be one of: {', '.join(enhancement_prompts.keys())}")

        return next(
            (
                await transcribe_with_llm(
                    [
                        TranscribeWithLLMInputParams(
                            file_path=input_data.file_path,
                            text_prompt=enhancement_prompts[input_data.enhancement_type],
                            model=input_data.model,
                        )
                    ]
                )
            )
        )

    return await asyncio.gather(*[process_single(input_data) for input_data in inputs])


def main() -> None:
    """Run main entrypoint."""
    mcp.run()


if __name__ == "__main__":
    main()
