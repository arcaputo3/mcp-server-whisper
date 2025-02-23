import base64
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
import os
import asyncio
import aiofiles
from pydub import AudioSegment
from typing import Literal

SupportedAudioFormat = Literal["mp3", "wav"]
AudioLLM = Literal["gpt-4o-audio-preview-2024-10-01"]
AudioTranscription = Literal["whisper-1"]
EnhancementType = Literal["detailed", "storytelling", "professional", "analytical"]

mcp = FastMCP("whisper", dependencies=["openai", "pydub", "aiofiles"])


async def convert_to_supported_format(
    input_file: str,
    output_path: str | None = None,
    target_format: SupportedAudioFormat = "mp3",
) -> str:
    """
    Async version of audio file conversion using pydub.
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
        
        await asyncio.to_thread(
            audio.export, output_path, format=target_format, parameters=["-ac", "2"]
        )
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {str(e)}")


async def compress_mp3_file(
    mp3_file_path: str, output_path: str | None = None, out_sample_rate: int = 11025
) -> str:
    """
    Downsample an existing mp3. If no output_path provided, returns a file named 'compressed_{original_stem}.mp3'.
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
        audio_file = await asyncio.to_thread(
            AudioSegment.from_file,
            mp3_file_path,
            format="mp3"
        )
        original_frame_rate = audio_file.frame_rate
        print(
            f"[Compression] Original frame rate: {original_frame_rate}, converting to {out_sample_rate}."
        )
        await asyncio.to_thread(
            audio_file.export,
            output_path,
            format="mp3",
            parameters=["-ar", str(out_sample_rate)],
        )
        return output_path
    except Exception as e:
        raise Exception(f"Error compressing mp3 file: {str(e)}")


async def maybe_compress_file(
    input_file: str, output_path: str | None = None, max_mb: int = 25
) -> str:
    """
    If file is above {max_mb}, convert to mp3 if needed, then compress.
    If no output_path provided, returns the compressed_{stem}.mp3 path if compression happens,
    otherwise returns the original path.
    """
    async with aiofiles.open(input_file, "rb") as f:
        file_size = len(await f.read())
    threshold_bytes = max_mb * 1024 * 1024

    if file_size <= threshold_bytes:
        return input_file  # No compression needed

    print(
        f"\n[maybe_compress_file] File '{input_file}' size > {max_mb}MB. Attempting compression..."
    )

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
async def convert_audio(
    input_file: str,
    output_path: Optional[str] = None,
    target_format: SupportedAudioFormat = "mp3",
) -> Dict[str, str]:
    """
    Convert an audio file to a supported format (mp3 or wav)
    """
    try:
        output_file = await convert_to_supported_format(
            input_file, output_path, target_format
        )
        return {"output_path": output_file}
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {str(e)}")


@mcp.tool()
async def compress_audio(
    input_file: str, output_path: Optional[str] = None, max_mb: int = 25
) -> Dict[str, str]:
    """
    Compress an audio file if it's larger than max_mb
    """
    try:
        output_file = await maybe_compress_file(input_file, output_path, max_mb)
        return {"output_path": output_file}
    except Exception as e:
        raise RuntimeError(f"Audio compression failed: {str(e)}")


@mcp.tool()
async def transcribe_audio(
    file_path: str,
    model: AudioTranscription = "whisper-1",
) -> Dict[str, Any]:
    """
    Basic audio transcription using Whisper API.
    Raises an exception on failure, so MCP returns a proper JSON error.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Prepare OpenAI client
    client = AsyncOpenAI()

    try:
        with open(file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model=model, file=audio_file, response_format="text"
            )
    except Exception as e:
        raise RuntimeError(f"Whisper processing failed: {e}") from e

    return {"text": transcript}


@mcp.tool()
async def transcribe_with_llm(
    file_path: str,
    text_prompt: Optional[str] = None,
    model: AudioLLM = "gpt-4o-audio-preview-2024-10-01",
) -> Dict[str, Any]:
    """
    Transcribe audio using GPT-4 with optional text prompt for context.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get format from extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().replace(".", "")

    # Load and encode audio file
    try:
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed reading audio file '{file_path}': {e}") from e

    # Prepare OpenAI client
    client = AsyncOpenAI()

    # Construct message content
    user_content = []
    if text_prompt:
        user_content.append({"type": "text", "text": text_prompt})
    user_content.append(
        {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": ext},
        }
    )

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            modalities=["text"],
        )
    except Exception as e:
        raise RuntimeError(f"GPT-4 processing failed: {e}") from e

    response = completion.choices[0].message
    return {"text": response.content}


@mcp.tool()
async def transcribe_with_enhancement(
    file_path: str,
    enhancement_type: EnhancementType = "detailed",
    model: AudioLLM = "gpt-4o-audio-preview-2024-10-01",
) -> Dict[str, Any]:
    """
    Transcribe audio with GPT-4 using specific enhancement prompts.
    Enhancement types:
    - detailed: Provides detailed description including tone, emotion, and background
    - storytelling: Transforms the transcription into a narrative
    - professional: Formats the transcription in a formal, business-appropriate way
    - analytical: Includes analysis of speech patterns, key points, and structure
    """
    enhancement_prompts = {
        "detailed": "Please transcribe this audio and include details about tone of voice, emotional undertones, and any background elements you notice. Make it rich and descriptive.",
        "storytelling": "Transform this audio into an engaging narrative. Maintain the core message but present it as a story.",
        "professional": "Transcribe this audio and format it in a professional, business-appropriate manner. Clean up any verbal fillers and structure it clearly.",
        "analytical": "Transcribe this audio and analyze the speech patterns, key discussion points, and overall structure. Include observations about delivery and organization.",
    }

    if enhancement_type not in enhancement_prompts:
        raise ValueError(
            f"Invalid enhancement_type. Must be one of: {', '.join(enhancement_prompts.keys())}"
        )

    return await transcribe_with_llm(
        file_path=file_path,
        text_prompt=enhancement_prompts[enhancement_type],
        model=model,
    )


def main():
    """Main entry point."""
    mcp.run()


if __name__ == "__main__":
    main()
