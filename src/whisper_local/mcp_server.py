import os
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP

# Import existing utility functions for file conversion and compression
from whisper_local.convert import convert_to_supported_format, maybe_compress_file

mcp = FastMCP("AudioGPT")


@mcp.tool()
async def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file to text using the OpenAI Whisper API via MCP.

    Args:
        file_path (str): Path to the local audio file.
    Returns:
        str: The transcribed text.
    Raises:
        ValueError: If the file is missing, or processing fails.
        RuntimeError: If the OpenAI transcription call fails.
    """
    from dotenv import load_dotenv

    load_dotenv()
    client = AsyncOpenAI()
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Attempt to convert unsupported formats to mp3
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().replace(".", "")
    if ext not in ("wav", "mp3"):
        try:
            file_path = await convert_to_supported_format(file_path, target_format="mp3")
        except Exception as e:
            raise ValueError(f"Error converting file: {str(e)}")

    # Attempt compression if over 25 MB
    try:
        file_path = await maybe_compress_file(file_path, max_mb=25)
    except Exception as e:
        raise ValueError(f"Error during compression: {str(e)}")

    # Final size check
    max_size = 25 * 1024 * 1024  # 25 MB
    if os.path.getsize(file_path) > max_size:
        raise ValueError(f"File '{file_path}' is still above 25MB after compression.")

    # Perform transcription via OpenAI Whisper API
    try:
        with open(file_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
            return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    # Run the MCP server over stdio (blocking call)
    mcp.run()
