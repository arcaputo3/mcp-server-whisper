import os
import base64
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI
from .convert import convert_to_supported_format, maybe_compress_file

mcp = FastMCP("AudioGPT")

@mcp.tool()
async def transcribe_audio(
    file_path: str,
    use_gpt4o: bool = False,
    model: str = "whisper-1",
    text_prompt: Optional[str] = None,
    modalities: Optional[List[str]] = None,
    audio_output: bool = False,
) -> str:
    """Transcribe audio using either Whisper API or GPT-4o."""

    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")

    # Get format and convert if needed
    _, ext = os.path.splitext(file_path)
    ext = ext.lower().replace(".", "")
    
    if ext not in ("wav", "mp3"):
        try:
            new_path = await convert_to_supported_format(file_path, "mp3")
            file_path = new_path
            ext = "mp3"
        except Exception as e:
            raise ValueError(f"Error converting file: {e}")

    # Check size and compress if needed
    try:
        file_path = await maybe_compress_file(file_path, max_mb=25)
    except Exception as e:
        raise ValueError(f"Error during compression: {e}")

    if os.path.getsize(file_path) > 25 * 1024 * 1024:
        raise ValueError(f"File '{file_path}' still exceeds 25MB limit after compression")

    client = AsyncOpenAI()

    if use_gpt4o:
        # Load and base64-encode audio
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Build message content
        user_content = []
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})
        user_content.append(
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": ext}}
        )

        # Configure audio output if requested
        audio_config = None
        modalities_list = modalities or ["text"]
        if audio_output:
            audio_config = {"voice": "alloy", "format": "wav"}
            if "audio" not in modalities_list:
                modalities_list.append("audio")

        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                modalities=modalities_list,
                audio=audio_config,
            )
            
            response = completion.choices[0].message

            # Handle different response formats
            if isinstance(response.content, list):
                text_output = ""
                for segment in response.content:
                    if segment.get("type") == "text":
                        text_output += segment.get("text", "") + "\n"
                    elif segment.get("type") == "audio" and audio_output:
                        audio_data = segment["audio"]["data"]
                        audio_format = segment["audio"]["format"]
                        with open(f"response.{audio_format}", "wb") as f:
                            f.write(base64.b64decode(audio_data))
                return text_output.strip()
            else:
                return response.content

        except Exception as e:
            raise ValueError(f"GPT-4o processing failed: {str(e)}")

    else:  # Use Whisper
        try:
            with open(file_path, "rb") as audio_file:
                transcript = await client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="text"
                )
                return transcript
        except Exception as e:
            raise ValueError(f"Whisper processing failed: {str(e)}")

def main():
    mcp.run()

if __name__ == "__main__":
    main()