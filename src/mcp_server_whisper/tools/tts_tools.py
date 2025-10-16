"""MCP tools for text-to-speech operations."""

from typing import Optional

from openai.types.audio.speech_model import SpeechModel

from ..config import check_and_get_audio_path
from ..constants import TTSVoice
from ..infrastructure import FileSystemRepository, MCPServer, OpenAIClientWrapper, SecurePathResolver
from ..models import TTSResult
from ..services import TTSService


def create_tts_tools(mcp: MCPServer) -> None:
    """Register TTS tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    openai_client = OpenAIClientWrapper()
    path_resolver = SecurePathResolver(audio_path)
    tts_service = TTSService(file_repo, openai_client, path_resolver)

    @mcp.tool(description="Create text-to-speech audio using OpenAI's TTS API with model and voice selection.")
    async def create_audio(
        text_prompt: str,
        model: SpeechModel = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: Optional[str] = None,
        speed: float = 1.0,
        output_file_name: Optional[str] = None,
    ) -> TTSResult:
        """Generate text-to-speech audio from text prompts with customizable voices.

        Args:
            text_prompt: Text to convert to speech
            model: TTS model to use (gpt-4o-mini-tts is preferred)
            voice: Voice for the TTS (alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer)
            instructions: Optional instructions for speech conversion (tonality, accent, style, etc.)
            speed: Speed of the speech conversion (0.25 to 4.0)
            output_file_name: Optional custom name for the output file (defaults to speech_<timestamp>.mp3)

        Returns:
        -------
            TTSResult with name of the generated audio file

        Note:
        ----
        Handles texts of any length by splitting into chunks at natural boundaries and
        concatenating the audio. OpenAI's TTS API has a limit of 4096 characters per request.

        """
        return await tts_service.create_speech(
            text_prompt=text_prompt,
            output_filename=output_file_name,
            model=model,
            voice=voice,
            instructions=instructions,
            speed=speed,
        )
