"""MCP tools for text-to-speech operations."""

from pathlib import Path

from ..config import check_and_get_audio_path
from ..infrastructure import FileSystemRepository, OpenAIClientWrapper
from ..models import CreateClaudecastInputParams
from ..services import TTSService


def create_tts_tools(mcp):
    """Register TTS tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    openai_client = OpenAIClientWrapper()
    tts_service = TTSService(file_repo, openai_client)

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
        return await tts_service.create_speech_batch(inputs)
