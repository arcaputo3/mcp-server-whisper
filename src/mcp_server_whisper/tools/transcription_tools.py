"""MCP tools for transcription operations."""

from typing import Any

from ..config import check_and_get_audio_path
from ..infrastructure import FileSystemRepository, OpenAIClientWrapper
from ..models import ChatWithAudioInputParams, TranscribeAudioInputParams, TranscribeWithEnhancementInputParams
from ..services import TranscriptionService


def create_transcription_tools(mcp):
    """Register transcription tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    openai_client = OpenAIClientWrapper()
    transcription_service = TranscriptionService(file_repo, openai_client)

    @mcp.tool(
        description=(
            "A tool used to transcribe audio files. "
            "It is recommended to use `gpt-4o-mini-transcribe` by default. "
            "If the user wants maximum performance, use `gpt-4o-transcribe`. "
            "Rarely should you use `whisper-1` as it is least performant, but it is available if needed. "
            "You can use prompts to guide the transcription process based on the users preference."
        )
    )
    async def transcribe_audio(inputs: list[TranscribeAudioInputParams]) -> list[dict[str, Any]]:
        """Transcribe audio using OpenAI's transcribe API for multiple files in parallel.

        Raises an exception on failure, so MCP returns a proper JSON error.
        """
        return await transcription_service.transcribe_audio_batch(inputs)

    @mcp.tool(
        description="A tool used to chat with audio files. The response will be a response to the audio file sent."
    )
    async def chat_with_audio(
        inputs: list[ChatWithAudioInputParams],
    ) -> list[dict[str, Any]]:
        """Transcribe multiple audio files using GPT-4 with optional text prompts in parallel."""
        return await transcription_service.chat_with_audio_batch(inputs)

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
        return await transcription_service.transcribe_with_enhancement_batch(inputs)
