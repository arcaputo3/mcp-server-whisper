"""MCP tools for transcription operations."""

from typing import Literal, Optional

from openai.types import AudioModel, AudioResponseFormat

from ..config import check_and_get_audio_path
from ..constants import ENHANCEMENT_PROMPTS, AudioChatModel, EnhancementType
from ..infrastructure import FileSystemRepository, MCPServer, OpenAIClientWrapper, SecurePathResolver
from ..models import ChatResult, TranscriptionResult
from ..services import TranscriptionService


def create_transcription_tools(mcp: MCPServer) -> None:
    """Register transcription tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    openai_client = OpenAIClientWrapper()
    path_resolver = SecurePathResolver(audio_path)
    transcription_service = TranscriptionService(file_repo, openai_client, path_resolver)

    @mcp.tool(
        description=(
            "A tool used to transcribe audio files. "
            "It is recommended to use `gpt-4o-mini-transcribe` by default. "
            "If the user wants maximum performance, use `gpt-4o-transcribe`. "
            "Rarely should you use `whisper-1` as it is least performant, but it is available if needed. "
            "You can use prompts to guide the transcription process based on the users preference."
        )
    )
    async def transcribe_audio(
        input_file_name: str,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        prompt: Optional[str] = None,
        timestamp_granularities: Optional[list[Literal["word", "segment"]]] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI's transcribe API.

        Args:
            input_file_name: Name of the input audio file to process
            model: The transcription model to use
            response_format: The response format (text, json, verbose_json, etc.)
            prompt: Optional prompt to guide the transcription
            timestamp_granularities: Optional timestamp granularities (word, segment)

        Returns:
        -------
            TranscriptionResult with transcribed text and metadata

        """
        return await transcription_service.transcribe_audio(
            filename=input_file_name,
            model=model,
            response_format=response_format,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )

    @mcp.tool(
        description=(
            "A tool used to chat with audio files. The response will be a response to the audio file sent. "
            "It is recommended to use `gpt-4o-audio-preview` by default for best results. "
            "Note: `gpt-4o-mini-audio-preview` has limitations with audio chat and may not process audio correctly."
        )
    )
    async def chat_with_audio(
        input_file_name: str,
        model: AudioChatModel = "gpt-4o-audio-preview",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> ChatResult:
        """Chat with audio using GPT-4o audio models.

        Args:
            input_file_name: Name of the input audio file to process
            model: The audio LLM model to use for transcription
            system_prompt: Optional system prompt
            user_prompt: Optional user prompt

        Returns:
        -------
            ChatResult with the response text

        """
        return await transcription_service.chat_with_audio(
            filename=input_file_name,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    @mcp.tool()
    async def transcribe_with_enhancement(
        input_file_name: str,
        enhancement_type: EnhancementType = "detailed",
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        timestamp_granularities: Optional[list[Literal["word", "segment"]]] = None,
    ) -> TranscriptionResult:
        """Transcribe audio with GPT-4 using specific enhancement prompts.

        Enhancement types:
        - detailed: Provides detailed description including tone, emotion, and background
        - storytelling: Transforms the transcription into a narrative
        - professional: Formats the transcription in a formal, business-appropriate way
        - analytical: Includes analysis of speech patterns, key points, and structure

        Args:
            input_file_name: Name of the input audio file to process
            enhancement_type: Type of enhancement to apply to the transcription
            model: The transcription model to use
            response_format: The response format
            timestamp_granularities: Optional timestamp granularities

        Returns:
        -------
            TranscriptionResult with enhanced transcription

        """
        # Get the enhancement prompt
        prompt = ENHANCEMENT_PROMPTS[enhancement_type]

        # Call transcribe_audio with the enhancement prompt
        return await transcription_service.transcribe_audio(
            filename=input_file_name,
            model=model,
            response_format=response_format,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )
