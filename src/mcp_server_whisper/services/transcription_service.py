"""Transcription service - orchestrates transcription operations."""

from typing import Literal

from ..infrastructure import FileSystemRepository, OpenAIClientWrapper, SecurePathResolver
from ..models import ChatResult, TranscriptionResult


class TranscriptionService:
    """Service for audio transcription operations."""

    def __init__(
        self, file_repo: FileSystemRepository, openai_client: OpenAIClientWrapper, path_resolver: SecurePathResolver
    ):
        """Initialize the transcription service.

        Args:
        ----
            file_repo: File system repository for I/O operations.
            openai_client: OpenAI API client wrapper.
            path_resolver: Secure path resolver for filename to path conversion.

        """
        self.file_repo = file_repo
        self.openai_client = openai_client
        self.path_resolver = path_resolver

    async def transcribe_audio(
        self,
        filename: str,
        model: str = "gpt-4o-mini-transcribe",
        response_format: str = "text",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio file using OpenAI's transcription API.

        Args:
        ----
            filename: Name of the audio file.
            model: Transcription model to use.
            response_format: Format of the response.
            prompt: Optional prompt to guide transcription.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            TranscriptionResult: Transcription result with typed fields.

        """
        # Resolve filename to path
        file_path = self.path_resolver.resolve_input(filename)

        # Read audio file
        audio_bytes = await self.file_repo.read_audio_file(file_path)

        # Transcribe using OpenAI
        result_dict = await self.openai_client.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            model=model,  # type: ignore
            response_format=response_format,  # type: ignore
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )

        # Convert dict to typed response model
        return TranscriptionResult(**result_dict)

    async def chat_with_audio(
        self,
        filename: str,
        model: str = "gpt-4o-audio-preview-2024-12-17",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> ChatResult:
        """Chat with audio using GPT-4o audio models.

        Args:
        ----
            filename: Name of the audio file.
            model: Audio chat model to use.
            system_prompt: Optional system prompt.
            user_prompt: Optional user text prompt.

        Returns:
        -------
            ChatResult: Chat response with typed text field.

        """
        # Resolve filename to path
        file_path = self.path_resolver.resolve_input(filename)

        # Validate format
        ext = file_path.suffix.lower().replace(".", "")
        if ext not in ["mp3", "wav"]:
            raise ValueError(f"Expected mp3 or wav extension, but got {ext}")

        # Read audio file
        audio_bytes = await self.file_repo.read_audio_file(file_path)

        # Chat with audio using OpenAI
        result_dict = await self.openai_client.chat_with_audio(
            audio_bytes=audio_bytes,
            audio_format=ext,  # type: ignore
            model=model,  # type: ignore
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Convert dict to typed response model
        return ChatResult(**result_dict)
