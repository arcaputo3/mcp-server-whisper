"""Transcription service - orchestrates transcription operations."""

import asyncio
from pathlib import Path
from typing import Any, Literal

from ..infrastructure import FileSystemRepository, OpenAIClientWrapper
from ..models import ChatWithAudioInputParams, TranscribeAudioInputParams, TranscribeWithEnhancementInputParams


class TranscriptionService:
    """Service for audio transcription operations."""

    def __init__(self, file_repo: FileSystemRepository, openai_client: OpenAIClientWrapper):
        """Initialize the transcription service.

        Args:
        ----
            file_repo: File system repository for I/O operations.
            openai_client: OpenAI API client wrapper.

        """
        self.file_repo = file_repo
        self.openai_client = openai_client

    async def transcribe_audio(
        self,
        file_path: Path,
        model: str = "gpt-4o-mini-transcribe",
        response_format: str = "text",
        prompt: str | None = None,
        timestamp_granularities: list[Literal["word", "segment"]] | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio file using OpenAI's transcription API.

        Args:
        ----
            file_path: Path to the audio file.
            model: Transcription model to use.
            response_format: Format of the response.
            prompt: Optional prompt to guide transcription.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            dict[str, Any]: Transcription result.

        """
        # Read audio file
        audio_bytes = await self.file_repo.read_audio_file(file_path)

        # Transcribe using OpenAI
        return await self.openai_client.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file_path.name,
            model=model,  # type: ignore
            response_format=response_format,  # type: ignore
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )

    async def transcribe_audio_batch(
        self,
        inputs: list[TranscribeAudioInputParams],
    ) -> list[dict[str, Any]]:
        """Transcribe multiple audio files in parallel.

        Args:
        ----
            inputs: List of transcription parameters.

        Returns:
        -------
            list[dict[str, Any]]: List of transcription results.

        """

        async def process_single(input_data: TranscribeAudioInputParams) -> dict[str, Any]:
            return await self.transcribe_audio(
                file_path=input_data.input_file_path,
                model=input_data.model,
                response_format=input_data.response_format,
                prompt=input_data.prompt,
                timestamp_granularities=input_data.timestamp_granularities,
            )

        return await asyncio.gather(*[process_single(input_data) for input_data in inputs])

    async def chat_with_audio(
        self,
        file_path: Path,
        model: str = "gpt-4o-audio-preview-2024-12-17",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Chat with audio using GPT-4o audio models.

        Args:
        ----
            file_path: Path to the audio file.
            model: Audio chat model to use.
            system_prompt: Optional system prompt.
            user_prompt: Optional user text prompt.

        Returns:
        -------
            dict[str, Any]: Chat completion response.

        """
        # Validate format
        ext = file_path.suffix.lower().replace(".", "")
        if ext not in ["mp3", "wav"]:
            raise ValueError(f"Expected mp3 or wav extension, but got {ext}")

        # Read audio file
        audio_bytes = await self.file_repo.read_audio_file(file_path)

        # Chat with audio using OpenAI
        return await self.openai_client.chat_with_audio(
            audio_bytes=audio_bytes,
            audio_format=ext,  # type: ignore
            model=model,  # type: ignore
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    async def chat_with_audio_batch(
        self,
        inputs: list[ChatWithAudioInputParams],
    ) -> list[dict[str, Any]]:
        """Chat with multiple audio files in parallel.

        Args:
        ----
            inputs: List of chat parameters.

        Returns:
        -------
            list[dict[str, Any]]: List of chat responses.

        """

        async def process_single(input_data: ChatWithAudioInputParams) -> dict[str, Any]:
            return await self.chat_with_audio(
                file_path=input_data.input_file_path,
                model=input_data.model,
                system_prompt=input_data.system_prompt,
                user_prompt=input_data.user_prompt,
            )

        return await asyncio.gather(*[process_single(input_data) for input_data in inputs])

    async def transcribe_with_enhancement_batch(
        self,
        inputs: list[TranscribeWithEnhancementInputParams],
    ) -> list[dict[str, Any]]:
        """Transcribe multiple audio files with enhancement templates in parallel.

        Args:
        ----
            inputs: List of enhancement parameters.

        Returns:
        -------
            list[dict[str, Any]]: List of enhanced transcription results.

        """
        # Convert enhancement params to transcribe params
        converted_inputs = [input_.to_transcribe_audio_input_params() for input_ in inputs]

        # Use the standard transcribe batch method
        return await self.transcribe_audio_batch(converted_inputs)
