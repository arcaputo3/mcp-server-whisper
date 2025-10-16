"""OpenAI API client wrapper for audio operations."""

import base64
from io import BytesIO
from typing import Any, Literal, Optional

import anyio
from aioresult import ResultCapture
from openai import AsyncOpenAI
from openai.types import AudioModel, AudioResponseFormat
from openai.types.audio.speech_model import SpeechModel
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import BaseModel

from ..constants import AudioChatModel, TTSVoice
from ..exceptions import TranscriptionAPIError, TTSAPIError


class OpenAIClientWrapper:
    """Wrapper for OpenAI API client with audio-specific methods."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client wrapper.

        Args:
        ----
            api_key: Optional OpenAI API key. If not provided, uses environment variable.

        """
        self.client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        model: AudioModel = "gpt-4o-mini-transcribe",
        response_format: AudioResponseFormat = "text",
        prompt: Optional[str] = None,
        timestamp_granularities: Optional[list[Literal["word", "segment"]]] = None,
    ) -> dict[str, Any]:
        """Transcribe audio using OpenAI's transcription API.

        Args:
        ----
            audio_bytes: Audio file content as bytes.
            filename: Name of the audio file (used by OpenAI API).
            model: Transcription model to use.
            response_format: Format of the response ('text', 'json', 'verbose_json').
            prompt: Optional prompt to guide transcription.
            timestamp_granularities: Optional timestamp granularities.

        Returns:
        -------
            dict[str, Any]: Transcription result.

        Raises:
        ------
            TranscriptionAPIError: If the API call fails.

        """
        try:
            # Create a file-like object from bytes for OpenAI API
            file_obj = BytesIO(audio_bytes)
            file_obj.name = filename  # OpenAI API needs a filename

            # Build request parameters
            params: dict[str, Any] = {
                "file": file_obj,
                "model": model,
                "response_format": response_format,
            }

            if prompt is not None:
                params["prompt"] = prompt

            if timestamp_granularities is not None:
                params["timestamp_granularities"] = timestamp_granularities

            transcript = await self.client.audio.transcriptions.create(**params)

            if isinstance(transcript, BaseModel):
                return transcript.model_dump()
            return {"text": transcript}

        except Exception as e:
            raise TranscriptionAPIError(f"Transcription failed: {e}") from e

    async def chat_with_audio(
        self,
        audio_bytes: bytes,
        audio_format: Literal["mp3", "wav"],
        model: AudioChatModel = "gpt-4o-audio-preview-2024-12-17",
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Chat with audio using GPT-4o audio models.

        Args:
        ----
            audio_bytes: Audio file content as bytes.
            audio_format: Audio format ('mp3' or 'wav').
            model: Audio chat model to use.
            system_prompt: Optional system prompt.
            user_prompt: Optional user text prompt to accompany the audio.

        Returns:
        -------
            dict[str, Any]: Chat completion response.

        Raises:
        ------
            TranscriptionAPIError: If the API call fails.

        """
        try:
            # Encode audio to base64
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            # Build messages
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            user_content: list[ChatCompletionContentPartParam] = []
            if user_prompt:
                user_content.append({"type": "text", "text": user_prompt})

            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": audio_format},
                }
            )
            messages.append({"role": "user", "content": user_content})

            completion = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                modalities=["text"],
            )

            return {"text": completion.choices[0].message.content}

        except Exception as e:
            raise TranscriptionAPIError(f"Audio chat failed: {e}") from e

    async def text_to_speech(
        self,
        text: str,
        model: SpeechModel = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """Generate speech from text using OpenAI's TTS API.

        Args:
        ----
            text: Text to convert to speech.
            model: TTS model to use.
            voice: Voice to use for TTS.
            instructions: Optional instructions for speech generation.
            speed: Speech speed (0.25 to 4.0).

        Returns:
        -------
            bytes: Audio content as bytes.

        Raises:
        ------
            TTSAPIError: If the API call fails.

        """
        try:
            params: dict[str, Any] = {
                "input": text,
                "model": model,
                "voice": voice,
                "speed": speed,
            }

            if instructions is not None:
                params["instructions"] = instructions

            response = await self.client.audio.speech.create(**params)
            return await response.aread()

        except Exception as e:
            raise TTSAPIError(f"Text-to-speech generation failed: {e}") from e

    async def generate_tts_chunks(
        self,
        text_chunks: list[str],
        model: SpeechModel = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: Optional[str] = None,
        speed: float = 1.0,
    ) -> list[bytes]:
        """Generate TTS for multiple text chunks in parallel.

        Args:
        ----
            text_chunks: List of text chunks to convert to speech.
            model: TTS model to use.
            voice: Voice to use for TTS.
            instructions: Optional instructions for speech generation.
            speed: Speech speed (0.25 to 4.0).

        Returns:
        -------
            list[bytes]: List of audio chunks as bytes.

        Raises:
        ------
            TTSAPIError: If any API call fails.

        """

        async def generate_chunk(chunk: str) -> bytes:
            return await self.text_to_speech(
                text=chunk,
                model=model,
                voice=voice,
                instructions=instructions,
                speed=speed,
            )

        try:
            async with anyio.create_task_group() as tg:
                captures = [ResultCapture.start_soon(tg, generate_chunk, chunk) for chunk in text_chunks]
            return [c.result() for c in captures]
        except Exception as e:
            raise TTSAPIError(f"Batch TTS generation failed: {e}") from e
