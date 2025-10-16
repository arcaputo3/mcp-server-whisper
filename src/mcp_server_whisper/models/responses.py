"""Response models for MCP tool outputs."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class AudioProcessingResult(BaseModel):
    """Result from audio processing operations (convert, compress)."""

    output_path: Path = Field(description="Path to the processed audio file")

    model_config = {"arbitrary_types_allowed": True}


class TranscriptionResult(BaseModel):
    """Result from transcription operations.

    Includes all fields that OpenAI's transcription API might return.
    """

    text: str = Field(description="The transcribed text")
    duration: Optional[float] = Field(default=None, description="Duration of the audio in seconds")
    language: Optional[str] = Field(default=None, description="Detected language of the audio")
    segments: Optional[list[dict[str, Any]]] = Field(default=None, description="Timestamped segments")
    words: Optional[list[dict[str, Any]]] = Field(default=None, description="Word-level timestamps")
    usage: Optional[dict[str, Any]] = Field(default=None, description="Token usage information")
    logprobs: Optional[Any] = Field(default=None, description="Log probabilities if requested")

    model_config = {"arbitrary_types_allowed": True}


class ChatResult(BaseModel):
    """Result from chat_with_audio operation."""

    text: str = Field(description="The response text from the audio chat")


class TTSResult(BaseModel):
    """Result from text-to-speech operations."""

    output_path: Path = Field(description="Path to the generated audio file")

    model_config = {"arbitrary_types_allowed": True}
