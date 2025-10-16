"""Audio processing related Pydantic models."""

from typing import Optional

from openai.types import AudioModel
from pydantic import BaseModel, Field

from ..constants import (
    DEFAULT_MAX_FILE_SIZE_MB,
    AudioChatModel,
    SortBy,
    SupportedChatWithAudioFormat,
)
from .base import BaseAudioInputParams


class ConvertAudioInputParams(BaseAudioInputParams):
    """Params for converting audio to mp3 or wav."""

    target_format: SupportedChatWithAudioFormat = Field(
        default="mp3", description="Target audio format to convert to (mp3 or wav)"
    )


class CompressAudioInputParams(BaseAudioInputParams):
    """Params for compressing audio."""

    max_mb: int = Field(
        default=DEFAULT_MAX_FILE_SIZE_MB,
        gt=0,
        description="Maximum file size in MB. Files larger than this will be compressed",
    )


class FilePathSupportParams(BaseModel):
    """Params for checking if a file supports transcription."""

    file_name: str = Field(description="Name of the audio file")
    transcription_support: Optional[list[AudioModel]] = Field(
        default=None, description="List of transcription models that support this file format"
    )
    chat_support: Optional[list[AudioChatModel]] = Field(
        default=None, description="List of audio LLM models that support this file format"
    )
    modified_time: float = Field(description="Last modified timestamp of the file (Unix time)")
    size_bytes: int = Field(description="Size of the file in bytes")
    format: str = Field(description="Audio format of the file (e.g., 'mp3', 'wav')")
    duration_seconds: Optional[float] = Field(
        default=None, description="Duration of the audio file in seconds, if available"
    )


class ListAudioFilesInputParams(BaseModel):
    """Input parameters for the list_audio_files tool."""

    pattern: Optional[str] = Field(default=None, description="Optional regex pattern to filter audio files by name")
    min_size_bytes: Optional[int] = Field(default=None, description="Minimum file size in bytes")
    max_size_bytes: Optional[int] = Field(default=None, description="Maximum file size in bytes")
    min_duration_seconds: Optional[float] = Field(default=None, description="Minimum audio duration in seconds")
    max_duration_seconds: Optional[float] = Field(default=None, description="Maximum audio duration in seconds")
    min_modified_time: Optional[float] = Field(
        default=None, description="Minimum file modification time (Unix timestamp)"
    )
    max_modified_time: Optional[float] = Field(
        default=None, description="Maximum file modification time (Unix timestamp)"
    )
    format: Optional[str] = Field(default=None, description="Specific audio format to filter by (e.g., 'mp3', 'wav')")
    sort_by: SortBy = Field(
        default=SortBy.NAME, description="Field to sort results by (name, size, duration, modified_time, format)"
    )
    reverse: bool = Field(default=False, description="Sort in reverse order if True")

    model_config = {"arbitrary_types_allowed": True}
