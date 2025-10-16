"""Configuration management for MCP Server Whisper."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class WhisperConfig(BaseSettings):
    """Configuration for MCP Server Whisper.

    Loads configuration from environment variables with validation.
    """

    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for accessing Whisper and GPT-4o models",
    )

    audio_files_path: Path = Field(
        ...,
        description="Path to the directory containing audio files",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "arbitrary_types_allowed": True,
    }

    @field_validator("audio_files_path")
    @classmethod
    def validate_audio_path(cls, v: Path) -> Path:
        """Validate that the audio path exists and is a directory."""
        resolved_path = v.resolve()
        if not resolved_path.exists():
            raise ConfigurationError(f"Audio path does not exist: {resolved_path}")
        if not resolved_path.is_dir():
            raise ConfigurationError(f"Audio path is not a directory: {resolved_path}")
        return resolved_path


@lru_cache
def get_config() -> WhisperConfig:
    """Get the application configuration (cached singleton).

    Returns
    -------
        WhisperConfig: The validated configuration object.

    Raises
    ------
        ConfigurationError: If configuration is invalid or missing.

    """
    try:
        return WhisperConfig()  # type: ignore
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


def check_and_get_audio_path() -> Path:
    """Check if the audio path environment variable is set and exists.

    This function maintains backward compatibility with the original implementation.

    Returns
    -------
        Path: The validated audio files path.

    Raises
    ------
        ValueError: If the audio path is not set or doesn't exist.

    """
    audio_path_str = os.getenv("AUDIO_FILES_PATH")
    if not audio_path_str:
        raise ValueError("AUDIO_FILES_PATH environment variable not set")

    audio_path = Path(audio_path_str).resolve()
    if not audio_path.exists():
        raise ValueError(f"Audio path does not exist: {audio_path}")
    return audio_path
