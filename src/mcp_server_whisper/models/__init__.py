"""Pydantic models for MCP Server Whisper."""

from .audio import (
    CompressAudioInputParams,
    ConvertAudioInputParams,
    FilePathSupportParams,
    ListAudioFilesInputParams,
)
from .base import BaseAudioInputParams, BaseInputPath
from .transcription import (
    ChatWithAudioInputParams,
    TranscribeAudioInputParams,
    TranscribeAudioInputParamsBase,
    TranscribeWithEnhancementInputParams,
)
from .tts import CreateClaudecastInputParams

__all__ = [
    # Base models
    "BaseInputPath",
    "BaseAudioInputParams",
    # Audio models
    "ConvertAudioInputParams",
    "CompressAudioInputParams",
    "FilePathSupportParams",
    "ListAudioFilesInputParams",
    # Transcription models
    "TranscribeAudioInputParamsBase",
    "TranscribeAudioInputParams",
    "ChatWithAudioInputParams",
    "TranscribeWithEnhancementInputParams",
    # TTS models
    "CreateClaudecastInputParams",
]
