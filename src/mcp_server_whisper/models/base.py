"""Base Pydantic models for MCP Server Whisper."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class BaseInputPath(BaseModel):
    """Base file path input."""

    input_file_path: Path = Field(description="Path to the input audio file to process")

    model_config = {"arbitrary_types_allowed": True}


class BaseAudioInputParams(BaseInputPath):
    """Base params for audio operations."""

    output_file_path: Optional[Path] = Field(
        default=None,
        description="Optional custom path for the output file. "
        "If not provided, defaults to input_file_path with appropriate extension",
    )
