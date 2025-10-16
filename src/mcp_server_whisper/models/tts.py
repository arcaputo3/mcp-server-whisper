"""Text-to-speech related Pydantic models."""

from pathlib import Path
from typing import Optional

from openai.types.audio.speech_model import SpeechModel
from pydantic import BaseModel, Field

from ..constants import TTSVoice


class CreateClaudecastInputParams(BaseModel):
    """Params for text-to-speech using OpenAI's API."""

    text_prompt: str = Field(description="Text to convert to speech")
    output_file_path: Optional[Path] = Field(
        default=None, description="Output file path (defaults to speech.mp3 in current directory)"
    )
    model: SpeechModel = Field(
        default="gpt-4o-mini-tts", description="TTS model to use. gpt-4o-mini-tts is always preferred."
    )
    voice: TTSVoice = Field(
        default="alloy",
        description="Voice for the TTS",
    )
    instructions: str | None = Field(
        default=None,
        description="Optional instructions for the speech conversion, such as tonality, accent, style, etc.",
    )
    speed: float = Field(
        default=1.0,
        gt=0.25,
        lt=4.0,
        description="Speed of the speech conversion. Use if the user prompts slow or fast speech.",
    )

    model_config = {"arbitrary_types_allowed": True}
