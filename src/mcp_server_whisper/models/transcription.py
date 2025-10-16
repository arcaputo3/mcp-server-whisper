"""Transcription related Pydantic models."""

from typing import Literal, Optional

from openai.types import AudioModel, AudioResponseFormat
from pydantic import Field

from ..constants import ENHANCEMENT_PROMPTS, AudioChatModel, EnhancementType
from .base import BaseInputPath


class TranscribeAudioInputParamsBase(BaseInputPath):
    """Base params for transcribing audio with audio-to-text model."""

    model: AudioModel = Field(
        default="gpt-4o-mini-transcribe",
        description="The transcription model to use (e.g., 'whisper-1', 'gpt-4o-transcribe', 'gpt-4o-mini-transcribe')",
    )
    response_format: AudioResponseFormat = Field(
        "text",
        description="The response format of the transcription model. "
        'Use `verbose_json` with `model="whisper-1"` for timestamps. '
        "`gpt-4o-transcribe` and `gpt-4o-mini-transcribe` only support `text` and `json`.",
    )
    timestamp_granularities: list[Literal["word", "segment"]] | None = Field(
        None,
        description="""The timestamp granularities to populate for this transcription.
`response_format` must be set `verbose_json` to use timestamp granularities.
Either or both of these options are supported: `word`, or `segment`.
Note: There is no additional latency for segment timestamps, but generating word timestamp incurs additional latency.
""",
    )


class TranscribeAudioInputParams(TranscribeAudioInputParamsBase):
    """Params for transcribing audio with audio-to-text model."""

    prompt: str | None = Field(
        None,
        description="""An optional prompt to guide the transcription model's output. Effective prompts can:

        1. Correct specific words/acronyms: Include technical terms or names that might be misrecognized
           Example: "The transcript discusses OpenAI's DALL·E and GPT-4 technology"

        2. Maintain context from previous segments: Include the last part of previous transcript
           Note: Model only considers final 224 tokens of the prompt

        3. Enforce punctuation: Include properly punctuated example text
           Example: "Hello, welcome to my lecture. Today, we'll discuss..."

        4. Preserve filler words: Include example with verbal hesitations
           Example: "Umm, let me think like, hmm... Okay, here's what I'm thinking"

        5. Set writing style: Use examples in desired format (simplified/traditional, formal/casual)

        The model will try to match the style and formatting of your prompt.""",
    )


class ChatWithAudioInputParams(BaseInputPath):
    """Params for transcribing audio with LLM using custom prompt."""

    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt to use.")
    user_prompt: Optional[str] = Field(default=None, description="Custom user prompt to use.")
    model: AudioChatModel = Field(
        default="gpt-4o-audio-preview-2024-12-17", description="The audio LLM model to use for transcription"
    )


class TranscribeWithEnhancementInputParams(TranscribeAudioInputParamsBase):
    """Params for transcribing audio with LLM using template prompt."""

    enhancement_type: EnhancementType = Field(
        default="detailed",
        description="Type of enhancement to apply to the transcription: "
        "detailed, storytelling, professional, or analytical.",
    )

    def to_transcribe_audio_input_params(self) -> TranscribeAudioInputParams:
        """Transfer audio with LLM using custom prompt."""
        return TranscribeAudioInputParams(
            input_file_path=self.input_file_path,
            prompt=ENHANCEMENT_PROMPTS[self.enhancement_type],
            model=self.model,
            timestamp_granularities=self.timestamp_granularities,
            response_format=self.response_format,
        )
