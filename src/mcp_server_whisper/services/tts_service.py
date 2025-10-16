"""Text-to-speech service - orchestrates TTS operations."""

import time
from pathlib import Path

from ..config import check_and_get_audio_path
from ..domain import AudioProcessor
from ..infrastructure import FileSystemRepository, OpenAIClientWrapper
from ..models import TTSResult
from ..utils import split_text_for_tts


class TTSService:
    """Service for text-to-speech operations."""

    def __init__(self, file_repo: FileSystemRepository, openai_client: OpenAIClientWrapper):
        """Initialize the TTS service.

        Args:
        ----
            file_repo: File system repository for I/O operations.
            openai_client: OpenAI API client wrapper.

        """
        self.file_repo = file_repo
        self.openai_client = openai_client
        self.audio_processor = AudioProcessor()

    async def create_speech(
        self,
        text_prompt: str,
        output_file_path: Path | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: str = "nova",
        instructions: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate text-to-speech audio from text.

        Args:
        ----
            text_prompt: Text to convert to speech.
            output_file_path: Optional output path.
            model: TTS model to use.
            voice: Voice to use for TTS.
            instructions: Optional instructions for speech generation.
            speed: Speech speed (0.25 to 4.0).

        Returns:
        -------
            TTSResult: Result with path to the generated audio file.

        """
        # Set default output path if not provided
        if output_file_path is None:
            audio_path = check_and_get_audio_path()
            output_file_path = audio_path / f"speech_{time.time_ns()}.mp3"

        # Ensure parent directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Split text if it exceeds the API limit
        text_chunks = split_text_for_tts(text_prompt)

        if len(text_chunks) == 1:
            # Single chunk - process directly
            audio_bytes = await self.openai_client.text_to_speech(
                text=text_chunks[0],
                model=model,  # type: ignore
                voice=voice,
                instructions=instructions,
                speed=speed,
            )

            # Write audio file
            await self.file_repo.write_audio_file(output_file_path, audio_bytes)

        else:
            # Multiple chunks - process in parallel and concatenate
            print(f"Text exceeds TTS API limit, splitting into {len(text_chunks)} chunks")

            # Generate TTS for all chunks in parallel
            audio_chunks = await self.openai_client.generate_tts_chunks(
                text_chunks=text_chunks,
                model=model,  # type: ignore
                voice=voice,
                instructions=instructions,
                speed=speed,
            )

            # Concatenate audio chunks using domain logic
            combined_audio = await self.audio_processor.concatenate_audio_segments(
                audio_chunks=audio_chunks,
                format="mp3",
            )

            # Write combined audio file
            await self.file_repo.write_audio_file(output_file_path, combined_audio)

        return TTSResult(output_path=output_file_path)
