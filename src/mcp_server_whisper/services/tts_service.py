"""Text-to-speech service - orchestrates TTS operations."""

import time

from ..constants import TTSVoice
from ..domain import AudioProcessor
from ..infrastructure import FileSystemRepository, OpenAIClientWrapper, SecurePathResolver
from ..models import TTSResult
from ..utils import split_text_for_tts


class TTSService:
    """Service for text-to-speech operations."""

    def __init__(
        self, file_repo: FileSystemRepository, openai_client: OpenAIClientWrapper, path_resolver: SecurePathResolver
    ):
        """Initialize the TTS service.

        Args:
        ----
            file_repo: File system repository for I/O operations.
            openai_client: OpenAI API client wrapper.
            path_resolver: Secure path resolver for filename to path conversion.

        """
        self.file_repo = file_repo
        self.openai_client = openai_client
        self.audio_processor = AudioProcessor()
        self.path_resolver = path_resolver

    async def create_speech(
        self,
        text_prompt: str,
        output_filename: str | None = None,
        model: str = "gpt-4o-mini-tts",
        voice: TTSVoice = "alloy",
        instructions: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate text-to-speech audio from text.

        Args:
        ----
            text_prompt: Text to convert to speech.
            output_filename: Optional name for output file.
            model: TTS model to use.
            voice: Voice to use for TTS.
            instructions: Optional instructions for speech generation.
            speed: Speech speed (0.25 to 4.0).

        Returns:
        -------
            TTSResult: Result with name of the generated audio file.

        """
        # Determine output filename
        if output_filename is None:
            default_name = f"speech_{time.time_ns()}.mp3"
        else:
            default_name = output_filename

        # Resolve to full path
        output_file_path = self.path_resolver.resolve_output(output_filename, default_name)

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

        return TTSResult(output_file=output_file_path.name)
