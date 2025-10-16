"""Audio processing domain logic (pure business logic, no I/O)."""

from pathlib import Path

import anyio
from pydub import AudioSegment  # type: ignore

from ..constants import DEFAULT_MAX_FILE_SIZE_MB, DEFAULT_TTS_SAMPLE_RATE, SupportedChatWithAudioFormat
from ..exceptions import AudioCompressionError, AudioConversionError


class AudioProcessor:
    """Domain logic for audio processing operations.

    This class contains pure business logic for audio manipulation
    without any I/O operations (those are handled by infrastructure layer).
    """

    @staticmethod
    async def convert_audio_format(
        audio_data: AudioSegment,
        target_format: SupportedChatWithAudioFormat,
        output_path: Path,
    ) -> bytes:
        """Convert audio to target format.

        Args:
        ----
            audio_data: Loaded AudioSegment object.
            target_format: Target format ('mp3' or 'wav').
            output_path: Path for temporary export.

        Returns:
        -------
            bytes: Converted audio data.

        Raises:
        ------
            AudioConversionError: If conversion fails.

        """
        try:
            # Export to temporary path
            await anyio.to_thread.run_sync(
                lambda: audio_data.export(
                    str(output_path),
                    format=target_format,
                    parameters=["-ac", "2"],
                )
            )

            # Read the converted file
            with open(output_path, "rb") as f:
                return f.read()

        except Exception as e:
            raise AudioConversionError(f"Audio conversion to {target_format} failed: {e}") from e

    @staticmethod
    async def compress_mp3(
        audio_data: AudioSegment,
        output_path: Path,
        target_sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
    ) -> bytes:
        """Compress MP3 audio by downsampling.

        Args:
        ----
            audio_data: Loaded AudioSegment object.
            output_path: Path for temporary export.
            target_sample_rate: Target sample rate for compression.

        Returns:
        -------
            bytes: Compressed audio data.

        Raises:
        ------
            AudioCompressionError: If compression fails.

        """
        try:
            original_frame_rate = audio_data.frame_rate
            print(f"[Compression] Original frame rate: {original_frame_rate}, converting to {target_sample_rate}.")

            await anyio.to_thread.run_sync(
                lambda: audio_data.export(
                    str(output_path),
                    format="mp3",
                    parameters=["-ar", str(target_sample_rate)],
                )
            )

            # Read the compressed file
            with open(output_path, "rb") as f:
                return f.read()

        except Exception as e:
            raise AudioCompressionError(f"MP3 compression failed: {e}") from e

    @staticmethod
    async def load_audio_from_path(file_path: Path) -> AudioSegment:
        """Load audio file into AudioSegment.

        Args:
        ----
            file_path: Path to the audio file.

        Returns:
        -------
            AudioSegment: Loaded audio segment.

        Raises:
        ------
            AudioConversionError: If loading fails.

        """
        try:
            format_str = file_path.suffix[1:]  # Remove leading dot
            return await anyio.to_thread.run_sync(lambda: AudioSegment.from_file(str(file_path), format=format_str))
        except Exception as e:
            raise AudioConversionError(f"Failed to load audio file {file_path}: {e}") from e

    @staticmethod
    def calculate_compression_needed(
        file_size_bytes: int,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> bool:
        """Determine if compression is needed based on file size.

        Args:
        ----
            file_size_bytes: Current file size in bytes.
            max_mb: Maximum allowed size in megabytes.

        Returns:
        -------
            bool: True if compression is needed, False otherwise.

        """
        threshold_bytes = max_mb * 1024 * 1024
        return file_size_bytes > threshold_bytes

    @staticmethod
    async def concatenate_audio_segments(audio_chunks: list[bytes], format: str = "mp3") -> bytes:
        """Concatenate multiple audio chunks into a single audio file.

        Args:
        ----
            audio_chunks: List of audio data as bytes.
            format: Audio format (default: 'mp3').

        Returns:
        -------
            bytes: Concatenated audio data.

        Raises:
        ------
            AudioProcessingError: If concatenation fails.

        """
        try:
            from io import BytesIO

            combined = AudioSegment.empty()

            for chunk in audio_chunks:
                # Load each chunk
                chunk_io = BytesIO(chunk)
                if format == "mp3":
                    segment = await anyio.to_thread.run_sync(lambda: AudioSegment.from_mp3(chunk_io))
                elif format == "wav":
                    segment = await anyio.to_thread.run_sync(lambda: AudioSegment.from_wav(chunk_io))
                else:
                    segment = await anyio.to_thread.run_sync(lambda: AudioSegment.from_file(chunk_io, format=format))

                combined += segment

            # Export combined audio to bytes
            output = BytesIO()
            await anyio.to_thread.run_sync(lambda: combined.export(output, format=format))
            return output.getvalue()

        except Exception as e:
            raise AudioConversionError(f"Audio concatenation failed: {e}") from e

    @staticmethod
    def generate_output_path(
        input_path: Path,
        output_path: Path | None,
        suffix: str,
        extension: str,
    ) -> Path:
        """Generate appropriate output path for processed audio.

        Args:
        ----
            input_path: Original input file path.
            output_path: User-specified output path (if any).
            suffix: Suffix to add to filename (e.g., 'compressed').
            extension: File extension (e.g., '.mp3').

        Returns:
        -------
            Path: Generated output path.

        """
        if output_path is not None:
            return output_path

        if suffix:
            return input_path.parent / f"{suffix}_{input_path.stem}{extension}"
        else:
            return input_path.with_suffix(extension)
