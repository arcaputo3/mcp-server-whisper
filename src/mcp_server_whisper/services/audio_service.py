"""Audio processing service - orchestrates domain and infrastructure."""

import tempfile
from pathlib import Path

from ..constants import DEFAULT_MAX_FILE_SIZE_MB, SupportedChatWithAudioFormat
from ..domain import AudioProcessor
from ..infrastructure import FileSystemRepository


class AudioService:
    """Service for audio conversion and compression operations."""

    def __init__(self, file_repo: FileSystemRepository):
        """Initialize the audio service.

        Args:
        ----
            file_repo: File system repository for I/O operations.

        """
        self.file_repo = file_repo
        self.processor = AudioProcessor()

    async def convert_audio(
        self,
        input_file: Path,
        output_path: Path | None = None,
        target_format: SupportedChatWithAudioFormat = "mp3",
    ) -> Path:
        """Convert audio file to supported format (mp3 or wav).

        Args:
        ----
            input_file: Path to input audio file.
            output_path: Optional output path.
            target_format: Target format ('mp3' or 'wav').

        Returns:
        -------
            Path: Path to the converted audio file.

        """
        # Determine output path
        if output_path is None:
            output_path = input_file.with_suffix(f".{target_format}")

        # Load audio
        audio_data = await self.processor.load_audio_from_path(input_file)

        # Convert format
        converted_bytes = await self.processor.convert_audio_format(
            audio_data=audio_data,
            target_format=target_format,
            output_path=output_path,
        )

        # Write converted file
        await self.file_repo.write_audio_file(output_path, converted_bytes)

        return output_path

    async def compress_audio(
        self,
        input_file: Path,
        output_path: Path | None = None,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> Path:
        """Compress audio file if it exceeds size limit.

        Args:
        ----
            input_file: Path to input audio file.
            output_path: Optional output path.
            max_mb: Maximum file size in MB.

        Returns:
        -------
            Path: Path to the compressed audio file (or original if no compression needed).

        """
        # Check if compression is needed
        file_size = await self.file_repo.get_file_size(input_file)
        needs_compression = self.processor.calculate_compression_needed(file_size, max_mb)

        if not needs_compression:
            return input_file  # No compression needed

        print(f"\n[AudioService] File '{input_file}' size > {max_mb}MB. Attempting compression...")

        # Convert to MP3 if not already
        if input_file.suffix.lower() != ".mp3":
            print(f"[AudioService] Converting to MP3 first...")
            input_file = await self.convert_audio(input_file, None, "mp3")

        # Determine output path
        if output_path is None:
            output_path = input_file.parent / f"compressed_{input_file.stem}.mp3"

        print(f"[AudioService] Original file: {input_file}")
        print(f"[AudioService] Output file: {output_path}")

        # Load and compress
        audio_data = await self.processor.load_audio_from_path(input_file)
        compressed_bytes = await self.processor.compress_mp3(audio_data, output_path)

        # Write compressed file
        await self.file_repo.write_audio_file(output_path, compressed_bytes)

        print(f"[AudioService] Compressed file size: {len(compressed_bytes)} bytes")

        return output_path

    async def maybe_compress_file(
        self,
        input_file: Path,
        output_path: Path | None = None,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> Path:
        """Compress file if needed, maintaining backward compatibility.

        This method provides the same interface as the original server.py function.

        Args:
        ----
            input_file: Path to input audio file.
            output_path: Optional output path.
            max_mb: Maximum file size in MB.

        Returns:
        -------
            Path: Path to the (possibly compressed) audio file.

        """
        return await self.compress_audio(input_file, output_path, max_mb)
