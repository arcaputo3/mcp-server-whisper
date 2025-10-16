"""Audio processing service - orchestrates domain and infrastructure."""

from pathlib import Path

from ..constants import DEFAULT_MAX_FILE_SIZE_MB, SupportedChatWithAudioFormat
from ..domain import AudioProcessor
from ..infrastructure import FileSystemRepository, SecurePathResolver
from ..models import AudioProcessingResult


class AudioService:
    """Service for audio conversion and compression operations."""

    def __init__(self, file_repo: FileSystemRepository, path_resolver: SecurePathResolver):
        """Initialize the audio service.

        Args:
        ----
            file_repo: File system repository for I/O operations.
            path_resolver: Secure path resolver for filename to path conversion.

        """
        self.file_repo = file_repo
        self.processor = AudioProcessor()
        self.path_resolver = path_resolver

    async def convert_audio(
        self,
        input_filename: str,
        output_filename: str | None = None,
        target_format: SupportedChatWithAudioFormat = "mp3",
    ) -> AudioProcessingResult:
        """Convert audio file to supported format (mp3 or wav).

        Args:
        ----
            input_filename: Name of input audio file.
            output_filename: Optional name for output file.
            target_format: Target format ('mp3' or 'wav').

        Returns:
        -------
            AudioProcessingResult: Result with name of the converted audio file.

        """
        # Resolve input filename to path
        input_file = self.path_resolver.resolve_input(input_filename)

        # Determine output path
        if output_filename is None:
            output_name = Path(input_filename).stem + f".{target_format}"
        else:
            output_name = output_filename
        output_path = self.path_resolver.resolve_output(output_name, f"{Path(input_filename).stem}.{target_format}")

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

        return AudioProcessingResult(output_file=output_path.name)

    async def compress_audio(
        self,
        input_filename: str,
        output_filename: str | None = None,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> AudioProcessingResult:
        """Compress audio file if it exceeds size limit.

        Args:
        ----
            input_filename: Name of input audio file.
            output_filename: Optional name for output file.
            max_mb: Maximum file size in MB.

        Returns:
        -------
            AudioProcessingResult: Result with name of the compressed audio file (or original if no compression needed).

        """
        # Resolve input filename to path
        input_file = self.path_resolver.resolve_input(input_filename)

        # Check if compression is needed
        file_size = await self.file_repo.get_file_size(input_file)
        needs_compression = self.processor.calculate_compression_needed(file_size, max_mb)

        if not needs_compression:
            return AudioProcessingResult(output_file=input_filename)  # No compression needed

        print(f"\n[AudioService] File '{input_filename}' size > {max_mb}MB. Attempting compression...")

        # Convert to MP3 if not already
        if input_file.suffix.lower() != ".mp3":
            print("[AudioService] Converting to MP3 first...")
            conversion_result = await self.convert_audio(input_filename, None, "mp3")
            # Update input to use the converted file
            input_filename = conversion_result.output_file
            input_file = self.path_resolver.resolve_input(input_filename)

        # Determine output path
        if output_filename is None:
            output_name = f"compressed_{input_file.stem}.mp3"
        else:
            output_name = output_filename
        output_path = self.path_resolver.resolve_output(output_name, f"compressed_{input_file.stem}.mp3")

        print(f"[AudioService] Original file: {input_filename}")
        print(f"[AudioService] Output file: {output_name}")

        # Load and compress
        audio_data = await self.processor.load_audio_from_path(input_file)
        compressed_bytes = await self.processor.compress_mp3(audio_data, output_path)

        # Write compressed file
        await self.file_repo.write_audio_file(output_path, compressed_bytes)

        print(f"[AudioService] Compressed file size: {len(compressed_bytes)} bytes")

        return AudioProcessingResult(output_file=output_path.name)

    async def maybe_compress_file(
        self,
        input_filename: str,
        output_filename: str | None = None,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    ) -> AudioProcessingResult:
        """Compress file if needed, maintaining backward compatibility.

        This method provides the same interface as the original server.py function.

        Args:
        ----
            input_filename: Name of input audio file.
            output_filename: Optional name for output file.
            max_mb: Maximum file size in MB.

        Returns:
        -------
            AudioProcessingResult: Result with name of the (possibly compressed) audio file.

        """
        return await self.compress_audio(input_filename, output_filename, max_mb)
