"""MCP tools for audio processing operations."""

from pathlib import Path
from typing import Optional

from ..config import check_and_get_audio_path
from ..constants import DEFAULT_MAX_FILE_SIZE_MB, SupportedChatWithAudioFormat
from ..infrastructure import FileSystemRepository
from ..models import AudioProcessingResult
from ..services import AudioService


def create_audio_tools(mcp):
    """Register audio processing tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    audio_service = AudioService(file_repo)

    @mcp.tool(description="A tool used to convert audio files to mp3 or wav which are gpt-4o compatible.")
    async def convert_audio(
        input_file_path: Path,
        target_format: SupportedChatWithAudioFormat = "mp3",
        output_file_path: Optional[Path] = None,
    ) -> AudioProcessingResult:
        """Convert audio file to supported format (mp3 or wav).

        Args:
            input_file_path: Path to the input audio file to process
            target_format: Target audio format to convert to (mp3 or wav)
            output_file_path: Optional custom path for the output file. If not provided,
                            defaults to input_file_path with appropriate extension

        Returns:
        -------
            AudioProcessingResult with path to the converted audio file

        """
        try:
            return await audio_service.convert_audio(
                input_file=input_file_path,
                output_path=output_file_path,
                target_format=target_format,
            )
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed for {input_file_path}: {str(e)}") from e

    @mcp.tool(
        description="A tool used to compress audio files which are >25mb. "
        "ONLY USE THIS IF THE USER REQUESTS COMPRESSION OR IF OTHER TOOLS FAIL DUE TO FILES BEING TOO LARGE."
    )
    async def compress_audio(
        input_file_path: Path,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        output_file_path: Optional[Path] = None,
    ) -> AudioProcessingResult:
        """Compress audio file if it's larger than max_mb.

        Args:
            input_file_path: Path to the input audio file to process
            max_mb: Maximum file size in MB. Files larger than this will be compressed
            output_file_path: Optional custom path for the output file. If not provided,
                            defaults to input_file_path with appropriate extension

        Returns:
        -------
            AudioProcessingResult with path to the compressed audio file (or original if no compression needed)

        """
        try:
            return await audio_service.compress_audio(
                input_file=input_file_path,
                output_path=output_file_path,
                max_mb=max_mb,
            )
        except Exception as e:
            raise RuntimeError(f"Audio compression failed for {input_file_path}: {str(e)}") from e
