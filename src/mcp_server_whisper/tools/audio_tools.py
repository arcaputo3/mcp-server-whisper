"""MCP tools for audio processing operations."""

from typing import Optional

from ..config import check_and_get_audio_path
from ..constants import DEFAULT_MAX_FILE_SIZE_MB, SupportedChatWithAudioFormat
from ..infrastructure import FileSystemRepository, MCPServer, SecurePathResolver
from ..models import AudioProcessingResult
from ..services import AudioService


def create_audio_tools(mcp: MCPServer) -> None:
    """Register audio processing tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    path_resolver = SecurePathResolver(audio_path)
    audio_service = AudioService(file_repo, path_resolver)

    @mcp.tool(description="A tool used to convert audio files to mp3 or wav which are gpt-4o compatible.")
    async def convert_audio(
        input_file_name: str,
        target_format: SupportedChatWithAudioFormat = "mp3",
        output_file_name: Optional[str] = None,
    ) -> AudioProcessingResult:
        """Convert audio file to supported format (mp3 or wav).

        Args:
            input_file_name: Name of the input audio file to process
            target_format: Target audio format to convert to (mp3 or wav)
            output_file_name: Optional custom name for the output file. If not provided,
                            defaults to input filename with appropriate extension

        Returns:
        -------
            AudioProcessingResult with name of the converted audio file

        """
        try:
            return await audio_service.convert_audio(
                input_filename=input_file_name,
                output_filename=output_file_name,
                target_format=target_format,
            )
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed for {input_file_name}: {str(e)}") from e

    @mcp.tool(
        description="A tool used to compress audio files which are >25mb. "
        "ONLY USE THIS IF THE USER REQUESTS COMPRESSION OR IF OTHER TOOLS FAIL DUE TO FILES BEING TOO LARGE."
    )
    async def compress_audio(
        input_file_name: str,
        max_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
        output_file_name: Optional[str] = None,
    ) -> AudioProcessingResult:
        """Compress audio file if it's larger than max_mb.

        Args:
            input_file_name: Name of the input audio file to process
            max_mb: Maximum file size in MB. Files larger than this will be compressed
            output_file_name: Optional custom name for the output file. If not provided,
                            defaults to input filename with appropriate extension

        Returns:
        -------
            AudioProcessingResult with name of the compressed audio file (or original if no compression needed)

        """
        try:
            return await audio_service.compress_audio(
                input_filename=input_file_name,
                output_filename=output_file_name,
                max_mb=max_mb,
            )
        except Exception as e:
            raise RuntimeError(f"Audio compression failed for {input_file_name}: {str(e)}") from e
