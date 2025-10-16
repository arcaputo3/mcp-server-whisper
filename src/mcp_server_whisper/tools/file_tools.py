"""MCP tools for file management operations."""

from typing import Optional

from ..config import check_and_get_audio_path
from ..constants import SortBy
from ..infrastructure import FileSystemRepository, MCPServer
from ..models import FilePathSupportParams
from ..services import FileService


def create_file_tools(mcp: MCPServer) -> None:
    """Register file management tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    file_service = FileService(file_repo)

    @mcp.tool(
        description="Get the most recent audio file from the audio path. "
        "ONLY USE THIS IF THE USER ASKS FOR THE LATEST FILE."
    )
    async def get_latest_audio() -> FilePathSupportParams:
        """Get the most recently modified audio file and returns its path with model support info.

        Supported formats:
        - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
        - GPT-4o: mp3, wav

        Returns detailed file information including size, format, and duration.
        """
        return await file_service.get_latest_audio_file()

    @mcp.tool(
        description="List, filter, and sort audio files from the audio path. Supports regex pattern matching, "
        "filtering by metadata (size, duration, date, format), and sorting."
    )
    async def list_audio_files(
        pattern: Optional[str] = None,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        min_duration_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
        min_modified_time: Optional[float] = None,
        max_modified_time: Optional[float] = None,
        format: Optional[str] = None,
        sort_by: SortBy = SortBy.NAME,
        reverse: bool = False,
    ) -> list[FilePathSupportParams]:
        """List, filter, and sort audio files in the AUDIO_FILES_PATH directory with comprehensive options.

        Supported formats:
        - Transcribe: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
        - Chat: mp3, wav

        Filtering options:
        - pattern: Regex pattern for file name/path matching
        - min/max_size_bytes: File size range in bytes
        - min/max_duration_seconds: Audio duration range in seconds
        - min/max_modified_time: File modification time range (Unix timestamps)
        - format: Specific audio format (e.g., 'mp3', 'wav')

        Sorting options:
        - sort_by: Field to sort by (name, size, duration, modified_time, format)
        - reverse: Set to true for descending order

        Returns detailed file information including size, format, duration, and transcription capabilities.
        """
        return await file_service.list_audio_files(
            pattern=pattern,
            min_size_bytes=min_size_bytes,
            max_size_bytes=max_size_bytes,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            min_modified_time=min_modified_time,
            max_modified_time=max_modified_time,
            format_filter=format,
            sort_by=sort_by,
            reverse=reverse,
        )
