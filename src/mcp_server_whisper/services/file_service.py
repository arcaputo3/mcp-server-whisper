"""File discovery and management service."""

from pathlib import Path
from typing import Optional

import anyio
from aioresult import ResultCapture

from ..constants import SortBy
from ..domain import FileFilterSorter
from ..infrastructure import FileSystemRepository, get_cached_audio_file_support
from ..models import FilePathSupportParams


class FileService:
    """Service for file discovery, filtering, and sorting operations."""

    def __init__(self, file_repo: FileSystemRepository):
        """Initialize the file service.

        Args:
        ----
            file_repo: File system repository for I/O operations.

        """
        self.file_repo = file_repo
        self.filter_sorter = FileFilterSorter()

    async def get_latest_audio_file(self) -> FilePathSupportParams:
        """Get the most recently modified audio file with model support info.

        Returns
        -------
            FilePathSupportParams: File metadata and model support information.

        """
        return await self.file_repo.get_latest_audio_file()

    async def list_audio_files(
        self,
        pattern: Optional[str] = None,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        min_duration_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
        min_modified_time: Optional[float] = None,
        max_modified_time: Optional[float] = None,
        format_filter: Optional[str] = None,
        sort_by: SortBy = SortBy.NAME,
        reverse: bool = False,
    ) -> list[FilePathSupportParams]:
        """List, filter, and sort audio files.

        Args:
        ----
            pattern: Optional regex pattern to filter files by name.
            min_size_bytes: Minimum file size in bytes.
            max_size_bytes: Maximum file size in bytes.
            min_duration_seconds: Minimum audio duration in seconds.
            max_duration_seconds: Maximum audio duration in seconds.
            min_modified_time: Minimum file modification time (Unix timestamp).
            max_modified_time: Maximum file modification time (Unix timestamp).
            format_filter: Specific audio format to filter by.
            sort_by: Field to sort results by.
            reverse: Sort in reverse order if True.

        Returns:
        -------
            list[FilePathSupportParams]: Filtered and sorted list of file metadata.

        """
        # Step 1: List files from filesystem (with basic filtering)
        file_paths = await self.file_repo.list_audio_files(
            pattern=pattern,
            min_size_bytes=min_size_bytes,
            max_size_bytes=max_size_bytes,
            format_filter=format_filter,
        )

        # Step 2: Get metadata for all files in parallel (with caching)
        async def get_support(path: Path) -> FilePathSupportParams:
            path_str = str(path)
            mtime = path.stat().st_mtime
            return await get_cached_audio_file_support(
                path_str,
                mtime,
                self.file_repo.get_audio_file_support,
            )

        async with anyio.create_task_group() as tg:
            captures = [ResultCapture.start_soon(tg, get_support, path) for path in file_paths]
        file_support_results = [c.result() for c in captures]

        # Step 3: Apply domain-level filters (duration, modified time)
        filtered_results = [
            file_info
            for file_info in file_support_results
            if self.filter_sorter.apply_all_filters(
                file_info,
                min_size_bytes=min_size_bytes,
                max_size_bytes=max_size_bytes,
                min_duration_seconds=min_duration_seconds,
                max_duration_seconds=max_duration_seconds,
                min_modified_time=min_modified_time,
                max_modified_time=max_modified_time,
            )
        ]

        # Step 4: Sort using domain logic
        sorted_results = self.filter_sorter.sort_files(filtered_results, sort_by, reverse)

        return sorted_results
