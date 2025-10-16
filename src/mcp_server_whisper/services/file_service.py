"""File discovery and management service."""

import asyncio
from typing import Optional

from ..constants import SortBy
from ..domain import FileFilterSorter
from ..infrastructure import FileSystemRepository, get_cached_audio_file_support
from ..models import FilePathSupportParams, ListAudioFilesInputParams


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
        cache_tasks = []
        for path in file_paths:
            path_str = str(path)
            mtime = path.stat().st_mtime
            cache_tasks.append(
                get_cached_audio_file_support(
                    path_str,
                    mtime,
                    self.file_repo.get_audio_file_support,
                )
            )

        file_support_results = await asyncio.gather(*cache_tasks)

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

    async def list_audio_files_batch(
        self,
        inputs: list[ListAudioFilesInputParams],
    ) -> list[list[FilePathSupportParams]]:
        """Process multiple list requests in parallel.

        Args:
        ----
            inputs: List of input parameters for each request.

        Returns:
        -------
            list[list[FilePathSupportParams]]: List of results for each request.

        """

        async def process_single(input_data: ListAudioFilesInputParams) -> list[FilePathSupportParams]:
            return await self.list_audio_files(
                pattern=input_data.pattern,
                min_size_bytes=input_data.min_size_bytes,
                max_size_bytes=input_data.max_size_bytes,
                min_duration_seconds=input_data.min_duration_seconds,
                max_duration_seconds=input_data.max_duration_seconds,
                min_modified_time=input_data.min_modified_time,
                max_modified_time=input_data.max_modified_time,
                format_filter=input_data.format,
                sort_by=input_data.sort_by,
                reverse=input_data.reverse,
            )

        return await asyncio.gather(*[process_single(input_data) for input_data in inputs])
