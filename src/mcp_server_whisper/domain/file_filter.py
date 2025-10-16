"""File filtering and sorting domain logic."""

from typing import Any, Callable, Optional

from ..constants import SortBy
from ..models import FilePathSupportParams


class FileFilterSorter:
    """Domain logic for filtering and sorting audio files.

    This class contains pure business logic for file filtering and sorting
    without any I/O operations.
    """

    @staticmethod
    def filter_by_size(
        file_info: FilePathSupportParams,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
    ) -> bool:
        """Check if file matches size criteria.

        Args:
        ----
            file_info: File metadata.
            min_size_bytes: Minimum file size in bytes.
            max_size_bytes: Maximum file size in bytes.

        Returns:
        -------
            bool: True if file matches criteria, False otherwise.

        """
        if min_size_bytes is not None and file_info.size_bytes < min_size_bytes:
            return False
        if max_size_bytes is not None and file_info.size_bytes > max_size_bytes:
            return False
        return True

    @staticmethod
    def filter_by_duration(
        file_info: FilePathSupportParams,
        min_duration_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> bool:
        """Check if file matches duration criteria.

        Args:
        ----
            file_info: File metadata.
            min_duration_seconds: Minimum audio duration in seconds.
            max_duration_seconds: Maximum audio duration in seconds.

        Returns:
        -------
            bool: True if file matches criteria, False otherwise.

        """
        # Skip duration filtering if duration info isn't available
        if file_info.duration_seconds is None:
            return True

        if min_duration_seconds is not None and file_info.duration_seconds < min_duration_seconds:
            return False
        if max_duration_seconds is not None and file_info.duration_seconds > max_duration_seconds:
            return False
        return True

    @staticmethod
    def filter_by_modified_time(
        file_info: FilePathSupportParams,
        min_modified_time: Optional[float] = None,
        max_modified_time: Optional[float] = None,
    ) -> bool:
        """Check if file matches modification time criteria.

        Args:
        ----
            file_info: File metadata.
            min_modified_time: Minimum file modification time (Unix timestamp).
            max_modified_time: Maximum file modification time (Unix timestamp).

        Returns:
        -------
            bool: True if file matches criteria, False otherwise.

        """
        if min_modified_time is not None and file_info.modified_time < min_modified_time:
            return False
        if max_modified_time is not None and file_info.modified_time > max_modified_time:
            return False
        return True

    @staticmethod
    def apply_all_filters(
        file_info: FilePathSupportParams,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        min_duration_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
        min_modified_time: Optional[float] = None,
        max_modified_time: Optional[float] = None,
    ) -> bool:
        """Apply all filters to a file.

        Args:
        ----
            file_info: File metadata.
            min_size_bytes: Minimum file size in bytes.
            max_size_bytes: Maximum file size in bytes.
            min_duration_seconds: Minimum audio duration in seconds.
            max_duration_seconds: Maximum audio duration in seconds.
            min_modified_time: Minimum file modification time (Unix timestamp).
            max_modified_time: Maximum file modification time (Unix timestamp).

        Returns:
        -------
            bool: True if file passes all filters, False otherwise.

        """
        if not FileFilterSorter.filter_by_size(file_info, min_size_bytes, max_size_bytes):
            return False

        if not FileFilterSorter.filter_by_duration(file_info, min_duration_seconds, max_duration_seconds):
            return False

        if not FileFilterSorter.filter_by_modified_time(file_info, min_modified_time, max_modified_time):
            return False

        return True

    @staticmethod
    def get_sort_key(sort_by: SortBy) -> Callable[[FilePathSupportParams], Any]:
        """Get the appropriate sort key function for a given sort field.

        Args:
        ----
            sort_by: Field to sort by.

        Returns:
        -------
            Callable: Function that extracts the sort key from FilePathSupportParams.

        """
        if sort_by == SortBy.NAME:
            return lambda x: x.file_name
        elif sort_by == SortBy.SIZE:
            return lambda x: x.size_bytes
        elif sort_by == SortBy.DURATION:
            # Use 0 for files with no duration to keep them at the beginning
            return lambda x: x.duration_seconds if x.duration_seconds is not None else 0
        elif sort_by == SortBy.MODIFIED_TIME:
            return lambda x: x.modified_time
        elif sort_by == SortBy.FORMAT:
            return lambda x: x.format
        else:
            # Default to sorting by name
            return lambda x: x.file_name

    @staticmethod
    def sort_files(
        files: list[FilePathSupportParams],
        sort_by: SortBy = SortBy.NAME,
        reverse: bool = False,
    ) -> list[FilePathSupportParams]:
        """Sort files according to the specified criteria.

        Args:
        ----
            files: List of file metadata to sort.
            sort_by: Field to sort by.
            reverse: Sort in reverse order if True.

        Returns:
        -------
            list[FilePathSupportParams]: Sorted list of file metadata.

        """
        sort_key = FileFilterSorter.get_sort_key(sort_by)
        return sorted(files, key=sort_key, reverse=reverse)
