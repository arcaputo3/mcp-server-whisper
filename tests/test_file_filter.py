"""Test file filtering and sorting domain logic."""

import pytest

from mcp_server_whisper.constants import SortBy
from mcp_server_whisper.domain.file_filter import FileFilterSorter
from mcp_server_whisper.models import FilePathSupportParams


class TestFileFilterSorter:
    """Test suite for FileFilterSorter domain logic."""

    @pytest.fixture
    def sample_files(self) -> list[FilePathSupportParams]:
        """Create sample file metadata for testing."""
        return [
            FilePathSupportParams(
                file_name="file1.mp3",
                size_bytes=1000,
                modified_time=100.0,
                duration_seconds=60.0,
                format="mp3",
                transcription_support=["whisper-1"],
                chat_support=["gpt-4o-audio-preview"],
            ),
            FilePathSupportParams(
                file_name="file2.wav",
                size_bytes=2000,
                modified_time=200.0,
                duration_seconds=120.0,
                format="wav",
                transcription_support=["whisper-1"],
                chat_support=["gpt-4o-audio-preview"],
            ),
            FilePathSupportParams(
                file_name="file3.mp4",
                size_bytes=500,
                modified_time=300.0,
                duration_seconds=None,  # No duration info
                format="mp4",
                transcription_support=["whisper-1"],
                chat_support=None,  # No chat support
            ),
            FilePathSupportParams(
                file_name="file4.flac",
                size_bytes=3000,
                modified_time=150.0,
                duration_seconds=30.0,
                format="flac",
                transcription_support=["whisper-1"],
                chat_support=None,  # No chat support
            ),
        ]

    def test_filter_by_size_min_only(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by minimum size."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_size(f, min_size_bytes=1000)]
        assert len(result) == 3  # 1000, 2000, 3000
        assert all(f.size_bytes >= 1000 for f in result)

    def test_filter_by_size_max_only(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by maximum size."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_size(f, max_size_bytes=2000)]
        assert len(result) == 3  # 500, 1000, 2000
        assert all(f.size_bytes <= 2000 for f in result)

    def test_filter_by_size_min_and_max(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by both min and max size."""
        result = [
            f for f in sample_files if FileFilterSorter.filter_by_size(f, min_size_bytes=1000, max_size_bytes=2000)
        ]
        assert len(result) == 2  # 1000, 2000
        assert all(1000 <= f.size_bytes <= 2000 for f in result)

    def test_filter_by_size_no_filters(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test that no filters returns all files."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_size(f)]
        assert len(result) == 4

    def test_filter_by_duration_min_only(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by minimum duration."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_duration(f, min_duration_seconds=60.0)]
        assert len(result) == 3  # 60.0, 120.0, and None (passes through)
        # None durations should pass through
        assert any(f.duration_seconds is None for f in result)

    def test_filter_by_duration_max_only(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by maximum duration."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_duration(f, max_duration_seconds=60.0)]
        assert len(result) == 3  # 30.0, 60.0, and None (passes through)

    def test_filter_by_duration_handles_none(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test that files with None duration pass all duration filters."""
        file_with_no_duration = sample_files[2]  # file3.mp4
        assert file_with_no_duration.duration_seconds is None
        assert FileFilterSorter.filter_by_duration(file_with_no_duration, min_duration_seconds=1000.0)
        assert FileFilterSorter.filter_by_duration(file_with_no_duration, max_duration_seconds=1.0)

    def test_filter_by_modified_time_min_only(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by minimum modification time."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_modified_time(f, min_modified_time=150.0)]
        assert len(result) == 3  # 150.0, 200.0, 300.0
        assert all(f.modified_time >= 150.0 for f in result)

    def test_filter_by_modified_time_max_only(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by maximum modification time."""
        result = [f for f in sample_files if FileFilterSorter.filter_by_modified_time(f, max_modified_time=200.0)]
        assert len(result) == 3  # 100.0, 150.0, 200.0
        assert all(f.modified_time <= 200.0 for f in result)

    def test_filter_by_modified_time_min_and_max(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test filtering by both min and max modification time."""
        result = [
            f
            for f in sample_files
            if FileFilterSorter.filter_by_modified_time(f, min_modified_time=150.0, max_modified_time=250.0)
        ]
        assert len(result) == 2  # 150.0, 200.0
        assert all(150.0 <= f.modified_time <= 250.0 for f in result)

    def test_apply_all_filters_combined(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test applying all filters together."""
        # Filter: size 1000-3000, duration 30-120, modified_time 100-200
        result = [
            f
            for f in sample_files
            if FileFilterSorter.apply_all_filters(
                f,
                min_size_bytes=1000,
                max_size_bytes=3000,
                min_duration_seconds=30.0,
                max_duration_seconds=120.0,
                min_modified_time=100.0,
                max_modified_time=200.0,
            )
        ]
        # Should match file1 (1000 bytes, 60s, 100) and file2 (2000 bytes, 120s, 200)
        # file4 has modified_time=150 which is in range, size=3000 in range, duration=30 in range
        assert len(result) == 3
        assert all(1000 <= f.size_bytes <= 3000 for f in result)
        assert all(100.0 <= f.modified_time <= 200.0 for f in result)

    def test_apply_all_filters_no_match(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test that impossible filter combination returns empty list."""
        result = [
            f
            for f in sample_files
            if FileFilterSorter.apply_all_filters(
                f,
                min_size_bytes=10000,  # All files are smaller
                max_size_bytes=20000,
            )
        ]
        assert len(result) == 0

    def test_apply_all_filters_no_filters_returns_all(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test that no filters returns all files."""
        result = [f for f in sample_files if FileFilterSorter.apply_all_filters(f)]
        assert len(result) == 4

    def test_get_sort_key_name(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sort key function for name sorting."""
        sort_key = FileFilterSorter.get_sort_key(SortBy.NAME)
        assert sort_key(sample_files[0]) == "file1.mp3"
        assert sort_key(sample_files[1]) == "file2.wav"

    def test_get_sort_key_size(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sort key function for size sorting."""
        sort_key = FileFilterSorter.get_sort_key(SortBy.SIZE)
        assert sort_key(sample_files[0]) == 1000
        assert sort_key(sample_files[1]) == 2000

    def test_get_sort_key_duration(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sort key function for duration sorting."""
        sort_key = FileFilterSorter.get_sort_key(SortBy.DURATION)
        assert sort_key(sample_files[0]) == 60.0
        assert sort_key(sample_files[2]) == 0  # None duration returns 0

    def test_get_sort_key_modified_time(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sort key function for modified time sorting."""
        sort_key = FileFilterSorter.get_sort_key(SortBy.MODIFIED_TIME)
        assert sort_key(sample_files[0]) == 100.0
        assert sort_key(sample_files[1]) == 200.0

    def test_get_sort_key_format(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sort key function for format sorting."""
        sort_key = FileFilterSorter.get_sort_key(SortBy.FORMAT)
        assert sort_key(sample_files[0]) == "mp3"
        assert sort_key(sample_files[1]) == "wav"

    def test_sort_files_by_name_ascending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by name in ascending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.NAME, reverse=False)
        assert result[0].file_name == "file1.mp3"
        assert result[1].file_name == "file2.wav"
        assert result[2].file_name == "file3.mp4"
        assert result[3].file_name == "file4.flac"

    def test_sort_files_by_name_descending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by name in descending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.NAME, reverse=True)
        assert result[0].file_name == "file4.flac"
        assert result[1].file_name == "file3.mp4"
        assert result[2].file_name == "file2.wav"
        assert result[3].file_name == "file1.mp3"

    def test_sort_files_by_size_ascending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by size in ascending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.SIZE, reverse=False)
        assert result[0].size_bytes == 500  # file3
        assert result[1].size_bytes == 1000  # file1
        assert result[2].size_bytes == 2000  # file2
        assert result[3].size_bytes == 3000  # file4

    def test_sort_files_by_size_descending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by size in descending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.SIZE, reverse=True)
        assert result[0].size_bytes == 3000
        assert result[1].size_bytes == 2000
        assert result[2].size_bytes == 1000
        assert result[3].size_bytes == 500

    def test_sort_files_by_duration_ascending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by duration in ascending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.DURATION, reverse=False)
        # None duration (file3) should be first with value 0
        assert result[0].duration_seconds is None
        assert result[1].duration_seconds == 30.0  # file4
        assert result[2].duration_seconds == 60.0  # file1
        assert result[3].duration_seconds == 120.0  # file2

    def test_sort_files_by_modified_time_ascending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by modified time in ascending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.MODIFIED_TIME, reverse=False)
        assert result[0].modified_time == 100.0  # file1
        assert result[1].modified_time == 150.0  # file4
        assert result[2].modified_time == 200.0  # file2
        assert result[3].modified_time == 300.0  # file3

    def test_sort_files_by_format_ascending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting files by format in ascending order."""
        result = FileFilterSorter.sort_files(sample_files, sort_by=SortBy.FORMAT, reverse=False)
        formats = [f.format for f in result]
        assert formats == sorted(formats)  # Should be alphabetically sorted

    def test_sort_files_default_is_name_ascending(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test that default sorting is by name, ascending."""
        result = FileFilterSorter.sort_files(sample_files)
        assert result[0].file_name == "file1.mp3"
        assert result[1].file_name == "file2.wav"

    def test_sort_files_empty_list(self) -> None:
        """Test sorting empty list returns empty list."""
        result = FileFilterSorter.sort_files([], sort_by=SortBy.NAME)
        assert result == []

    def test_sort_files_single_file(self, sample_files: list[FilePathSupportParams]) -> None:
        """Test sorting single file returns that file."""
        single_file = [sample_files[0]]
        result = FileFilterSorter.sort_files(single_file, sort_by=SortBy.SIZE)
        assert len(result) == 1
        assert result[0] == sample_files[0]
