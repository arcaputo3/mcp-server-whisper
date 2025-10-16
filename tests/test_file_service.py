"""Test file service orchestration layer."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_server_whisper.constants import SortBy
from mcp_server_whisper.infrastructure.file_system import FileSystemRepository
from mcp_server_whisper.models import FilePathSupportParams
from mcp_server_whisper.services.file_service import FileService


class TestFileService:
    """Test suite for FileService."""

    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        """Create temporary audio directory."""
        audio_path = tmp_path / "audio"
        audio_path.mkdir()
        return audio_path

    @pytest.fixture
    def mock_repo(self, audio_dir: Path) -> MagicMock:
        """Create mock FileSystemRepository."""
        repo = MagicMock(spec=FileSystemRepository)
        repo.audio_files_path = audio_dir
        return repo

    @pytest.fixture
    def service(self, mock_repo: MagicMock) -> FileService:
        """Create FileService instance with mocked repository."""
        return FileService(mock_repo)

    @pytest.fixture
    def sample_file_info(self) -> FilePathSupportParams:
        """Create sample file metadata."""
        return FilePathSupportParams(
            file_name="test.mp3",
            transcription_support=["whisper-1"],
            chat_support=["gpt-4o-audio-preview"],
            modified_time=100.0,
            size_bytes=1000,
            format="mp3",
            duration_seconds=60.0,
        )

    @pytest.mark.anyio
    async def test_get_latest_audio_file_delegates_to_repo(
        self, service: FileService, mock_repo: MagicMock, sample_file_info: FilePathSupportParams
    ) -> None:
        """Test that get_latest_audio_file delegates to repository."""
        mock_repo.get_latest_audio_file = AsyncMock(return_value=sample_file_info)

        result = await service.get_latest_audio_file()

        assert result == sample_file_info
        mock_repo.get_latest_audio_file.assert_called_once()

    @pytest.mark.anyio
    async def test_list_audio_files_with_no_filters(
        self, service: FileService, mock_repo: MagicMock, audio_dir: Path, sample_file_info: FilePathSupportParams
    ) -> None:
        """Test listing files with no filters applied."""
        # Create real files for stat() calls
        file1 = audio_dir / "file1.mp3"
        file2 = audio_dir / "file2.mp3"
        file1.write_bytes(b"data1")
        file2.write_bytes(b"data2")

        mock_paths = [file1, file2]
        mock_repo.list_audio_files = AsyncMock(return_value=mock_paths)

        # Mock get_audio_file_support to be called for each file
        mock_repo.get_audio_file_support = AsyncMock(return_value=sample_file_info)

        result = await service.list_audio_files()

        assert len(result) == 2
        # Should have called repo list with no filters
        mock_repo.list_audio_files.assert_called_once_with(
            pattern=None,
            min_size_bytes=None,
            max_size_bytes=None,
            format_filter=None,
        )

    @pytest.mark.anyio
    async def test_list_audio_files_with_all_filters(
        self, service: FileService, mock_repo: MagicMock, audio_dir: Path
    ) -> None:
        """Test listing files with all filters applied."""
        # Create real file
        filtered_file = audio_dir / "filtered.mp3"
        filtered_file.write_bytes(b"data")

        mock_paths = [filtered_file]

        file_info = FilePathSupportParams(
            file_name="filtered.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=150.0,
            size_bytes=1500,
            format="mp3",
            duration_seconds=90.0,
        )

        mock_repo.list_audio_files = AsyncMock(return_value=mock_paths)
        mock_repo.get_audio_file_support = AsyncMock(return_value=file_info)

        result = await service.list_audio_files(
            pattern="filtered",
            min_size_bytes=1000,
            max_size_bytes=2000,
            min_duration_seconds=60.0,
            max_duration_seconds=120.0,
            min_modified_time=100.0,
            max_modified_time=200.0,
            format_filter="mp3",
            sort_by=SortBy.SIZE,
            reverse=True,
        )

        assert len(result) == 1
        assert result[0].file_name == "filtered.mp3"

        # Verify repo was called with filesystem-level filters
        mock_repo.list_audio_files.assert_called_once_with(
            pattern="filtered",
            min_size_bytes=1000,
            max_size_bytes=2000,
            format_filter="mp3",
        )

    @pytest.mark.anyio
    async def test_list_audio_files_applies_domain_filters(
        self, service: FileService, mock_repo: MagicMock, audio_dir: Path
    ) -> None:
        """Test that domain-level filters (duration, mtime) are applied after fetching."""
        # Create real files
        file1_path = audio_dir / "file1.mp3"
        file2_path = audio_dir / "file2.mp3"
        file1_path.write_bytes(b"data1")
        file2_path.write_bytes(b"data2")

        mock_paths = [file1_path, file2_path]

        # Create files with different durations
        file1 = FilePathSupportParams(
            file_name="file1.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=100.0,
            size_bytes=1000,
            format="mp3",
            duration_seconds=30.0,  # Too short
        )

        file2 = FilePathSupportParams(
            file_name="file2.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=200.0,
            size_bytes=2000,
            format="mp3",
            duration_seconds=90.0,  # In range
        )

        mock_repo.list_audio_files = AsyncMock(return_value=mock_paths)
        mock_repo.get_audio_file_support = AsyncMock(side_effect=[file1, file2])

        result = await service.list_audio_files(
            min_duration_seconds=60.0,
            max_duration_seconds=120.0,
        )

        # Should filter out file1 (too short)
        assert len(result) == 1
        assert result[0].file_name == "file2.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_sorting_by_name(
        self, service: FileService, mock_repo: MagicMock, audio_dir: Path
    ) -> None:
        """Test sorting files by name."""
        # Create real files
        zebra_file = audio_dir / "zebra.mp3"
        alpha_file = audio_dir / "alpha.mp3"
        zebra_file.write_bytes(b"data")
        alpha_file.write_bytes(b"data")

        mock_paths = [zebra_file, alpha_file]

        file_z = FilePathSupportParams(
            file_name="zebra.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=100.0,
            size_bytes=1000,
            format="mp3",
            duration_seconds=60.0,
        )

        file_a = FilePathSupportParams(
            file_name="alpha.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=100.0,
            size_bytes=1000,
            format="mp3",
            duration_seconds=60.0,
        )

        mock_repo.list_audio_files = AsyncMock(return_value=mock_paths)
        mock_repo.get_audio_file_support = AsyncMock(side_effect=[file_z, file_a])

        result = await service.list_audio_files(sort_by=SortBy.NAME, reverse=False)

        # Should be sorted alphabetically
        assert result[0].file_name == "alpha.mp3"
        assert result[1].file_name == "zebra.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_sorting_by_size_descending(
        self, service: FileService, mock_repo: MagicMock, audio_dir: Path
    ) -> None:
        """Test sorting files by size in descending order."""
        # Create real files
        small_file = audio_dir / "small.mp3"
        large_file = audio_dir / "large.mp3"
        small_file.write_bytes(b"data")
        large_file.write_bytes(b"data")

        mock_paths = [small_file, large_file]

        file_small = FilePathSupportParams(
            file_name="small.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=100.0,
            size_bytes=500,
            format="mp3",
            duration_seconds=30.0,
        )

        file_large = FilePathSupportParams(
            file_name="large.mp3",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=100.0,
            size_bytes=3000,
            format="mp3",
            duration_seconds=90.0,
        )

        mock_repo.list_audio_files = AsyncMock(return_value=mock_paths)
        mock_repo.get_audio_file_support = AsyncMock(side_effect=[file_small, file_large])

        result = await service.list_audio_files(sort_by=SortBy.SIZE, reverse=True)

        # Should be sorted by size, descending
        assert result[0].size_bytes == 3000  # large first
        assert result[1].size_bytes == 500  # small second

    @pytest.mark.anyio
    async def test_list_audio_files_empty_result(self, service: FileService, mock_repo: MagicMock) -> None:
        """Test listing when no files match criteria."""
        mock_repo.list_audio_files = AsyncMock(return_value=[])

        result = await service.list_audio_files()

        assert result == []
