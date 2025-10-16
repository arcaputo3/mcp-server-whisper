"""Test file system repository for audio file operations."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_whisper.exceptions import AudioFileNotFoundError
from mcp_server_whisper.infrastructure.file_system import FileSystemRepository


class TestFileSystemRepository:
    """Test suite for FileSystemRepository."""

    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        """Create temporary audio directory."""
        audio_path = tmp_path / "audio"
        audio_path.mkdir()
        return audio_path

    @pytest.fixture
    def repo(self, audio_dir: Path) -> FileSystemRepository:
        """Create FileSystemRepository instance."""
        return FileSystemRepository(audio_dir)

    @pytest.mark.anyio
    async def test_get_audio_file_support_mp3(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test getting file support for MP3 file."""
        test_file = audio_dir / "test.mp3"
        test_file.write_bytes(b"fake audio")

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=60000)  # 60 seconds in milliseconds

        with patch("mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file", return_value=mock_audio):
            result = await repo.get_audio_file_support(test_file)

            assert result.file_name == "test.mp3"
            assert result.format == "mp3"
            assert result.size_bytes == 10  # len(b"fake audio")
            assert result.transcription_support == ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
            assert result.chat_support and "gpt-4o-audio-preview" in result.chat_support[0]  # MP3 supports chat
            assert result.duration_seconds == 60.0

    @pytest.mark.anyio
    async def test_get_audio_file_support_wav(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test getting file support for WAV file."""
        test_file = audio_dir / "test.wav"
        test_file.write_bytes(b"wav data")

        mock_audio = MagicMock()
        mock_audio.__len__ = MagicMock(return_value=30000)  # 30 seconds

        with patch("mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file", return_value=mock_audio):
            result = await repo.get_audio_file_support(test_file)

            assert result.format == "wav"
            assert result.transcription_support is not None
            assert result.chat_support is not None  # WAV supports chat
            assert result.duration_seconds == 30.0

    @pytest.mark.anyio
    async def test_get_audio_file_support_unsupported_format(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that unsupported formats have no model support."""
        test_file = audio_dir / "test.txt"
        test_file.write_bytes(b"not audio")

        with patch(
            "mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file", side_effect=Exception("Invalid")
        ):
            result = await repo.get_audio_file_support(test_file)

            assert result.transcription_support is None
            assert result.chat_support is None
            assert result.duration_seconds is None  # Failed to load

    @pytest.mark.anyio
    async def test_get_audio_file_support_mp4_no_chat(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that MP4 supports transcription but not chat."""
        test_file = audio_dir / "video.mp4"
        test_file.write_bytes(b"mp4 data")

        with patch(
            "mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file",
            side_effect=Exception("Skip duration"),
        ):
            result = await repo.get_audio_file_support(test_file)

            assert result.format == "mp4"
            assert result.transcription_support is not None
            assert result.chat_support is None  # MP4 doesn't support chat
            assert result.duration_seconds is None

    @pytest.mark.anyio
    async def test_get_audio_file_support_duration_extraction_fails_gracefully(
        self, repo: FileSystemRepository, audio_dir: Path
    ) -> None:
        """Test that duration extraction failure doesn't crash the function."""
        test_file = audio_dir / "test.mp3"
        test_file.write_bytes(b"data")

        with patch(
            "mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file", side_effect=Exception("Load failed")
        ):
            result = await repo.get_audio_file_support(test_file)

            # Should still return result, just without duration
            assert result.file_name == "test.mp3"
            assert result.duration_seconds is None

    @pytest.mark.anyio
    async def test_get_latest_audio_file_finds_most_recent(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test finding the most recently modified audio file."""
        # Create files with different mtimes
        old_file = audio_dir / "old.mp3"
        old_file.write_bytes(b"old")

        import time

        time.sleep(0.01)

        new_file = audio_dir / "new.mp3"
        new_file.write_bytes(b"new")

        with patch("mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file"):
            result = await repo.get_latest_audio_file()
            assert result.file_name == "new.mp3"

    @pytest.mark.anyio
    async def test_get_latest_audio_file_no_files_raises_error(self, repo: FileSystemRepository) -> None:
        """Test that AudioFileNotFoundError is raised when no audio files exist."""
        with pytest.raises(AudioFileNotFoundError, match="No supported audio files found"):
            await repo.get_latest_audio_file()

    @pytest.mark.anyio
    async def test_get_latest_audio_file_skips_unsupported_formats(
        self, repo: FileSystemRepository, audio_dir: Path
    ) -> None:
        """Test that only supported formats are considered."""
        # Create unsupported file
        txt_file = audio_dir / "notes.txt"
        txt_file.write_bytes(b"text")

        # Should raise error because txt is not supported
        with pytest.raises(AudioFileNotFoundError):
            await repo.get_latest_audio_file()

    @pytest.mark.anyio
    async def test_get_latest_audio_file_skips_directories(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that directories are skipped."""
        # Create a directory with audio extension (edge case)
        fake_dir = audio_dir / "fake.mp3"
        fake_dir.mkdir()

        # Should not find any files
        with pytest.raises(AudioFileNotFoundError):
            await repo.get_latest_audio_file()

    @pytest.mark.anyio
    async def test_list_audio_files_returns_all_supported(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test listing all supported audio files."""
        # Create various formats
        (audio_dir / "file1.mp3").write_bytes(b"mp3")
        (audio_dir / "file2.wav").write_bytes(b"wav")
        (audio_dir / "file3.mp4").write_bytes(b"mp4")
        (audio_dir / "unsupported.txt").write_bytes(b"txt")

        result = await repo.list_audio_files()

        assert len(result) == 3
        filenames = {p.name for p in result}
        assert filenames == {"file1.mp3", "file2.wav", "file3.mp4"}

    @pytest.mark.anyio
    async def test_list_audio_files_with_regex_pattern(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test filtering by regex pattern."""
        (audio_dir / "interview_1.mp3").write_bytes(b"data")
        (audio_dir / "interview_2.mp3").write_bytes(b"data")
        (audio_dir / "music.mp3").write_bytes(b"data")

        result = await repo.list_audio_files(pattern="interview")

        assert len(result) == 2
        assert all("interview" in p.name for p in result)

    @pytest.mark.anyio
    async def test_list_audio_files_with_format_filter(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test filtering by specific format."""
        (audio_dir / "file1.mp3").write_bytes(b"mp3")
        (audio_dir / "file2.wav").write_bytes(b"wav")
        (audio_dir / "file3.mp3").write_bytes(b"mp3")

        result = await repo.list_audio_files(format_filter="mp3")

        assert len(result) == 2
        assert all(p.suffix == ".mp3" for p in result)

    @pytest.mark.anyio
    async def test_list_audio_files_with_min_size(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test filtering by minimum file size."""
        (audio_dir / "small.mp3").write_bytes(b"a" * 100)
        (audio_dir / "large.mp3").write_bytes(b"a" * 2000)

        result = await repo.list_audio_files(min_size_bytes=1000)

        assert len(result) == 1
        assert result[0].name == "large.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_with_max_size(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test filtering by maximum file size."""
        (audio_dir / "small.mp3").write_bytes(b"a" * 100)
        (audio_dir / "large.mp3").write_bytes(b"a" * 2000)

        result = await repo.list_audio_files(max_size_bytes=500)

        assert len(result) == 1
        assert result[0].name == "small.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_with_size_range(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test filtering by size range."""
        (audio_dir / "tiny.mp3").write_bytes(b"a" * 50)
        (audio_dir / "medium.mp3").write_bytes(b"a" * 500)
        (audio_dir / "large.mp3").write_bytes(b"a" * 2000)

        result = await repo.list_audio_files(min_size_bytes=100, max_size_bytes=1000)

        assert len(result) == 1
        assert result[0].name == "medium.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_skips_directories(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that directories are skipped in listing."""
        (audio_dir / "file.mp3").write_bytes(b"data")
        (audio_dir / "subdir").mkdir()

        result = await repo.list_audio_files()

        assert len(result) == 1
        assert result[0].name == "file.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_empty_directory(self, repo: FileSystemRepository) -> None:
        """Test listing in empty directory returns empty list."""
        result = await repo.list_audio_files()
        assert result == []

    @pytest.mark.anyio
    async def test_read_audio_file_success(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test successfully reading audio file."""
        test_file = audio_dir / "test.mp3"
        test_content = b"audio content here"
        test_file.write_bytes(test_content)

        result = await repo.read_audio_file(test_file)

        assert result == test_content

    @pytest.mark.anyio
    async def test_read_audio_file_not_found(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test error when reading non-existent file."""
        missing_file = audio_dir / "missing.mp3"

        with pytest.raises(AudioFileNotFoundError, match="File not found"):
            await repo.read_audio_file(missing_file)

    @pytest.mark.anyio
    async def test_read_audio_file_is_directory(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test error when trying to read a directory."""
        directory = audio_dir / "subdir"
        directory.mkdir()

        with pytest.raises(AudioFileNotFoundError, match="File not found"):
            await repo.read_audio_file(directory)

    @pytest.mark.anyio
    async def test_write_audio_file_success(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test successfully writing audio file."""
        output_file = audio_dir / "output.mp3"
        test_content = b"generated audio"

        await repo.write_audio_file(output_file, test_content)

        assert output_file.exists()
        assert output_file.read_bytes() == test_content

    @pytest.mark.anyio
    async def test_write_audio_file_creates_parent_directory(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that write creates parent directories if needed."""
        nested_dir = audio_dir / "subdir" / "nested"
        output_file = nested_dir / "output.mp3"
        test_content = b"audio"

        await repo.write_audio_file(output_file, test_content)

        assert output_file.exists()
        assert output_file.read_bytes() == test_content
        assert nested_dir.exists()

    @pytest.mark.anyio
    async def test_get_file_size_success(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test getting file size."""
        test_file = audio_dir / "test.mp3"
        test_content = b"a" * 1234
        test_file.write_bytes(test_content)

        result = await repo.get_file_size(test_file)

        assert result == 1234

    @pytest.mark.anyio
    async def test_get_file_size_not_found(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test error when file doesn't exist."""
        missing_file = audio_dir / "missing.mp3"

        with pytest.raises(AudioFileNotFoundError, match="File not found"):
            await repo.get_file_size(missing_file)

    @pytest.mark.anyio
    async def test_list_audio_files_combined_filters(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test using multiple filters together."""
        (audio_dir / "interview_small.mp3").write_bytes(b"a" * 100)
        (audio_dir / "interview_large.mp3").write_bytes(b"a" * 2000)
        (audio_dir / "music_large.mp3").write_bytes(b"a" * 2000)
        (audio_dir / "interview.wav").write_bytes(b"a" * 1500)

        result = await repo.list_audio_files(
            pattern="interview", min_size_bytes=500, max_size_bytes=2500, format_filter="mp3"
        )

        assert len(result) == 1
        assert result[0].name == "interview_large.mp3"

    @pytest.mark.anyio
    async def test_list_audio_files_case_insensitive_format(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that format filter is case insensitive."""
        (audio_dir / "test.MP3").write_bytes(b"data")
        (audio_dir / "test.wav").write_bytes(b"data")

        # Filter for "MP3" (uppercase)
        result = await repo.list_audio_files(format_filter="MP3")

        assert len(result) == 1
        # File extension is lowercase .mp3, but filter is case insensitive
        assert result[0].suffix.lower() == ".mp3"

    @pytest.mark.anyio
    async def test_get_audio_file_support_flac_format(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test FLAC file support (transcription only)."""
        test_file = audio_dir / "audio.flac"
        test_file.write_bytes(b"flac data")

        with patch(
            "mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file", side_effect=Exception("Skip")
        ):
            result = await repo.get_audio_file_support(test_file)

            assert result.format == "flac"
            assert result.transcription_support is not None
            assert result.chat_support is None  # FLAC doesn't support chat

    @pytest.mark.anyio
    async def test_get_audio_file_support_preserves_mtime(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that modification time is correctly captured."""
        test_file = audio_dir / "test.mp3"
        test_file.write_bytes(b"data")

        # Get the actual mtime
        expected_mtime = test_file.stat().st_mtime

        with patch(
            "mcp_server_whisper.infrastructure.file_system.AudioSegment.from_file", side_effect=Exception("Skip")
        ):
            result = await repo.get_audio_file_support(test_file)

            assert result.modified_time == expected_mtime

    @pytest.mark.anyio
    async def test_list_audio_files_regex_pattern_full_path(self, repo: FileSystemRepository, audio_dir: Path) -> None:
        """Test that regex pattern matches against full path."""
        (audio_dir / "test.mp3").write_bytes(b"data")
        (audio_dir / "audio.mp3").write_bytes(b"data")

        # Pattern should match full path string
        result = await repo.list_audio_files(pattern=r".*test\.mp3$")

        assert len(result) == 1
        assert result[0].name == "test.mp3"
