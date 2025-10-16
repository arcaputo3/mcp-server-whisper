"""Test audio service orchestration layer."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from mcp_server_whisper.domain.audio_processor import AudioProcessor
from mcp_server_whisper.infrastructure.file_system import FileSystemRepository
from mcp_server_whisper.infrastructure.path_resolver import SecurePathResolver
from mcp_server_whisper.models import AudioProcessingResult
from mcp_server_whisper.services.audio_service import AudioService


class TestAudioService:
    """Test suite for AudioService."""

    @pytest.fixture
    def audio_dir(self, tmp_path: Path) -> Path:
        """Create temporary audio directory."""
        audio_path = tmp_path / "audio"
        audio_path.mkdir()
        return audio_path

    @pytest.fixture
    def mock_repo(self) -> MagicMock:
        """Create mock FileSystemRepository."""
        return MagicMock(spec=FileSystemRepository)

    @pytest.fixture
    def path_resolver(self, audio_dir: Path) -> SecurePathResolver:
        """Create real SecurePathResolver."""
        return SecurePathResolver(audio_dir)

    @pytest.fixture
    def service(self, mock_repo: MagicMock, path_resolver: SecurePathResolver) -> AudioService:
        """Create AudioService instance."""
        return AudioService(mock_repo, path_resolver)

    @pytest.mark.anyio
    async def test_convert_audio_success(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test successful audio conversion."""
        # Create input file
        input_file = audio_dir / "input.wav"
        input_file.write_bytes(b"wav data")

        # Mock AudioProcessor methods using pytest-mock
        mock_audio_data = MagicMock()
        converted_bytes = b"mp3 data"

        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_convert = mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock)

        mock_load.return_value = mock_audio_data
        mock_convert.return_value = converted_bytes
        mock_repo.write_audio_file = AsyncMock()

        result = await service.convert_audio(
            input_filename="input.wav",
            target_format="mp3",
        )

        assert isinstance(result, AudioProcessingResult)
        assert result.output_file == "input.mp3"

        mock_load.assert_called_once()
        mock_convert.assert_called_once()
        mock_repo.write_audio_file.assert_called_once()

    @pytest.mark.anyio
    async def test_convert_audio_with_custom_output_filename(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test conversion with custom output filename."""
        input_file = audio_dir / "source.wav"
        input_file.write_bytes(b"wav")

        mock_audio_data = MagicMock()

        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_convert = mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock)

        mock_load.return_value = mock_audio_data
        mock_convert.return_value = b"mp3"
        mock_repo.write_audio_file = AsyncMock()

        result = await service.convert_audio(
            input_filename="source.wav",
            output_filename="custom_output.mp3",
            target_format="mp3",
        )

        assert result.output_file == "custom_output.mp3"

    @pytest.mark.anyio
    async def test_compress_audio_below_threshold(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock
    ) -> None:
        """Test that files below threshold are not compressed."""
        input_file = audio_dir / "small.mp3"
        input_file.write_bytes(b"small file")

        file_size = 10 * 1024 * 1024  # 10 MB (below 25MB threshold)
        mock_repo.get_file_size = AsyncMock(return_value=file_size)

        result = await service.compress_audio(
            input_filename="small.mp3",
            max_mb=25,
        )

        # Should return original filename without compression
        assert result.output_file == "small.mp3"
        # Should not have called write_audio_file
        mock_repo.write_audio_file.assert_not_called()

    @pytest.mark.anyio
    async def test_compress_audio_above_threshold(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test that files above threshold are compressed."""
        input_file = audio_dir / "large.mp3"
        input_file.write_bytes(b"large file")

        file_size = 30 * 1024 * 1024  # 30 MB (above 25MB threshold)
        compressed_bytes = b"compressed data"

        mock_repo.get_file_size = AsyncMock(return_value=file_size)
        mock_audio_data = MagicMock()

        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)

        mock_load.return_value = mock_audio_data
        mock_compress.return_value = compressed_bytes
        mock_repo.write_audio_file = AsyncMock()

        result = await service.compress_audio(
            input_filename="large.mp3",
            max_mb=25,
        )

        assert result.output_file == "compressed_large.mp3"
        mock_compress.assert_called_once()
        mock_repo.write_audio_file.assert_called_once()

    @pytest.mark.anyio
    async def test_compress_audio_non_mp3_converts_first(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test that non-MP3 files are converted to MP3 before compression."""
        # Create WAV file that needs compression
        input_file = audio_dir / "large.wav"
        input_file.write_bytes(b"large wav")

        file_size = 30 * 1024 * 1024  # Large file
        mock_audio_data = MagicMock()

        # Mock get_file_size to return different values for different calls
        size_calls = [file_size, file_size]  # First for WAV, second for converted MP3
        mock_repo.get_file_size = AsyncMock(side_effect=size_calls)
        mock_repo.write_audio_file = AsyncMock()

        # Need to create the converted MP3 after first conversion
        async def write_and_create_file(path, content):
            path.write_bytes(content)

        mock_repo.write_audio_file.side_effect = write_and_create_file

        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_convert = mocker.patch.object(AudioProcessor, "convert_audio_format", new_callable=AsyncMock)
        mock_compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)

        mock_load.return_value = mock_audio_data
        mock_convert.return_value = b"converted mp3"
        mock_compress.return_value = b"compressed"

        await service.compress_audio(
            input_filename="large.wav",
            max_mb=25,
        )

        # Should have converted to MP3 first
        assert mock_convert.call_count >= 1
        # Then compressed
        assert mock_compress.call_count >= 1

    @pytest.mark.anyio
    async def test_compress_audio_with_custom_output_filename(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock, mocker: MockerFixture
    ) -> None:
        """Test compression with custom output filename."""
        input_file = audio_dir / "large.mp3"
        input_file.write_bytes(b"large")

        file_size = 30 * 1024 * 1024
        mock_repo.get_file_size = AsyncMock(return_value=file_size)
        mock_audio_data = MagicMock()

        mock_load = mocker.patch.object(AudioProcessor, "load_audio_from_path", new_callable=AsyncMock)
        mock_compress = mocker.patch.object(AudioProcessor, "compress_mp3", new_callable=AsyncMock)

        mock_load.return_value = mock_audio_data
        mock_compress.return_value = b"compressed"
        mock_repo.write_audio_file = AsyncMock()

        result = await service.compress_audio(
            input_filename="large.mp3",
            output_filename="custom_compressed.mp3",
            max_mb=25,
        )

        assert result.output_file == "custom_compressed.mp3"

    @pytest.mark.anyio
    async def test_maybe_compress_file_delegates_to_compress_audio(
        self, service: AudioService, audio_dir: Path, mock_repo: MagicMock
    ) -> None:
        """Test that maybe_compress_file delegates to compress_audio."""
        input_file = audio_dir / "test.mp3"
        input_file.write_bytes(b"data")

        file_size = 10 * 1024 * 1024  # Small file
        mock_repo.get_file_size = AsyncMock(return_value=file_size)

        result = await service.maybe_compress_file("test.mp3", max_mb=25)

        # Should not compress (below threshold)
        assert result.output_file == "test.mp3"
