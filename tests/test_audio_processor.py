"""Test audio processor domain logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_whisper.domain.audio_processor import AudioProcessor
from mcp_server_whisper.exceptions import AudioCompressionError, AudioConversionError


class TestAudioProcessor:
    """Test suite for AudioProcessor domain logic."""

    def test_calculate_compression_needed_below_threshold(self) -> None:
        """Test that files below threshold don't need compression."""
        file_size = 20 * 1024 * 1024  # 20 MB
        result = AudioProcessor.calculate_compression_needed(file_size, max_mb=25)
        assert result is False

    def test_calculate_compression_needed_at_threshold(self) -> None:
        """Test that files exactly at threshold don't need compression."""
        file_size = 25 * 1024 * 1024  # Exactly 25 MB
        result = AudioProcessor.calculate_compression_needed(file_size, max_mb=25)
        assert result is False

    def test_calculate_compression_needed_above_threshold(self) -> None:
        """Test that files above threshold need compression."""
        file_size = 30 * 1024 * 1024  # 30 MB
        result = AudioProcessor.calculate_compression_needed(file_size, max_mb=25)
        assert result is True

    def test_calculate_compression_needed_custom_threshold(self) -> None:
        """Test compression calculation with custom threshold."""
        file_size = 15 * 1024 * 1024  # 15 MB
        result = AudioProcessor.calculate_compression_needed(file_size, max_mb=10)
        assert result is True

    def test_generate_output_path_with_custom_path(self, tmp_path: Path) -> None:
        """Test generating output path when custom path is provided."""
        input_path = tmp_path / "input.mp3"
        custom_path = tmp_path / "custom_output.mp3"

        result = AudioProcessor.generate_output_path(
            input_path=input_path, output_path=custom_path, suffix="compressed", extension=".mp3"
        )

        assert result == custom_path

    def test_generate_output_path_with_suffix(self, tmp_path: Path) -> None:
        """Test generating output path with suffix."""
        input_path = tmp_path / "audio.mp3"

        result = AudioProcessor.generate_output_path(
            input_path=input_path, output_path=None, suffix="compressed", extension=".mp3"
        )

        assert result == tmp_path / "compressed_audio.mp3"

    def test_generate_output_path_without_suffix(self, tmp_path: Path) -> None:
        """Test generating output path without suffix (extension change only)."""
        input_path = tmp_path / "audio.mp3"

        result = AudioProcessor.generate_output_path(
            input_path=input_path, output_path=None, suffix="", extension=".wav"
        )

        assert result == tmp_path / "audio.wav"

    def test_generate_output_path_preserves_parent_directory(self, tmp_path: Path) -> None:
        """Test that generated path stays in same parent directory."""
        parent = tmp_path / "audio_files"
        parent.mkdir()
        input_path = parent / "test.mp3"

        result = AudioProcessor.generate_output_path(
            input_path=input_path, output_path=None, suffix="converted", extension=".wav"
        )

        assert result.parent == parent
        assert result == parent / "converted_test.wav"

    @pytest.mark.anyio
    async def test_load_audio_from_path_success(self, tmp_path: Path) -> None:
        """Test successfully loading audio from path."""
        test_file = tmp_path / "test.mp3"
        test_file.write_bytes(b"fake audio data")

        mock_audio = MagicMock()

        with patch(
            "mcp_server_whisper.domain.audio_processor.AudioSegment.from_file", return_value=mock_audio
        ) as mock_from_file:
            result = await AudioProcessor.load_audio_from_path(test_file)

            assert result == mock_audio
            mock_from_file.assert_called_once()
            # Check that format was extracted from file extension
            call_args = mock_from_file.call_args
            assert call_args[0][0] == str(test_file)
            assert call_args[1]["format"] == "mp3"

    @pytest.mark.anyio
    async def test_load_audio_from_path_failure(self, tmp_path: Path) -> None:
        """Test error handling when loading audio fails."""
        test_file = tmp_path / "invalid.mp3"
        test_file.write_bytes(b"not audio")

        with patch(
            "mcp_server_whisper.domain.audio_processor.AudioSegment.from_file", side_effect=Exception("Invalid audio")
        ):
            with pytest.raises(AudioConversionError, match="Failed to load audio file"):
                await AudioProcessor.load_audio_from_path(test_file)

    @pytest.mark.anyio
    async def test_convert_audio_format_success(self, tmp_path: Path) -> None:
        """Test successfully converting audio format."""
        mock_audio = MagicMock()
        mock_export = MagicMock()
        mock_audio.export = mock_export

        output_path = tmp_path / "output.wav"
        test_data = b"converted audio data"
        output_path.write_bytes(test_data)

        result = await AudioProcessor.convert_audio_format(mock_audio, "wav", output_path)

        assert result == test_data
        mock_export.assert_called_once()
        call_args = mock_export.call_args
        assert call_args[0][0] == str(output_path)
        assert call_args[1]["format"] == "wav"

    @pytest.mark.anyio
    async def test_convert_audio_format_failure(self, tmp_path: Path) -> None:
        """Test error handling when conversion fails."""
        mock_audio = MagicMock()
        mock_audio.export = MagicMock(side_effect=Exception("Conversion failed"))

        output_path = tmp_path / "output.mp3"

        with pytest.raises(AudioConversionError, match="Audio conversion to mp3 failed"):
            await AudioProcessor.convert_audio_format(mock_audio, "mp3", output_path)

    @pytest.mark.anyio
    async def test_compress_mp3_success(self, tmp_path: Path) -> None:
        """Test successfully compressing MP3."""
        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_export = MagicMock()
        mock_audio.export = mock_export

        output_path = tmp_path / "compressed.mp3"
        test_data = b"compressed audio"
        output_path.write_bytes(test_data)

        result = await AudioProcessor.compress_mp3(mock_audio, output_path, target_sample_rate=22050)

        assert result == test_data
        mock_export.assert_called_once()
        call_args = mock_export.call_args
        assert call_args[0][0] == str(output_path)
        assert call_args[1]["format"] == "mp3"
        assert call_args[1]["parameters"] == ["-ar", "22050"]

    @pytest.mark.anyio
    async def test_compress_mp3_failure(self, tmp_path: Path) -> None:
        """Test error handling when compression fails."""
        mock_audio = MagicMock()
        mock_audio.frame_rate = 44100
        mock_audio.export = MagicMock(side_effect=Exception("Compression failed"))

        output_path = tmp_path / "compressed.mp3"

        with pytest.raises(AudioCompressionError, match="MP3 compression failed"):
            await AudioProcessor.compress_mp3(mock_audio, output_path)

    @pytest.mark.anyio
    async def test_concatenate_audio_segments_mp3(self) -> None:
        """Test concatenating multiple MP3 audio segments."""
        chunk1 = b"chunk1 data"
        chunk2 = b"chunk2 data"

        mock_segment = MagicMock()
        mock_output = b"concatenated audio"

        with (
            patch("mcp_server_whisper.domain.audio_processor.AudioSegment.from_mp3") as mock_from_mp3,
            patch("mcp_server_whisper.domain.audio_processor.AudioSegment.empty") as mock_empty,
        ):
            # Return same mock for simplicity
            mock_empty.return_value = mock_segment
            mock_from_mp3.return_value = mock_segment

            # Mock += operation
            mock_segment.__iadd__ = MagicMock(return_value=mock_segment)

            # Mock the export
            def mock_export(output, format):
                output.write(mock_output)
                return None

            mock_segment.export = mock_export

            result = await AudioProcessor.concatenate_audio_segments([chunk1, chunk2], format="mp3")

            assert result == mock_output
            assert mock_from_mp3.call_count == 2

    @pytest.mark.anyio
    async def test_concatenate_audio_segments_wav(self) -> None:
        """Test concatenating WAV audio segments."""
        chunks = [b"chunk1", b"chunk2"]

        mock_segment = MagicMock()
        mock_output = b"concatenated wav"

        with (
            patch("mcp_server_whisper.domain.audio_processor.AudioSegment.from_wav") as mock_from_wav,
            patch("mcp_server_whisper.domain.audio_processor.AudioSegment.empty") as mock_empty,
        ):
            mock_empty.return_value = mock_segment
            mock_from_wav.return_value = mock_segment

            # Mock += operation
            mock_segment.__iadd__ = MagicMock(return_value=mock_segment)

            def mock_export(output, format):
                output.write(mock_output)
                return None

            mock_segment.export = mock_export

            result = await AudioProcessor.concatenate_audio_segments(chunks, format="wav")

            assert result == mock_output
            assert mock_from_wav.call_count == 2

    @pytest.mark.anyio
    async def test_concatenate_audio_segments_failure(self) -> None:
        """Test error handling when concatenation fails."""
        chunks = [b"chunk1"]

        with patch(
            "mcp_server_whisper.domain.audio_processor.AudioSegment.empty", side_effect=Exception("Concat failed")
        ):
            with pytest.raises(AudioConversionError, match="Audio concatenation failed"):
                await AudioProcessor.concatenate_audio_segments(chunks)

    @pytest.mark.anyio
    async def test_concatenate_audio_segments_empty_list(self) -> None:
        """Test concatenating empty list of chunks."""
        mock_combined = MagicMock()
        mock_output = b"empty"

        with patch("mcp_server_whisper.domain.audio_processor.AudioSegment.empty") as mock_empty:
            mock_empty.return_value = mock_combined

            def mock_export(output, format):
                output.write(mock_output)
                return None

            mock_combined.export = mock_export

            result = await AudioProcessor.concatenate_audio_segments([])

            assert result == mock_output
