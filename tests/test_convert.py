import os
import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock
from whisper_local.convert import (
    convert_to_supported_format,
    maybe_compress_file,
    compress_mp3_file,
)


@pytest.fixture
def mock_audio_segment(monkeypatch):
    """
    Fixture to replace AudioSegment.from_file with a mock
    so we don't need real audio files to test conversion logic.
    """
    mock_audio = MagicMock()
    mock_audio.export = MagicMock()
    monkeypatch.setattr(
        "pydub.AudioSegment.from_file", MagicMock(return_value=mock_audio)
    )
    return mock_audio


@pytest.mark.asyncio(loop_scope="session")
async def test_convert_to_supported_format(mock_audio_segment, tmp_path):
    """Test that convert_to_supported_format calls pydub's export and returns new path."""
    input_file = tmp_path / "test.m4a"
    input_file.write_text("fake data")

    output = await convert_to_supported_format(str(input_file), "mp3")
    assert output.endswith(".mp3")
    mock_audio_segment.export.assert_called_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_convert_to_supported_format_exception(monkeypatch, tmp_path):
    """If pydub fails, we should raise an exception."""

    def mock_export_fail(*args, **kwargs):
        raise Exception("Export failure")

    mock_audio = MagicMock()
    mock_audio.export = mock_export_fail
    monkeypatch.setattr(
        "pydub.AudioSegment.from_file", MagicMock(return_value=mock_audio)
    )

    input_file = tmp_path / "test.wav"
    input_file.write_text("fake data")

    with pytest.raises(Exception) as exc:
        await convert_to_supported_format(str(input_file), "mp3")
    assert "Export failure" in str(exc.value)


@patch("os.path.getsize")
@pytest.mark.asyncio(loop_scope="session")
async def test_maybe_compress_file_not_needed(
    mock_getsize, mock_audio_segment, tmp_path
):
    """If the file is below threshold, we do not compress."""
    mock_getsize.return_value = 10 * 1024 * 1024
    input_file = tmp_path / "test.wav"
    input_file.write_text("fake data")

    result = await maybe_compress_file(str(input_file), max_mb=25)
    assert result == str(input_file)
    mock_audio_segment.export.assert_not_called()


@patch("os.path.getsize")
@pytest.mark.asyncio
async def test_maybe_compress_file_conversion(
    mock_getsize, mock_audio_segment, tmp_path
):
    """If file above threshold and not mp3, convert to mp3 then compress => return compressed_test.mp3."""
    mock_getsize.side_effect = [30 * 1024 * 1024, 20 * 1024 * 1024]
    input_file = tmp_path / "test.m4a"
    input_file.write_text("fake data")

    result = await maybe_compress_file(str(input_file), max_mb=25)
    assert "compressed_test.mp3" in result


@pytest.mark.asyncio
async def test_compress_mp3_file(mock_audio_segment, tmp_path):
    """Test compress_mp3_file calls export with lowered sample rate."""
    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake data")

    result = await compress_mp3_file(str(input_file), out_sample_rate=8000)
    mock_audio_segment.export.assert_called_with(
        result, format="mp3", parameters=["-ar", "8000"]
    )
    assert "compressed_test.mp3" in result


@patch("os.path.getsize")
@pytest.mark.asyncio
async def test_maybe_compress_file_still_too_large(
    mock_getsize, mock_audio_segment, tmp_path
):
    """Test handling of files that remain too large after compression."""
    mock_getsize.side_effect = [30 * 1024 * 1024, 29 * 1024 * 1024]
    input_file = tmp_path / "test.wav"
    input_file.write_text("fake data")

    result = await maybe_compress_file(str(input_file), max_mb=25)
    assert "compressed_test.mp3" in result
