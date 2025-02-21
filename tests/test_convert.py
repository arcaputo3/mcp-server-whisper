import os
import pytest
from unittest.mock import patch, MagicMock
from whisper_local.convert import (
    convert_to_supported_format,
    maybe_compress_file,
    compress_mp3_file
)

@pytest.fixture
def mock_audio_segment(monkeypatch):
    """
    Fixture to replace AudioSegment.from_file with a mock
    so we don't need real audio files to test conversion logic.
    """
    mock_audio = MagicMock()
    mock_audio.export = MagicMock()
    monkeypatch.setattr("pydub.AudioSegment.from_file", MagicMock(return_value=mock_audio))
    return mock_audio

def test_convert_to_supported_format(mock_audio_segment, tmp_path):
    """
    Test that convert_to_supported_format calls pydub's export and returns new path.
    """
    input_file = tmp_path / "test.m4a"
    input_file.write_text("fake data")  # Just to have a file

    output = convert_to_supported_format(str(input_file), "mp3")
    assert output.endswith(".mp3")
    # pydub export called
    mock_audio_segment.export.assert_called_once()

def test_convert_to_supported_format_exception(monkeypatch, tmp_path):
    """
    If pydub fails, we should raise an exception.
    """
    def mock_export_fail(*args, **kwargs):
        raise Exception("Export failure")

    mock_audio = MagicMock()
    mock_audio.export = mock_export_fail
    monkeypatch.setattr("pydub.AudioSegment.from_file", MagicMock(return_value=mock_audio))

    input_file = tmp_path / "test.wav"
    input_file.write_text("fake data")

    with pytest.raises(Exception) as exc:
        convert_to_supported_format(str(input_file), "mp3")
    assert "Export failure" in str(exc.value)

@patch("os.path.getsize")
def test_maybe_compress_file_not_needed(mock_getsize, mock_audio_segment, tmp_path):
    """
    If the file is below the threshold, we do not compress.
    """
    mock_getsize.return_value = 10 * 1024 * 1024  # 10MB
    input_file = tmp_path / "test.wav"
    input_file.write_text("fake data")

    result = maybe_compress_file(str(input_file), max_mb=25)
    # Should return the same path, no compression
    assert result == str(input_file)
    mock_audio_segment.export.assert_not_called()

@patch("os.path.getsize")
def test_maybe_compress_file_conversion(mock_getsize, mock_audio_segment, tmp_path):
    """
    If the file is above threshold and not mp3, we convert to mp3 then compress.
    """
    mock_getsize.side_effect = [30 * 1024 * 1024,  # original
                                20 * 1024 * 1024] # after compress
    input_file = tmp_path / "test.m4a"
    input_file.write_text("fake data")

    result = maybe_compress_file(str(input_file), max_mb=25)
    # Should now be "compressed_test.mp3"
    assert "compressed_test.mp3" in result
    assert result != str(input_file)

def test_compress_mp3_file(mock_audio_segment, tmp_path):
    """
    Test compress_mp3_file calls export with lowered sample rate.
    """
    from whisper_local.convert import compress_mp3_file
    input_file = tmp_path / "test.mp3"
    input_file.write_text("fake data")

    new_file = compress_mp3_file(str(input_file), out_sample_rate=8000)
    mock_audio_segment.export.assert_called_with(
        new_file,
        format="mp3",
        parameters=["-ar", "8000"]
    )
    assert "compressed_test.mp3" in new_file

@patch("os.path.getsize")
def test_maybe_compress_file_still_too_large(mock_getsize, mock_audio_segment, tmp_path):
    """
    If after compression the file is still above max_mb, we do not do multiple passes,
    the function returns the compressed path but leaves further handling to the caller.
    """
    # large, then large again
    mock_getsize.side_effect = [30 * 1024 * 1024, 29 * 1024 * 1024]
    input_file = tmp_path / "test.wav"
    input_file.write_text("fake data")

    result = maybe_compress_file(str(input_file), max_mb=25)
    # Should have tried compressing
    assert "compressed_test.mp3" in result
    # The function doesn't raise an error if it is still too large
    # That check is supposed to happen afterwards.