import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from whisper_local.transcribe import do_whisper_transcription

@patch("whisper_local.transcribe.openai")
def test_do_whisper_transcription_success(mock_openai, tmp_path):
    """
    Test that do_whisper_transcription calls openai.audio.transcriptions.create.
    """
    input_file = tmp_path / "input.wav"
    input_file.write_text("fake audio data")

    class Args:
        input = str(input_file)
        output = None
        model = "whisper-1"

    do_whisper_transcription(Args())
    mock_openai.audio.transcriptions.create.assert_called_once()
    args, kwargs = mock_openai.audio.transcriptions.create.call_args
    assert kwargs["model"] == "whisper-1"


def test_do_whisper_transcription_file_not_found(tmp_path, capsys):
    """
    If input file is missing, exit with error.
    """
    class Args:
        input = str(tmp_path / "nope.wav")
        output = None
        model = "whisper-1"

    with pytest.raises(SystemExit) as e:
        do_whisper_transcription(Args())
    assert e.type == SystemExit
    assert e.value.code == 1

    out = capsys.readouterr().out
    assert "Could not find audio file" in out


@patch("whisper_local.transcribe.convert_to_supported_format")
@patch("whisper_local.transcribe.maybe_compress_file")
@patch("os.path.getsize", return_value=10 * 1024 * 1024)  # final size check passes
@patch("whisper_local.transcribe.openai.audio.transcriptions.create")
def test_do_whisper_transcription_conversion(
    mock_transcribe,
    mock_getsize,
    mock_compress,
    mock_convert,
    tmp_path
):
    """
    If the extension is not mp3 or wav, we convert it, possibly compress, then transcribe.
    """
    input_file = tmp_path / "input.m4a"
    input_file.write_text("fake audio data")

    # Return absolute paths from the mocks
    converted_mp3 = tmp_path / "converted.mp3"
    compressed_mp3 = tmp_path / "compressed.mp3"

    # We create dummy files so open(...) won't fail
    converted_mp3.write_text("fake mp3 data")
    compressed_mp3.write_text("fake mp3 data")

    mock_convert.return_value = str(converted_mp3)
    mock_compress.return_value = str(compressed_mp3)

    class Args:
        input = str(input_file)
        output = None
        model = "whisper-1"

    do_whisper_transcription(Args())

    mock_convert.assert_called_once()
    mock_compress.assert_called_once()
    mock_transcribe.assert_called_once()


@patch("os.path.getsize", return_value=30*1024*1024)  # final file remains 30MB
@patch("whisper_local.transcribe.convert_to_supported_format", return_value="converted.mp3")
@patch("whisper_local.transcribe.maybe_compress_file", return_value="compressed.mp3")
def test_do_whisper_transcription_too_big(
    mock_compress,
    mock_convert,
    mock_getsize,
    tmp_path,
    capsys
):
    """
    If after conversion/compression the file is still > 25MB, do_whisper_transcription should exit.
    """
    input_file = tmp_path / "large.wav"
    input_file.write_text("fake audio data")

    # create a dummy "compressed.mp3" to avoid not found error
    (tmp_path / "compressed.mp3").write_text("fake mp3 data")

    class Args:
        input = str(input_file)
        output = None
        model = "whisper-1"

    with pytest.raises(SystemExit) as e:
        do_whisper_transcription(Args())
    assert e.type == SystemExit
    assert e.value.code == 1

    out = capsys.readouterr().out
    # Because size is still 30MB, we expect the final error message
    assert "still above 25MB" in out