import pytest
import os
import sys
from unittest.mock import patch, MagicMock
from whisper_local.main import main


def test_main_whisper(tmp_path, monkeypatch):
    """
    If --use-gpt4o is not provided, main should call do_whisper_transcription.
    """
    test_audio = tmp_path / "input.wav"
    test_audio.write_text("fake audio")

    mocked_transcribe = MagicMock()
    mocked_gpt4o = MagicMock()

    monkeypatch.setattr(
        "whisper_local.main.do_whisper_transcription", mocked_transcribe
    )
    monkeypatch.setattr("whisper_local.main.do_gpt4o_audio", mocked_gpt4o)

    # Use monkeypatch to set sys.argv
    monkeypatch.setattr(sys, "argv", ["prog", str(test_audio)])
    main()

    mocked_transcribe.assert_called_once()
    mocked_gpt4o.assert_not_called()


def test_main_gpt4o(tmp_path, monkeypatch):
    """
    If --use-gpt4o is passed, main should call do_gpt4o_audio.
    """
    test_audio = tmp_path / "input.wav"
    test_audio.write_text("fake audio")

    mocked_transcribe = MagicMock()
    mocked_gpt4o = MagicMock()

    monkeypatch.setattr(
        "whisper_local.main.do_whisper_transcription", mocked_transcribe
    )
    monkeypatch.setattr("whisper_local.main.do_gpt4o_audio", mocked_gpt4o)

    monkeypatch.setattr(sys, "argv", ["prog", str(test_audio), "--use-gpt4o"])
    main()

    mocked_transcribe.assert_not_called()
    mocked_gpt4o.assert_called_once()


def test_main_file_not_found(tmp_path, monkeypatch, capsys):
    """
    If the input file doesn't exist, main should exit with code 1 and print an error.
    """
    non_existent = tmp_path / "no_such_file.wav"

    monkeypatch.setattr(sys, "argv", ["prog", str(non_existent)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.type == SystemExit
    assert e.value.code == 1

    captured = capsys.readouterr()
    assert "Could not find audio file" in captured.out
