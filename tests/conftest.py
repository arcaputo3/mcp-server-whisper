import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_openai_api_key(monkeypatch):
    """
    Automatically set a dummy OPENAI_API_KEY for any test that invokes openai
    so we don't get an error about missing credentials.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")


@pytest.fixture
def mock_audio_segment_all(monkeypatch):
    """
    Global fixture that mocks pydub.AudioSegment.from_file
    so that it does NOT actually parse the file.
    By default it will not raise errors for invalid data.
    """
    mock_audio = MagicMock()
    mock_audio.export = MagicMock()
    monkeypatch.setattr("pydub.AudioSegment.from_file", MagicMock(return_value=mock_audio))
    return mock_audio