import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from openai import AsyncOpenAI

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Set dummy OpenAI API key for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")


@pytest.fixture
def mock_audio_segment(monkeypatch):
    """Mock pydub.AudioSegment."""
    mock_audio = MagicMock()
    mock_audio.export = MagicMock()
    monkeypatch.setattr(
        "pydub.AudioSegment.from_file", MagicMock(return_value=mock_audio)
    )
    return mock_audio


@pytest_asyncio.fixture
async def mock_openai_async():
    """Create AsyncMock for OpenAI client with both Whisper and GPT-4o support."""
    mock = AsyncMock(spec=AsyncOpenAI)

    # Setup audio transcription mock
    mock.audio = AsyncMock()
    mock.audio.transcriptions = AsyncMock()
    mock.audio.transcriptions.create = AsyncMock()

    # Setup chat completion mock
    mock.chat = AsyncMock()
    mock.chat.completions = AsyncMock()
    mock.chat.completions.create = AsyncMock()

    # Patch the AsyncOpenAI class
    with patch("openai.AsyncOpenAI", return_value=mock):
        yield mock


@pytest.fixture
def mock_async_openai():
    """Create mock AsyncOpenAI client."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.audio.transcriptions.create = AsyncMock()
    return mock_client


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set dummy OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
