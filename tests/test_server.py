"""Test the whisper server functionality."""

from mcp_server_whisper.constants import SortBy
from mcp_server_whisper.models import ListAudioFilesInputParams


def test_sort_by_enum() -> None:
    """Test the SortBy enum has all expected values."""
    assert SortBy.NAME.value == "name"
    assert SortBy.SIZE.value == "size"
    assert SortBy.DURATION.value == "duration"
    assert SortBy.MODIFIED_TIME.value == "modified_time"
    assert SortBy.FORMAT.value == "format"


def test_list_audio_files_params_defaults() -> None:
    """Test the default parameters for ListAudioFilesInputParams."""
    params: ListAudioFilesInputParams = ListAudioFilesInputParams()
    assert params.pattern is None
    assert params.min_size_bytes is None
    assert params.max_size_bytes is None
    assert params.min_duration_seconds is None
    assert params.max_duration_seconds is None
    assert params.min_modified_time is None
    assert params.max_modified_time is None
    assert params.format is None
    assert params.sort_by == SortBy.NAME
    assert params.reverse is False
