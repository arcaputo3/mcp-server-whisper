"""Test audio file listing and filtering capabilities."""

from mcp_server_whisper.constants import SortBy
from mcp_server_whisper.models import ListAudioFilesInputParams


def test_list_audio_files_input_params() -> None:
    """Test the ListAudioFilesInputParams class with various configurations."""
    # Test with default values
    params: ListAudioFilesInputParams = ListAudioFilesInputParams()
    assert params.sort_by == SortBy.NAME
    assert params.reverse is False
    assert params.pattern is None

    # Test with custom values
    params = ListAudioFilesInputParams(
        sort_by=SortBy.DURATION,
        reverse=True,
        pattern=".*\\.mp3",
        min_size_bytes=1000,
        max_size_bytes=10000,
        min_duration_seconds=30.0,
        max_duration_seconds=300.0,
        min_modified_time=1000000.0,
        max_modified_time=2000000.0,
        format="mp3",
    )
    assert params.sort_by == SortBy.DURATION
    assert params.reverse is True
    assert params.pattern == ".*\\.mp3"
    assert params.min_size_bytes == 1000
    assert params.max_size_bytes == 10000
    assert params.min_duration_seconds == 30.0
    assert params.max_duration_seconds == 300.0
    assert params.min_modified_time == 1000000.0
    assert params.max_modified_time == 2000000.0
    assert params.format == "mp3"
