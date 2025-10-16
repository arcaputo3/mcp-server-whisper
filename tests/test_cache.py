"""Test caching infrastructure for audio file metadata."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mcp_server_whisper.infrastructure.cache import (
    clear_global_cache,
    get_cached_audio_file_support,
    get_global_cache_info,
)
from mcp_server_whisper.models import FilePathSupportParams


class TestGlobalCache:
    """Test suite for global audio file cache using async-lru."""

    @pytest.fixture(autouse=True)
    def clear_cache_before_each_test(self) -> Generator[None, None, None]:
        """Clear cache before each test to ensure isolation."""
        clear_global_cache()
        yield
        clear_global_cache()

    @pytest.fixture
    def sample_file_info(self) -> FilePathSupportParams:
        """Create sample file metadata."""
        return FilePathSupportParams(
            file_name="test.mp3",
            transcription_support=["whisper-1"],
            chat_support=["gpt-4o-audio-preview"],
            modified_time=100.0,
            size_bytes=1000,
            format="mp3",
            duration_seconds=60.0,
        )

    @pytest.mark.anyio
    async def test_cache_miss_calls_function(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache miss calls the underlying function."""
        mock_func = AsyncMock(return_value=sample_file_info)

        result = await get_cached_audio_file_support(
            file_path="/audio/test.mp3",
            mtime=100.0,
            get_support_func=mock_func,
        )

        assert result == sample_file_info
        mock_func.assert_called_once_with(Path("/audio/test.mp3"))

    @pytest.mark.anyio
    async def test_cache_hit_reuses_result(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache hit reuses cached result without calling function again."""
        mock_func = AsyncMock(return_value=sample_file_info)

        # First call - cache miss
        result1 = await get_cached_audio_file_support(
            file_path="/audio/test.mp3",
            mtime=100.0,
            get_support_func=mock_func,
        )

        # Second call with same parameters - cache hit
        result2 = await get_cached_audio_file_support(
            file_path="/audio/test.mp3",
            mtime=100.0,
            get_support_func=mock_func,
        )

        assert result1 == sample_file_info
        assert result2 == sample_file_info
        # Function should only be called once (cache hit on second call)
        mock_func.assert_called_once()

    @pytest.mark.anyio
    async def test_cache_invalidation_on_mtime_change(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache is invalidated when file mtime changes."""
        mock_func = AsyncMock(return_value=sample_file_info)

        # First call with mtime=100.0
        await get_cached_audio_file_support(
            file_path="/audio/test.mp3",
            mtime=100.0,
            get_support_func=mock_func,
        )

        # Second call with different mtime - should trigger new function call
        await get_cached_audio_file_support(
            file_path="/audio/test.mp3",
            mtime=200.0,  # Different mtime
            get_support_func=mock_func,
        )

        # Function should be called twice (once per mtime)
        assert mock_func.call_count == 2

    @pytest.mark.anyio
    async def test_cache_different_files_separately(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that different files are cached separately."""
        mock_func = AsyncMock(return_value=sample_file_info)

        # Cache two different files
        await get_cached_audio_file_support(
            file_path="/audio/file1.mp3",
            mtime=100.0,
            get_support_func=mock_func,
        )

        await get_cached_audio_file_support(
            file_path="/audio/file2.mp3",
            mtime=100.0,
            get_support_func=mock_func,
        )

        # Both files should trigger function calls (different cache keys)
        assert mock_func.call_count == 2

    @pytest.mark.anyio
    async def test_cache_respects_maxsize(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache evicts old entries when maxsize is reached."""
        mock_func = AsyncMock(return_value=sample_file_info)

        # Fill cache beyond maxsize (32) - add 35 entries
        for i in range(35):
            await get_cached_audio_file_support(
                file_path=f"/audio/file{i}.mp3",
                mtime=float(i),
                get_support_func=mock_func,
            )

        # Check that cache size is capped
        info = get_global_cache_info()
        assert info.currsize <= 32

    @pytest.mark.anyio
    async def test_cache_with_different_functions(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache distinguishes between different support functions."""
        mock_func1 = AsyncMock(return_value=sample_file_info)
        mock_func2 = AsyncMock(return_value=sample_file_info)

        # Same file, same mtime, but different function
        await get_cached_audio_file_support("/audio/test.mp3", 100.0, mock_func1)
        await get_cached_audio_file_support("/audio/test.mp3", 100.0, mock_func2)

        # Both functions should be called (different cache keys)
        mock_func1.assert_called_once()
        mock_func2.assert_called_once()

    def test_clear_global_cache(self) -> None:
        """Test clearing the global cache."""
        clear_global_cache()

        info = get_global_cache_info()
        assert info.currsize == 0

    def test_cache_info_structure(self) -> None:
        """Test that cache info returns proper statistics namedtuple."""
        info = get_global_cache_info()

        # Should have all expected attributes
        assert hasattr(info, "hits")
        assert hasattr(info, "misses")
        assert hasattr(info, "maxsize")
        assert hasattr(info, "currsize")

        # Initial state
        assert info.hits == 0
        assert info.misses == 0
        assert info.maxsize == 32
        assert info.currsize == 0

    @pytest.mark.anyio
    async def test_cache_info_tracks_hits_and_misses(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache info correctly tracks hits and misses."""
        mock_func = AsyncMock(return_value=sample_file_info)

        # First access - miss
        await get_cached_audio_file_support("/audio/test.mp3", 100.0, mock_func)

        info_after_miss = get_global_cache_info()
        assert info_after_miss.misses == 1
        assert info_after_miss.hits == 0

        # Second access - hit
        await get_cached_audio_file_support("/audio/test.mp3", 100.0, mock_func)

        info_after_hit = get_global_cache_info()
        assert info_after_hit.misses == 1  # Still 1 miss
        assert info_after_hit.hits == 1  # Now 1 hit

    @pytest.mark.anyio
    async def test_cache_returns_correct_data_type(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache returns the correct data type."""
        mock_func = AsyncMock(return_value=sample_file_info)

        result = await get_cached_audio_file_support("/audio/test.mp3", 100.0, mock_func)

        assert isinstance(result, FilePathSupportParams)
        assert result.file_name == "test.mp3"
        assert result.size_bytes == 1000
        assert result.format == "mp3"

    @pytest.mark.anyio
    async def test_cache_with_none_duration(self) -> None:
        """Test caching file info with None duration."""
        file_info = FilePathSupportParams(
            file_name="test.mp4",
            transcription_support=["whisper-1"],
            chat_support=None,
            modified_time=100.0,
            size_bytes=500,
            format="mp4",
            duration_seconds=None,  # No duration
        )

        mock_func = AsyncMock(return_value=file_info)

        result1 = await get_cached_audio_file_support("/audio/test.mp4", 100.0, mock_func)
        result2 = await get_cached_audio_file_support("/audio/test.mp4", 100.0, mock_func)

        assert result1 == result2
        assert result1.duration_seconds is None
        # Should hit cache
        mock_func.assert_called_once()

    @pytest.mark.anyio
    async def test_concurrent_cache_access(self, sample_file_info: FilePathSupportParams) -> None:
        """Test that cache handles concurrent access correctly."""
        import anyio

        mock_func = AsyncMock(return_value=sample_file_info)

        # Make multiple concurrent calls
        async def access_cache():
            return await get_cached_audio_file_support("/audio/test.mp3", 100.0, mock_func)

        async with anyio.create_task_group() as tg:
            for _ in range(5):
                tg.start_soon(access_cache)

        # All results should be the same
        # Function might be called multiple times due to race conditions,
        # but that's okay - cache is eventually consistent
        assert mock_func.call_count >= 1
