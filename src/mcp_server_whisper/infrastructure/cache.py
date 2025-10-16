"""Caching utilities for audio file metadata."""

from collections.abc import Awaitable, Callable
from functools import _CacheInfo
from pathlib import Path

from async_lru import alru_cache

from ..models import FilePathSupportParams


# Global cached function with maxsize=32
@alru_cache(maxsize=32)
async def get_cached_audio_file_support(
    file_path: str,
    mtime: float,
    get_support_func: Callable[[Path], Awaitable[FilePathSupportParams]],
) -> FilePathSupportParams:
    """Get cached audio file support using async LRU cache.

    Uses file path and modification time as cache key to ensure
    cache invalidation when files are modified.

    Args:
    ----
        file_path: Path to the audio file as string.
        mtime: File modification time (Unix timestamp) - used as cache key.
        get_support_func: Async function to get file support info.

    Returns:
    -------
        FilePathSupportParams: Cached or freshly computed file support info.

    """
    return await get_support_func(Path(file_path))


def clear_global_cache() -> None:
    """Clear the global audio file cache."""
    get_cached_audio_file_support.cache_clear()


def get_global_cache_info() -> _CacheInfo:
    """Get statistics for the global cache.

    Returns
    -------
        _CacheInfo: Named tuple with cache statistics (hits, misses, maxsize, currsize).

    """
    return get_cached_audio_file_support.cache_info()
