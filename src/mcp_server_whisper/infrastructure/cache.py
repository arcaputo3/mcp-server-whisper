"""Caching utilities for audio file metadata."""

from functools import lru_cache
from pathlib import Path

from ..models import FilePathSupportParams


class AudioFileCache:
    """Cache for audio file support information."""

    def __init__(self, maxsize: int = 32):
        """Initialize the audio file cache.

        Args:
        ----
            maxsize: Maximum number of cached entries.

        """
        self.maxsize = maxsize
        # Create the cached function with the specified maxsize
        self._get_cached_support = lru_cache(maxsize=maxsize)(self._get_support_impl)

    async def get_cached_audio_file_support(
        self,
        file_path: str,
        mtime: float,
        get_support_func,
    ) -> FilePathSupportParams:
        """Get cached audio file support information.

        Uses file path and modification time as cache key to ensure
        cache invalidation when files are modified.

        Args:
        ----
            file_path: Path to the audio file as string.
            mtime: File modification time (Unix timestamp).
            get_support_func: Async function to get file support info.

        Returns:
        -------
            FilePathSupportParams: Cached or freshly computed file support info.

        """
        # The LRU cache will use (file_path, mtime) as key
        return await self._get_cached_support(file_path, mtime, get_support_func)

    async def _get_support_impl(
        self,
        file_path: str,
        mtime: float,  # noqa: ARG002 - Used as cache key
        get_support_func,
    ) -> FilePathSupportParams:
        """Internal implementation for getting file support.

        Args:
        ----
            file_path: Path to the audio file as string.
            mtime: File modification time (used as cache key).
            get_support_func: Async function to get file support info.

        Returns:
        -------
            FilePathSupportParams: File support information.

        """
        return await get_support_func(Path(file_path))

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._get_cached_support.cache_clear()

    def cache_info(self):
        """Get cache statistics.

        Returns
        -------
            CacheInfo: Named tuple with cache statistics (hits, misses, maxsize, currsize).

        """
        return self._get_cached_support.cache_info()


# Global cache instance (can be configured or injected)
_global_audio_file_cache = AudioFileCache(maxsize=32)


async def get_cached_audio_file_support(
    file_path: str,
    mtime: float,
    get_support_func,
) -> FilePathSupportParams:
    """Get cached audio file support using the global cache instance.

    This is a convenience function that uses the global cache singleton.

    Args:
    ----
        file_path: Path to the audio file as string.
        mtime: File modification time (Unix timestamp).
        get_support_func: Async function to get file support info.

    Returns:
    -------
        FilePathSupportParams: Cached or freshly computed file support info.

    """
    return await _global_audio_file_cache.get_cached_audio_file_support(
        file_path=file_path,
        mtime=mtime,
        get_support_func=get_support_func,
    )


def clear_global_cache() -> None:
    """Clear the global audio file cache."""
    _global_audio_file_cache.clear_cache()


def get_global_cache_info():
    """Get statistics for the global cache.

    Returns
    -------
        CacheInfo: Named tuple with cache statistics.

    """
    return _global_audio_file_cache.cache_info()
