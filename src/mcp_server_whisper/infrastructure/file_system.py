"""File system operations for audio file management."""

import re
from pathlib import Path
from typing import Optional

import aiofiles
import anyio
from openai.types import AudioModel
from pydub import AudioSegment  # type: ignore

from ..constants import CHAT_WITH_AUDIO_FORMATS, TRANSCRIBE_AUDIO_FORMATS, AudioChatModel
from ..exceptions import AudioFileError, AudioFileNotFoundError
from ..models import FilePathSupportParams


class FileSystemRepository:
    """Repository for file system operations related to audio files."""

    def __init__(self, audio_files_path: Path):
        """Initialize the file system repository.

        Args:
        ----
            audio_files_path: Path to the directory containing audio files.

        """
        self.audio_files_path = audio_files_path

    async def get_audio_file_support(self, file_path: Path) -> FilePathSupportParams:
        """Determine audio transcription file format support and metadata.

        Includes file size, format, and duration information where available.

        Args:
        ----
            file_path: Path to the audio file.

        Returns:
        -------
            FilePathSupportParams: File metadata and model support information.

        """
        file_ext = file_path.suffix.lower()

        transcription_support: list[AudioModel] | None = (
            ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
            if file_ext in TRANSCRIBE_AUDIO_FORMATS
            else None
        )
        chat_support: list[AudioChatModel] | None = (
            [
                "gpt-4o-audio-preview-2024-10-01",
                "gpt-4o-audio-preview-2024-12-17",
                "gpt-4o-mini-audio-preview-2024-12-17",
            ]
            if file_ext in CHAT_WITH_AUDIO_FORMATS
            else None
        )

        # Get file stats (including size - much faster than reading entire file!)
        file_stats = file_path.stat()
        size_bytes = file_stats.st_size

        # Get audio format (remove the dot from extension)
        audio_format = file_ext[1:] if file_ext.startswith(".") else file_ext

        # Get duration if possible (could be expensive for large files)
        duration_seconds = None
        try:
            # Load just the metadata to get duration
            audio = await anyio.to_thread.run_sync(lambda: AudioSegment.from_file(str(file_path), format=audio_format))
            # Convert from milliseconds to seconds
            duration_seconds = len(audio) / 1000.0
        except Exception:
            # If we can't get duration, just continue without it
            pass

        return FilePathSupportParams(
            file_name=file_path.name,
            transcription_support=transcription_support,
            chat_support=chat_support,
            modified_time=file_stats.st_mtime,
            size_bytes=size_bytes,
            format=audio_format,
            duration_seconds=duration_seconds,
        )

    async def get_latest_audio_file(self) -> FilePathSupportParams:
        """Get the most recently modified audio file with model support info.

        Supported formats:
        - Whisper: mp3, mp4, mpeg, mpga, m4a, wav, webm
        - GPT-4o: mp3, wav

        Returns
        -------
            FilePathSupportParams: File metadata and model support information.

        Raises
        ------
            AudioFileNotFoundError: If no supported audio files are found.
            AudioFileError: If there's an error accessing audio files.

        """
        try:
            files = []
            for file_path in self.audio_files_path.iterdir():
                if not file_path.is_file():
                    continue

                file_ext = file_path.suffix.lower()
                if file_ext in TRANSCRIBE_AUDIO_FORMATS or file_ext in CHAT_WITH_AUDIO_FORMATS:
                    files.append((file_path, file_path.stat().st_mtime))

            if not files:
                raise AudioFileNotFoundError("No supported audio files found")

            latest_file = max(files, key=lambda x: x[1])[0]
            return await self.get_audio_file_support(latest_file)

        except AudioFileNotFoundError:
            raise
        except Exception as e:
            raise AudioFileError(f"Failed to get latest audio file: {e}") from e

    async def list_audio_files(
        self,
        pattern: Optional[str] = None,
        min_size_bytes: Optional[int] = None,
        max_size_bytes: Optional[int] = None,
        format_filter: Optional[str] = None,
    ) -> list[Path]:
        """List audio files matching the given criteria.

        Args:
        ----
            pattern: Optional regex pattern to filter files by name.
            min_size_bytes: Minimum file size in bytes.
            max_size_bytes: Maximum file size in bytes.
            format_filter: Specific audio format to filter by (e.g., 'mp3', 'wav').

        Returns:
        -------
            list[Path]: List of file paths matching the criteria.

        """
        file_paths = []

        for file_path in self.audio_files_path.iterdir():
            if not file_path.is_file():
                continue

            file_ext = file_path.suffix.lower()
            if file_ext in TRANSCRIBE_AUDIO_FORMATS or file_ext in CHAT_WITH_AUDIO_FORMATS:
                # Apply regex pattern filtering if provided
                if pattern and not re.search(pattern, str(file_path)):
                    continue

                # Apply format filtering if provided
                if format_filter and file_ext[1:].lower() != format_filter.lower():
                    continue

                # Apply size filtering if provided
                if min_size_bytes is not None or max_size_bytes is not None:
                    file_size = file_path.stat().st_size
                    if min_size_bytes is not None and file_size < min_size_bytes:
                        continue
                    if max_size_bytes is not None and file_size > max_size_bytes:
                        continue

                file_paths.append(file_path)

        return file_paths

    async def read_audio_file(self, file_path: Path) -> bytes:
        """Read an audio file asynchronously.

        Args:
        ----
            file_path: Path to the audio file.

        Returns:
        -------
            bytes: The file content as bytes.

        Raises:
        ------
            AudioFileNotFoundError: If the file doesn't exist.
            AudioFileError: If there's an error reading the file.

        """
        if not file_path.exists() or not file_path.is_file():
            raise AudioFileNotFoundError(f"File not found: {file_path}")

        try:
            async with aiofiles.open(file_path, "rb") as f:
                return await f.read()
        except Exception as e:
            raise AudioFileError(f"Failed to read audio file '{file_path}': {e}") from e

    async def write_audio_file(self, file_path: Path, content: bytes) -> None:
        """Write audio content to a file asynchronously.

        Args:
        ----
            file_path: Path where the file should be written.
            content: Audio content as bytes.

        Raises:
        ------
            AudioFileError: If there's an error writing the file.

        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
        except Exception as e:
            raise AudioFileError(f"Failed to write audio file '{file_path}': {e}") from e

    async def get_file_size(self, file_path: Path) -> int:
        """Get the size of a file in bytes.

        Args:
        ----
            file_path: Path to the file.

        Returns:
        -------
            int: File size in bytes.

        Raises:
        ------
            AudioFileNotFoundError: If the file doesn't exist.

        """
        if not file_path.exists():
            raise AudioFileNotFoundError(f"File not found: {file_path}")

        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()
        return len(content)
