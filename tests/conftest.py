"""Pytest configuration for mcp_server_whisper tests."""

import os
from pathlib import Path
from typing import List, Tuple

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def mock_audio_path(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    """Mock audio path environment variable and create test directory."""
    # Create a temporary directory for audio files
    audio_path = tmp_path / "audio"
    audio_path.mkdir()

    # Mock the environment variable
    monkeypatch.setenv("AUDIO_FILES_PATH", str(audio_path))

    return audio_path


@pytest.fixture
def sample_audio_files(mock_audio_path: Path) -> List[Tuple[Path, int, int]]:
    """Create sample audio files for testing."""
    # Create a few test files with different formats
    files = [
        ("test1.mp3", 1000, 100),  # name, size, mtime
        ("test2.wav", 2000, 200),
        ("test3.mp4", 3000, 300),
        ("test_large.mp3", 26 * 1024 * 1024, 400),  # > 25MB
    ]

    created_files: List[Tuple[Path, int, int]] = []
    for name, size, mtime in files:
        file_path = mock_audio_path / name
        # Create file with specific size
        with open(file_path, "wb") as f:
            f.write(b"0" * size)

        # Set modification time
        os.utime(file_path, (mtime, mtime))
        created_files.append((file_path, size, mtime))

    return created_files
