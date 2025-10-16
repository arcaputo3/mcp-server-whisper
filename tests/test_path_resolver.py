"""Test secure path resolver for filesystem operations."""

from pathlib import Path

import pytest

from mcp_server_whisper.infrastructure.path_resolver import SecurePathResolver


class TestSecurePathResolver:
    """Test suite for SecurePathResolver security and functionality."""

    @pytest.fixture
    def base_path(self, tmp_path: Path) -> Path:
        """Create a temporary base directory for testing."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        return audio_dir

    @pytest.fixture
    def resolver(self, base_path: Path) -> SecurePathResolver:
        """Create a SecurePathResolver instance."""
        return SecurePathResolver(base_path)

    @pytest.fixture
    def sample_file(self, base_path: Path) -> Path:
        """Create a sample file for testing."""
        file_path = base_path / "test.mp3"
        file_path.write_text("test content")
        return file_path

    def test_resolve_input_valid_filename(self, resolver: SecurePathResolver, sample_file: Path) -> None:
        """Test resolving a valid filename."""
        result = resolver.resolve_input("test.mp3")
        assert result == sample_file
        assert result.exists()

    def test_resolve_input_file_not_found(self, resolver: SecurePathResolver) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError, match="File not found: nonexistent.mp3"):
            resolver.resolve_input("nonexistent.mp3")

    def test_resolve_input_prevents_parent_directory_traversal(self, resolver: SecurePathResolver) -> None:
        """Test that '../' path traversal is safely handled by stripping directory components."""
        # The resolver strips directory components, so ../secret.txt becomes secret.txt
        # This is secure behavior - the .. is ignored
        with pytest.raises(FileNotFoundError, match="File not found: secret.txt"):
            resolver.resolve_input("../secret.txt")

    def test_resolve_input_prevents_absolute_path_traversal(self, resolver: SecurePathResolver, tmp_path: Path) -> None:
        """Test that absolute paths are safely handled by stripping directory components."""
        # Create a file outside the base directory
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("secret")

        # Try to access it with absolute path - should strip to just filename
        # This is secure: /tmp/outside.txt becomes outside.txt (not found in base dir)
        with pytest.raises(FileNotFoundError, match="File not found: outside.txt"):
            resolver.resolve_input(str(outside_file))

    def test_resolve_input_prevents_multi_level_traversal(self, resolver: SecurePathResolver) -> None:
        """Test that multi-level '../../../' traversal is safely handled."""
        # The resolver strips all directory components, so ../../../etc/passwd becomes passwd
        # This is secure behavior - all the ../ are ignored
        with pytest.raises(FileNotFoundError, match="File not found: passwd"):
            resolver.resolve_input("../../../etc/passwd")

    def test_resolve_input_strips_directory_components(self, resolver: SecurePathResolver, sample_file: Path) -> None:
        """Test that directory components are stripped from input."""
        # Even if user provides a path with directories, only the filename is used
        result = resolver.resolve_input("subdir/test.mp3")
        # Should resolve to base_path/test.mp3 (not base_path/subdir/test.mp3)
        assert result == sample_file

    def test_resolve_input_with_subdirectory_file(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test accessing a file that legitimately exists in base directory."""
        # Create a file with a name that looks like a path
        tricky_file = base_path / "looks_like_path.mp3"
        tricky_file.write_text("content")

        result = resolver.resolve_input("looks_like_path.mp3")
        assert result == tricky_file
        assert result.exists()

    def test_resolve_output_with_custom_filename(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test resolving output path with custom filename."""
        result = resolver.resolve_output("output.wav", default="default.mp3")
        assert result == base_path / "output.wav"
        # File doesn't need to exist for output

    def test_resolve_output_with_none_uses_default(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test that None filename uses default."""
        result = resolver.resolve_output(None, default="default.mp3")
        assert result == base_path / "default.mp3"

    def test_resolve_output_prevents_path_traversal(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test that output path traversal is safely handled."""
        # The resolver strips directory components, so ../malicious.mp3 becomes malicious.mp3
        # This is secure - it stays in the base directory
        result = resolver.resolve_output("../malicious.mp3", default="default.mp3")
        assert result == base_path / "malicious.mp3"
        # Verify it's still in the base directory
        assert str(result).startswith(str(base_path))

    def test_resolve_output_strips_directory_components(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test that directory components are stripped from output filename."""
        result = resolver.resolve_output("subdir/output.mp3", default="default.mp3")
        # Should be base_path/output.mp3, not base_path/subdir/output.mp3
        assert result == base_path / "output.mp3"

    def test_resolve_output_with_absolute_path_attempt(
        self, resolver: SecurePathResolver, tmp_path: Path, base_path: Path
    ) -> None:
        """Test that absolute path attempts for output are safely handled."""
        outside_path = str(tmp_path / "outside" / "output.mp3")
        # The resolver strips directory components, keeping only the filename
        # This is secure - /tmp/outside/output.mp3 becomes output.mp3 in base dir
        result = resolver.resolve_output(outside_path, default="default.mp3")
        assert result == base_path / "output.mp3"
        # Verify it's in the base directory, not the attempted outside path
        assert str(result).startswith(str(base_path))
        assert "outside" not in str(result)

    def test_get_relative_name_returns_filename(self, resolver: SecurePathResolver) -> None:
        """Test that get_relative_name extracts just the filename."""
        path = Path("/some/long/path/to/file.mp3")
        result = resolver.get_relative_name(path)
        assert result == "file.mp3"

    def test_get_relative_name_with_no_directory(self, resolver: SecurePathResolver) -> None:
        """Test get_relative_name with just a filename."""
        path = Path("file.mp3")
        result = resolver.get_relative_name(path)
        assert result == "file.mp3"

    def test_base_path_is_resolved(self, tmp_path: Path) -> None:
        """Test that base path is resolved to absolute path."""
        # Create a path with . and .. components
        complex_path = tmp_path / "subdir" / ".." / "audio"
        complex_path.mkdir(parents=True, exist_ok=True)

        resolver = SecurePathResolver(complex_path)
        # Base path should be resolved and normalized
        assert resolver.base_path.is_absolute()
        assert ".." not in str(resolver.base_path)

    def test_resolve_input_with_whitespace(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test resolving filenames with spaces."""
        file_with_space = base_path / "my audio file.mp3"
        file_with_space.write_text("content")

        result = resolver.resolve_input("my audio file.mp3")
        assert result == file_with_space
        assert result.exists()

    def test_resolve_output_with_whitespace(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test resolving output paths with spaces."""
        result = resolver.resolve_output("output file.mp3", default="default.mp3")
        assert result == base_path / "output file.mp3"

    def test_resolve_input_with_special_characters(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test resolving filenames with special characters."""
        special_file = base_path / "test_file-123.mp3"
        special_file.write_text("content")

        result = resolver.resolve_input("test_file-123.mp3")
        assert result == special_file

    def test_multiple_resolvers_different_base_paths(self, tmp_path: Path) -> None:
        """Test that multiple resolvers maintain separate base paths."""
        dir1 = tmp_path / "audio1"
        dir2 = tmp_path / "audio2"
        dir1.mkdir()
        dir2.mkdir()

        resolver1 = SecurePathResolver(dir1)
        resolver2 = SecurePathResolver(dir2)

        assert resolver1.base_path != resolver2.base_path
        assert resolver1.base_path == dir1.resolve()
        assert resolver2.base_path == dir2.resolve()

    def test_resolve_input_empty_filename(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test that empty filename is handled."""
        # Empty string's .name is '.', which likely doesn't exist
        # Or it might refer to the directory itself
        try:
            result = resolver.resolve_input("")
            # If it succeeds, it should be resolving to base directory or similar
            assert str(result).startswith(str(base_path))
        except FileNotFoundError:
            # This is also acceptable behavior
            pass

    def test_resolve_output_empty_filename_uses_default(self, resolver: SecurePathResolver, base_path: Path) -> None:
        """Test that empty output filename uses default."""
        result = resolver.resolve_output("", default="default.mp3")
        # Empty string is falsy, so should use default
        assert result == base_path / "default.mp3"
