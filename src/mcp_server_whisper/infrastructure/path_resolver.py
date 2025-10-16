"""Secure path resolver for filesystem operations."""

from pathlib import Path


class SecurePathResolver:
    """Resolves filenames to paths while enforcing security constraints.

    This class ensures all file operations are confined to a base directory,
    preventing path traversal attacks and protecting user privacy by working
    only with filenames (not full paths).
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize the secure path resolver.

        Args:
        ----
            base_path: Base directory for all file operations.

        """
        self.base_path = base_path.resolve()

    def resolve_input(self, filename: str) -> Path:
        """Resolve an input filename to a full path within the base directory.

        Args:
        ----
            filename: Name of the file to resolve.

        Returns:
        -------
            Path: Resolved path within the base directory.

        Raises:
        ------
            ValueError: If the resolved path would be outside the base directory.
            FileNotFoundError: If the file doesn't exist.

        """
        # Strip any directory components to prevent path traversal
        safe_filename = Path(filename).name
        full_path = (self.base_path / safe_filename).resolve()

        # Ensure resolved path is still within base_path
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(f"Access denied: path traversal attempt detected in '{filename}'")

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {safe_filename}")

        return full_path

    def resolve_output(self, filename: str | None, default: str) -> Path:
        """Resolve an output filename to a full path within the base directory.

        Args:
        ----
            filename: Optional custom filename. If None, uses default.
            default: Default filename to use if filename is None.

        Returns:
        -------
            Path: Resolved path within the base directory.

        Raises:
        ------
            ValueError: If the resolved path would be outside the base directory.

        """
        name = filename or default
        # Strip any directory components to prevent path traversal
        safe_filename = Path(name).name
        full_path = (self.base_path / safe_filename).resolve()

        # Ensure resolved path is still within base_path
        if not str(full_path).startswith(str(self.base_path)):
            raise ValueError(f"Access denied: path traversal attempt detected in '{name}'")

        return full_path

    def get_relative_name(self, path: Path) -> str:
        """Get the filename from a path (strips directory components).

        Args:
        ----
            path: Path to extract filename from.

        Returns:
        -------
            str: Just the filename without directory components.

        """
        return path.name
