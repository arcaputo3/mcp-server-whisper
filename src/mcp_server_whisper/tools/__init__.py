"""MCP tools for Whisper server."""

from ..infrastructure import MCPServer
from .audio_tools import create_audio_tools
from .file_tools import create_file_tools
from .transcription_tools import create_transcription_tools
from .tts_tools import create_tts_tools

__all__ = [
    "create_file_tools",
    "create_audio_tools",
    "create_transcription_tools",
    "create_tts_tools",
]


def register_all_tools(mcp: MCPServer) -> None:
    """Register all MCP tools with the server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    create_file_tools(mcp)
    create_audio_tools(mcp)
    create_transcription_tools(mcp)
    create_tts_tools(mcp)
