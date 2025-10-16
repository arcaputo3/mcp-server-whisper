"""Whisper MCP server - Minimal server setup with tool registration."""

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("whisper", dependencies=["openai", "pydub", "aiofiles"])


def main() -> None:
    """Run main entrypoint."""
    # Import and register tools at runtime (not at module import time)
    from .tools import register_all_tools

    register_all_tools(mcp)
    mcp.run()


if __name__ == "__main__":
    main()
