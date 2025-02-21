import pytest
import os
import asyncio
from unittest.mock import patch, MagicMock
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# We'll assume mcp_server.py is in src/whisper_local. Adjust the path if needed.
MCP_SERVER_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "src", "whisper_local", "mcp_server.py"
)


@pytest.mark.asyncio
async def test_transcribe_audio_tool(tmp_path):
    """
    Basic integration test for the transcribe_audio MCP tool.
    We mock the openai.audio.transcriptions.create call to avoid external API requests.
    """
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_text("Fake data")  # not real audio, but enough for file existence

    # Patch openai call to return a dummy transcription
    with patch(
        "openai.audio.transcriptions.create", return_value="Dummy transcript"
    ) as mock_transcribe:
        # Patch getsize to be under 25MB
        with patch("os.path.getsize", return_value=10 * 1024 * 1024):
            server_params = StdioServerParameters(
                command="python", args=[MCP_SERVER_SCRIPT]
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # Now call the tool
                    result = await session.call_tool(
                        "transcribe_audio", {"file_path": str(fake_audio)}
                    )
                    assert result == "Dummy transcript"
                    mock_transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_transcribe_audio_file_not_found():
    """
    If the provided file doesn't exist, the tool should raise ValueError.
    """
    # We'll intercept the JSON error. We'll attempt to pass a non-existent file.
    server_params = StdioServerParameters(command="python", args=[MCP_SERVER_SCRIPT])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            with pytest.raises(Exception) as exc_info:
                await session.call_tool(
                    "transcribe_audio", {"file_path": "no_such_file.wav"}
                )
            assert "File not found" in str(exc_info.value)
