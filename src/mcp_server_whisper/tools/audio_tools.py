"""MCP tools for audio processing operations."""

from pathlib import Path

from ..config import check_and_get_audio_path
from ..infrastructure import FileSystemRepository
from ..models import CompressAudioInputParams, ConvertAudioInputParams
from ..services import AudioService


def create_audio_tools(mcp):
    """Register audio processing tools with the MCP server.

    Args:
    ----
        mcp: FastMCP server instance.

    """
    # Initialize services
    audio_path = check_and_get_audio_path()
    file_repo = FileSystemRepository(audio_path)
    audio_service = AudioService(file_repo)

    @mcp.tool(description="A tool used to convert audio files to mp3 or wav which are gpt-4o compatible.")
    async def convert_audio(inputs: list[ConvertAudioInputParams]) -> list[dict[str, Path]]:
        """Convert multiple audio files to supported formats (mp3 or wav) in parallel."""

        async def process_single(input_data: ConvertAudioInputParams) -> dict[str, Path]:
            try:
                output_file = await audio_service.convert_audio(
                    input_file=input_data.input_file_path,
                    output_path=input_data.output_file_path,
                    target_format=input_data.target_format,
                )
                return {"output_path": output_file}
            except Exception as e:
                raise RuntimeError(f"Audio conversion failed for {input_data.input_file_path}: {str(e)}")

        import asyncio

        return await asyncio.gather(*[process_single(input_data) for input_data in inputs])

    @mcp.tool(
        description="A tool used to compress audio files which are >25mb. "
        "ONLY USE THIS IF THE USER REQUESTS COMPRESSION OR IF OTHER TOOLS FAIL DUE TO FILES BEING TOO LARGE."
    )
    async def compress_audio(inputs: list[CompressAudioInputParams]) -> list[dict[str, Path]]:
        """Compress multiple audio files in parallel if they're larger than max_mb."""

        async def process_single(input_data: CompressAudioInputParams) -> dict[str, Path]:
            try:
                output_file = await audio_service.compress_audio(
                    input_file=input_data.input_file_path,
                    output_path=input_data.output_file_path,
                    max_mb=input_data.max_mb,
                )
                return {"output_path": output_file}
            except Exception as e:
                raise RuntimeError(f"Audio compression failed for {input_data.input_file_path}: {str(e)}")

        import asyncio

        return await asyncio.gather(*[process_single(input_data) for input_data in inputs])
