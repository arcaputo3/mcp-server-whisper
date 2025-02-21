import sys
import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .cli import parse_args
from .transcribe import do_whisper_transcription
from .gpt4o import do_gpt4o_audio


def print_error_and_exit(msg: str):
    print(f"Error: {msg}")
    sys.exit(1)


async def run_mcp_transcription_tool(audio_path: str) -> str:
    """
    Launch the MCP server as a subprocess and call the transcribe_audio tool.
    Returns the transcription text.
    """
    # We assume 'mcp_server.py' is in the same directory as this file
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    if not os.path.isfile(server_script):
        raise FileNotFoundError(
            "Could not locate mcp_server.py for MCP server startup."
        )

    server_params = StdioServerParameters(command="python", args=[server_script])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Now call the 'transcribe_audio' tool
            result = await session.call_tool(
                "transcribe_audio", {"file_path": audio_path}
            )
            return result


async def main_async():
    load_dotenv()
    args = parse_args()
    client = AsyncOpenAI()

    if not os.path.isfile(args.input):
        print_error_and_exit(f"Could not find audio file '{args.input}'")

    # If user wants direct GPT-4o usage (audio in GPT-4), we skip MCP
    if args.use_gpt4o:
        # Existing logic for GPT-4o direct audio
        await do_gpt4o_audio(client, args)
        return

    # Otherwise, run standard Whisper transcription OR use the MCP-based approach
    # We'll demonstrate using the new MCP approach for audio
    if args.input.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        # Use the MCP-based tool to transcribe the audio
        try:
            transcript = await run_mcp_transcription_tool(args.input)
        except Exception as e:
            print(f"[Error during MCP transcription: {e}]")
            sys.exit(1)

        # If user provided an --output file, write the transcript there
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(transcript)
            except Exception as f_err:
                print(f"Error writing transcript to file: {str(f_err)}")
            else:
                print("Transcript saved to", args.output)
        else:
            print("Transcript:\n", transcript)
    else:
        # Non-audio input => fallback to do_whisper_transcription or handle as error
        # We'll do a standard whisper call for demonstration:
        await do_whisper_transcription(args)

    # Optionally demonstrate calling GPT-4 after we have a transcription
    # We can keep a simple conversation approach if desired
    # For example, to show how you'd create a ChatCompletion with the transcript:
    conversation = []
    if args.output and os.path.isfile(args.output):
        # read transcript from file
        with open(args.output, "r", encoding="utf-8") as f:
            user_text = f.read()
    else:
        # use transcript in memory
        user_text = transcript if "transcript" in locals() else ""

    print(user_text)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
