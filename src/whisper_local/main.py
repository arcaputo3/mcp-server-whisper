import os
import asyncio
from dotenv import load_dotenv
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from .cli import parse_args


async def main_async():
    load_dotenv()
    args = parse_args()

    # Setup MCP server parameters
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "server"],
        env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"), **os.environ},
    )

    # Connect to MCP server and call transcribe_audio tool
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_params = {
                "file_path": args.input,
                "use_gpt4o": args.use_gpt4o,
                "model": args.model if not args.use_gpt4o else args.gpt4o_model,
                "text_prompt": args.text_prompt if args.use_gpt4o else None,
                "modalities": args.modalities.split(",") if args.use_gpt4o else None,
                "audio_output": args.audio_output if args.use_gpt4o else False,
                "output_path": args.output,
            }

            result = await session.call_tool("transcribe_audio", tool_params)

            # Handle output
            text_output = result.content[0].text
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(text_output)
            else:
                print(text_output)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
