import argparse
import os
import sys
import copy
import base64
from openai import AsyncOpenAI


from .conversation import load_conversation, save_conversation
from .convert import convert_to_supported_format, maybe_compress_file


async def do_gpt4o_audio(client: AsyncOpenAI, args: argparse.Namespace) -> None:
    """
    Send audio to GPT-4o via ChatCompletion API. Supports:
        - Checking file size limits (<=25MB)
        - Automatic compression if size is above 25MB
        - Base64 encoding
        - Adding optional text prompt
        - Conversation context loading & saving
        - Requesting text or audio output

    Args:
        client: AsyncOpenAI
        args (argparse.Namespace): Parsed CLI arguments.
            - args.input: Path to input audio file
            - args.output: Path to output text file (optional)
            - args.gpt4o_model: Name of GPT-4o model
            - args.modalities: Comma-separated string of requested modalities (e.g., "text,audio")
            - args.text_prompt: Optional text prompt to include with the audio
            - args.audio_output: Boolean; if True, request audio in the response
            - args.conversation_file: JSON file to load/save conversation context
    """
    if not os.path.isfile(args.input):
        print(f"Error: Could not find audio file '{args.input}'")
        sys.exit(1)

    # Determine audio format from file extension
    _, ext = os.path.splitext(args.input)
    ext = ext.lower().replace(".", "")

    # If the extension is not mp3 or wav, convert to mp3
    if ext not in ("wav", "mp3"):
        print(f"Converting {args.input} from {ext} to a supported format (mp3)...")
        try:
            new_path = await convert_to_supported_format(args.input, target_format="mp3")
            args.input = new_path
            ext = "mp3"
        except Exception as e:
            print(f"Error converting file: {e}")
            sys.exit(1)

    # Check size and compress if needed
    try:
        args.input = await maybe_compress_file(args.input, max_mb=25)
    except Exception as e:
        print(f"Error during compression: {str(e)}")
        sys.exit(1)

    # After possible compression, final size check
    file_size = os.path.getsize(args.input)
    if file_size > 25 * 1024 * 1024:
        print(
            f"Error: Even after compression, audio file '{args.input}' exceeds 25MB limit."
        )
        sys.exit(1)

    # Load and base64-encode the (now guaranteed <=25MB) audio
    try:
        with open(args.input, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except FileNotFoundError:
        print(
            f"Error: Could not find audio file '{args.input}' after conversion/compression."
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error reading audio file: {str(e)}")
        sys.exit(1)

    # Build conversation context
    conversation = load_conversation(args.conversation_file)

    # Construct the user message content
    user_content_parts = []
    if args.text_prompt.strip():
        user_content_parts.append({"type": "text", "text": args.text_prompt.strip()})

    user_content_parts.append(
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": ext}}
    )

    # Append user message
    conversation.append({"role": "user", "content": user_content_parts})

    # Parse modalities from user input
    modalities_list = [m.strip() for m in args.modalities.split(",") if m.strip()]

    # If user wants an audio response, define audio config
    audio_config = None
    if args.audio_output:
        audio_config = {
            "voice": "alloy",
            "format": "wav",
        }
        if "audio" not in modalities_list:
            modalities_list.append("audio")

    # Create a copy of conversation for the API call
    messages_for_api = copy.deepcopy(conversation)

    # Now call GPT-4o
    try:
        completion = await client.chat.completions.create(
            model=args.gpt4o_model,
            modalities=modalities_list,
            messages=messages_for_api,
            audio=audio_config,
        )
    except Exception as e:
        print(f"Error calling GPT-4o: {str(e)}")
        sys.exit(1)

    # Extract the assistant message
    assistant_message = completion.choices[0].message

    # Append to conversation
    conversation.append({"role": "assistant", "content": assistant_message.content})

    # If the response is a list of content parts (text/audio), collect text for printing
    text_output = ""
    if isinstance(assistant_message.content, list):
        # Process each segment
        for segment in assistant_message.content:
            if segment.get("type") == "text":
                text_output += segment.get("text", "") + "\n"

        # Check for audio in the response
        for segment in assistant_message.content:
            if segment.get("type") == "audio":
                audio_data_b64 = segment["audio"]["data"]
                audio_format = segment["audio"]["format"]
                print(f"Assistant returned audio in format: {audio_format}")
                with open(f"assistant_response.{audio_format}", "wb") as f:
                    f.write(base64.b64decode(audio_data_b64))
    else:
        # Possibly the assistant returned a single text string
        text_output = assistant_message.content

    # Print or save text response
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text_output)
        except Exception as e:
            print(f"Error writing output to file: {str(e)}")
    else:
        print(text_output.strip())

    # Save updated conversation
    save_conversation(conversation, args.conversation_file)
