import pytest
from whisper_local.cli import parse_args

def test_parse_args_defaults():
    """
    Test that the default arguments are set properly
    when no optional flags are passed.
    """
    args = parse_args(["input_file.wav"])
    assert args.input == "input_file.wav"
    assert args.output is None
    assert args.model == "whisper-1"
    assert args.use_gpt4o is False
    assert args.gpt4o_model == "gpt-4o-audio-preview-2024-10-01"
    assert args.modalities == "text"
    assert args.text_prompt == ""
    assert args.audio_output is False
    assert args.conversation_file is None

def test_parse_args_gpt4o():
    """
    Test that GPT-4o flags are parsed correctly.
    """
    args = parse_args([
        "sample.mp3",
        "--use-gpt4o",
        "--gpt4o-model", "custom-gpt4o-model",
        "--modalities", "text,audio",
        "--text-prompt", "Hello, world!",
        "--audio-output",
        "--conversation-file", "conv.json"
    ])
    assert args.input == "sample.mp3"
    assert args.use_gpt4o is True
    assert args.gpt4o_model == "custom-gpt4o-model"
    assert args.modalities == "text,audio"
    assert args.text_prompt == "Hello, world!"
    assert args.audio_output is True
    assert args.conversation_file == "conv.json"