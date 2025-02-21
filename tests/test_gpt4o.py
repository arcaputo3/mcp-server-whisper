import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock
from whisper_local.gpt4o import do_gpt4o_audio
from whisper_local.conversation import load_conversation

@pytest.mark.usefixtures("mock_audio_segment_all")
@patch("os.path.getsize", return_value=30*1024*1024)
def test_do_gpt4o_audio_above_limit(mock_getsize, tmp_path, capsys):
    """
    If the final compressed file is still above 25MB, do_gpt4o_audio should exit.
    """
    input_file = tmp_path / "large.wav"
    input_file.write_text("fake data")

    class Args:
        input = str(input_file)
        output = None
        gpt4o_model = "gpt-4o-audio-preview-2024-10-01"
        modalities = "text"
        text_prompt = ""
        audio_output = False
        conversation_file = None

    with pytest.raises(SystemExit) as e:
        do_gpt4o_audio(Args())
    assert e.type == SystemExit
    assert e.value.code == 1

    captured = capsys.readouterr()
    # Because we forced the file to remain 30MB, we should see the final error message
    assert "exceeds 25MB limit" in captured.out


@pytest.mark.usefixtures("mock_audio_segment_all")
@patch("whisper_local.gpt4o.openai.chat.completions.create")
def test_do_gpt4o_audio_text_only(mock_create, tmp_path, monkeypatch):
    """
    Test sending audio to GPT-4o with text output only.
    """
    input_file = tmp_path / "sample.wav"
    input_file.write_text("fake audio")

    class Args:
        input = str(input_file)
        output = None
        gpt4o_model = "gpt-4o-audio-preview-2024-10-01"
        modalities = "text"
        text_prompt = "Hello?"
        audio_output = False
        conversation_file = None

    # Mock completion
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Assistant response"))]
    )

    do_gpt4o_audio(Args())
    # Check that openai.chat.completions.create was called
    mock_create.assert_called_once()
    kwargs = mock_create.call_args[1]
    assert kwargs["model"] == "gpt-4o-audio-preview-2024-10-01"
    assert kwargs["modalities"] == ["text"]
    assert kwargs["messages"][-1]["role"] == "user"
    assert "input_audio" in kwargs["messages"][-1]["content"][-1]


@pytest.mark.usefixtures("mock_audio_segment_all")
@patch("whisper_local.gpt4o.openai.chat.completions.create")
def test_do_gpt4o_audio_audio_output(mock_create, tmp_path):
    """
    If audio_output is True, we add audio config and expect the returned data to contain audio.
    """
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("fake audio")

    class Args:
        input = str(input_file)
        output = None
        gpt4o_model = "gpt-4o-audio-preview-2024-10-01"
        modalities = "text,audio"
        text_prompt = ""
        audio_output = True
        conversation_file = None

    mock_create.return_value = MagicMock(choices=[
        MagicMock(message=MagicMock(content=[
            {"type": "text", "text": "Here is some text"},
            {"type": "audio", "audio": {"data": "U29tZUJhc2U2NA==", "format": "wav"}}
        ]))
    ])
    do_gpt4o_audio(Args())

    # After the function, we expect "assistant_response.wav" to exist
    out_file = "assistant_response.wav"
    assert os.path.isfile(out_file)
    with open(out_file, "rb") as f:
        content = f.read()
    # "U29tZUJhc2U2NA==" is "SomeBase64" => decode => "???" as bytes.
    assert len(content) > 0
    # Cleanup
    os.remove(out_file)


@pytest.mark.usefixtures("mock_audio_segment_all")
@patch("whisper_local.gpt4o.openai.chat.completions.create")
def test_do_gpt4o_audio_save_output(mock_create, tmp_path):
    """
    If --output is specified, do_gpt4o_audio should save the text portion of the response to that file.
    """
    input_file = tmp_path / "sample.mp3"
    input_file.write_text("fake audio")

    output_file = tmp_path / "resp.txt"

    class Args:
        input = str(input_file)
        output = str(output_file)
        gpt4o_model = "gpt-4o-audio-preview-2024-10-01"
        modalities = "text"
        text_prompt = "Hello?"
        audio_output = False
        conversation_file = None

    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Assistant text reply."))]
    )
    do_gpt4o_audio(Args())

    assert output_file.exists()
    assert output_file.read_text() == "Assistant text reply."


@pytest.mark.usefixtures("mock_audio_segment_all")
@patch("whisper_local.gpt4o.openai.chat.completions.create")
def test_do_gpt4o_audio_conversation(mock_create, tmp_path):
    """
    If a conversation file is provided, do_gpt4o_audio should load it,
    append user input, append assistant output, and save.
    """
    conv_file = tmp_path / "conv.json"
    initial_data = [
        {"role": "user", "content": "Hello from before!"},
        {"role": "assistant", "content": "Hello user!"}
    ]
    conv_file.write_text(json.dumps(initial_data), encoding="utf-8")

    input_file = tmp_path / "sample.mp3"
    input_file.write_text("fake audio")

    class Args:
        input = str(input_file)
        output = None
        gpt4o_model = "gpt-4o-audio-preview-2024-10-01"
        modalities = "text"
        text_prompt = "New question"
        audio_output = False
        conversation_file = str(conv_file)

    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Assistant's new answer"))]
    )

    do_gpt4o_audio(Args())

    updated = json.loads(conv_file.read_text())
    assert len(updated) == 4
    assert updated[2]["role"] == "user"
    assert updated[3]["role"] == "assistant"
    assert updated[3]["content"] == "Assistant's new answer"


@pytest.mark.usefixtures("mock_audio_segment_all")
@patch("whisper_local.gpt4o.openai.chat.completions.create", side_effect=Exception("GPT-4o error"))
def test_do_gpt4o_audio_openai_fail(mock_create, tmp_path, capsys):
    """
    If openai call fails, do_gpt4o_audio should exit with error code 1 and print the error.
    """
    input_file = tmp_path / "sample.wav"
    input_file.write_text("fake audio")

    class Args:
        input = str(input_file)
        output = None
        gpt4o_model = "gpt-4o-audio-preview-2024-10-01"
        modalities = "text"
        text_prompt = ""
        audio_output = False
        conversation_file = None

    with pytest.raises(SystemExit) as e:
        do_gpt4o_audio(Args())
    assert e.type == SystemExit
    assert e.value.code == 1

    captured = capsys.readouterr()
    assert "GPT-4o error" in captured.out