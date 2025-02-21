import json
import pytest
import os
from whisper_local.conversation import load_conversation, save_conversation


def test_load_conversation_nonexistent(tmp_path):
    """
    If the conversation file does not exist, load_conversation should return an empty list.
    """
    non_existent_file = tmp_path / "no_such_file.json"
    conversation = load_conversation(str(non_existent_file))
    assert conversation == []


def test_load_conversation_valid(tmp_path):
    """
    If the conversation file exists and contains valid JSON, load_conversation
    should return that JSON data.
    """
    test_file = tmp_path / "conversation.json"
    data = [{"role": "user", "content": "Hello"}]
    test_file.write_text(json.dumps(data), encoding="utf-8")

    conversation = load_conversation(str(test_file))
    assert conversation == data


def test_load_conversation_invalid_json(tmp_path):
    """
    If the conversation file exists but is invalid JSON, load_conversation should return an empty list.
    """
    test_file = tmp_path / "conversation.json"
    test_file.write_text("{invalid json", encoding="utf-8")

    conversation = load_conversation(str(test_file))
    assert conversation == []


def test_save_conversation(tmp_path):
    """
    Test that save_conversation writes JSON to the file.
    """
    test_file = tmp_path / "conversation.json"
    conversation_data = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    save_conversation(conversation_data, str(test_file))
    assert test_file.exists()

    with open(test_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == conversation_data


def test_save_conversation_failure(tmp_path, monkeypatch):
    """
    If the file cannot be opened for writing, save_conversation should not raise,
    but print a warning. We'll mock open to simulate an exception.
    """
    test_file = tmp_path / "conversation.json"
    conversation_data = [{"role": "user", "content": "Hi"}]

    def mock_open(*args, **kwargs):
        raise IOError("Cannot open file for writing")

    monkeypatch.setattr("builtins.open", mock_open)
    # We'll just ensure it doesn't crash
    save_conversation(conversation_data, str(test_file))
    # There's no easy assertion except that no exception is raised.
