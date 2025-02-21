import json
import os
from typing import Any


def load_conversation(conversation_file: str) -> Any:
    """
    Load conversation context from a JSON file if it exists, else return an empty list.

    Args:
        conversation_file (str): Path to the conversation JSON file.

    Returns:
        list: The conversation messages as a list of dicts.
    """
    if conversation_file and os.path.isfile(conversation_file):
        try:
            with open(conversation_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # If file not valid JSON or can't be read, return empty conversation
            return []
    return []


def save_conversation(conversation: list[dict[str, Any]], conversation_file: str) -> None:
    """
    Save the conversation context to a JSON file.

    Args:
        conversation (list): The conversation messages as a list of dicts.
        conversation_file (str): Path to the conversation JSON file.
    """
    if conversation_file:
        try:
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save conversation context: {str(e)}")
