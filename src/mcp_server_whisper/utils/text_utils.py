"""Text processing utilities."""


def split_text_for_tts(text: str, max_length: int = 4000) -> list[str]:
    """Split text into chunks that don't exceed the TTS API limit.

    The function splits text at sentence boundaries (periods, question marks, exclamation points)
    to create natural-sounding chunks. If a sentence is too long, it falls back to
    splitting at commas, then spaces.

    Args:
    ----
        text: The text to split.
        max_length: Maximum character length for each chunk (default 4000 to provide buffer).

    Returns:
    -------
        List of text chunks, each below the maximum length.

    """
    # If text is already under the limit, return it as a single chunk
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining_text = text

    # Define boundary markers in order of preference
    sentence_boundaries = [". ", "? ", "! ", ".\n", "?\n", "!\n"]
    secondary_boundaries = [", ", ";\n", ";\n", ":\n", "\n", " "]

    while len(remaining_text) > max_length:
        # Try to find the best split point starting from max_length and working backward
        split_index = -1

        # First try sentence boundaries (most preferred)
        for boundary in sentence_boundaries:
            last_boundary = remaining_text[:max_length].rfind(boundary)
            if last_boundary != -1:
                split_index = last_boundary + len(boundary)
                break

        # If no sentence boundary found, try secondary boundaries
        if split_index == -1:
            for boundary in secondary_boundaries:
                last_boundary = remaining_text[:max_length].rfind(boundary)
                if last_boundary != -1:
                    split_index = last_boundary + len(boundary)
                    break

        # If still no boundary found, just cut at max_length (least preferred)
        if split_index == -1 or split_index == 0:
            split_index = max_length

        # Add the chunk and update remaining text
        chunks.append(remaining_text[:split_index])
        remaining_text = remaining_text[split_index:]

    # Add any remaining text as the final chunk
    if remaining_text:
        chunks.append(remaining_text)

    return chunks
