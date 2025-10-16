"""Test text processing utilities."""

from mcp_server_whisper.utils.text_utils import split_text_for_tts


class TestSplitTextForTTS:
    """Test suite for text splitting functionality."""

    def test_short_text_returns_single_chunk(self) -> None:
        """Test that text under max_length returns as single chunk."""
        text = "This is a short text."
        result = split_text_for_tts(text, max_length=100)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_string_returns_single_chunk(self) -> None:
        """Test that empty string returns as single chunk."""
        result = split_text_for_tts("", max_length=100)
        assert len(result) == 1
        assert result[0] == ""

    def test_split_at_sentence_boundary_period(self) -> None:
        """Test splitting at sentence boundaries with periods."""
        text = "First sentence. Second sentence. Third sentence."
        result = split_text_for_tts(text, max_length=30)
        # Should split at period boundaries
        assert len(result) >= 2
        # Verify all text is preserved
        assert "".join(result) == text
        # Each chunk should be under max_length
        for chunk in result:
            assert len(chunk) <= 30

    def test_split_at_sentence_boundary_question_mark(self) -> None:
        """Test splitting at question mark boundaries."""
        text = "First question? Second question? Third question?"
        result = split_text_for_tts(text, max_length=35)
        # Should split at question mark boundaries
        assert len(result) >= 2
        # Verify all text is preserved
        assert "".join(result) == text
        # Each chunk should be under max_length
        for chunk in result:
            assert len(chunk) <= 35

    def test_split_at_sentence_boundary_exclamation(self) -> None:
        """Test splitting at exclamation mark boundaries."""
        text = "First exclamation! Second exclamation! Third!"
        result = split_text_for_tts(text, max_length=40)
        # Should split at exclamation mark boundaries
        assert len(result) >= 2
        # Verify all text is preserved
        assert "".join(result) == text
        # Each chunk should be under max_length
        for chunk in result:
            assert len(chunk) <= 40

    def test_split_at_comma_when_no_sentence_boundary(self) -> None:
        """Test fallback to comma splitting when no sentence boundary exists."""
        text = "First part, second part, third part, fourth part"
        result = split_text_for_tts(text, max_length=30)
        assert len(result) >= 2
        # Should split at commas
        for chunk in result:
            assert len(chunk) <= 30

    def test_split_at_space_when_no_comma(self) -> None:
        """Test fallback to space splitting when no comma exists."""
        text = "word " * 100  # Long text with only spaces
        result = split_text_for_tts(text, max_length=50)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 50

    def test_hard_cut_when_no_boundaries(self) -> None:
        """Test hard cut at max_length when no boundaries exist."""
        text = "a" * 100  # Very long word with no boundaries
        result = split_text_for_tts(text, max_length=30)
        assert len(result) == 4  # 100 / 30 = 3.33, so 4 chunks
        assert len(result[0]) == 30
        assert len(result[1]) == 30
        assert len(result[2]) == 30
        assert len(result[3]) == 10

    def test_preserves_newlines_in_sentence_boundaries(self) -> None:
        """Test that newline sentence boundaries are respected."""
        text = "First sentence.\nSecond sentence.\nThird sentence."
        result = split_text_for_tts(text, max_length=35)
        assert len(result) >= 2
        # Should split at period+newline boundaries

    def test_multiple_chunks_complex_text(self) -> None:
        """Test splitting complex text with multiple boundary types."""
        text = (
            "This is the first sentence. This is the second sentence! "
            "And here's a question? Finally, we have some text with commas, "
            "and more commas, and even more text that needs to be split properly."
        )
        result = split_text_for_tts(text, max_length=60)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 60
        # Verify all text is preserved
        assert "".join(result) == text

    def test_exact_max_length_no_split(self) -> None:
        """Test that text exactly at max_length doesn't get split."""
        text = "a" * 50
        result = split_text_for_tts(text, max_length=50)
        assert len(result) == 1
        assert result[0] == text

    def test_max_length_plus_one_splits(self) -> None:
        """Test that text one character over max_length gets split."""
        text = "a" * 51
        result = split_text_for_tts(text, max_length=50)
        assert len(result) == 2

    def test_default_max_length_is_4000(self) -> None:
        """Test that default max_length parameter is 4000."""
        text = "a" * 4000
        result = split_text_for_tts(text)  # No max_length specified
        assert len(result) == 1

        text = "a" * 4001
        result = split_text_for_tts(text)
        assert len(result) == 2

    def test_whitespace_preservation(self) -> None:
        """Test that whitespace is preserved in chunks."""
        text = "First.  Second.  Third."  # Double spaces
        result = split_text_for_tts(text, max_length=15)
        # Verify spaces are maintained
        reassembled = "".join(result)
        assert reassembled == text

    def test_very_long_text_many_chunks(self) -> None:
        """Test splitting very long text into many chunks."""
        # Create a long text with sentence boundaries
        sentences = ["This is sentence number {}. ".format(i) for i in range(100)]
        text = "".join(sentences)
        result = split_text_for_tts(text, max_length=100)
        assert len(result) > 10
        assert "".join(result) == text
