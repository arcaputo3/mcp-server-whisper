"""Test OpenAI API client wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from mcp_server_whisper.exceptions import TranscriptionAPIError, TTSAPIError
from mcp_server_whisper.infrastructure.openai_client import OpenAIClientWrapper


class MockTranscription(BaseModel):
    """Mock transcription response."""

    text: str
    duration: float
    language: str


class TestOpenAIClientWrapper:
    """Test suite for OpenAIClientWrapper."""

    @pytest.fixture
    def client(self, monkeypatch) -> OpenAIClientWrapper:
        """Create OpenAIClientWrapper instance with test API key."""
        # Use a fake API key to avoid env var requirement
        return OpenAIClientWrapper(api_key="sk-test-fake-key-for-testing")

    @pytest.mark.anyio
    async def test_transcribe_audio_success_with_basemodel(self, client: OpenAIClientWrapper) -> None:
        """Test successful transcription with BaseModel response."""
        audio_data = b"fake audio bytes"
        mock_response = MockTranscription(text="Hello world", duration=5.0, language="en")

        with patch.object(client.client.audio.transcriptions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.transcribe_audio(
                audio_bytes=audio_data,
                filename="test.mp3",
                model="whisper-1",
                response_format="verbose_json",
            )

            assert result["text"] == "Hello world"
            assert result["duration"] == 5.0
            assert result["language"] == "en"
            mock_create.assert_called_once()

    @pytest.mark.anyio
    async def test_transcribe_audio_success_with_string(self, client: OpenAIClientWrapper) -> None:
        """Test successful transcription with string response."""
        audio_data = b"fake audio"
        mock_response = "Transcribed text here"

        with patch.object(client.client.audio.transcriptions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.transcribe_audio(
                audio_bytes=audio_data,
                filename="test.mp3",
                model="whisper-1",
                response_format="text",
            )

            assert result == {"text": "Transcribed text here"}

    @pytest.mark.anyio
    async def test_transcribe_audio_with_prompt(self, client: OpenAIClientWrapper) -> None:
        """Test transcription with custom prompt."""
        audio_data = b"audio"
        mock_response = "Transcribed"

        with patch.object(client.client.audio.transcriptions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await client.transcribe_audio(
                audio_bytes=audio_data,
                filename="test.mp3",
                prompt="This is a technical discussion",
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["prompt"] == "This is a technical discussion"

    @pytest.mark.anyio
    async def test_transcribe_audio_with_timestamp_granularities(self, client: OpenAIClientWrapper) -> None:
        """Test transcription with timestamp granularities."""
        audio_data = b"audio"
        mock_response = MockTranscription(text="Text", duration=5.0, language="en")

        with patch.object(client.client.audio.transcriptions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await client.transcribe_audio(
                audio_bytes=audio_data,
                filename="test.mp3",
                timestamp_granularities=["word", "segment"],
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["timestamp_granularities"] == ["word", "segment"]

    @pytest.mark.anyio
    async def test_transcribe_audio_api_failure(self, client: OpenAIClientWrapper) -> None:
        """Test error handling when transcription API fails."""
        audio_data = b"audio"

        with patch.object(client.client.audio.transcriptions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(TranscriptionAPIError, match="Transcription failed"):
                await client.transcribe_audio(audio_bytes=audio_data, filename="test.mp3")

    @pytest.mark.anyio
    async def test_chat_with_audio_success(self, client: OpenAIClientWrapper) -> None:
        """Test successful audio chat."""
        audio_data = b"audio bytes"

        mock_message = MagicMock()
        mock_message.content = "The audio says hello"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_completion

            result = await client.chat_with_audio(
                audio_bytes=audio_data,
                audio_format="mp3",
                user_prompt="What does this say?",
            )

            assert result["text"] == "The audio says hello"
            mock_create.assert_called_once()

            # Verify base64 encoding was used
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["modalities"] == ["text"]

    @pytest.mark.anyio
    async def test_chat_with_audio_with_system_prompt(self, client: OpenAIClientWrapper) -> None:
        """Test audio chat with system prompt."""
        audio_data = b"audio"

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_completion

            await client.chat_with_audio(
                audio_bytes=audio_data,
                audio_format="wav",
                system_prompt="You are a helpful assistant",
                user_prompt="Analyze this",
            )

            call_kwargs = mock_create.call_args.kwargs
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant"

    @pytest.mark.anyio
    async def test_chat_with_audio_api_failure(self, client: OpenAIClientWrapper) -> None:
        """Test error handling when chat API fails."""
        audio_data = b"audio"

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(TranscriptionAPIError, match="Audio chat failed"):
                await client.chat_with_audio(audio_bytes=audio_data, audio_format="mp3")

    @pytest.mark.anyio
    async def test_text_to_speech_success(self, client: OpenAIClientWrapper) -> None:
        """Test successful text-to-speech generation."""
        text = "Hello, this is a test."
        expected_audio = b"audio bytes here"

        mock_response = AsyncMock()
        mock_response.aread = AsyncMock(return_value=expected_audio)

        with patch.object(client.client.audio.speech, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.text_to_speech(
                text=text,
                model="gpt-4o-mini-tts",
                voice="alloy",
                speed=1.0,
            )

            assert result == expected_audio
            mock_create.assert_called_once()

    @pytest.mark.anyio
    async def test_text_to_speech_with_instructions(self, client: OpenAIClientWrapper) -> None:
        """Test TTS with custom instructions."""
        text = "Test text"
        mock_response = AsyncMock()
        mock_response.aread = AsyncMock(return_value=b"audio")

        with patch.object(client.client.audio.speech, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await client.text_to_speech(
                text=text,
                voice="shimmer",
                instructions="Speak in a professional tone",
                speed=1.5,
            )

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["instructions"] == "Speak in a professional tone"
            assert call_kwargs["voice"] == "shimmer"
            assert call_kwargs["speed"] == 1.5

    @pytest.mark.anyio
    async def test_text_to_speech_api_failure(self, client: OpenAIClientWrapper) -> None:
        """Test error handling when TTS API fails."""
        text = "Test"

        with patch.object(client.client.audio.speech, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(TTSAPIError, match="Text-to-speech generation failed"):
                await client.text_to_speech(text=text)

    @pytest.mark.anyio
    async def test_generate_tts_chunks_success(self, client: OpenAIClientWrapper) -> None:
        """Test generating TTS for multiple chunks in parallel."""
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        expected_audio = [b"audio1", b"audio2", b"audio3"]

        mock_response = AsyncMock()
        mock_response.aread = AsyncMock(side_effect=expected_audio)

        with patch.object(client.client.audio.speech, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.generate_tts_chunks(
                text_chunks=chunks,
                model="gpt-4o-mini-tts",
                voice="alloy",
            )

            assert len(result) == 3
            # Should have called create for each chunk
            assert mock_create.call_count == 3

    @pytest.mark.anyio
    async def test_generate_tts_chunks_single_chunk(self, client: OpenAIClientWrapper) -> None:
        """Test generating TTS for single chunk."""
        chunks = ["Single chunk"]
        expected_audio = b"audio"

        mock_response = AsyncMock()
        mock_response.aread = AsyncMock(return_value=expected_audio)

        with patch.object(client.client.audio.speech, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            result = await client.generate_tts_chunks(text_chunks=chunks)

            assert len(result) == 1
            assert result[0] == expected_audio

    @pytest.mark.anyio
    async def test_generate_tts_chunks_failure(self, client: OpenAIClientWrapper) -> None:
        """Test error handling when batch TTS fails."""
        chunks = ["Chunk 1", "Chunk 2"]

        with patch.object(client.client.audio.speech, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(TTSAPIError, match="Batch TTS generation failed"):
                await client.generate_tts_chunks(text_chunks=chunks)

    @pytest.mark.anyio
    async def test_transcribe_audio_file_object_has_name(self, client: OpenAIClientWrapper) -> None:
        """Test that BytesIO file object has proper name attribute."""
        audio_data = b"audio"
        mock_response = "Text"

        with patch.object(client.client.audio.transcriptions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response

            await client.transcribe_audio(
                audio_bytes=audio_data,
                filename="myfile.mp3",
            )

            # Verify the file object passed has the correct name
            call_args = mock_create.call_args.kwargs
            file_obj = call_args["file"]
            assert file_obj.name == "myfile.mp3"

    @pytest.mark.anyio
    async def test_chat_with_audio_wav_format(self, client: OpenAIClientWrapper) -> None:
        """Test audio chat with WAV format."""
        audio_data = b"wav audio"

        mock_message = MagicMock()
        mock_message.content = "WAV response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_completion

            result = await client.chat_with_audio(
                audio_bytes=audio_data,
                audio_format="wav",
            )

            assert result["text"] == "WAV response"

            # Verify audio format was passed correctly
            call_kwargs = mock_create.call_args.kwargs
            user_message = call_kwargs["messages"][-1]
            audio_part = [p for p in user_message["content"] if p["type"] == "input_audio"][0]
            assert audio_part["input_audio"]["format"] == "wav"

    def test_client_initialization_with_api_key(self) -> None:
        """Test client initialization with explicit API key."""
        with patch("mcp_server_whisper.infrastructure.openai_client.AsyncOpenAI") as mock_openai:
            OpenAIClientWrapper(api_key="test-key-123")
            mock_openai.assert_called_once_with(api_key="test-key-123")

    def test_client_initialization_without_api_key(self) -> None:
        """Test client initialization without API key (uses env var)."""
        with patch("mcp_server_whisper.infrastructure.openai_client.AsyncOpenAI") as mock_openai:
            OpenAIClientWrapper()
            # Should be called without api_key param (uses environment)
            mock_openai.assert_called_once_with()
