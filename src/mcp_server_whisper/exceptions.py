"""Custom exceptions for MCP Server Whisper."""


class WhisperServerError(Exception):
    """Base exception for all MCP Server Whisper errors."""

    pass


class ConfigurationError(WhisperServerError):
    """Raised when there is a configuration issue."""

    pass


class AudioFileError(WhisperServerError):
    """Base exception for audio file-related errors."""

    pass


class AudioFileNotFoundError(AudioFileError):
    """Raised when an audio file is not found."""

    pass


class UnsupportedAudioFormatError(AudioFileError):
    """Raised when an audio format is not supported."""

    pass


class AudioProcessingError(AudioFileError):
    """Raised when audio processing fails."""

    pass


class AudioConversionError(AudioProcessingError):
    """Raised when audio format conversion fails."""

    pass


class AudioCompressionError(AudioProcessingError):
    """Raised when audio compression fails."""

    pass


class TranscriptionError(WhisperServerError):
    """Base exception for transcription-related errors."""

    pass


class TranscriptionAPIError(TranscriptionError):
    """Raised when the transcription API call fails."""

    pass


class TTSError(WhisperServerError):
    """Base exception for text-to-speech errors."""

    pass


class TTSAPIError(TTSError):
    """Raised when the TTS API call fails."""

    pass
