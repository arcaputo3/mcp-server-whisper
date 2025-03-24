# MCP Server Whisper

<div align="center">

A Model Context Protocol (MCP) server for advanced audio transcription and processing using OpenAI's Whisper and GPT-4o models.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![CI Status](https://github.com/arcaputo3/mcp-server-whisper/workflows/CI/CD%20Pipeline/badge.svg)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-a240e6)](https://github.com/astral-sh/uv)

</div>

## Overview

MCP Server Whisper provides a standardized way to process audio files through OpenAI's latest transcription and speech services. By implementing the [Model Context Protocol](https://modelcontextprotocol.io/), it enables AI assistants like Claude to seamlessly interact with audio processing capabilities.

Key features:
- 🔍 **Advanced file searching** with regex patterns, file metadata filtering, and sorting capabilities
- 🔄 **Parallel batch processing** for multiple audio files
- 🔄 **Format conversion** between supported audio types
- 📦 **Automatic compression** for oversized files
- 🎯 **Multi-model transcription** with support for all OpenAI audio models
- 🗣️ **Interactive audio chat** with GPT-4o audio models
- ✏️ **Enhanced transcription** with specialized prompts and timestamp support
- 🎙️ **Text-to-speech generation** with customizable voices, instructions, and speed
- 📊 **Comprehensive metadata** including duration, file size, and format support
- 🚀 **High-performance caching** for repeated operations

## Installation

```bash
# Clone the repository
git clone https://github.com/arcaputo3/mcp-server-whisper.git
cd mcp-server-whisper

# Using uv 
uv sync

# Set up pre-commit hooks
uv run pre-commit install
```

## Environment Setup

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
AUDIO_FILES_PATH=/path/to/your/audio/files
```

## Usage

### Starting the Server

To run the MCP server in development mode:

```bash
mcp dev src/mcp_server_whisper/server.py
```

To install the server for use with Claude Desktop or other MCP clients:

```bash
mcp install src/mcp_server_whisper/server.py [--env-file .env]
```

### Exposed MCP Tools

#### Audio File Management

- `list_audio_files` - Lists audio files with comprehensive filtering and sorting options:
  - Filter by regex pattern matching on filenames
  - Filter by file size, duration, modification time, or format
  - Sort by name, size, duration, modification time, or format
  - All operations support parallelized batch processing
- `get_latest_audio` - Gets the most recently modified audio file with model support info

#### Audio Processing

- `convert_audio` - Converts audio files to supported formats (mp3 or wav)
- `compress_audio` - Compresses audio files that exceed size limits

#### Transcription

- `transcribe_audio` - Advanced transcription using OpenAI's models:
  - Supports `whisper-1`, `gpt-4o-transcribe`, and `gpt-4o-mini-transcribe`
  - Custom prompts for guided transcription
  - Optional timestamp granularities for word and segment-level timing
  - JSON response format option
 
- `chat_with_audio` - Interactive audio analysis using GPT-4o audio models:
  - Supports `gpt-4o-audio-preview-2024-10-01`, `gpt-4o-audio-preview-2024-12-17`, and `gpt-4o-mini-audio-preview-2024-12-17`
  - Custom system and user prompts
  - Provides conversational responses to audio content
  
- `transcribe_with_enhancement` - Enhanced transcription with specialized templates:
  - `detailed` - Includes tone, emotion, and background details
  - `storytelling` - Transforms the transcript into a narrative form
  - `professional` - Creates formal, business-appropriate transcriptions
  - `analytical` - Adds analysis of speech patterns and key points

#### Text-to-Speech

- `create_claudecast` - Generate text-to-speech audio using OpenAI's TTS API:
  - Supports `gpt-4o-mini-tts` (preferred) and other speech models
  - Multiple voice options (alloy, ash, coral, echo, fable, onyx, nova, sage, shimmer)
  - Speed adjustment and custom instructions
  - Customizable output file paths
  - Handles texts of any length by automatically splitting and joining audio segments

## Supported Audio Formats

| Model | Supported Formats |
|-------|-------------------|
| Transcribe | flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm |
| Chat | mp3, wav |

**Note:** Files larger than 25MB are automatically compressed to meet API limits.

## Example Usage with Claude

<details>
<summary>Basic Audio Transcription</summary>

```
Claude, please transcribe my latest audio file with detailed insights.
```

Claude will automatically:
1. Find the latest audio file using `get_latest_audio`
2. Determine the appropriate transcription method
3. Process the file with `transcribe_with_enhancement` using the "detailed" template
4. Return the enhanced transcription
</details>

<details>
<summary>Advanced Audio File Search and Filtering</summary>

```
Claude, list all my audio files that are longer than 5 minutes and were created after January 1st, 2024, sorted by size.
```

Claude will:
1. Convert the date to a timestamp
2. Use `list_audio_files` with appropriate filters:
   - `min_duration_seconds: 300`  (5 minutes)
   - `min_modified_time: <timestamp for Jan 1, 2024>`
   - `sort_by: "size"`
3. Return a sorted list of matching audio files with comprehensive metadata
</details>

<details>
<summary>Batch Processing Multiple Files</summary>

```
Claude, find all MP3 files with "interview" in the filename and create professional transcripts for each one.
```

Claude will:
1. Search for files using `list_audio_files` with:
   - `pattern: ".*interview.*\\.mp3"`
   - `format: "mp3"`
2. Process all matching files in parallel using `transcribe_with_enhancement`
   - `enhancement_type: "professional"`
   - `model: "gpt-4o-mini-transcribe"` (for efficiency)
3. Return all transcriptions in a well-formatted output
</details>

<details>
<summary>Generating Text-to-Speech with Claudecast</summary>

```
Claude, create a claudecast with this script: "Welcome to our podcast! Today we'll be discussing artificial intelligence trends in 2025." Use the shimmer voice.
```

Claude will:
1. Use the `create_claudecast` tool with:
   - `text_prompt` containing the script
   - `voice: "shimmer"`
   - `model: "gpt-4o-mini-tts"` (default high-quality model)
   - `instructions: "Speak in an enthusiastic, podcast host style"` (optional)
   - `speed: 1.0` (default, can be adjusted)
2. Generate the audio file and save it to the configured audio directory
3. Provide the path to the generated audio file
</details>

## Configuration with Claude Desktop

Add this to your `claude_desktop_config.json`:

### UVX

```json 
{
  "mcpServers": {
    "whisper": {
      "command": "uvx",
      "args": [
        "--with",
        "aiofiles",
        "--with",
        "mcp[cli]",
        "--with",
        "openai",
        "--with",
        "pydub",
        "mcp-server-whisper"
      ],
      "env": {
        "OPENAI_API_KEY": "your_openai_api_key",
        "AUDIO_FILES_PATH": "/path/to/your/audio/files"
      }
    }
  }
}
```

### Recommendation (Mac OS Only)

- Install [Screen Recorder By Omi](https://apps.apple.com/us/app/screen-recorder-by-omi/id1592987853?mt=12) (free)
- Set `AUDIO_FILES_PATH` to `/Users/<user>/Movies/Omi Screen Recorder` and replace `<user>` with your username
- As you record audio with the app, you can transcribe large batches directly with Claude

## Development

This project uses modern Python development tools including `uv`, `pytest`, `ruff`, and `mypy`.

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Format code
uv run ruff format src

# Lint code
uv run ruff check src

# Run type checking (strict mode)
uv run mypy --strict src

# Run the pre-commit hooks
pre-commit run --all-files
```

### CI/CD Workflow

The project uses GitHub Actions for CI/CD:

1. **Lint & Type Check**: Ensures code quality with ruff and strict mypy type checking
2. **Tests**: Runs tests on multiple Python versions (3.10, 3.11)
3. **Build**: Creates distribution packages
4. **Publish**: Automatically publishes to PyPI when a new version tag is pushed

To create a new release version:
```bash
git checkout main
# Make sure everything is up to date
git pull
# Create a new version tag
git tag v0.1.1
# Push the tag
git push origin v0.1.1
```

## How It Works

For detailed architecture information, see [Architecture Documentation](docs/architecture.md).

MCP Server Whisper is built on the Model Context Protocol, which standardizes how AI models interact with external tools and data sources. The server:

1. **Exposes Audio Processing Capabilities**: Through standardized MCP tool interfaces
2. **Implements Parallel Processing**: Using asyncio and batch operations for performance
3. **Manages File Operations**: Handles detection, validation, conversion, and compression
4. **Provides Rich Transcription**: Via different OpenAI models and enhancement templates
5. **Optimizes Performance**: With caching mechanisms for repeated operations

**Under the hood, it uses:**
- `pydub` for audio file manipulation
- `asyncio` for concurrent processing
- OpenAI's latest transcription models (including gpt-4o-transcribe)
- OpenAI's GPT-4o audio models for enhanced understanding
- OpenAI's gpt-4o-mini-tts for high-quality speech synthesis
- FastMCP for simplified MCP server implementation
- Type hints and strict mypy validation throughout the codebase

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests and linting (`uv run pytest && uv run ruff check src && uv run mypy --strict src`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) - For the protocol specification
- [pydub](https://github.com/jiaaro/pydub) - For audio processing
- [OpenAI Whisper](https://openai.com/research/whisper) - For audio transcription
- [FastMCP](https://github.com/anthropics/FastMCP) - For MCP server implementation
- [Anthropic Claude](https://claude.ai/) - For natural language interaction

---

<div align="center">
Made with ❤️ by <a href="https://github.com/arcaputo3">Richie Caputo</a>
</div>