# MCP Server Whisper

<div align="center">

A Model Context Protocol (MCP) server for advanced audio transcription and processing using OpenAI's Whisper and GPT-4o models.

[![PyPI version](https://img.shields.io/pypi/v/mcp-server-whisper.svg)](https://pypi.org/project/mcp-server-whisper/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org/downloads/)
![CI Status](https://github.com/arcaputo3/mcp-server-whisper/workflows/CI/CD%20Pipeline/badge.svg)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-a240e6)](https://github.com/astral-sh/uv)

</div>

## Overview

MCP Server Whisper provides a standardized way to process audio files through OpenAI's latest transcription and speech services. By implementing the [Model Context Protocol](https://modelcontextprotocol.io/), it enables AI assistants like Claude to seamlessly interact with audio processing capabilities.

Key features:
- 🔍 **Advanced file searching** with regex patterns, file metadata filtering, and sorting capabilities
- ⚡ **MCP-native parallel processing** - call multiple tools simultaneously
- 🔄 **Format conversion** between supported audio types
- 📦 **Automatic compression** for oversized files
- 🎯 **Multi-model transcription** with support for all OpenAI audio models
- 🗣️ **Interactive audio chat** with GPT-4o audio models
- ✏️ **Enhanced transcription** with specialized prompts and timestamp support
- 🎙️ **Text-to-speech generation** with customizable voices, instructions, and speed
- 📊 **Comprehensive metadata** including duration, file size, and format support
- 🚀 **High-performance caching** for repeated operations
- 🔒 **Type-safe responses** with Pydantic models for all tool outputs

> **Note:** This project is unofficial and not affiliated with, endorsed by, or sponsored by OpenAI. It provides a Model Context Protocol interface to OpenAI's publicly available APIs.

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

Create a `.env` file based on the provided `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```
OPENAI_API_KEY=your_openai_api_key
AUDIO_FILES_PATH=/path/to/your/audio/files
```

**Note:** Environment variables must be available at runtime. For local development with Claude, use a tool like `dotenv-cli` to load them (see Usage section below).

## Usage

### Local Development with Claude

The project includes a `.mcp.json` configuration file for local development with Claude. To use it:

1. Ensure your `.env` file is configured with the required environment variables
2. Launch Claude with environment variables loaded:

```bash
bunx dotenv-cli -- claude
```

This will:
- Load environment variables from your `.env` file
- Launch Claude with the MCP server configured per `.mcp.json`
- Enable hot-reloading during development

The `.mcp.json` configuration:

```json
{
  "mcpServers": {
    "whisper": {
      "command": "uv",
      "args": ["run", "mcp-server-whisper"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "AUDIO_FILES_PATH": "${AUDIO_FILES_PATH}"
      }
    }
  }
}
```

### Exposed MCP Tools

#### Audio File Management

- `list_audio_files` - Lists audio files with comprehensive filtering and sorting options:
  - Filter by regex pattern matching on filenames
  - Filter by file size, duration, modification time, or format
  - Sort by name, size, duration, modification time, or format
  - Returns type-safe `FilePathSupportParams` with full metadata
- `get_latest_audio` - Gets the most recently modified audio file with model support info

#### Audio Processing

- `convert_audio` - Converts audio files to supported formats (mp3 or wav)
  - Returns `AudioProcessingResult` with output path
- `compress_audio` - Compresses audio files that exceed size limits
  - Returns `AudioProcessingResult` with output path

#### Transcription

- `transcribe_audio` - Advanced transcription using OpenAI's models:
  - Supports `whisper-1`, `gpt-4o-transcribe`, and `gpt-4o-mini-transcribe`
  - Custom prompts for guided transcription
  - Optional timestamp granularities for word and segment-level timing
  - JSON response format option
  - Returns `TranscriptionResult` with text, usage data, and optional timestamps

- `chat_with_audio` - Interactive audio analysis using GPT-4o audio models:
  - Supports `gpt-4o-audio-preview` (recommended) and dated versions
  - Note: `gpt-4o-mini-audio-preview` has limitations with audio chat and is not recommended
  - Custom system and user prompts
  - Provides conversational responses to audio content
  - Returns `ChatResult` with response text

- `transcribe_with_enhancement` - Enhanced transcription with specialized templates:
  - `detailed` - Includes tone, emotion, and background details
  - `storytelling` - Transforms the transcript into a narrative form
  - `professional` - Creates formal, business-appropriate transcriptions
  - `analytical` - Adds analysis of speech patterns and key points
  - Returns `TranscriptionResult` with enhanced output

#### Text-to-Speech

- `create_audio` - Generate text-to-speech audio using OpenAI's TTS API:
  - Supports `gpt-4o-mini-tts` (preferred) and other speech models
  - Multiple voice options (alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar)
  - Speed adjustment and custom instructions
  - Customizable output file paths
  - Handles texts of any length by automatically splitting and joining audio segments
  - Returns `TTSResult` with output path

## Supported Audio Formats

| Model      | Supported Formats                               |
|------------|-------------------------------------------------|
| Transcribe | flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm |
| Chat       | mp3, wav                                        |

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
1. Search for files using `list_audio_files` with pattern and format filters
2. Make multiple parallel `transcribe_with_enhancement` tool calls (MCP handles parallelism natively)
3. Each call uses `enhancement_type: "professional"` and returns a typed `TranscriptionResult`
4. Return all transcriptions with full metadata in a well-formatted output
</details>

<details>
<summary>Generating Text-to-Speech Audio</summary>

```
Claude, create audio with this script: "Welcome to our podcast! Today we'll be discussing artificial intelligence trends in 2025." Use the shimmer voice.
```

Claude will:
1. Use the `create_audio` tool with:
   - `text_prompt` containing the script
   - `voice: "shimmer"`
   - `model: "gpt-4o-mini-tts"` (default high-quality model)
   - `instructions: "Speak in an enthusiastic, podcast host style"` (optional)
   - `speed: 1.0` (default, can be adjusted)
2. Generate the audio file and save it to the configured audio directory
3. Provide the path to the generated audio file
</details>

## Configuration with Claude Desktop

For production use with Claude Desktop (as opposed to local development), add this to your `claude_desktop_config.json`:

### UVX

```json
{
  "mcpServers": {
    "whisper": {
      "command": "uvx",
      "args": ["mcp-server-whisper"],
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
- As you record audio with the app, you can transcribe multiple files in parallel with Claude

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
2. **Tests**: Runs tests on multiple Python versions (3.10, 3.11, 3.12, 3.13, 3.14, 3.14t)
3. **Release & Publish**: Dual-trigger workflow for flexible release management

**Note:** Python 3.14t is the free-threaded build (without GIL) for testing true parallelism.

#### Creating a New Release

The release workflow supports two approaches:

**Option 1: Automated Release (Recommended)**

Push a tag to automatically create a release and publish to PyPI:

```bash
# 1. Update version in pyproject.toml
# Edit the version field manually, e.g., "1.0.0" -> "1.1.0"

# 2. Update __version__ in src/mcp_server_whisper/__init__.py to match

# 3. Update the lock file
uv lock

# 4. Commit the version bump
git add pyproject.toml src/mcp_server_whisper/__init__.py uv.lock
git commit -m "chore: bump version to 1.1.0"

# 5. Create and push the version tag
git tag v1.1.0
git push origin main
git push origin v1.1.0
```

This will:
- Verify the tag version matches pyproject.toml
- Build the package
- Create a GitHub release with auto-generated notes
- Automatically publish to PyPI

**Option 2: Manual Release**

Create a release manually via GitHub UI, then publish optionally:

1. Go to [Releases](https://github.com/arcaputo3/mcp-server-whisper/releases) on GitHub
2. Click "Draft a new release"
3. Create a new tag or select an existing one
4. Fill in release details
5. Click "Publish release"

When you publish the release, the workflow will automatically publish to PyPI. You can also create a draft release to delay publishing.

## API Design Philosophy

MCP Server Whisper follows a **flat, type-safe API design** optimized for MCP clients:

- **Flat Arguments**: All tools accept flat parameters instead of nested objects for simpler, more intuitive calls
- **Type-Safe Responses**: Every tool returns a strongly-typed Pydantic model (`TranscriptionResult`, `ChatResult`, `AudioProcessingResult`, `TTSResult`)
- **Single-Item Operations**: One call processes one file, with MCP protocol handling parallelism natively
- **Per-File Error Handling**: Failures are isolated to individual operations, not entire batches
- **Self-Documenting**: Type hints provide autocomplete and validation in IDEs and AI models

This design makes it significantly easier for AI assistants to use the tools correctly and handle results reliably.

## How It Works

For detailed architecture information, see [Architecture Documentation](docs/architecture.md).

MCP Server Whisper is built on the Model Context Protocol, which standardizes how AI models interact with external tools and data sources. The server:

1. **Exposes Audio Processing Capabilities**: Through standardized MCP tool interfaces with flat, type-safe APIs
2. **Implements Parallel Processing**: Using anyio structured concurrency; MCP clients handle parallelism natively
3. **Manages File Operations**: Handles detection, validation, conversion, and compression
4. **Provides Rich Transcription**: Via different OpenAI models and enhancement templates
5. **Optimizes Performance**: With caching mechanisms for repeated operations
6. **Ensures Type Safety**: All responses use Pydantic models for validation and IDE support

**Under the hood, it uses:**
- `pydub` for audio file manipulation (with `audioop-lts` for Python 3.13+)
- `anyio` for structured concurrency and task group management
- `aioresult` for collecting results from parallel task groups
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
- [MCP Review](https://mcpreview.com/mcp-servers/arcaputo3/mcp-server-whisper) - This MCP Server is certified by MCP Review

---

<div align="center">
Made with ❤️ by <a href="https://github.com/arcaputo3">Richie Caputo</a>
</div>
