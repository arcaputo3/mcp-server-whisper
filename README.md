# MCP Server Whisper

A Model Context Protocol (MCP) server for advanced audio transcription and processing using OpenAI's Whisper and GPT-4o models.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

MCP Server Whisper provides a standardized way to process audio files through OpenAI's transcription services. By implementing the [Model Context Protocol](https://modelcontextprotocol.io/), it enables AI assistants like Claude to seamlessly interact with audio processing capabilities.

Key features:
- Automatic audio file detection and format validation
- Parallel batch processing for multiple audio files
- Format conversion between supported audio types
- Automatic compression for oversized files
- Enhanced transcription with specialized prompts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-server-whisper.git
cd mcp-server-whisper

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
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
mcp install src/mcp_server_whisper/server.py
```

### Exposed MCP Tools and Resources

#### Resources

- `dir://audio` - Lists all supported audio files with format compatibility information

#### Tools

- `get_latest_audio` - Gets the most recently modified audio file with model support info
- `convert_audio` - Converts audio files to supported formats (mp3 or wav)
- `compress_audio` - Compresses audio files that exceed size limits
- `transcribe_audio` - Basic transcription using OpenAI's Whisper model
- `transcribe_with_llm` - Transcription with custom prompts using GPT-4o
- `transcribe_with_enhancement` - Enhanced transcription with specialized templates:
  - `detailed` - Includes tone, emotion, and background details
  - `storytelling` - Transforms the transcript into a narrative form
  - `professional` - Creates formal, business-appropriate transcriptions
  - `analytical` - Adds analysis of speech patterns and key points

## Supported Audio Formats

| Model | Supported Formats |
|-------|-------------------|
| Whisper | mp3, mp4, mpeg, mpga, m4a, wav, webm |
| GPT-4o | mp3, wav |

**Note:** Files larger than 25MB are automatically compressed to meet API limits.

## Example Usage with Claude

When using with Claude or other MCP-compatible assistants:

```
Claude, please transcribe my latest audio file with detailed insights.
```

Claude will automatically:
1. Find the latest audio file using `get_latest_audio`
2. Determine the appropriate transcription method
3. Process the file with `transcribe_with_enhancement` using the "detailed" template
4. Return the enhanced transcription

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Format code
ruff format src

# Lint code
ruff check src

# Run type checking
mypy src
```

## How It Works

MCP Server Whisper is built on the Model Context Protocol, which standardizes how AI models interact with external tools and data sources. The server:

1. Exposes audio processing capabilities as MCP tools and resources
2. Accepts batch operations for efficient parallel processing
3. Handles file management, format validation, and size constraints
4. Provides rich transcription options via different OpenAI models

Under the hood, it uses:
- `pydub` for audio file manipulation
- OpenAI's Whisper API for base transcription
- GPT-4o for enhanced audio understanding
- FastMCP for simplified MCP server implementation

## Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.