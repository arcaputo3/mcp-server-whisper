# MCP Server Whisper Architecture

## Overview

MCP Server Whisper is an MCP-compatible server that provides audio transcription and processing capabilities using OpenAI's Whisper and GPT-4o models. It follows a modular architecture with several key components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   MCP Client    │     │   MCP Server    │     │  Audio Storage  │
│   (Claude)      │◄────►   (FastMCP)     │◄────►                 │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
        ┌────────▼───────┐    ┌▼────────────┐    ┌────────────────┐
        │                │    │             │    │                │
        │ File Processing│    │Transcription│    │ OpenAI API     │
        │ & Management   │    │Services     │◄───►                │
        │                │    │             │    │                │
        └────────────────┘    └─────────────┘    └────────────────┘
```

## Components

### 1. MCP Server
- Built using FastMCP framework
- Exposes tools and resources via MCP protocol
- Handles authentication, request routing, and error handling

### 2. File Processing & Management
- Audio file detection and validation
- Format conversion between supported audio types
- Compression for oversized files
- Advanced search/filtering capabilities

### 3. Transcription Services
- Basic transcription with Whisper
- Enhanced transcription with GPT-4o
- Multiple enhancement templates

### 4. OpenAI Integration
- API client for Whisper and GPT-4o
- File preparation and submission
- Response processing

## Data Flow

1. **Client Request**: An MCP client (like Claude) makes a request to use a transcription tool
2. **Tool Invocation**: The MCP server receives the request and invokes the appropriate tool
3. **File Processing**: The system locates, validates, and prepares the audio file
4. **API Submission**: The prepared file is submitted to OpenAI's API
5. **Transcription Generation**: OpenAI processes the audio and returns the transcription
6. **Response Formatting**: The system formats and enhances the response if needed
7. **Response Delivery**: The MCP server returns the response to the client

## Key Features

- **Parallel Processing**: Batch operations using asyncio for efficient processing
- **Caching**: LRU caching for audio file metadata to improve performance
- **Enhanced Search**: Advanced file search capabilities with regex and metadata filtering
- **Format Handling**: Automatic format detection, validation, conversion, and compression

## Technology Stack

- **Language**: Python 3.10+
- **Audio Processing**: pydub
- **Async Framework**: asyncio, aiofiles
- **MCP Framework**: FastMCP
- **API Integration**: OpenAI Python client
- **Type Checking**: mypy with strict mode
- **Code Quality**: ruff for linting and formatting