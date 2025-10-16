# MCP Server Whisper Architecture

## Overview

MCP Server Whisper is an MCP-compatible server that provides audio transcription and processing capabilities using OpenAI's Whisper and GPT-4o models. It follows a **layered architecture** based on Domain-Driven Design principles with clean separation of concerns.

```
┌─────────────────┐
│   MCP Client    │     (e.g., Claude Desktop)
│   (Claude)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              MCP Server (FastMCP)                       │
│                   server.py (19 lines)                  │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    Tools Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ file_tools   │  │ audio_tools  │  │transcription │ │
│  │              │  │              │  │   tools      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                 Services Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ FileService  │  │AudioService  │  │ Transcription│ │
│  │              │  │              │  │   Service    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼──────────────────┼──────────────────┼─────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│                  Domain Layer                           │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │ FileFilterSorter │      │  AudioProcessor  │        │
│  │ (pure logic)     │      │  (pure logic)    │        │
│  └──────────────────┘      └──────────────────┘        │
└─────────────────────────────────────────────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│              Infrastructure Layer                       │
│  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │FileSystemRepo    │  │ OpenAIClientWrapper      │   │
│  │(File I/O)        │  │ (API Integration)        │   │
│  └────────┬─────────┘  └──────────┬───────────────┘   │
└───────────┼────────────────────────┼───────────────────┘
            │                        │
            ▼                        ▼
   ┌────────────────┐      ┌────────────────┐
   │ Audio Storage  │      │  OpenAI API    │
   │ (File System)  │      │                │
   └────────────────┘      └────────────────┘
```

## Layered Architecture

### 1. MCP Server Layer (server.py - 19 lines)
- Minimal FastMCP server initialization
- Tool registration only
- No business logic

### 2. Tools Layer (tools/)
- **file_tools.py**: File management MCP tools
- **audio_tools.py**: Audio conversion/compression MCP tools
- **transcription_tools.py**: Transcription MCP tools
- **tts_tools.py**: Text-to-speech MCP tools
- Thin layer: validation + service orchestration + response formatting

### 3. Services Layer (services/)
- **FileService**: File discovery, filtering, sorting orchestration
- **AudioService**: Audio conversion/compression workflows
- **TranscriptionService**: Transcription API coordination
- **TTSService**: Text-to-speech generation workflows
- Orchestrates domain logic + infrastructure

### 4. Domain Layer (domain/)
- **AudioProcessor**: Pure audio processing algorithms (conversion, compression, concatenation)
- **FileFilterSorter**: Pure filtering and sorting logic
- **Zero external dependencies** - only business rules
- Highly testable and reusable

### 5. Infrastructure Layer (infrastructure/)
- **FileSystemRepository**: All file I/O operations
- **OpenAIClientWrapper**: OpenAI API integration
- **AudioFileCache**: LRU caching for metadata
- Abstracts external dependencies

### 6. Supporting Modules
- **models/**: Pydantic models for data validation
- **constants.py**: All constants, enums, type aliases
- **exceptions.py**: Custom exception hierarchy
- **config.py**: Configuration management
- **utils/**: Pure utility functions

## Data Flow Example: Transcribe Audio

1. **MCP Client Request** → Claude calls `transcribe_audio` tool
2. **Tools Layer** → `transcription_tools.py` validates input parameters
3. **Services Layer** → `TranscriptionService` orchestrates the workflow:
   - Uses `FileSystemRepository` to read audio file
   - Uses `OpenAIClientWrapper` to call API
4. **Infrastructure Layer** → Executes I/O and API calls:
   - `FileSystemRepository.read_audio_file()` → async file read
   - `OpenAIClientWrapper.transcribe_audio()` → API call to OpenAI
5. **Response** → Flows back up: Infrastructure → Service → Tool → MCP → Client

## Architectural Benefits

### Separation of Concerns
- Each layer has a single, well-defined responsibility
- Changes in one layer don't affect others
- Easy to understand and navigate

### Testability
- **Domain Layer**: Pure functions, no mocking needed
- **Infrastructure Layer**: Easy to mock external dependencies
- **Services Layer**: Test orchestration logic with mocked infrastructure
- **Tools Layer**: Integration tests with real services

### Maintainability
- **server.py**: Reduced from 892 → 19 lines (97.6% reduction)
- **Clear boundaries**: Each module has a specific purpose
- **Type safety**: Full mypy compliance with native types
- **Extensibility**: Add new features without modifying existing code

### Performance
- **Structured Concurrency**: Task groups using anyio for safe parallel processing
- **LRU Caching**: Smart caching with mtime-based invalidation
- **Async I/O**: Non-blocking file operations with aiofiles
- **Concurrent API Calls**: Parallel OpenAI requests with result collection

## Project Structure

```
src/mcp_server_whisper/
├── server.py              # MCP server (19 lines)
├── config.py              # Configuration management
├── constants.py           # Constants & enums
├── exceptions.py          # Custom exceptions
│
├── models/                # Pydantic models (4 files)
│   ├── base.py
│   ├── audio.py
│   ├── transcription.py
│   └── tts.py
│
├── domain/                # Business logic (2 files)
│   ├── audio_processor.py
│   └── file_filter.py
│
├── infrastructure/        # External dependencies (4 files)
│   ├── file_system.py
│   ├── openai_client.py
│   └── cache.py
│
├── services/              # Orchestration (4 files)
│   ├── file_service.py
│   ├── audio_service.py
│   ├── transcription_service.py
│   └── tts_service.py
│
├── tools/                 # MCP tools (4 files)
│   ├── file_tools.py
│   ├── audio_tools.py
│   ├── transcription_tools.py
│   └── tts_tools.py
│
└── utils/                 # Utilities (1 file)
    └── text_utils.py
```

**Total: 31 files, 2,453 lines** (vs original: 1 file, 892 lines)

## Technology Stack

- **Language**: Python 3.10+
- **Audio Processing**: pydub, audioop-lts (Python 3.13+)
- **Async Framework**: anyio (structured concurrency), aiofiles
- **Concurrency**: aioresult for result collection from task groups
- **MCP Framework**: FastMCP
- **API Integration**: OpenAI Python client
- **Data Validation**: Pydantic models
- **Configuration**: pydantic-settings
- **Type Checking**: mypy (100% compliant, native types only)
- **Code Quality**: ruff for linting and formatting

## Design Patterns

- **Repository Pattern**: FileSystemRepository abstracts file I/O
- **Dependency Injection**: Services receive dependencies via constructor
- **Strategy Pattern**: Enhancement templates for different transcription styles
- **Facade Pattern**: Services provide simple interfaces to complex operations
- **Singleton Pattern**: Configuration and cache instances