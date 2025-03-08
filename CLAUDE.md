# MCP Server Whisper - Development Guidelines

## Commands
- Run tests: `pytest`
- Run single test: `pytest tests/path_to_test.py::test_function_name -v`
- Run with coverage: `pytest --cov=src`
- Type checking: `mypy src`
- Linting: `ruff check src`
- Format code: `ruff format src`
- Run MCP server: `mcp dev src/mcp_server_whisper/server.py`
- Install MCP server: `mcp install src/mcp_server_whisper/server.py`
- Setup pre-commit: `pre-commit install`
- Run pre-commit: `pre-commit run --all-files`

## Code Style
- Line length: 120 characters
- Use type hints for all functions and variables
- Import order: standard library → typing → third-party → local
- Use async/await for I/O operations
- Use Pydantic models for data validation
- Function/variable naming: snake_case
- Class naming: PascalCase
- Constants: UPPER_SNAKE_CASE
- Follow Google docstring style for documentation

## Error Handling
- Use specific exceptions with descriptive messages
- Propagate exceptions with `raise ... from e`
- Handle errors at appropriate abstraction levels
- Log errors with context information

## MCP Implementation
- Tools are defined with `@mcp.tool()` decorator
- Resources are defined with `@mcp.resource()` decorator
- Use batch processing with lists of inputs for parallel operations
- Process multiple files concurrently using `asyncio.gather()`
- Return structured data (typically Dict[str, Any]) for consistent handling

## Audio Processing
- Supports multiple formats: mp3, wav, mp4, mpeg, mpga, m4a, webm
- GPT-4o direct access supports: mp3, wav
- Files over 25MB are automatically compressed
- Parallel processing for batch transcription jobs
- Transcription options: basic Whisper, GPT-4o with prompt, enhanced transcription