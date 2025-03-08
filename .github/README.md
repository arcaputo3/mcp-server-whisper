# GitHub Workflows for MCP Server Whisper

This directory contains GitHub Actions workflows for automated testing, building, and deployment of the MCP Server Whisper package.

## Workflows

### CI/CD Pipeline (`ci-cd.yml`)

This workflow runs on push to main and mcp-experimental branches, as well as on all pull requests to main.

The workflow uses the modern `uv` Python toolchain for faster dependency installation, testing, and package management.

The workflow consists of the following jobs:

1. **Lint**: Runs Ruff and MyPy to ensure code quality and type correctness using `uv`.
2. **Test**: Runs pytest with coverage on multiple Python versions (3.10 and 3.11) using `uv`.
3. **Build**: Creates distribution packages using `uv build` when code is pushed.
4. **PyPI Publish**: Publishes the package to PyPI using `uv publish` when a new tag with the format `v*` is created.

### Release Creation (`release.yml`)

This workflow is triggered manually and helps automate the version bumping and release creation process. It also uses the `uv` toolchain for Python operations.

1. Select the version part to bump (major, minor, or patch)
2. The workflow updates the version in `pyproject.toml` using Python with `toml` library
3. Commits and pushes the change
4. Creates a git tag for the new version
5. Creates a GitHub release with automatically generated notes
6. The CI/CD pipeline will then automatically build and publish the package using `uv`

## Usage

### Creating a New Release

1. Go to the "Actions" tab in your GitHub repository
2. Select the "Create Release" workflow
3. Click "Run workflow"
4. Select the version part to bump (patch, minor, or major)
5. Click "Run workflow" to start the process

The CI/CD workflow will automatically build and publish the package to PyPI when the new tag is created.