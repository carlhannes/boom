# AI Coding Agent (BOOM)

An AI coding agent based on Learn-by-interact principles that autonomously performs coding tasks in Git repositories. The agent learns from its interactions with the codebase and improves its performance through trajectory-based learning.

## Key Features

- Task generation from documentation
- Execution trajectory recording
- Backward construction of instructions
- Similarity-based trajectory retrieval
- Git repository interaction
- OpenAI GPT integration

## Installation

The project uses Poetry for dependency management. Key dependencies are specified in `pyproject.toml`.

Required Python version: ^3.9

```bash
poetry install
```

Main dependencies:
- openai ^1.0.0
- gitpython ^3.1.0
- numpy ^1.26.0
- pytest ^7.0.0

## Getting Started

See the `examples/demo.py` file for example usage of the agent.

## Project Structure

- `ai_agent/core/` - Core agent and learning components
- `ai_agent/environment/` - Git repository interface
- `ai_agent/data/` - Trajectory storage and management
- `docs/` - Additional documentation
- `examples/` - Usage examples
- `tests/` - Test suite

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Usage Guide](docs/usage.md)

## License

See [LICENSE.md](LICENSE.md) for details.