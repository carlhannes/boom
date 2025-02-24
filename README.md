# AI Agent

AI coding agent based on Learn-by-interact principles.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-agent
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## ARM64 (Apple Silicon) Users

For ARM64 architectures (e.g., Apple Silicon Macs), if you encounter any issues with scipy installation, pre-compiled wheels are available in the `wheels` directory. These can be installed with:

```bash
pip install wheels/numpy-2.2.3-cp312-cp312-macosx_14_0_arm64.whl
pip install wheels/scipy-1.13.1-cp312-cp312-macosx_12_0_arm64.whl
```

## Development

1. Setup development environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest
```

3. Format code:
```bash
black .
isort .
```

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

See [LICENSE](LICENSE) file.