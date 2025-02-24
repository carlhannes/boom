# AI Agent with Learn-by-interact

An AI coding agent that implements Learn-by-interact principles for autonomous learning and task execution in software development environments.

## Features

- **Autonomous Learning**: Learns from documentation, codebase patterns, and successful interactions
- **Quality-Driven**: Automatically filters and maintains high-quality action patterns
- **State-Aware Matching**: Uses repository state and context for better action selection
- **Pattern-Based Recovery**: Learns from successful error recovery patterns
- **Continuous Improvement**: Refines instructions through backward construction

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Learning from a Repository

```bash
ai-agent learn /path/to/repo --docs-path /path/to/docs
```

### Executing Tasks

```bash
ai-agent execute /path/to/repo "Add unit tests for the user module"
```

To see planned actions without executing:
```bash
ai-agent execute /path/to/repo "Refactor authentication logic" --dry-run
```

### Analyzing a Repository

```bash
ai-agent analyze /path/to/repo
```

### Viewing Status

```bash
ai-agent status /path/to/repo
```

## How it Works

1. **Task Generation**
   - Extracts tasks from documentation
   - Analyzes codebase patterns
   - Generates framework-specific tasks

2. **Pattern Learning**
   - Records successful action sequences
   - Builds pattern library from high-quality executions
   - Learns state transitions and context patterns

3. **Quality Assessment**
   - Evaluates trajectory success rate
   - Measures action efficiency and relevance
   - Filters low-quality patterns automatically

4. **Agentic Retrieval**
   - Matches current state with successful patterns
   - Uses context-aware similarity scoring
   - Adapts to repository-specific patterns

5. **Error Recovery**
   - Learns from successful recovery patterns
   - Builds pattern-based recovery strategies
   - Maintains safety through quality filtering

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see LICENSE file for details