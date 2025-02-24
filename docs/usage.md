# Usage Guide

## Setup and Installation
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# For ARM64 (Apple Silicon) users, if needed:
pip install wheels/numpy-2.2.3-cp312-cp312-macosx_14_0_arm64.whl
pip install wheels/scipy-1.13.1-cp312-cp312-macosx_12_0_arm64.whl
```

## Basic Usage

```bash
# Execute a task with default settings
ai-agent execute "Add input validation to user form"

# Execute with custom BM25 retrieval settings
ai-agent execute "Refactor authentication module" --bm25-candidates 100

# Specify custom paths
ai-agent execute "Fix database connection" -r /path/to/repo -s /path/to/storage
```

## Retrieval System

The agent uses a two-stage hybrid retrieval system for efficient and accurate code trajectory search:

1. **First Stage (BM25)**
   - Fast keyword-based retrieval using inverted index
   - Configurable number of candidates (--bm25-candidates)
   - Code-aware tokenization (preserves file extensions, code terms)
   - Highly efficient for large codebases by using token intersection
   - No need to scan entire corpus for each query

2. **Second Stage (Embedding + State)**
   - Re-ranks BM25 candidates using:
     - Semantic similarity (40% weight) - handles synonyms and paraphrasing
     - BM25 score (30% weight) - maintains keyword relevance
     - Repository state matching (30% weight) - considers code context
   - Returns top 5 most relevant examples
   - Computed only on BM25 candidates, making it efficient

The two-tier strategy combines the speed of BM25 with the semantic understanding of embeddings, making it both efficient and accurate for code trajectory retrieval.

## Retrieval System Configuration

The agent uses an intelligent two-stage retrieval system that adapts to query characteristics:

### Basic Configuration

```bash
# Default settings (balanced approach)
ai-agent execute "Add input validation" --bm25-candidates 50

# Force re-ranking for complex queries
ai-agent execute "Implement secure user authentication following best practices" --force-rerank

# Skip re-ranking for simple searches
ai-agent execute "Find getUserProfile function" --skip-rerank
```

### Query Types and Optimal Settings

1. **Simple Keyword Queries**
   - Function names: `getUserData`
   - File paths: `src/auth/login.js`
   - Error messages: `TypeError: Cannot read property`
   → Use BM25 only (faster, equally accurate)

2. **Conceptual/Complex Queries**
   - Natural language: "How to handle async state updates"
   - Best practices: "Implement secure password storage"
   - Design patterns: "Use factory pattern for object creation"
   → Enable re-ranking for better semantic matching

### Performance Optimization

1. **Repository Size Considerations**
   - Small (< 1000 files)
     * BM25 candidates: 20-50
     * Re-ranking: Optional, fast enough for most queries
   
   - Medium (1000-10000 files)
     * BM25 candidates: 50-100
     * Re-ranking: Selective based on query type
   
   - Large (> 10000 files)
     * BM25 candidates: 100-200
     * Re-ranking: Only for complex queries
     * Consider caching embeddings

2. **Resource Usage**
   - Memory: Primarily affected by number of stored trajectories
   - CPU: Re-ranking more intensive than BM25
   - Storage: Embeddings cached for frequent queries

3. **Response Time Targets**
   - BM25 only: < 100ms typical
   - With re-ranking: 200-500ms typical
   - Can be optimized with caching and selective re-ranking

### Best Practices

1. **Query Formulation**
   - Start with specific keywords when possible
   - Use natural language for conceptual searches
   - Include relevant code terms for better matching

2. **Configuration Strategy**
   - Start with BM25-only for basic operations
   - Enable re-ranking for:
     * Learning new patterns
     * Complex refactoring tasks
     * Design-related queries
   - Monitor and adjust based on result quality

3. **Performance Monitoring**
   - Track query response times
   - Monitor re-ranking frequency
   - Adjust thresholds based on feedback
   - Cache frequently used embeddings

## Workflow Patterns

### 1. Task Generation from Documentation

```python
learner = SelfLearner()
docs = [
    "Implement input validation for user data",
    "Add error handling to the database connections"
]
tasks = learner.generate_tasks_from_docs(docs)
```

### 2. Batch Task Execution

```python
for task in tasks:
    print(f"Executing: {task.instruction}")
    trajectory = agent.execute_task(task.instruction)
    print(f"Actions taken: {len(trajectory.actions)}")
```

## Environment State

The agent observes the following from the Git environment:
- List of tracked files
- Git status (modified, staged, untracked files)
- Current branch
- Last commit information
- Merge conflicts

## Storage Location

Trajectories are stored at:
```
~/.ai-agent/trajectories/*.json
```

Each trajectory file contains:
- Original instruction
- Executed actions
- Environment observations
- Final state
- Instruction embedding

## Best Practices

1. **Start Small**: Begin with simple, well-defined tasks
2. **Review Changes**: The agent commits changes but review before pushing
3. **Provide Context**: Include relevant documentation for better task generation
4. **Monitor Trajectories**: Review stored trajectories to understand agent behavior
5. **Tune Retrieval**: Adjust --bm25-candidates based on repository size:
   - Small repos (< 1000 files): 20-50 candidates
     → Smaller search space allows fewer candidates while maintaining accuracy
   - Medium repos: 50-100 candidates
     → Balance between search speed and result quality
   - Large repos (> 10000 files): 100-200 candidates
     → More candidates needed to ensure good coverage of larger codebase
   
   The two-stage approach (BM25 + embedding re-ranking) keeps search efficient even with larger candidate sets, as the more expensive embedding similarity is only computed on the BM25 candidates.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| --repo-path, -r | . | Path to Git repository |
| --storage-path, -s | ~/.ai-agent/trajectories | Path to store agent data |
| --bm25-candidates, -k | 50 | Number of first-stage retrieval candidates |

## Development Workflow

1. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

2. Running tests:
```bash
pytest
```

3. Code formatting:
```bash
black .
isort .
```

4. Cleaning up:
```bash
# Remove cache files and virtual environment
make clean
```

## CLI Usage

```bash
# Execute a task
ai-agent "Add error handling to login function"

# Generate tasks from docs
ai-agent generate-tasks --docs path/to/docs.md
```

## Example Workflows

### 1. Basic Task Execution
```bash
ai-agent execute "Add error handling to login function"
```

### 2. Large Codebase with More Candidates
```bash
ai-agent execute "Implement new API endpoint" --bm25-candidates 150
```

### 3. Custom Storage Location
```bash
ai-agent execute "Write unit tests" -s ./my-trajectories