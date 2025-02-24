# Technical Architecture

## Component Architecture

```
┌─────────────────┐
│   CodingAgent   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼────────┐
│GitEnv │ │Trajectory  │
└───────┘ │Manager     │
          └───┬────────┘
              │
          ┌───▼────┐
          │Self    │
          │Learner │
          └────────┘
```

## Component Responsibilities

### CodingAgent
- Entry point for task execution
- Coordinates between environment and learning components
- Manages task execution flow
- Location: `ai_agent/core/agent.py`

### GitEnvironment
- Provides Git repository interface
- Handles file operations
- Tracks repository state
- Manages Git commands execution
- Location: `ai_agent/environment/git_env.py`

### TrajectoryManager
- Stores execution trajectories
- Handles trajectory serialization
- Implements hybrid retrieval system:
  - BM25 inverted index for first-pass retrieval
  - Embedding-based re-ranking of top candidates
  - State similarity scoring
- Manages trajectory storage on disk
- Location: `ai_agent/data/trajectory_manager.py`

### SelfLearner
- Generates tasks from documentation
- Performs backward construction
- Computes embeddings for similarity matching
- Integrates with OpenAI's API
- Location: `ai_agent/core/learner.py`

## Data Structures

### Trajectory
- Represents a single execution trace
- Contains:
  - Original instruction
  - Sequence of actions
  - Observations from actions
  - Final environment state
  - Pre-computed embedding for similarity matching
  - BM25 tokens for inverted index
- Used for learning and retrieval

## Storage

### Trajectory Storage
Trajectories are stored in JSON format at `~/.ai-agent/trajectories/` with the following structure:
```json
{
  "instruction": "string",
  "actions": [{}],
  "observations": [{}],
  "final_state": {},
  "embedding": []
}
```

### Search Index Structure
The system maintains two types of indices:
1. **BM25 Inverted Index**
   - Maps tokens to trajectory IDs
   - Built using rank-bm25 library
   - Rebuilt when new trajectories are added
   - Optimized for fast keyword search
   - Primary retrieval mechanism for most queries

2. **Embedding Store**
   - Stores pre-computed embeddings for each trajectory
   - Used for semantic similarity during re-ranking
   - Computed using Sentence Transformers' all-MiniLM-L6-v2 model
   - No specialized vector database needed due to two-stage approach
   - Applied selectively based on query characteristics

### Retrieval Strategy

The system uses a smart hybrid approach combining BM25 and embedding-based re-ranking:

1. **Initial BM25 Retrieval**
   - Fast keyword-based filtering using inverted index
   - Always performed as first-pass retrieval
   - Highly efficient for exact matches and keyword queries
   - Returns configurable number of candidates (default 50-100)

2. **Conditional Re-ranking**
   The system decides whether to apply embedding re-ranking based on several heuristics:
   
   - **BM25 Score Analysis**
     - High BM25 scores → Skip re-ranking (exact matches found)
     - Low/scattered scores → Apply re-ranking (semantic matching needed)
   
   - **Query Characteristics**
     - Short/keyword queries → BM25 sufficient
     - Long/conceptual queries → Re-ranking beneficial
     - Function names/error messages → BM25 preferred
     - Natural language/synonyms → Re-ranking needed

   - **Resource Considerations**
     - System load
     - Response time requirements
     - Query complexity
     - Repository size

3. **Re-ranking Process** (when applied)
   - Compute query embedding
   - Fetch pre-computed embeddings for BM25 candidates
   - Calculate similarity scores:
     - Semantic similarity (40%)
     - BM25 score (30%)
     - Repository state (30%)
   - Re-sort candidates by combined score

This adaptive approach ensures:
- Fast responses for straightforward queries
- Semantic understanding when needed
- Efficient resource utilization
- Balance between speed and accuracy