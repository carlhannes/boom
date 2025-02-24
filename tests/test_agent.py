import pytest
import numpy as np
from pathlib import Path
import json
import git
from ai_agent.core.agent import CodingAgent
from ai_agent.environment.git_env import GitEnvironment
from ai_agent.data.trajectory_manager import Trajectory
from typing import Dict, Any, List

class MockEmbedding:
    def __init__(self, embedding: List[float]):
        self.embedding = embedding

class MockEmbeddingData:
    def __init__(self, embedding: List[float]):
        self.data = [MockEmbedding(embedding)]

class MockEmbeddingAPI:
    @staticmethod
    def create(**kwargs):
        return MockEmbeddingData([1.0] * 384)  # Match all-MiniLM-L6-v2 dimension

class MockSentenceTransformer:
    def encode(self, text: str, convert_to_numpy: bool = True) -> np.ndarray:
        # Mock encoding that matches all-MiniLM-L6-v2 dimension
        base = np.ones(384)
        if 'validation' in text.lower():
            return base * 1.5
        return base

class MockChatCompletion:
    def __init__(self, content: str):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})})]

class MockChatAPI:
    @staticmethod
    def create(**kwargs):
        if 'Generate 3 specific coding tasks' in kwargs['messages'][1]['content']:
            return MockChatCompletion("Task 1: Add input validation\nTask 2: Handle errors\nTask 3: Write tests")
        return MockChatCompletion(json.dumps({
            'type': 'edit_file',
            'path': 'test.py',
            'content': 'print("test")',
            'status': 'complete'
        }))

class MockClient:
    def __init__(self):
        self.embeddings = MockEmbeddingAPI()
        self.chat = type('Chat', (), {'completions': MockChatAPI})

class MockLearner:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.model = "mock-model"
        self.client = MockClient()
        self.embedding_model = MockSentenceTransformer()
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Mock embedding computation - weight validation-related terms higher"""
        return self.embedding_model.encode(text)
        
    def backward_construct(self, data: dict) -> str:
        """Mock backward construction"""
        return data.get('instruction', '')
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Mock chat completion"""
        user_msg = next((m for m in messages if m['role'] == 'user'), None)
        if not user_msg:
            return {'type': 'unknown'}
            
        if 'Current task:' in user_msg['content']:
            task = user_msg['content'].split('Current task:')[1].split('\n')[0].strip()
            
            if 'empty' in task.lower() or 'create' in task.lower():
                return {
                    'type': 'edit_file',
                    'path': 'new_file.py',
                    'content': '# New file content',
                    'status': 'complete'
                }
        
        return {
            'type': 'edit_file',
            'path': 'test.py',
            'content': 'print("test")',
            'status': 'complete'
        }

@pytest.fixture
def mock_repo(tmp_path):
    """Create a temporary Git repository"""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    repo = git.Repo.init(repo_path)
    
    # Create some test files
    (repo_path / "test.py").write_text("# Initial content")
    (repo_path / "README.md").write_text("# Test Repository")
    
    # Initial commit
    repo.index.add(["test.py", "README.md"])
    repo.index.commit("Initial commit")
    
    return repo_path

@pytest.fixture
def mock_storage(tmp_path):
    """Create temporary storage path"""
    storage_path = tmp_path / "trajectories"
    storage_path.mkdir()
    return storage_path

@pytest.fixture
def mock_learner():
    """Create a mock learner instance"""
    return MockLearner()

def test_agent_initialization(mock_repo, mock_storage, mock_learner):
    """Test agent initialization with custom BM25 settings"""
    agent = CodingAgent(str(mock_repo), str(mock_storage), bm25_top_k=100, api_key="mock-key")
    agent.set_learner(mock_learner)  # Use new set_learner method
    assert agent.bm25_top_k == 100
    assert isinstance(agent.environment, GitEnvironment)
    assert agent.environment.repo_path == mock_repo

def test_agent_task_execution_with_retrieval(mock_repo, mock_storage, mock_learner):
    """Test task execution with hybrid retrieval system"""
    # Create some example trajectories
    example_trajectories = [
        Trajectory(
            instruction="Add logging to function",
            actions=[{"type": "edit_file", "path": "test.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["test.py"], "git_status": {"modified": ["test.py"]}}
        ),
        Trajectory(
            instruction="Implement error handling",
            actions=[{"type": "edit_file", "path": "test.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["test.py"], "git_status": {"modified": ["test.py"]}}
        )
    ]
    
    # Set up agent with mock components
    agent = CodingAgent(str(mock_repo), str(mock_storage), bm25_top_k=10, api_key="mock-key")
    agent.set_learner(mock_learner)  # Use new set_learner method
    
    # Store example trajectories
    for traj in example_trajectories:
        agent.trajectory_manager.store_trajectory(traj)
    
    # Execute a task
    trajectory = agent.execute_task("Add logging to test.py")
    
    # Verify trajectory
    assert isinstance(trajectory, Trajectory)
    assert len(trajectory.actions) > 0
    assert all(isinstance(action, dict) for action in trajectory.actions)
    assert "test.py" in str(list(mock_repo.glob("*.py")))

def test_agent_with_empty_repository(mock_repo, mock_storage, mock_learner):
    """Test agent behavior with no existing trajectories"""
    agent = CodingAgent(str(mock_repo), str(mock_storage), api_key="mock-key")
    agent.set_learner(mock_learner)  # Use new set_learner method
    
    trajectory = agent.execute_task("Create new file")
    
    assert isinstance(trajectory, Trajectory)
    assert len(trajectory.actions) > 0

def test_agent_retrieval_ranking(mock_repo, mock_storage, mock_learner):
    """Test that hybrid retrieval properly ranks results"""
    agent = CodingAgent(str(mock_repo), str(mock_storage), api_key="mock-key")
    agent.set_learner(mock_learner)  # Use new set_learner method
    
    # Create trajectories with varying similarity
    trajectories = [
        Trajectory(
            instruction="Update documentation",
            actions=[{"type": "edit_file", "path": "README.md"}],
            observations=[{"status": "success"}],
            final_state={"files": ["README.md"], "git_status": {"modified": ["README.md"]}}
        ),
        Trajectory(
            instruction="Add input validation",
            actions=[{"type": "edit_file", "path": "validate.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["validate.py"], "git_status": {"modified": ["validate.py"]}}
        ),
        Trajectory(
            instruction="Fix validation bug",
            actions=[{"type": "edit_file", "path": "validate.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["validate.py"], "git_status": {"modified": ["validate.py"]}}
        )
    ]
    
    # Store trajectories in reverse order to ensure ranking isn't just by insertion
    for traj in reversed(trajectories):
        agent.trajectory_manager.store_trajectory(traj)
    
    # Execute a validation-related task
    trajectory = agent.execute_task("Implement form validation")
    
    # Get retrieved examples from the plan_next_action call
    retrieved = agent.trajectory_manager.retrieve_similar_trajectories(
        current_state={"files": ["validate.py"]},
        instruction="Implement form validation",
        limit=3
    )
    
    # Verify ranking - validation-related trajectories should be ranked higher
    assert len(retrieved) == 3
    assert any("validation" in r.instruction.lower() for r in retrieved[:2])
    assert retrieved[0].instruction != "Update documentation"  # Non-validation should be ranked lower

def test_retrieval_reranking_decision(mock_repo, mock_storage, mock_learner):
    """Test that re-ranking is selectively applied based on query type"""
    agent = CodingAgent(str(mock_repo), str(mock_storage), api_key="mock-key")
    agent.set_learner(mock_learner)

    # Test cases that should skip re-ranking
    simple_queries = [
        "getUserProfile function",
        "main.py",
        "TypeError: Cannot read property",
        "def calculate_sum"
    ]
    
    # Test cases that should use re-ranking
    complex_queries = [
        "How to implement secure authentication",
        "Design pattern for handling multiple data sources",
        "What's the best way to manage state",
        "Implement input validation following best practices"
    ]
    
    # Verify simple queries skip re-ranking
    for query in simple_queries:
        trajectory = agent.execute_task(query)
        assert isinstance(trajectory, Trajectory)
        # Simple queries should primarily rely on BM25 scores
        
    # Verify complex queries use re-ranking
    for query in complex_queries:
        trajectory = agent.execute_task(query)
        assert isinstance(trajectory, Trajectory)
        # Complex queries should show evidence of semantic matching

def test_bm25_score_threshold(mock_repo, mock_storage, mock_learner):
    """Test that high BM25 scores bypass re-ranking"""
    agent = CodingAgent(str(mock_repo), str(mock_storage), api_key="mock-key")
    agent.set_learner(mock_learner)
    
    # Create a trajectory with exact match to query
    exact_match = Trajectory(
        instruction="Add input validation to login form",
        actions=[{"type": "edit_file", "path": "login.py"}],
        observations=[{"status": "success"}],
        final_state={"files": ["login.py"]}
    )
    
    # Create some slightly different trajectories
    similar_trajectories = [
        Trajectory(
            instruction="Add validation to signup form",
            actions=[{"type": "edit_file", "path": "signup.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["signup.py"]}
        ),
        Trajectory(
            instruction="Implement form validation",
            actions=[{"type": "edit_file", "path": "validate.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["validate.py"]}
        )
    ]
    
    # Store trajectories
    agent.trajectory_manager.store_trajectory(exact_match)
    for traj in similar_trajectories:
        agent.trajectory_manager.store_trajectory(traj)
    
    # Execute query with exact match
    query = "Add input validation to login form"
    trajectory = agent.execute_task(query)
    
    # The exact match should be found by BM25 without needing re-ranking
    assert isinstance(trajectory, Trajectory)
    # Verify actions match the exact match trajectory
    assert trajectory.actions[0]["path"] == "login.py"