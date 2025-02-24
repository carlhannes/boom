import pytest
import numpy as np
from pathlib import Path
from ai_agent.data.trajectory_manager import TrajectoryManager, Trajectory

class MockLearner:
    def compute_embedding(self, text: str) -> np.ndarray:
        """Mock embedding computation for testing"""
        return np.ones(10)  # Return a simple embedding vector

def test_hybrid_retrieval(tmp_path):
    manager = TrajectoryManager(str(tmp_path), learner=MockLearner())
    
    # Create test trajectories
    trajectories = [
        Trajectory(
            instruction="Add input validation to the login form",
            actions=[{"type": "edit_file", "path": "login.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["login.py"], "git_status": {"modified": ["login.py"]}}
        ),
        Trajectory(
            instruction="Implement user authentication system",
            actions=[{"type": "edit_file", "path": "auth.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["auth.py"], "git_status": {"modified": ["auth.py"]}}
        ),
        Trajectory(
            instruction="Fix bug in data validation",
            actions=[{"type": "edit_file", "path": "validator.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["validator.py"], "git_status": {"modified": ["validator.py"]}}
        )
    ]
    
    # Store test trajectories
    for traj in trajectories:
        manager.store_trajectory(traj)
    
    # Test retrieval with a query that should match validation-related trajectories
    current_state = {"files": ["login.py", "validator.py"], "git_status": {"modified": []}}
    results = manager.retrieve_similar_trajectories(
        current_state=current_state,
        instruction="How to validate user input?",
        limit=2
    )
    
    # Verify results
    assert len(results) == 2
    # First results should be validation-related trajectories
    assert any("validation" in r.instruction.lower() for r in results)
    
def test_bm25_index_rebuild(tmp_path):
    manager = TrajectoryManager(str(tmp_path), learner=MockLearner())
    
    # Test empty index
    assert manager.bm25_index is None
    
    # Add trajectory and verify index is built
    traj = Trajectory(
        instruction="Test trajectory",
        actions=[],
        observations=[],
        final_state={}
    )
    manager.store_trajectory(traj)
    
    assert manager.bm25_index is not None
    
def test_simple_tokenize():
    from ai_agent.data.trajectory_manager import simple_tokenize
    
    text = "Add input validation to login.py!"
    tokens = simple_tokenize(text)
    
    assert all(t.islower() for t in tokens)
    assert "validation" in tokens
    assert "login.py" in tokens  # Now preserves file extension
    assert "add" in tokens