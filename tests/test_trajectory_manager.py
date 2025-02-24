import pytest
import numpy as np
from pathlib import Path
from ai_agent.data.trajectory_manager import TrajectoryManager, Trajectory
from ai_agent.data.sequence import ActionSequence, SequencePattern

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

@pytest.fixture
def trajectory_manager_with_data(tmp_path):
    """Create a trajectory manager with test data"""
    manager = TrajectoryManager(str(tmp_path))
    
    # Add test trajectories for different patterns
    create_trajectory = Trajectory(
        instruction="Create new validation module",
        actions=[
            {"type": "create_file", "path": "src/validation.py"},
            {"type": "edit_file", "path": "src/validation.py", "content": "def validate(): pass"},
            {"type": "run_tests", "path": "tests/"}
        ],
        observations=[
            {"status": "success", "state": {}},
            {"status": "success", "state": {}},
            {"status": "success", "state": {}}
        ],
        final_state={"files": ["src/validation.py"], "git_status": {"modified": []}}
    )
    
    modify_trajectory = Trajectory(
        instruction="Update input validation",
        actions=[
            {"type": "edit_file", "path": "src/validation.py", "content": "def validate_input(): pass"},
            {"type": "run_tests", "path": "tests/"}
        ],
        observations=[
            {"status": "success", "state": {}},
            {"status": "success", "state": {}}
        ],
        final_state={"files": ["src/validation.py"], "git_status": {"modified": ["src/validation.py"]}}
    )
    
    test_trajectory = Trajectory(
        instruction="Add validation tests",
        actions=[
            {"type": "create_file", "path": "tests/test_validation.py"},
            {"type": "edit_file", "path": "tests/test_validation.py", "content": "def test_validate(): pass"},
            {"type": "run_tests", "path": "tests/"}
        ],
        observations=[
            {"status": "success", "state": {}},
            {"status": "success", "state": {}},
            {"status": "success", "state": {}}
        ],
        final_state={"files": ["tests/test_validation.py"], "git_status": {"modified": []}}
    )
    
    # Store test trajectories
    manager.store_trajectory(create_trajectory)
    manager.store_trajectory(modify_trajectory)
    manager.store_trajectory(test_trajectory)
    
    return manager

def test_extract_action_patterns(trajectory_manager_with_data):
    """Test extraction of common action patterns from trajectories"""
    patterns = trajectory_manager_with_data._extract_action_patterns()
    
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
    
    # Check pattern types
    for pattern_type, pattern in patterns.items():
        assert isinstance(pattern, SequencePattern)
        assert pattern.semantic_type == pattern_type
        assert pattern.frequency >= 1
        assert 0 <= pattern.avg_success_rate <= 1
        
def test_find_matching_patterns(trajectory_manager_with_data):
    """Test finding patterns that match query and state"""
    current_state = {
        'files': ['src/app.py', 'tests/test_app.py'],
        'git_status': {'modified': []},
        'frameworks': ['python'],
        'languages': ['python'],
        'patterns': ['testing']
    }
    
    # Test create pattern
    create_matches = trajectory_manager_with_data._find_matching_patterns(
        "Create new validation module",
        current_state
    )
    assert len(create_matches) > 0
    assert any(seq.semantic_type == 'create' for seq in create_matches)
    
    # Test modify pattern
    modify_matches = trajectory_manager_with_data._find_matching_patterns(
        "Update input validation",
        current_state
    )
    assert len(modify_matches) > 0
    assert any(seq.semantic_type == 'modify' for seq in modify_matches)
    
def test_pattern_based_retrieval(trajectory_manager_with_data):
    """Test enhanced retrieval with pattern matching"""
    current_state = {
        'files': ['src/app.py', 'tests/test_app.py'],
        'git_status': {'modified': []},
        'frameworks': ['python'],
        'languages': ['python'],
        'patterns': ['testing']
    }
    
    # Create a query that should match patterns
    trajectories = trajectory_manager_with_data.retrieve_similar_trajectories(
        current_state=current_state,
        instruction="Create new test module",
        limit=5
    )
    
    assert len(trajectories) > 0
    # First result should be from pattern matching for "create" type
    first_sequence = ActionSequence.from_trajectory(trajectories[0])
    assert first_sequence.semantic_type == 'create'
    
def test_hybrid_retrieval_ranking(trajectory_manager_with_data):
    """Test that pattern matches are prioritized over standard retrieval"""
    current_state = {
        'files': ['src/app.py'],
        'git_status': {'modified': []},
        'frameworks': ['python'],
        'languages': ['python']
    }
    
    # Should trigger both pattern and standard matching
    trajectories = trajectory_manager_with_data.retrieve_similar_trajectories(
        current_state=current_state,
        instruction="Update validation and add tests",
        limit=5
    )
    
    assert len(trajectories) > 0
    # First results should be pattern matches
    first_sequence = ActionSequence.from_trajectory(trajectories[0])
    assert first_sequence.success_rate > 0.8

def test_pattern_priority(trajectory_manager_with_data):
    """Test that pattern matches are prioritized over standard retrieval"""
    current_state = {
        'files': ['src/validation.py'],
        'git_status': {'modified': []},
        'frameworks': ['python'],
        'languages': ['python']
    }
    
    trajectories = trajectory_manager_with_data.retrieve_similar_trajectories(
        current_state=current_state,
        instruction="Add tests for validation module",
        limit=5
    )
    
    assert len(trajectories) > 0
    # Should prioritize test pattern trajectory
    first_sequence = ActionSequence.from_trajectory(trajectories[0])
    assert 'test' in first_sequence.semantic_type.lower()
    assert first_sequence.success_rate > 0.8