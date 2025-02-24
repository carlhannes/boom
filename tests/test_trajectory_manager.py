import pytest
import numpy as np
from pathlib import Path
from ai_agent.data.trajectory_manager import TrajectoryManager, Trajectory
from ai_agent.data.sequence import ActionSequence, SequencePattern
from ai_agent.environment.git_env import GitEnvironment
from ai_agent.data.quality_metrics import QualityScore
from tests.test_learner import MockLearner

@pytest.fixture
def mock_learner():
    return MockLearner()

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
def trajectory_manager_with_data(tmp_path, mock_learner):
    """Create a trajectory manager with test data"""
    manager = TrajectoryManager(str(tmp_path), learner=mock_learner)
    
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

def create_test_trajectory(instruction: str, actions: list, observations: list, final_state: dict):
    """Helper to create test trajectories"""
    return type('Trajectory', (), {
        'instruction': instruction,
        'actions': actions,
        'observations': observations,
        'final_state': final_state,
        'compute_quality_metrics': lambda self: type('Metrics', (), {'success_rate': 1.0})()
    })

def test_error_pattern_extraction():
    tm = TrajectoryManager("test_storage")
    
    # Create a test trajectory with error and recovery
    trajectory = create_test_trajectory(
        instruction="Add a new feature",
        actions=[
            {'type': 'edit_file', 'file': 'test.py'},
            {'type': 'run_tests'},  # Causes error
            {'type': 'fix_imports'},  # Recovery action
            {'type': 'run_tests'}  # Success
        ],
        observations=[
            {'status': 'success'},
            {'status': 'error', 'error': 'ImportError'},
            {'status': 'success'},
            {'status': 'success'}
        ],
        final_state={'files': ['test.py']}
    )
    
    # Store trajectory and extract patterns
    tm.store_trajectory(trajectory)
    patterns = tm.extract_error_patterns()
    
    assert 'ImportError' in patterns
    assert len(patterns['ImportError']) == 1
    
    # Verify recovery sequence
    recovery_seq = patterns['ImportError'][0]
    assert isinstance(recovery_seq, ActionSequence)
    assert len(recovery_seq.steps) == 2  # fix_imports + run_tests

def test_environment_pattern_integration(tmp_path):
    tm = TrajectoryManager("test_storage")
    env = GitEnvironment(str(tmp_path))
    
    # Create and store a trajectory with successful error recovery
    trajectory = create_test_trajectory(
        instruction="Fix merge conflict",
        actions=[
            {'type': 'merge'},
            {'type': 'resolve_conflict', 'file': 'test.py'},
            {'type': 'commit'}
        ],
        observations=[
            {'status': 'error', 'error': 'MergeConflict'},
            {'status': 'success'},
            {'status': 'success'}
        ],
        final_state={'branch': 'main'}
    )
    
    tm.store_trajectory(trajectory)
    
    # Update environment patterns
    tm.update_environment_patterns(env)
    
    # Verify pattern was transferred to environment
    assert 'MergeConflict' in env.error_patterns
    pattern = env.error_patterns['MergeConflict'][0]
    assert pattern.success_rate > 0

def test_quality_filtered_storage():
    tm = TrajectoryManager("test_storage")
    
    # Try storing a high-quality trajectory
    good_trajectory = create_test_trajectory(
        "Add tests",
        [
            {'type': 'create_file', 'file': 'test.py'},
            {'type': 'write_test', 'file': 'test.py'},
            {'type': 'run_tests'}
        ],
        [
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'}
        ]
    )
    
    stored = tm.store_trajectory(good_trajectory)
    assert stored  # Should store high-quality trajectory
    assert len(tm.trajectories) == 1
    
    # Try storing a low-quality trajectory
    bad_trajectory = create_test_trajectory(
        "Delete files",
        [
            {'type': 'delete_file', 'file': '*'},
            {'type': 'delete_file', 'file': '*'}
        ],
        [
            {'status': 'error'},
            {'status': 'error'}
        ],
        success_rate=0.0
    )
    
    stored = tm.store_trajectory(bad_trajectory)
    assert not stored  # Should reject low-quality trajectory
    assert len(tm.trajectories) == 1

def test_quality_maintenance():
    tm = TrajectoryManager("test_storage")
    
    # Add mix of high and low quality trajectories
    trajectories = [
        create_test_trajectory(
            "Good example",
            [{'type': 'good_action'}],
            [{'status': 'success'}],
            success_rate=1.0
        ),
        create_test_trajectory(
            "Bad example",
            [{'type': 'bad_action'}],
            [{'status': 'error'}],
            success_rate=0.2
        ),
        create_test_trajectory(
            "Another good example",
            [{'type': 'good_action'}],
            [{'status': 'success'}],
            success_rate=0.9
        )
    ]
    
    for t in trajectories:
        tm.store_trajectory(t)
    
    # Run quality maintenance
    removed = tm.maintain_quality(min_score=0.7)
    
    assert removed == 1  # Should remove one low-quality trajectory
    assert len(tm.trajectories) == 2  # Should keep two good trajectories

def test_quality_filtered_retrieval():
    tm = TrajectoryManager("test_storage")
    
    # Add trajectories with varying quality
    trajectories = [
        create_test_trajectory(
            "High quality example",
            [{'type': 'good_action', 'file': 'test.py'}],
            [{'status': 'success'}],
            success_rate=1.0
        ),
        create_test_trajectory(
            "Medium quality example",
            [{'type': 'ok_action', 'file': 'test.py'}],
            [{'status': 'success'}, {'status': 'error'}],
            success_rate=0.5
        ),
        create_test_trajectory(
            "Another high quality example",
            [{'type': 'good_action', 'file': 'other.py'}],
            [{'status': 'success'}],
            success_rate=0.9
        )
    ]
    
    for t in trajectories:
        tm.store_trajectory(t)
    
    # Test quality-filtered retrieval
    similar = tm.retrieve_similar_trajectories(
        current_state={'files': ['test.py']},
        instruction="Run good action",
        min_quality=0.7
    )
    
    assert len(similar) == 2  # Should only return high-quality matches
    assert all(t.compute_quality_metrics().success_rate >= 0.7 for t in similar)