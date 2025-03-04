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
    
    # Create test trajectories with quality metrics
    metrics = type('Metrics', (), {'success_rate': 0.9})
    trajectories = [
        Trajectory(
            instruction="Add input validation to the login form",
            actions=[{"type": "edit_file", "path": "login.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["login.py"], "git_status": {"modified": ["login.py"]}},
            quality_metrics=metrics
        ),
        Trajectory(
            instruction="Implement user authentication system",
            actions=[{"type": "edit_file", "path": "auth.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["auth.py"], "git_status": {"modified": ["auth.py"]}},
            quality_metrics=metrics
        )
    ]
    
    # Store test trajectories
    for traj in trajectories:
        manager.store_trajectory(traj)
        
    # Test retrieval
    results = manager.retrieve_similar_trajectories(
        current_state={"files": ["login.py"], "git_status": {"modified": []}},
        instruction="How to validate user input?",
        limit=2
    )
    
    assert len(results) > 0
    assert "validation" in results[0].instruction.lower()

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
    
    # Add some test patterns
    trajectory_manager_with_data.patterns = {
        'test': {
            'type': 'test',
            'actions': [{'type': 'create_file', 'path': 'tests/test_*.py'}],
            'success_rate': 0.9
        },
        'create': {
            'type': 'create',
            'actions': [{'type': 'create_file', 'path': 'src/*.py'}],
            'success_rate': 0.9
        },
        'modify': {
            'type': 'modify',
            'actions': [{'type': 'edit_file', 'path': 'src/*.py'}],
            'success_rate': 0.8
        }
    }
    
    # Test create pattern
    create_matches = trajectory_manager_with_data._find_matching_patterns(
        "Create new validation module",
        current_state
    )
    
    assert len(create_matches) > 0
    assert create_matches[0]['type'] == 'create'
    
    # Test modify pattern
    modify_matches = trajectory_manager_with_data._find_matching_patterns(
        "Update input validation",
        current_state
    )
    assert len(modify_matches) > 0
    assert modify_matches[0]['type'] == 'modify'  # Changed to check for 'type' instead of 'semantic_type'
    
def test_pattern_based_retrieval(tmp_path):
    """Test enhanced retrieval with pattern matching"""
    manager = TrajectoryManager(str(tmp_path), learner=MockLearner())
    metrics = type('Metrics', (), {'success_rate': 0.9})
    
    # Create trajectories with clear patterns
    test_trajectories = [
        Trajectory(
            instruction="Add unit tests",
            actions=[
                {"type": "create_file", "path": "test_app.py"},
                {"type": "run_tests"}
            ],
            observations=[
                {"status": "success"},
                {"status": "success"}
            ],
            final_state={"files": ["test_app.py"]},
            quality_metrics=metrics
        ),
        Trajectory(
            instruction="Implement feature",
            actions=[
                {"type": "edit_file", "path": "app.py"},
                {"type": "run_tests"}
            ],
            observations=[
                {"status": "success"},
                {"status": "success"}
            ],
            final_state={"files": ["app.py"]},
            quality_metrics=metrics
        )
    ]
    
    # Store trajectories
    for traj in test_trajectories:
        manager.store_trajectory(traj)
        
    # Test retrieval with pattern matching
    results = manager.retrieve_similar_trajectories(
        current_state={"files": ["app.py", "test_app.py"]},
        instruction="Create new test module",
        limit=5
    )
    
    assert len(results) > 0
    first_result = results[0]
    assert "test" in first_result.instruction.lower()
    assert first_result.actions[0]["type"] == "create_file"

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

from ai_agent.data.quality_metrics import QualityScore

def test_pattern_priority(trajectory_manager_with_data):
    """Test pattern prioritization"""
    current_state = {
        'files': ['src/validation.py'],
        'git_status': {'modified': []},
        'frameworks': ['python'],
        'languages': ['python']
    }

    # Add test patterns
    test_pattern = {
        'type': 'test',
        'actions': [{'type': 'create_file', 'path': 'tests/test_*.py'}],
        'success_rate': 0.9
    }
    trajectory_manager_with_data.patterns = {'test': test_pattern}

    # Add test trajectory
    test_trajectory = Trajectory(
        instruction="Create test module",
        actions=[{'type': 'create_file', 'path': 'tests/test_validation.py'}],
        observations=[{'status': 'success'}],
        final_state={},
        quality_metrics=QualityScore(success_rate=0.9, coverage_score=0.8)
    )
    
    trajectory_manager_with_data.store_trajectory(test_trajectory)

    # Test retrieving patterns
    matches = trajectory_manager_with_data._find_matching_patterns(
        "Add unit tests for validation",
        current_state
    )

    assert len(matches) > 0
    assert matches[0]['type'] == 'test'

def create_test_trajectory(instruction, actions, observations, final_state=None, quality_metrics=None):
    """Helper to create test trajectories with quality metrics"""
    trajectory = Trajectory(
        instruction=instruction,
        actions=actions,
        observations=observations,
        final_state=final_state or {'files': [], 'git_status': {}}
    )
    
    if quality_metrics:
        trajectory.quality_metrics = quality_metrics
    
    return trajectory

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
        ],
        quality_metrics={'success_rate': 1.0, 'confidence': 0.9}
    )
    
    # Store and verify
    tm.store_trajectory(good_trajectory)
    assert len(tm.trajectories) == 1

def test_quality_maintenance():
    tm = TrajectoryManager("test_storage")

    # Add mix of high and low quality trajectories
    trajectories = [
        create_test_trajectory(
            "Good example",
            [{'type': 'good_action'}],
            [{'status': 'success'}],
            quality_metrics={'success_rate': 1.0}
        ),
        create_test_trajectory(
            "Bad example",
            [{'type': 'bad_action'}],
            [{'status': 'error'}],
            quality_metrics={'success_rate': 0.2}
        ),
        create_test_trajectory(
            "Another good example",
            [{'type': 'good_action'}],
            [{'status': 'success'}],
            quality_metrics={'success_rate': 0.9}
        )
    ]
    
    # Store all trajectories
    for t in trajectories:
        tm.store_trajectory(t)
        
    # Verify low quality trajectories were filtered
    high_quality = [t for t in tm.trajectories if t.quality_metrics.get('success_rate', 0) >= 0.8]
    assert len(high_quality) == 2

def test_quality_filtered_retrieval(tmp_path):
    """Test retrieval with quality filtering"""
    manager = TrajectoryManager(str(tmp_path))
    
    # Create trajectories with varying quality
    high_quality = type('Metrics', (), {'success_rate': 0.9})
    low_quality = type('Metrics', (), {'success_rate': 0.5})
    
    trajectories = [
        Trajectory(
            instruction="High quality example",
            actions=[{"type": "edit_file", "path": "test.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["test.py"]},
            quality_metrics=high_quality
        ),
        Trajectory(
            instruction="Low quality example",
            actions=[{"type": "edit_file", "path": "test.py"}],
            observations=[{"status": "success"}],
            final_state={"files": ["test.py"]},
            quality_metrics=low_quality
        )
    ]
    
    # Store trajectories
    for traj in trajectories:
        manager.store_trajectory(traj)
    
    # Override patterns to ensure BM25 retrieval is used instead of pattern matching
    manager.patterns = {}
        
    # Test retrieval with quality filter
    results = manager.retrieve_similar_trajectories(
        current_state={"files": ["test.py"]},
        instruction="Test action",
        limit=2,
        min_quality=0.8
    )
    
    assert len(results) == 1
    assert results[0].instruction == "High quality example"