import pytest
from ai_agent.data.quality_metrics import QualityMetrics, QualityScore

def create_test_trajectory(actions, observations, instruction="test"):
    """Helper to create test trajectories"""
    return type('Trajectory', (), {
        'actions': actions,
        'observations': observations,
        'instruction': instruction
    })

def test_quality_score_calculation():
    metrics = QualityMetrics()
    
    # Test successful trajectory
    good_trajectory = create_test_trajectory(
        actions=[
            {'type': 'create_file', 'file': 'test.py'},
            {'type': 'write_test', 'file': 'test.py'},
            {'type': 'run_tests'}
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'}
        ],
        instruction="Create and run tests"
    )
    
    score = metrics.compute_trajectory_quality(good_trajectory, good_trajectory.instruction)
    assert score.success_rate == 1.0
    assert score.total_score >= 0.8  # High quality trajectory
    
    # Test inefficient trajectory
    inefficient_trajectory = create_test_trajectory(
        actions=[
            {'type': 'create_file', 'file': 'test.py'},
            {'type': 'create_file', 'file': 'test.py'},  # Redundant
            {'type': 'write_test', 'file': 'test.py'},
            {'type': 'write_test', 'file': 'test.py'},  # Redundant
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'}
        ]
    )
    
    score = metrics.compute_trajectory_quality(inefficient_trajectory, inefficient_trajectory.instruction)
    assert score.efficiency < 0.7  # Should penalize redundant actions

def test_safety_scoring():
    metrics = QualityMetrics()
    
    # Test risky actions
    risky_trajectory = create_test_trajectory(
        actions=[
            {'type': 'delete_file', 'file': 'important.py'},
            {'type': 'drop_table', 'table': 'users'}
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'}
        ]
    )
    
    score = metrics.compute_trajectory_quality(risky_trajectory, risky_trajectory.instruction)
    assert score.safety < 0.5  # Should flag risky operations

def test_relevance_scoring():
    metrics = QualityMetrics()
    
    # Test relevant actions
    relevant_trajectory = create_test_trajectory(
        actions=[
            {'type': 'create_test', 'file': 'user_test.py'},
            {'type': 'write_test', 'file': 'user_test.py'}
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'}
        ],
        instruction="Create user tests"
    )
    
    score = metrics.compute_trajectory_quality(relevant_trajectory, relevant_trajectory.instruction)
    assert score.relevance >= 0.8  # Actions match instruction terms
    
    # Test irrelevant actions
    irrelevant_trajectory = create_test_trajectory(
        actions=[
            {'type': 'update_docs', 'file': 'readme.md'}
        ],
        observations=[
            {'status': 'success'}
        ],
        instruction="Fix database connection"
    )
    
    score = metrics.compute_trajectory_quality(irrelevant_trajectory, irrelevant_trajectory.instruction)
    assert score.relevance < 0.5  # Actions don't match instruction

def test_pattern_learning():
    metrics = QualityMetrics()
    
    # Add some successful patterns
    good_pattern = create_test_trajectory(
        actions=[
            {'type': 'create_file'},
            {'type': 'write_code'},
            {'type': 'run_tests'}
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'}
        ]
    )
    
    metrics.update_patterns(good_pattern)
    
    # Test consistency scoring
    similar_trajectory = create_test_trajectory(
        actions=[
            {'type': 'create_file'},
            {'type': 'write_code'},
            {'type': 'run_tests'}
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'}
        ]
    )
    
    score = metrics.compute_trajectory_quality(similar_trajectory, "test")
    assert score.consistency == 1.0  # Matches known good pattern

def test_filtering():
    metrics = QualityMetrics(min_quality_threshold=0.7)
    
    # Test low quality trajectory
    bad_trajectory = create_test_trajectory(
        actions=[
            {'type': 'delete_file', 'file': '*'},
            {'type': 'delete_file', 'file': '*'}  # Redundant and dangerous
        ],
        observations=[
            {'status': 'error'},
            {'status': 'error'}
        ]
    )
    
    score = metrics.compute_trajectory_quality(bad_trajectory, bad_trajectory.instruction)
    assert metrics.should_filter_trajectory(score)  # Should be filtered out