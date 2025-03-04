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
    
    # Test inefficient trajectory with duplicates
    inefficient_trajectory = create_test_trajectory(
        actions=[
            {'type': 'create_file', 'file': 'test.py'},
            {'type': 'create_file', 'file': 'test.py'},
            {'type': 'write_test', 'file': 'test.py'},
            {'type': 'write_test', 'file': 'test.py'}
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

def test_complexity_scoring():
    metrics = QualityMetrics()
    
    # Test simple trajectory
    simple_trajectory = create_test_trajectory(
        actions=[
            {'type': 'edit_file', 'file': 'simple.py'},
            {'type': 'run_tests'}
        ],
        observations=[
            {'status': 'success'},
            {'status': 'success'}
        ]
    )
    
    score = metrics.assess_trajectory(simple_trajectory)
    assert score.complexity_score >= 0.8  # Should be high for simple changes
    
    # Test complex trajectory
    complex_trajectory = create_test_trajectory(
        actions=[
            {'type': 'create_file', 'file': 'a.py'},
            {'type': 'edit_file', 'file': 'b.py'},
            {'type': 'move_file', 'from': 'c.py', 'to': 'd.py'},
            {'type': 'run_tests'},
            {'type': 'git_commit'},
            {'type': 'git_push'}
        ],
        observations=[{'status': 'success'}] * 6
    )
    
    score = metrics.assess_trajectory(complex_trajectory)
    assert score.complexity_score < 0.5  # Should penalize complexity

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
    
    # Test similar pattern recognition
    similar_pattern = create_test_trajectory(
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
    
    metrics.update_patterns(similar_pattern)
    
    # Pattern should be recognized and have high success rate
    pattern_key = tuple(a.get('type', '') for a in good_pattern.actions)
    assert pattern_key in metrics.patterns
    pattern_stats = metrics.patterns[pattern_key]
    assert pattern_stats['success_count'] == 2
    assert pattern_stats['count'] == 2