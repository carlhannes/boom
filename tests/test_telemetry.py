import pytest
from pathlib import Path
from datetime import datetime, timedelta
from ai_agent.core.telemetry import TelemetryManager, LearningMetrics

def test_telemetry_recording(tmp_path):
    # Setup
    telemetry = TelemetryManager(str(tmp_path))
    
    # Create test trajectory
    trajectory = type('Trajectory', (), {
        'compute_quality_metrics': lambda self: type('Metrics', (), {'success_rate': 0.9})(),
        'state_changes': [
            type('StateChange', (), {'impact': 0.5})(),
            type('StateChange', (), {'impact': 0.7})()
        ]
    })()
    
    # Record trajectory execution
    telemetry.record_trajectory(trajectory, "test_pattern", was_exploration=True)
    
    # Verify metrics were recorded
    assert len(telemetry.metrics_history) == 1
    assert telemetry.pattern_stats["test_pattern"]["uses"] == 1
    assert telemetry.pattern_stats["test_pattern"]["successes"] == 1

def test_metrics_persistence(tmp_path):
    telemetry = TelemetryManager(str(tmp_path))
    
    # Add some test metrics
    metrics = LearningMetrics(
        timestamp=datetime.now(),
        total_trajectories=10,
        successful_trajectories=8,
        pattern_confidence=0.8,
        average_impact=0.5,
        exploration_rate=0.3
    )
    telemetry.metrics_history.append(metrics)
    telemetry._save_history()
    
    # Create new manager and verify metrics loaded
    new_telemetry = TelemetryManager(str(tmp_path))
    assert len(new_telemetry.metrics_history) == 1
    loaded = new_telemetry.metrics_history[0]
    assert loaded.total_trajectories == 10
    assert loaded.successful_trajectories == 8

def test_learning_summary(tmp_path):
    telemetry = TelemetryManager(str(tmp_path))
    
    # Add test patterns
    patterns = ["pattern1", "pattern2"]
    for pattern in patterns:
        telemetry.pattern_stats[pattern] = {
            'uses': 5,
            'successes': 4,
            'impact_sum': 2.0
        }
    
    # Get summary
    summary = telemetry.get_learning_summary()
    
    # Verify summary stats
    assert summary['total_patterns'] == 2
    assert summary['total_trajectories'] == 10
    assert summary['success_rate'] == 0.8

def test_pattern_performance(tmp_path):
    telemetry = TelemetryManager(str(tmp_path))
    
    # Record some pattern executions
    pattern = "test_pattern"
    telemetry.pattern_stats[pattern] = {
        'uses': 10,
        'successes': 8,
        'impact_sum': 4.0
    }
    
    # Get performance metrics
    performance = telemetry.get_pattern_performance()
    
    # Verify pattern metrics
    assert pattern in performance
    assert performance[pattern]['success_rate'] == 0.8
    assert performance[pattern]['average_impact'] == 0.4
    assert performance[pattern]['usage_count'] == 10

def test_exploration_rate_calculation(tmp_path):
    telemetry = TelemetryManager(str(tmp_path))
    
    # Add historical metrics with varying exploration rates
    base_time = datetime.now()
    for i in range(10):
        metrics = LearningMetrics(
            timestamp=base_time + timedelta(minutes=i),
            total_trajectories=i+1,
            successful_trajectories=i,
            pattern_confidence=0.8,
            average_impact=0.5,
            exploration_rate=1.0 - (i/10)  # Decreasing exploration rate
        )
        telemetry.metrics_history.append(metrics)
    
    # Calculate current exploration rate
    rate = telemetry._calculate_exploration_rate()
    
    # Should be average of historical rates
    assert 0 < rate < 1
    assert isinstance(rate, float)