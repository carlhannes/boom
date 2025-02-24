from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

@dataclass
class LearningMetrics:
    """Metrics tracking learning progress"""
    timestamp: datetime
    total_trajectories: int
    successful_trajectories: int
    pattern_confidence: float
    average_impact: float
    exploration_rate: float

class TelemetryManager:
    """Manages learning telemetry and analytics"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.pattern_stats = {}  # Now initialized as empty dict
        self.total_executions = 0
        self.successful_executions = 0
        self.metrics_history: List[LearningMetrics] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Load existing metrics history"""
        metrics_file = Path(self.storage_path) / 'metrics_history.json'
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                    self.metrics_history = [
                        LearningMetrics(
                            timestamp=datetime.fromisoformat(m['timestamp']),
                            total_trajectories=m['total_trajectories'],
                            successful_trajectories=m['successful_trajectories'],
                            pattern_confidence=m['pattern_confidence'],
                            average_impact=m['average_impact'],
                            exploration_rate=m['exploration_rate']
                        )
                        for m in data
                    ]
            except Exception as e:
                print(f"Error loading metrics history: {e}")
    
    def record_trajectory(self,
                        trajectory: 'Trajectory',
                        pattern_key: str,
                        was_exploration: bool) -> None:
        """Record trajectory execution metrics"""
        success = trajectory.compute_quality_metrics().success_rate >= 0.8
        impact = np.mean([c.impact for c in getattr(trajectory, 'state_changes', [])])
        
        # Update pattern statistics
        if pattern_key not in self.pattern_stats:
            self.pattern_stats[pattern_key] = {
                'uses': 0,
                'successes': 0,
                'impact_sum': 0.0
            }
        self.pattern_stats[pattern_key]['uses'] += 1
        if success:
            self.pattern_stats[pattern_key]['successes'] += 1
            self.successful_executions += 1  # Track successful executions
        self.pattern_stats[pattern_key]['impact_sum'] += impact
        self.total_executions += 1  # Track total executions
        
        # Record overall metrics
        self._record_metrics(success, impact, was_exploration)
    
    def _record_metrics(self,
                       success: bool,
                       impact: float,
                       was_exploration: bool) -> None:
        """Record current learning metrics"""
        current_metrics = LearningMetrics(
            timestamp=datetime.now(),
            total_trajectories=sum(p['uses'] for p in self.pattern_stats.values()),
            successful_trajectories=sum(p['successes'] for p in self.pattern_stats.values()),
            pattern_confidence=self._calculate_confidence(),
            average_impact=self._calculate_impact(),
            exploration_rate=self._calculate_exploration_rate()
        )
        
        self.metrics_history.append(current_metrics)
        self._save_history()
    
    def _calculate_confidence(self) -> float:
        """Calculate overall pattern confidence"""
        if not self.pattern_stats:
            return 0.0
            
        confidences = [
            stats['successes'] / max(1, stats['uses'])
            for stats in self.pattern_stats.values()
        ]
        return np.mean(confidences)
    
    def _calculate_impact(self) -> float:
        """Calculate average action impact"""
        if not self.pattern_stats:
            return 0.0
            
        impacts = [
            stats['impact_sum'] / max(1, stats['uses'])
            for stats in self.pattern_stats.values()
        ]
        return np.mean(impacts)
    
    def _calculate_exploration_rate(self) -> float:
        """Calculate current exploration rate"""
        if not self.metrics_history:
            return 1.0
            
        # Use last 100 metrics points
        recent = self.metrics_history[-100:]
        return np.mean([m.exploration_rate for m in recent])
    
    def _save_history(self) -> None:
        """Save metrics history to file"""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        metrics_file = Path(self.storage_path) / 'metrics_history.json'
        
        with open(metrics_file, 'w') as f:
            json.dump([{
                'timestamp': m.timestamp.isoformat(),
                'total_trajectories': m.total_trajectories,
                'successful_trajectories': m.successful_trajectories,
                'pattern_confidence': m.pattern_confidence,
                'average_impact': m.average_impact,
                'exploration_rate': m.exploration_rate
            } for m in self.metrics_history], f, indent=2)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary statistics of learning progress"""
        if not self.pattern_stats:
            return {
                'total_patterns': 0,
                'successful_patterns': 0,
                'avg_impact': 0.0,
                'success_rate': 0.0,
                'total_trajectories': 0
            }

        total_uses = sum(stats['uses'] for stats in self.pattern_stats.values())
        total_successes = sum(stats['successes'] for stats in self.pattern_stats.values())
        total_impact = sum(stats['impact_sum'] for stats in self.pattern_stats.values())
        
        successful_patterns = sum(1 for stats in self.pattern_stats.values() 
                                if stats['successes'] > 0)
        
        return {
            'total_patterns': len(self.pattern_stats),
            'successful_patterns': successful_patterns,
            'avg_impact': total_impact / total_uses if total_uses > 0 else 0.0,
            'success_rate': total_successes / total_uses if total_uses > 0 else 0.0,
            'total_trajectories': total_uses
        }
    
    def get_pattern_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each pattern"""
        return {
            pattern: {
                'success_rate': stats['successes'] / max(1, stats['uses']),
                'average_impact': stats['impact_sum'] / max(1, stats['uses']),
                'usage_count': stats['uses']
            }
            for pattern, stats in self.pattern_stats.items()
        }