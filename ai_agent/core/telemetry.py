import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class LearningMetrics:
    def __init__(self, pattern_success_rate=0.0, exploration_rate=0.0,
                 pattern_confidence=0.0, avg_confidence=None, pattern_count=0, timestamp=None,
                 total_trajectories=None, successful_trajectories=None,
                 average_impact=None):
        self.pattern_success_rate = pattern_success_rate
        self.exploration_rate = exploration_rate
        # Handle both parameter names for backward compatibility
        self.pattern_confidence = pattern_confidence if avg_confidence is None else avg_confidence
        self.pattern_count = pattern_count
        self.timestamp = timestamp or datetime.now()
        self.total_trajectories = total_trajectories or 0
        self.successful_trajectories = successful_trajectories or 0
        self.average_impact = average_impact or 0.0
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'pattern_success_rate': self.pattern_success_rate,
            'exploration_rate': self.exploration_rate,
            'pattern_confidence': self.pattern_confidence,
            'pattern_count': self.pattern_count,
            'total_trajectories': self.total_trajectories,
            'successful_trajectories': self.successful_trajectories,
            'average_impact': self.average_impact
        }
    
    @classmethod
    def from_dict(cls, data):
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ExecutionMetrics:
    """Metrics tracking execution performance"""
    def __init__(self, success_rate=0.0, avg_duration=0.0, error_rate=0.0, recovery_rate=0.0, timestamp=None):
        self.success_rate = success_rate
        self.avg_duration = avg_duration
        self.error_rate = error_rate
        self.recovery_rate = recovery_rate
        self.timestamp = timestamp or datetime.now().timestamp()
        
    def to_dict(self):
        """Convert metrics to dictionary"""
        return {
            'success_rate': self.success_rate,
            'avg_duration': self.avg_duration,
            'error_rate': self.error_rate,
            'recovery_rate': self.recovery_rate,
            'timestamp': self.timestamp
        }

class TelemetryManager:
    """Manages collection and analysis of telemetry data"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {'execution': [], 'learning': []}
        self.pattern_stats = {}
        self._load_metrics()
    
    def record_learning_metrics(self, metrics: LearningMetrics) -> None:
        """Record learning performance metrics"""
        if not isinstance(metrics, LearningMetrics):
            # Convert dict to LearningMetrics if needed
            metrics = LearningMetrics(**metrics)
        metrics.timestamp = metrics.timestamp or datetime.now()
        self.metrics_history['learning'].append(metrics)
        self._save_metrics()
    
    def record_execution_metrics(self, metrics: ExecutionMetrics) -> None:
        """Record execution performance metrics"""
        if not isinstance(metrics, ExecutionMetrics):
            metrics = ExecutionMetrics(**metrics)
        metrics.timestamp = metrics.timestamp or datetime.now().timestamp()
        self.metrics_history['execution'].append(metrics)
        self._save_metrics()
    
    def get_learning_trends(self, window: int = 10) -> Dict[str, float]:
        """Calculate trends in learning metrics"""
        if not self.metrics_history['learning']:
            return {}
            
        recent = self.metrics_history['learning'][-window:]
        if not recent:
            return {}
            
        # Calculate trends using simple linear regression
        x = np.arange(len(recent))
        trends = {}
        
        for field in ['pattern_success_rate', 'exploration_rate', 'pattern_confidence']:
            y = np.array([getattr(m, field) for m in recent])
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trends[f"{field}_trend"] = float(slope)
            else:
                trends[f"{field}_trend"] = 0.0
                
        return trends
    
    def get_execution_trends(self, window: int = 10) -> Dict[str, float]:
        """Calculate trends in execution metrics"""
        if not self.metrics_history['execution']:
            return {}
            
        recent = self.metrics_history['execution'][-window:]
        if not recent:
            return {}
            
        trends = {}
        x = np.arange(len(recent))
        
        for field in ['success_rate', 'avg_duration', 'error_rate', 'recovery_rate']:
            y = np.array([getattr(m, field) for m in recent])
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trends[f"{field}_trend"] = float(slope)
            else:
                trends[f"{field}_trend"] = 0.0
                
        return trends
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        summary = {
            'learning': self._summarize_metrics('learning'),
            'execution': self._summarize_metrics('execution'),
            'trends': {
                **self.get_learning_trends(),
                **self.get_execution_trends()
            }
        }
        return summary
    
    def _summarize_metrics(self, metric_type: str) -> Dict[str, float]:
        """Summarize metrics of a given type"""
        metrics = self.metrics_history.get(metric_type, [])
        if not metrics:
            return {}
            
        summary = {}
        
        # Get latest metrics
        latest = metrics[-1]
        
        # Get fields to summarize
        if metric_type == 'learning':
            fields = {
                'pattern_success_rate': latest.pattern_success_rate,
                'exploration_rate': latest.exploration_rate,
                'pattern_confidence': latest.pattern_confidence,
                'pattern_count': latest.pattern_count
            }
        else:  # execution
            fields = {
                'success_rate': latest.success_rate,
                'avg_duration': latest.avg_duration,
                'error_rate': latest.error_rate,
                'recovery_rate': latest.recovery_rate
            }
        
        # Calculate averages over all history
        for field, value in fields.items():
            if field != 'timestamp' and value is not None:
                try:
                    values = [getattr(m, field) for m in metrics if getattr(m, field, None) is not None]
                    if values:
                        summary[f"avg_{field}"] = float(np.mean(values))
                        summary[f"max_{field}"] = float(np.max(values))
                        summary[f"min_{field}"] = float(np.min(values))
                except (TypeError, ValueError):
                    continue
                    
        return summary
    
    def _load_history(self) -> None:
        """Load metrics history from storage"""
        learning_path = self.storage_path / 'learning_metrics.json'
        execution_path = self.storage_path / 'execution_metrics.json'
        
        try:
            if learning_path.exists():
                with open(learning_path, 'r') as f:
                    data = json.load(f)
                    self.metrics_history['learning'] = [
                        LearningMetrics.from_dict(m) for m in data
                    ]
                    
            if execution_path.exists():
                with open(execution_path, 'r') as f:
                    data = json.load(f)
                    self.metrics_history['execution'] = [
                        ExecutionMetrics(**m) for m in data
                    ]
        except Exception as e:
            print(f"Error loading metrics history: {e}")
    
    def _save_history(self) -> None:
        """Save metrics history to storage"""
        try:
            learning_path = self.storage_path / 'learning_metrics.json'
            with open(learning_path, 'w') as f:
                json.dump(
                    [m.to_dict() for m in self.metrics_history['learning']],
                    f,
                    cls=DateTimeEncoder
                )
                
            execution_path = self.storage_path / 'execution_metrics.json'
            with open(execution_path, 'w') as f:
                json.dump(
                    [m.to_dict() for m in self.metrics_history['execution']],
                    f,
                    cls=DateTimeEncoder
                )
        except Exception as e:
            print(f"Error saving metrics history: {e}")
            
    def analyze_learning_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of learning process"""
        if not self.metrics_history['learning']:
            return {
                'learning_stability': 0.0,
                'exploration_efficiency': 0.0,
                'pattern_quality': 0.0,
                'recommendations': ['No learning metrics available - start recording metrics']
            }
            
        analysis = {
            'learning_stability': 0.0,
            'exploration_efficiency': 0.0,
            'pattern_quality': 0.0,
            'recommendations': []
        }
        
        recent = self.metrics_history['learning']  # Use all metrics for better analysis
        
        # Analyze learning stability
        if len(recent) > 1:
            success_rates = [m.pattern_success_rate for m in recent]
            stability = 1.0 - np.std(success_rates)  # Lower variance = more stable
            analysis['learning_stability'] = float(stability)
            
            if stability < 0.5:
                analysis['recommendations'].append(
                    "Consider adjusting learning rate decay to improve stability"
                )
                
        # Analyze exploration efficiency
        if len(recent) > 0:
            exploration_rates = [m.exploration_rate for m in recent]
            avg_exploration = float(np.mean(exploration_rates))
            analysis['exploration_efficiency'] = avg_exploration
            
            if avg_exploration > 0.8:
                analysis['recommendations'].append(
                    "High exploration rate - consider exploiting learned patterns more"
                )
            elif avg_exploration < 0.2:
                analysis['recommendations'].append(
                    "Low exploration rate - consider exploring new patterns"
                )
                
        # Analyze pattern quality
        if len(recent) > 0:
            confidences = [m.pattern_confidence for m in recent]
            avg_confidence = float(np.mean(confidences))
            analysis['pattern_quality'] = avg_confidence
            
            if avg_confidence < 0.6:
                analysis['recommendations'].append(
                    "Low pattern confidence - review pattern validation criteria"
                )
        
        # Always add at least one recommendation
        if not analysis['recommendations']:
            analysis['recommendations'].append(
                "Learning process is stable - continue current approach"
            )
            
        return analysis
        
    def record_trajectory(self, trajectory, pattern_name: str, was_exploration: bool):
        """Record trajectory execution metrics"""
        if not trajectory:
            return
            
        # Get quality metrics
        if hasattr(trajectory, 'compute_quality_metrics'):
            metrics = trajectory.compute_quality_metrics()
            success_rate = metrics.success_rate if hasattr(metrics, 'success_rate') else 0.0
        else:
            success_rate = getattr(trajectory, 'success_rate', 0.0)
            
        # Calculate impact
        impact = 0.0
        if hasattr(trajectory, 'state_changes'):
            impact = sum(getattr(change, 'impact', 0) for change in trajectory.state_changes)
        
        # Record pattern stats
        if pattern_name not in self.pattern_stats:
            self.pattern_stats[pattern_name] = {
                'uses': 0,
                'successes': 0,
                'impact_sum': 0.0
            }
            
        stats = self.pattern_stats[pattern_name]
        stats['uses'] += 1
        if success_rate >= 0.7:
            stats['successes'] += 1
        stats['impact_sum'] += impact
        
        # Record learning metrics
        pattern_confidence = stats['successes'] / stats['uses'] if stats['uses'] > 0 else 0.0
        learning_metrics = LearningMetrics(
            pattern_success_rate=success_rate,
            exploration_rate=1.0 if was_exploration else 0.0,
            pattern_confidence=pattern_confidence,
            pattern_count=len(self.pattern_stats),
            average_impact=impact
        )
        self.record_learning_metrics(learning_metrics)
        self._save_metrics()
        
    def _load_metrics(self):
        """Load metrics from storage"""
        metrics_file = self.storage_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                self.pattern_stats = data.get('pattern_stats', {})
                
                # Convert learning metrics back to objects
                learning_metrics = data.get('learning_metrics', [])
                self.metrics_history['learning'] = [
                    LearningMetrics.from_dict(m) for m in learning_metrics
                ]
                
                # Convert execution metrics back to objects
                execution_metrics = data.get('execution_metrics', [])
                self.metrics_history['execution'] = [
                    ExecutionMetrics(**m) for m in execution_metrics
                ]
                
    def _save_metrics(self):
        """Save metrics to storage"""
        metrics_file = self.storage_path / "metrics.json"
        data = {
            'pattern_stats': self.pattern_stats,
            'learning_metrics': [
                m.to_dict() for m in self.metrics_history['learning']
            ],
            'execution_metrics': [
                m.to_dict() for m in self.metrics_history['execution']
            ]
        }
        with open(metrics_file, 'w') as f:
            json.dump(data, f, cls=DateTimeEncoder)
            
    def get_pattern_performance(self, pattern_name: str = None) -> Dict:
        """Get performance metrics for a pattern or overall pattern performance"""
        if pattern_name is None:
            # Return aggregate stats for all patterns
            total_uses = sum(p.get('uses', 0) for p in self.pattern_stats.values())
            if total_uses == 0:
                return {
                    'success_rate': 0.0,
                    'avg_impact': 0.0,
                    'total_uses': 0
                }
            return {
                'success_rate': sum(p.get('successes', 0) for p in self.pattern_stats.values()) / total_uses,
                'avg_impact': sum(p.get('impact_sum', 0.0) for p in self.pattern_stats.values()) / total_uses,
                'total_uses': total_uses
            }
        else:
            stats = self.pattern_stats.get(pattern_name, {})
            uses = stats.get('uses', 0)
            if uses == 0:
                return {
                    'success_rate': 0.0,
                    'avg_impact': 0.0,
                    'total_uses': 0
                }
            return {
                'success_rate': stats.get('successes', 0) / uses,
                'avg_impact': stats.get('impact_sum', 0.0) / uses,
                'total_uses': uses
            }
            
    def _calculate_stability(self, success_rates: List[float]) -> float:
        """Calculate learning stability"""
        if len(success_rates) < 2:
            return 0.0
        variations = np.diff(success_rates)
        return 1.0 - min(1.0, np.std(variations))
        
    def _calculate_exploration_efficiency(self, exploration_rates: List[float], success_rates: List[float]) -> float:
        """Calculate exploration efficiency"""
        if len(exploration_rates) < 2 or len(success_rates) < 2:
            return 0.0
        
        # Higher score if success increases while exploration decreases
        success_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
        exploration_trend = np.polyfit(range(len(exploration_rates)), exploration_rates, 1)[0]
        
        if success_trend <= 0:
            return 0.0
        if exploration_trend >= 0:
            return 0.3
            
        return min(1.0, success_trend / abs(exploration_trend))
        
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning progress"""
        summary = {
            'total_patterns': len(self.pattern_stats),
            'total_trajectories': 0,
            'success_rate': 0.0,
            'pattern_stats': {
                'total': 0,
                'successful': 0,
                'avg_impact': 0.0
            }
        }
        
        if not self.metrics_history['learning']:
            return summary
            
        recent = self.metrics_history['learning'][-10:]  # Last 10 entries
        current = recent[-1]
        
        successful_patterns = sum(1 for p in self.pattern_stats.values() 
                                if p.get('successes', 0) > 0 and 
                                p.get('uses', 0) > 0 and
                                p.get('successes', 0) / p.get('uses', 1) > 0.7)
        
        avg_impacts = [p.get('impact_sum', 0.0) / max(p.get('uses', 1), 1) 
                       for p in self.pattern_stats.values()]
        avg_impact = float(np.mean(avg_impacts)) if avg_impacts else 0.0
        
        pattern_stats = {
            'total': len(self.pattern_stats),
            'successful': successful_patterns,
            'avg_impact': avg_impact
        }
        
        summary = {
            'total_patterns': len(self.pattern_stats),
            'current_metrics': current.to_dict(),
            'trends': self.get_learning_trends(),
            'pattern_stats': pattern_stats
        }
        
        # Add effectiveness analysis
        effectiveness = self.analyze_learning_effectiveness()
        summary.update(effectiveness)
        
        return summary