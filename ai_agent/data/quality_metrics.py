from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np

@dataclass
class QualityScore:
    """Represents the quality score components for a trajectory"""
    success_rate: float
    consistency: float
    efficiency: float
    relevance: float
    safety: float
    
    @property
    def total_score(self) -> float:
        """Calculate weighted total quality score"""
        weights = {
            'success_rate': 0.3,
            'consistency': 0.2,
            'efficiency': 0.2,
            'relevance': 0.2,
            'safety': 0.1
        }
        
        score = (
            self.success_rate * weights['success_rate'] +
            self.consistency * weights['consistency'] +
            self.efficiency * weights['efficiency'] +
            self.relevance * weights['relevance'] +
            self.safety * weights['safety']
        )
        
        return score

class QualityMetrics:
    """Analyzes and scores trajectory quality"""
    
    def __init__(self, min_quality_threshold: float = 0.7):
        self.min_quality_threshold = min_quality_threshold
        self.action_patterns = Counter()
        self.state_patterns = Counter()
    
    def compute_trajectory_quality(self, 
                                trajectory: 'Trajectory',
                                instruction: str) -> QualityScore:
        """Compute quality metrics for a trajectory"""
        # Calculate success rate
        success_rate = self._compute_success_rate(trajectory)
        
        # Calculate consistency with similar trajectories
        consistency = self._compute_consistency(trajectory)
        
        # Calculate efficiency (avoid unnecessary actions)
        efficiency = self._compute_efficiency(trajectory)
        
        # Calculate relevance to instruction
        relevance = self._compute_relevance(trajectory, instruction)
        
        # Calculate safety score
        safety = self._compute_safety(trajectory)
        
        return QualityScore(
            success_rate=success_rate,
            consistency=consistency,
            efficiency=efficiency,
            relevance=relevance,
            safety=safety
        )
    
    def should_filter_trajectory(self, score: QualityScore) -> bool:
        """Determine if trajectory should be filtered out based on quality"""
        return score.total_score < self.min_quality_threshold
    
    def _compute_success_rate(self, trajectory: 'Trajectory') -> float:
        """Calculate success rate of trajectory actions"""
        if not trajectory.observations:
            return 0.0
            
        successes = sum(
            1 for obs in trajectory.observations
            if isinstance(obs, dict) and obs.get('status') == 'success'
        )
        return successes / len(trajectory.observations)
    
    def _compute_consistency(self, trajectory: 'Trajectory') -> float:
        """Calculate consistency with known good patterns"""
        if not trajectory.actions:
            return 0.0
            
        # Convert actions to pattern
        pattern = tuple(a.get('type', '') for a in trajectory.actions)
        total_patterns = sum(self.action_patterns.values())
        
        if total_patterns == 0:
            return 1.0  # First pattern is considered consistent
            
        # Check how common this pattern is
        pattern_frequency = self.action_patterns[pattern]
        return pattern_frequency / total_patterns
    
    def _compute_efficiency(self, trajectory: 'Trajectory') -> float:
        """Calculate action efficiency"""
        if not trajectory.actions:
            return 0.0
            
        # Look for redundant or unnecessary actions
        action_types = [a.get('type', '') for a in trajectory.actions]
        unique_types = set(action_types)
        
        if len(unique_types) == 0:
            return 0.0
            
        # Penalize excessive repetition of same action type
        repetition_score = len(unique_types) / len(action_types)
        
        # Check for common inefficient patterns
        has_unnecessary_actions = any(
            action_types[i] == action_types[i-1]  # Immediate repetition
            for i in range(1, len(action_types))
        )
        
        efficiency_score = repetition_score
        if has_unnecessary_actions:
            efficiency_score *= 0.8
            
        return efficiency_score
    
    def _compute_relevance(self, trajectory: 'Trajectory', instruction: str) -> float:
        """Calculate relevance to original instruction"""
        if not trajectory.actions or not instruction:
            return 0.0
            
        # Extract key terms from instruction
        instruction_terms = set(instruction.lower().split())
        
        # Look for alignment between instruction terms and actions
        relevance_score = 0.0
        for action in trajectory.actions:
            # Check action type
            action_type = action.get('type', '').lower()
            if any(term in action_type for term in instruction_terms):
                relevance_score += 1
                
            # Check file names and other parameters
            for value in action.values():
                if isinstance(value, str):
                    if any(term in value.lower() for term in instruction_terms):
                        relevance_score += 0.5
                        
        return min(1.0, relevance_score / len(trajectory.actions))
    
    def _compute_safety(self, trajectory: 'Trajectory') -> float:
        """Calculate safety score based on potentially risky actions"""
        if not trajectory.actions:
            return 0.0
            
        # Define risky action patterns
        risky_patterns = {
            'delete': 0.3,  # File deletion
            'remove': 0.3,  # Resource removal
            'drop': 0.2,    # Database operations
            'truncate': 0.2 # Database operations
        }
        
        safety_score = 1.0
        for action in trajectory.actions:
            action_str = str(action).lower()
            for pattern, penalty in risky_patterns.items():
                if pattern in action_str:
                    safety_score -= penalty
                    
        return max(0.0, safety_score)
    
    def update_patterns(self, trajectory: 'Trajectory') -> None:
        """Update known patterns from a successful trajectory"""
        if self._compute_success_rate(trajectory) >= 0.8:
            # Add action pattern
            pattern = tuple(a.get('type', '') for a in trajectory.actions)
            self.action_patterns[pattern] += 1
            
            # Add state transition pattern if available
            if trajectory.observations and len(trajectory.observations) > 1:
                state_pattern = (
                    str(trajectory.observations[0].get('state_before')),
                    str(trajectory.observations[-1].get('state_after'))
                )
                self.state_patterns[state_pattern] += 1