from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

class QualityScore:
    def __init__(self, success_rate=0.0, coverage_score=0.0, complexity_score=0.0, 
                 risk_score=0.0, efficiency=0.0, relevance=0.0, safety=1.0):
        self._success_rate = success_rate
        self._coverage_score = coverage_score
        self._complexity_score = complexity_score
        self._risk_score = risk_score
        self._efficiency = efficiency
        self._relevance = relevance
        self._safety = 1.0 - risk_score  # Safety is inverse of risk
        self.total_score = self._compute_total()

    @property
    def success_rate(self):
        return self._success_rate

    @property
    def coverage_score(self):
        return self._coverage_score
        
    @property
    def complexity_score(self):
        return self._complexity_score
        
    @property
    def risk_score(self):
        return self._risk_score
        
    @property
    def efficiency(self):
        return self._efficiency
        
    @property
    def relevance(self):
        return self._relevance
        
    @property
    def safety(self):
        return self._safety

    def _compute_total(self) -> float:
        """Compute total quality score with weighted components"""
        weights = {
            'success_rate': 0.45,  # High weight for success
            'coverage_score': 0.2,
            'complexity_score': 0.15,
            'safety': 0.1,  # Use safety instead of risk
            'efficiency': 0.05,
            'relevance': 0.05
        }
        
        # Success rate bonus for perfect execution
        success_bonus = 0.1 if self._success_rate >= 0.95 else 0.0
        
        base_score = (
            weights['success_rate'] * self._success_rate +
            weights['coverage_score'] * self._coverage_score +
            weights['complexity_score'] * self._complexity_score +
            weights['safety'] * self._safety +
            weights['efficiency'] * self._efficiency +
            weights['relevance'] * self._relevance
        )
        
        return min(1.0, base_score + success_bonus)

class QualityMetrics:
    def __init__(self, min_quality_threshold=0.6):
        self.min_quality_threshold = min_quality_threshold
        self.risk_thresholds = {
            'delete_file': 0.9,
            'drop_table': 0.95,
            'git_reset': 0.8,
            'git_clean': 0.8,
            'edit_file': 0.4,
            'move_file': 0.5,
            'run_tests': 0.1
        }
        self.patterns = {}  # Initialize patterns dictionary
        
    def _calculate_success_rate(self, trajectory) -> float:
        """Calculate success rate from trajectory observations"""
        if not hasattr(trajectory, 'observations') or not trajectory.observations:
            return 0.0
        
        successes = sum(1 for obs in trajectory.observations 
                       if isinstance(obs, dict) and obs.get('status') == 'success')
        return successes / len(trajectory.observations)
        
    def update_patterns(self, trajectory) -> None:
        """Update pattern database with new trajectory"""
        if not trajectory.actions:
            return
        
        # Extract pattern key
        pattern_key = tuple(a.get('type', '') for a in trajectory.actions
                          if isinstance(a, dict))
        
        if not pattern_key:
            return
            
        # Initialize pattern entry if needed
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {
                'count': 0,
                'success_count': 0,
                'total_actions': 0,
                'total_success_rate': 0.0
            }
            
        # Update pattern stats
        pattern = self.patterns[pattern_key]
        pattern['count'] += 1
        pattern['total_actions'] += len(trajectory.actions)
        
        # Calculate success rate from observations
        success_rate = self._calculate_success_rate(trajectory)
        if success_rate >= 0.8:  # Only count high success
            pattern['success_count'] += 1
            pattern['total_success_rate'] += success_rate

    def _calculate_complexity(self, trajectory) -> float:
        """Calculate complexity score. Higher score means less complex (simpler)."""
        if not trajectory.actions:
            return 0.0
            
        num_actions = len(trajectory.actions)
        unique_types = len({a.get('type', '') for a in trajectory.actions})
        
        # Simple trajectories (1-2 actions)
        if num_actions <= 2:
            return 0.9
            
        # Calculate base penalty from number of actions
        action_penalty = 0.15 * num_actions  # 15% per action
        
        # Additional penalty for unique action types
        type_penalty = 0.2 * unique_types  # 20% per unique type
        
        # Penalty for non-grouped similar actions
        grouped_penalty = 0.0
        action_types = [a.get('type', '') for a in trajectory.actions]
        type_switches = sum(1 for i in range(1, len(action_types))
                          if action_types[i] != action_types[i-1])
        grouped_penalty = 0.1 * type_switches  # 10% per type switch
        
        # Calculate final score
        total_penalty = action_penalty + type_penalty + grouped_penalty
        score = max(0.1, 1.0 - total_penalty)
        
        # Hard caps based on action count
        if num_actions >= 5:
            score = min(score, 0.4)
        if num_actions >= 6:
            score = min(score, 0.3)
            
        return score

    def _calculate_risk(self, trajectory) -> float:
        """Calculate risk score based on action types and patterns"""
        if not trajectory.actions:
            return 0.0
            
        # Calculate risk based on highest risk action
        risk_scores = []
        for action in trajectory.actions:
            action_type = action.get('type', '')
            risk = self.risk_thresholds.get(action_type, 0.3)
            risk_scores.append(risk)
            
        # Use maximum risk as base
        base_risk = max(risk_scores) if risk_scores else 0.0
        
        # Increase risk for longer sequences of risky actions
        high_risk_actions = sum(1 for r in risk_scores if r >= 0.7)
        risk_multiplier = 1.0 + (0.1 * high_risk_actions)  # 10% increase per high-risk action
        
        return min(1.0, base_risk * risk_multiplier)

    def assess_trajectory(self, trajectory) -> QualityScore:
        """Assess quality metrics for a trajectory"""
        if not trajectory.actions:
            return QualityScore()

        success_rate = (getattr(trajectory, 'success_rate', None) or 
                       self._calculate_success_rate(trajectory))
        coverage_score = self._calculate_coverage(trajectory)
        complexity_score = self._calculate_complexity(trajectory)
        risk_score = self._calculate_risk(trajectory)

        return QualityScore(
            success_rate=success_rate,
            coverage_score=coverage_score,
            complexity_score=complexity_score,
            risk_score=risk_score
        )

    def _calculate_coverage(self, trajectory) -> float:
        """Calculate coverage score based on action completeness"""
        if not trajectory.actions:
            return 0.0
            
        # Check test coverage
        test_actions = sum(1 for a in trajectory.actions 
                         if 'test' in str(a.get('type', '')).lower())
        has_tests = test_actions > 0
        
        # Check validation coverage
        validation_actions = sum(1 for a in trajectory.actions
                               if 'validate' in str(a.get('type', '')).lower())
        has_validation = validation_actions > 0
        
        # Calculate score components
        test_score = 0.5 if has_tests else 0.0
        validation_score = 0.3 if has_validation else 0.0
        action_score = min(1.0, len(trajectory.actions) / 5) * 0.2
        
        return test_score + validation_score + action_score

    def _calculate_efficiency(self, trajectory) -> float:
        """Calculate efficiency of actions"""
        if not trajectory.actions:
            return 0.0

        # Calculate based on action count and success rate
        success_rate = getattr(trajectory, 'success_rate', 0.0)
        action_count = len(trajectory.actions)
        
        # Penalize for too many actions
        action_penalty = max(0.0, 1.0 - (action_count / 10))
        
        return (success_rate * 0.7 + action_penalty * 0.3)

    def _calculate_relevance(self, trajectory, instruction: str) -> float:
        """Calculate how relevant the actions were to the instruction"""
        if not instruction or not trajectory.actions:
            return 0.0
        
        # Convert instruction to lower case terms
        instruction_terms = set(instruction.lower().split())
        action_terms = set()
        exact_matches = 0
        
        for action in trajectory.actions:
            if not isinstance(action, dict):
                continue
                
            # Track exact matches for action types
            if 'type' in action:
                action_type = action['type'].lower()
                if action_type in instruction.lower():
                    exact_matches += 1
                # Add type terms with duplicates for higher weight
                type_parts = action_type.split('_')
                action_terms.update(type_parts)
                for part in type_parts:
                    action_terms.add(part)  # Add again for more weight
                
            # Include file names without extensions
            if 'file' in action:
                filename = action['file']
                if isinstance(filename, str):
                    base_name = filename.rsplit('.', 1)[0].lower()
                    name_parts = base_name.split('_')
                    action_terms.update(name_parts)
                    if any(part in instruction.lower() for part in name_parts):
                        exact_matches += 1
        
        if not action_terms or not instruction_terms:
            return 0.0
        
        # Calculate term overlap with exact match bonus
        common_terms = instruction_terms.intersection(action_terms)
        base_relevance = len(common_terms) / len(instruction_terms)
        exact_match_bonus = min(0.3, exact_matches * 0.15)  # Up to 30% bonus
        
        # Additional boost for key terms
        key_terms = {'create', 'test', 'fix', 'implement', 'update', 'add', 'write', 'run'}
        key_term_bonus = sum(0.1 for term in key_terms 
                            if term in common_terms)  # 10% per key term
        
        return min(1.0, base_relevance + exact_match_bonus + key_term_bonus)

    def compute_trajectory_quality(self, trajectory, instruction: str) -> QualityScore:
        """Compute comprehensive quality metrics for a trajectory"""
        if not trajectory.actions:
            return QualityScore()

        # Calculate individual metrics using assess_trajectory as base
        base_score = self.assess_trajectory(trajectory)
        
        # Add relevance calculation since we have instruction
        relevance = self._calculate_relevance(trajectory, instruction)

        return QualityScore(
            success_rate=base_score.success_rate,
            coverage_score=base_score.coverage_score,
            complexity_score=base_score.complexity_score,
            risk_score=base_score.risk_score,
            efficiency=base_score.efficiency,
            relevance=relevance
        )

    def _has_clear_pattern(self, actions) -> bool:
        """Check if actions follow a clear pattern"""
        if len(actions) < 2:
            return False
            
        types = [a.get('type', '') for a in actions]
        
        # Check for repeating sequences
        for length in range(1, len(types) // 2 + 1):
            pattern = types[:length]
            if self._is_repeating_pattern(types, pattern):
                return True
                
        return False

    def _is_repeating_pattern(self, sequence, pattern) -> bool:
        """Check if a sequence follows a repeating pattern"""
        if not pattern:
            return False
            
        pattern_len = len(pattern)
        for i in range(0, len(sequence) - pattern_len + 1, pattern_len):
            if sequence[i:i + pattern_len] != pattern:
                return False
        return True

    def get(self, key: str, default: float = 0.0) -> float:
        """Get a metric value"""
        return self._metrics.get(key, default)

    def meets_quality_threshold(self, score: QualityScore) -> bool:
        """Check if quality score meets minimum threshold"""
        return score.total_score >= self.min_quality_threshold

    def _has_repeating_subsequence(self, sequence, length: int) -> bool:
        """Check if sequence contains repeating subsequences of given length"""
        if len(sequence) < length * 2:
            return False
            
        for i in range(len(sequence) - length + 1):
            pattern = sequence[i:i+length]
            # Look for the same pattern later in the sequence
            for j in range(i + length, len(sequence) - length + 1, length):
                if sequence[j:j+length] == pattern:
                    return True
        return False