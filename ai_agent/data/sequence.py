from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence
import numpy as np

@dataclass
class ActionStep:
    """Represents a single step in an action sequence"""
    action: Dict[str, Any]
    observation: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    success: bool

@dataclass
class ActionSequence:
    """Represents a sequence of actions with their outcomes"""
    steps: List[ActionStep]
    success_rate: float
    complexity: float
    semantic_type: str  # e.g., 'create', 'modify', 'test', etc.
    
    @classmethod
    def from_trajectory(cls, trajectory: 'Trajectory') -> 'ActionSequence':
        """Create an ActionSequence from a Trajectory"""
        steps = []
        current_state = trajectory.final_state
        
        # Build steps in reverse order to track states
        for action, obs in zip(reversed(trajectory.actions), reversed(trajectory.observations)):
            step = ActionStep(
                action=action,
                observation=obs,
                state_after=current_state.copy(),
                state_before={},  # Will be filled in next iteration
                success=isinstance(obs, dict) and obs.get('status') == 'success'
            )
            steps.insert(0, step)
            current_state = obs.get('state', {})
            if len(steps) > 1:
                steps[1].state_before = current_state.copy()
        
        if steps:
            steps[0].state_before = current_state
        
        # Calculate metrics
        success_rate = sum(1 for s in steps if s.success) / len(steps) if steps else 0
        complexity = len(set(s.action.get('type') for s in steps)) / len(steps) if steps else 0
        
        # Determine semantic type
        semantic_type = cls._determine_semantic_type(steps)
        
        return cls(
            steps=steps,
            success_rate=success_rate,
            complexity=complexity,
            semantic_type=semantic_type
        )
    
    @staticmethod
    def _determine_semantic_type(steps: List[ActionStep]) -> str:
        """Determine the semantic type of an action sequence"""
        action_types = [s.action.get('type', '') for s in steps]
        
        if 'create_file' in action_types:
            return 'create'
        elif 'edit_file' in action_types:
            return 'modify'
        elif 'run_tests' in action_types:
            return 'test'
        elif 'resolve_conflict' in action_types:
            return 'fix'
        else:
            return 'other'
    
    def matches_pattern(self, other: 'ActionSequence', fuzzy: bool = True) -> float:
        """
        Check if this sequence matches another sequence pattern
        Returns similarity score between 0 and 1
        """
        if not self.steps or not other.steps:
            return 0.0
        
        # Exact matching
        if not fuzzy:
            return 1.0 if all(
                s1.action.get('type') == s2.action.get('type')
                for s1, s2 in zip(self.steps, other.steps)
            ) else 0.0
        
        # Fuzzy matching
        matches = 0
        total = max(len(self.steps), len(other.steps))
        
        for i in range(min(len(self.steps), len(other.steps))):
            s1, s2 = self.steps[i], other.steps[i]
            
            # Type match
            if s1.action.get('type') == s2.action.get('type'):
                matches += 0.5
                
                # State transition similarity
                if s1.success == s2.success:
                    matches += 0.3
                    
                # Action parameter similarity
                shared_keys = set(s1.action.keys()) & set(s2.action.keys())
                if shared_keys:
                    param_matches = sum(
                        1 for k in shared_keys
                        if s1.action[k] == s2.action[k]
                    )
                    matches += 0.2 * (param_matches / len(shared_keys))
        
        return matches / total
    
    def is_subsequence_of(self, other: 'ActionSequence') -> bool:
        """Check if this sequence is a subsequence of another sequence"""
        if len(self.steps) > len(other.steps):
            return False
            
        for i in range(len(other.steps) - len(self.steps) + 1):
            if all(
                s1.action.get('type') == s2.action.get('type')
                for s1, s2 in zip(self.steps, other.steps[i:i+len(self.steps)])
            ):
                return True
        return False
    
    def get_success_pattern(self) -> Optional['ActionSequence']:
        """Extract the successful part of the sequence"""
        successful_steps = [
            step for step in self.steps
            if step.success
        ]
        
        if successful_steps:
            return ActionSequence(
                steps=successful_steps,
                success_rate=1.0,
                complexity=len(set(s.action.get('type') for s in successful_steps)) / len(successful_steps),
                semantic_type=self._determine_semantic_type(successful_steps)
            )
        return None

class SequencePattern:
    """Represents a common pattern of successful actions"""
    def __init__(self, sequences: List[ActionSequence]):
        self.sequences = sequences
        self.frequency = len(sequences)
        self.avg_success_rate = np.mean([s.success_rate for s in sequences])
        self.avg_complexity = np.mean([s.complexity for s in sequences])
        self.semantic_type = self._get_common_type()
    
    def _get_common_type(self) -> str:
        """Get the most common semantic type"""
        type_counts = {}
        for seq in self.sequences:
            type_counts[seq.semantic_type] = type_counts.get(seq.semantic_type, 0) + 1
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def matches_sequence(self, sequence: ActionSequence, threshold: float = 0.7) -> bool:
        """Check if a sequence matches this pattern"""
        return any(
            sequence.matches_pattern(s, fuzzy=True) >= threshold
            for s in self.sequences
        )
    
    def get_best_example(self) -> ActionSequence:
        """Get the most successful example of this pattern"""
        return max(self.sequences, key=lambda s: s.success_rate)