from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence
import numpy as np
from collections import defaultdict

@dataclass
class ActionStep:
    """Represents a single step in an action sequence"""
    action: Dict[str, Any]
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    success: bool
    error: Optional[str] = None

class ActionSequence:
    """Represents a sequence of actions with their states and outcomes"""
    def __init__(self, steps: List[ActionStep], semantic_type: str):
        self.steps = steps
        self.semantic_type = semantic_type
        self._success_rate = self._calculate_success_rate()

    @classmethod
    def from_trajectory(cls, trajectory: 'Trajectory') -> 'ActionSequence':
        """Create an action sequence from a trajectory"""
        steps = []
        states = [trajectory.final_state]  # Start with final state
        
        # Work backwards through actions and observations
        for action, obs in zip(reversed(trajectory.actions), reversed(trajectory.observations)):
            if isinstance(obs, dict):
                success = obs.get('status') == 'success'
                error = obs.get('error') if not success else None
                steps.insert(0, ActionStep(
                    action=action,
                    state_before=obs.get('state_before', {}),
                    state_after=obs.get('state_after', {}),
                    success=success,
                    error=error
                ))
                
        # Infer semantic type from action patterns
        semantic_type = cls._infer_semantic_type(trajectory.actions)
        
        return cls(steps, semantic_type)

    @staticmethod
    def _infer_semantic_type(actions: List[Dict[str, Any]]) -> str:
        """Infer the semantic type of an action sequence"""
        action_types = [a.get('type', '') for a in actions]
        
        # Common patterns for different semantic types
        patterns = {
            'create': {'create_file', 'write_file', 'add_dependency'},
            'modify': {'edit_file', 'update_file', 'rename_file'},
            'test': {'run_test', 'check_test', 'verify'},
            'fix': {'fix_error', 'resolve_conflict', 'update_imports'}
        }
        
        # Score each semantic type based on action overlap
        type_scores = defaultdict(int)
        for action in action_types:
            for sem_type, pattern_actions in patterns.items():
                if any(pattern in action.lower() for pattern in pattern_actions):
                    type_scores[sem_type] += 1
                    
        # Return the most likely type, or 'other' if no clear match
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return 'other'

    @property
    def success_rate(self) -> float:
        """Get the success rate of the sequence"""
        return self._success_rate

    def _calculate_success_rate(self) -> float:
        """Calculate the success rate of the sequence"""
        if not self.steps:
            return 0.0
        return sum(1 for step in self.steps if step.success) / len(self.steps)

class SequencePattern:
    """Represents a common pattern extracted from multiple action sequences"""
    def __init__(self, sequences: List[ActionSequence]):
        self.sequences = sequences
        self.success_rate = self._calculate_pattern_success()
        self.avg_success_rate = self.success_rate  # Alias for test compatibility
        self.common_steps = self._extract_common_steps()
        self.frequency = len(sequences)  # Add frequency attribute
        # Inherit semantic type from sequences if they share one
        self.semantic_type = sequences[0].semantic_type if sequences and all(s.semantic_type == sequences[0].semantic_type for s in sequences) else 'mixed'

    def _calculate_pattern_success(self) -> float:
        """Calculate overall success rate of the pattern"""
        if not self.sequences:
            return 0.0
        return sum(seq.success_rate for seq in self.sequences) / len(self.sequences)

    def _extract_common_steps(self) -> List[Dict[str, Any]]:
        """Extract common steps across sequences"""
        if not self.sequences:
            return []
            
        # Group similar actions by position in sequence
        step_groups = defaultdict(list)
        for seq in self.sequences:
            for i, step in enumerate(seq.steps):
                step_groups[i].append(step.action)
                
        # Find most common action type at each position
        common_steps = []
        for pos in sorted(step_groups.keys()):
            actions = step_groups[pos]
            if actions:
                # Group by action type
                type_groups = defaultdict(list)
                for action in actions:
                    type_groups[action.get('type', '')].append(action)
                    
                # Get most common action type
                most_common_type = max(type_groups.items(), key=lambda x: len(x[1]))[0]
                common_steps.append({
                    'type': most_common_type,
                    'frequency': len(type_groups[most_common_type]) / len(actions)
                })
                
        return common_steps

    def get_best_example(self) -> Optional[ActionSequence]:
        """Get the most successful sequence that follows this pattern"""
        if not self.sequences:
            return None
            
        return max(self.sequences, key=lambda seq: seq.success_rate)

    def matches_state(self, state: Dict[str, Any], threshold: float = 0.6) -> bool:
        """Check if a state matches the typical starting state for this pattern"""
        if not self.sequences:
            return False
            
        # Get starting states from all sequences
        starting_states = [seq.steps[0].state_before for seq in self.sequences if seq.steps]
        
        # Compare state similarity with each starting state
        for start_state in starting_states:
            # Simple comparison of key features
            matches = 0
            total = 0
            
            for key in set(state.keys()) | set(start_state.keys()):
                if key in state and key in start_state:
                    if state[key] == start_state[key]:
                        matches += 1
                total += 1
                
            if total > 0 and matches / total >= threshold:
                return True
                
        return False