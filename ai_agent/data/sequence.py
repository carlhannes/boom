from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence
import numpy as np
from collections import defaultdict

class ActionStep:
    """Represents a single action step in a sequence"""
    def __init__(self, action: Dict[str, Any]):
        self.action_data = action  # Store the action data
        
    @property
    def action(self) -> Dict[str, Any]:
        """Get the action data"""
        return self.action_data
        
    @property
    def action_type(self) -> str:
        """Get the action type"""
        return self.action_data.get('type', '')

class ActionSequence:
    def __init__(self, actions=None, semantic_type=None):
        self.actions = actions or []
        self._semantic_type = semantic_type
        self._cached_score = None
        self.steps: List[ActionStep] = []
        self.success_rate: float = 0.0
        self.pattern_type: str = ""

    @property
    def semantic_type(self):
        if not self._semantic_type:
            self._infer_semantic_type()
        return self._semantic_type

    def _infer_semantic_type(self):
        """Infer semantic type from actions"""
        if not self.actions:
            self._semantic_type = 'empty'
            return

        first_action = self.actions[0]
        action_types = [a['type'] for a in self.actions]
        
        if 'test' in ' '.join(action_types).lower():
            self._semantic_type = 'test'
        elif first_action['type'] == 'create_file':
            self._semantic_type = 'create'
        elif first_action['type'] == 'edit_file':
            self._semantic_type = 'modify'
        elif first_action['type'] == 'git_commit':
            self._semantic_type = 'commit'
        else:
            self._semantic_type = first_action['type']

    @classmethod
    def from_trajectory(cls, trajectory) -> 'ActionSequence':
        """Create sequence from trajectory"""
        sequence = cls()
        if not hasattr(trajectory, 'actions') or not hasattr(trajectory, 'observations'):
            return sequence

        for i, action in enumerate(trajectory.actions):
            observation = trajectory.observations[i] if i < len(trajectory.observations) else None
            step = ActionStep(action)
            sequence.steps.append(step)

        # Calculate success rate
        if trajectory.observations:
            successes = sum(1 for obs in trajectory.observations 
                          if isinstance(obs, dict) and obs.get('status') == 'success')
            sequence.success_rate = successes / len(trajectory.observations)

        # Determine pattern type
        sequence.pattern_type = sequence._determine_pattern_type()
        return sequence

    def _determine_pattern_type(self) -> str:
        """Determine the type of pattern this sequence represents"""
        if not self.steps:
            return ""

        # Common patterns
        test_actions = {'create_test', 'run_tests', 'write_test'}
        model_actions = {'create_model', 'edit_model', 'update_schema'}
        git_actions = {'git_commit', 'git_push', 'git_checkout'}

        action_types = {step.action_type for step in self.steps}

        if any(a in test_actions for a in action_types):
            return "test"
        elif any(a in model_actions for a in action_types):
            return "model"
        elif any(a in git_actions for a in action_types):
            return "git"
        return "general"

    def matches_state(self, state: Dict) -> bool:
        """Check if sequence matches given state"""
        if not self.steps:
            return False

        # Check file existence requirements
        required_files = {
            param.get('path') for step in self.steps 
            for param in [step.action]
            if isinstance(param, dict) and 'path' in param
        }

        state_files = set(state.get('files', []))
        return all(any(req_file in f for f in state_files) for req_file in required_files)

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