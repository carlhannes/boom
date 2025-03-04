from typing import Dict, Any, List, Optional, Set, Tuple, TYPE_CHECKING
from pathlib import Path
import json
import time
import os
from collections import defaultdict
from datetime import datetime

from .learner import SelfLearner, TaskExample
from ..environment.git_env import GitEnvironment
from ..data.trajectory_manager import TrajectoryManager, Trajectory

class CodingAgent:
    def __init__(self, repo_path: str, storage_path: str, bm25_top_k: int = 50, api_key: str = None):
        """Initialize the coding agent
        
        Args:
            repo_path: Path to Git repository
            storage_path: Path to store trajectories
            bm25_top_k: Number of candidates to retrieve in first-stage BM25 retrieval
            api_key: Optional OpenAI API key for testing
        """
        self.environment = GitEnvironment(repo_path)
        self.trajectory_manager = TrajectoryManager(storage_path)
        self.bm25_top_k = bm25_top_k
        self.learner = None
        self.api_key = api_key
        
    def set_learner(self, learner):
        """Set the learner component"""
        self.learner = learner
        self.learner.agent = self
        
    def generate_tasks_from_docs(self, docs: List[str]) -> List[TaskExample]:
        """Generate tasks by analyzing documentation and current repository state"""
        current_state = self.environment.get_state()
        return self.learner.generate_tasks_from_docs(docs, repo_state=current_state)
        
    def _handle_action_error(self, error: Exception, trajectory: Trajectory, current_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle errors during action execution and attempt recovery
        Returns a recovery action if possible, None otherwise
        """
        error_msg = str(error).lower()
        
        # Check for common error patterns
        if 'permission denied' in error_msg:
            return {
                'type': 'fix_permissions',
                'description': 'Attempting to fix permission issues'
            }
        elif 'not found' in error_msg and trajectory.actions:
            last_action = trajectory.actions[-1]
            if last_action.get('type') == 'edit_file':
                return {
                    'type': 'create_file',
                    'path': last_action.get('path'),
                    'description': 'Creating missing file before edit'
                }
        elif 'merge conflict' in error_msg:
            return {
                'type': 'resolve_conflict',
                'path': trajectory.actions[-1].get('path'),
                'description': 'Attempting to resolve merge conflict'
            }
            
        return None

    def execute_task(self, instruction: str) -> Trajectory:
        """Execute a coding task based on the instruction with error handling and recovery"""
        current_state = self.environment.get_state()
        max_retries = 3
        retry_delay = 1  # seconds
        
        # Get similar trajectories using hybrid retrieval
        similar_trajectories = self.trajectory_manager.retrieve_similar_trajectories(
            current_state=current_state,
            instruction=instruction,
            limit=5,
            bm25_top_k=self.bm25_top_k
        )
        
        # Initialize trajectory
        trajectory = Trajectory(
            instruction=instruction,
            actions=[],
            observations=[],
            final_state={}
        )
        
        # For empty repository/no examples, create default action
        if not similar_trajectories:
            # Extract target filename from instruction
            words = instruction.lower().split()
            target = next((w for w in words if '.py' in w), None)
            if not target:
                target = next((w for w in words if w not in ['create', 'new', 'add']), 'module')
                if not target.endswith('.py'):
                    target = f"{target}.py"

            # Create default action
            default_action = {
                'type': 'create_file' if 'create' in instruction.lower() else 'edit_file',
                'path': f"src/{target}",
                'content': '# TODO: Implement'
            }
            observation = self.environment.execute_action(default_action)
            trajectory.actions.append(default_action)
            trajectory.observations.append(observation)
            trajectory.final_state = self.environment.get_state()
            return trajectory

        # Track failed actions to avoid cycles
        failed_actions = set()
        attempts = 0
        
        while not self._task_complete(trajectory) and attempts < max_retries:
            try:
                next_action = self._plan_next_action(
                    current_state,
                    trajectory,
                    similar_trajectories
                )

                if not next_action:
                    break

                # Generate action key for tracking failures
                action_key = f"{next_action.get('type')}:{next_action.get('path', next_action.get('file', ''))}"
                if action_key in failed_actions:
                    attempts += 1
                    continue

                # Execute action with retry logic
                observation = self.environment.execute_action(next_action)
                trajectory.actions.append(next_action)
                trajectory.observations.append(observation)
                current_state = self.environment.get_state()

                if observation['status'] == 'error':
                    failed_actions.add(action_key)
                    attempts += 1

            except Exception as e:
                attempts += 1
                time.sleep(retry_delay)
        
        trajectory.final_state = current_state
        
        # Do backward construction with error context
        refined_instruction = self.learner.backward_construct({
            'instruction': instruction,
            'actions': trajectory.actions,
            'observations': trajectory.observations,
            'similar_trajectories': [t.to_dict() for t in similar_trajectories],
            'failed_actions': list(failed_actions)
        })
        trajectory.instruction = refined_instruction
        
        # Only store if we had some success
        if any(isinstance(obs, dict) and obs.get('status') == 'success'
               for obs in trajectory.observations):
            self.trajectory_manager.store_trajectory(trajectory)
        
        return trajectory

    def _detect_action_cycle(self, trajectory: Trajectory, window: int = 3) -> bool:
        """
        Detect if the agent is in a cycle of repeating actions
        
        Args:
            trajectory: The current trajectory
            window: Size of the window to check for cycles
        """
        if len(trajectory.actions) < window * 2:
            return False
            
        # Get the last n actions and the n before them
        recent = trajectory.actions[-window:]
        previous = trajectory.actions[-2*window:-window]
        
        # Compare action types and targets
        return all(
            a1.get('type') == a2.get('type') and 
            a1.get('path') == a2.get('path')
            for a1, a2 in zip(recent, previous)
        )

    def _check_goal_alignment(self, trajectory: Trajectory) -> float:
        """
        Check how well the trajectory aligns with its intended goal
        Returns a score between 0 and 1
        """
        # Get the initial task instruction and completed actions
        instruction = trajectory.instruction.lower()
        actions = [a.get('description', '') for a in trajectory.actions]
        action_text = ' '.join(actions).lower()
        
        # Common completion indicators in the instruction
        goal_verbs = {
            'add': ['add', 'create', 'implement'],
            'modify': ['update', 'change', 'modify', 'refactor'],
            'remove': ['remove', 'delete', 'clean'],
            'fix': ['fix', 'repair', 'solve'],
            'test': ['test', 'verify', 'validate']
        }
        
        # Check if action types match instruction intent
        score = 0.0
        for category, verbs in goal_verbs.items():
            if any(verb in instruction for verb in verbs):
                # Check if actions match the intended category
                if category == 'add' and ('create_file' in action_text or 'edit_file' in action_text):
                    score += 0.4
                elif category == 'modify' and 'edit_file' in action_text:
                    score += 0.4
                elif category == 'remove' and 'delete_file' in action_text:
                    score += 0.4
                elif category == 'fix' and 'edit_file' in action_text:
                    score += 0.4
                elif category == 'test' and ('run_tests' in action_text or 'verify' in action_text):
                    score += 0.4
                    
        # Check if mentioned files/paths were acted upon
        file_mentions = [word for word in instruction.split() if '.' in word]
        for file in file_mentions:
            if any(file in str(action) for action in trajectory.actions):
                score += 0.3
                
        # Additional points for successful outcomes
        if any(isinstance(obs, dict) and obs.get('status') == 'success'
               for obs in trajectory.observations[-3:]):  # Look at last 3 observations
            score += 0.3
            
        return min(1.0, score)

    def _task_complete(self, trajectory: Trajectory) -> bool:
        """
        Determine if the current task is complete using multiple heuristics:
        1. Explicit completion signals
        2. Error states
        3. Goal alignment
        4. Action cycles
        5. Success signals
        """
        if not trajectory.observations:
            return False
            
        last_obs = trajectory.observations[-1]
        
        # 1. Check explicit completion signals
        if isinstance(last_obs, dict):
            if last_obs.get('status') == 'complete':
                return True
                
            # Handle terminal error states
            if last_obs.get('error'):
                error_msg = str(last_obs['error']).lower()
                # Only stop on serious errors, not temporary failures
                terminal_indicators = ['permission denied', 'not found', 'invalid']
                if any(ind in error_msg for ind in terminal_indicators):
                    return True
        
        # 2. Check for action cycles (avoid infinite loops)
        if self._detect_action_cycle(trajectory):
            return True
        
        # 3. Check goal alignment
        alignment_score = self._check_goal_alignment(trajectory)
        if alignment_score > 0.8:  # High confidence of completion
            return True
            
        # 4. Check recent success signals
        recent_obs = trajectory.observations[-3:]  # Look at last 3 observations
        success_rate = sum(
            1 for obs in recent_obs
            if isinstance(obs, dict) and obs.get('status') == 'success'
        ) / len(recent_obs) if recent_obs else 0
        
        if success_rate > 0.7 and len(trajectory.actions) >= 3:
            return True
        
        # 5. Safety limit on number of actions
        return len(trajectory.actions) >= 15  # Increased from 10 to allow more complex tasks
        
    def _analyze_similar_trajectories(self, current_state: Dict[str, Any], examples: List[Trajectory]) -> Dict[str, Any]:
        """
        Analyze similar trajectories to extract action patterns and success indicators
        """
        patterns = {
            'common_sequences': [],
            'successful_paths': [],
            'state_transitions': []
        }
        
        for example in examples:
            if not example.actions:
                continue
                
            # Record successful action sequences
            if all(isinstance(obs, dict) and obs.get('status') == 'success' 
                   for obs in example.observations):
                action_sequence = [
                    {'type': a.get('type'), 'target': a.get('path')}
                    for a in example.actions
                ]
                patterns['successful_paths'].append(action_sequence)
                
            # Analyze state transitions
            for i, action in enumerate(example.actions):
                if i > 0:  # Look at state before and after action
                    prev_state = example.observations[i-1].get('state', {})
                    curr_state = example.observations[i].get('state', {})
                    patterns['state_transitions'].append({
                        'action': action,
                        'before': prev_state,
                        'after': curr_state,
                        'success': example.observations[i].get('status') == 'success'
                    })
        
        # Find common action sequences (n-grams)
        if patterns['successful_paths']:
            for size in range(2, 4):  # Look for sequences of 2-3 actions
                for path in patterns['successful_paths']:
                    if len(path) >= size:
                        for i in range(len(path) - size + 1):
                            sequence = path[i:i+size]
                            patterns['common_sequences'].append(sequence)
        
        return patterns

    def _match_state_pattern(self, current_state: Dict[str, Any], pattern_state: Dict[str, Any]) -> float:
        """
        Calculate how well the current state matches a pattern state
        """
        score = 0.0
        total_checks = 0
        
        # Compare files
        pattern_files = set(pattern_state.get('files', []))
        current_files = set(current_state.get('files', []))
        if pattern_files:
            total_checks += 1
            overlap = len(pattern_files & current_files) / len(pattern_files)
            score += overlap
            
        # Compare git status
        pattern_status = pattern_state.get('git_status', {})
        current_status = current_state.get('git_status', {})
        for key in ['modified', 'staged', 'untracked']:
            pattern_set = set(pattern_status.get(key, []))
            if pattern_set:
                total_checks += 1
                current_set = set(current_status.get(key, []))
                overlap = len(pattern_set & current_set) / len(pattern_set)
                score += overlap
                
        return score / max(total_checks, 1)

    def _plan_next_action(self,
                         current_state: Dict[str, Any],
                         trajectory: Trajectory,
                         examples: List[Trajectory]) -> Optional[Dict[str, Any]]:
        """
        Plan next action using pattern matching, state analysis, and LLM guidance
        """
        if not self.learner:
            return None

        # Get analysis of similar trajectories
        patterns = self._analyze_similar_trajectories(current_state, examples)
        
        # Find matching state transitions
        matching_transitions = []
        for transition in patterns['state_transitions']:
            if self._match_state_pattern(current_state, transition['before']) > 0.7:
                matching_transitions.append(transition)
        
        # Look for ongoing action sequences
        current_sequence = [
            {'type': a.get('type'), 'target': a.get('path')}
            for a in trajectory.actions[-2:] if a  # Last 2 actions
        ]
        
        matching_sequences = []
        if current_sequence:
            for seq in patterns['common_sequences']:
                if len(seq) > len(current_sequence) and \
                   all(a1['type'] == a2['type'] for a1, a2 in zip(current_sequence, seq)):
                    matching_sequences.append(seq[len(current_sequence):])  # Get next actions
        
        # Prepare rich context for LLM
        context = {
            'instruction': trajectory.instruction,
            'current_state': current_state,
            'trajectory_so_far': trajectory.to_dict(),
            'examples': [e.to_dict() for e in examples],
            'analysis': {
                'matching_transitions': matching_transitions,
                'potential_next_steps': matching_sequences,
                'successful_patterns': patterns['successful_paths']
            }
        }
        
        # Get next action recommendation from LLM
        response = self.learner.chat_completion([
            {"role": "system", "content": (
                "You are a coding agent that plans the next action in a repository. "
                "Consider the current state, action history, similar examples, and "
                "detected patterns to decide the optimal next step.\n\n"
                "Focus on:\n"
                "1. Following successful patterns from similar tasks\n"
                "2. Maintaining state consistency\n"
                "3. Making progress toward the goal\n"
                "4. Avoiding cycles or redundant actions"
            )},
            {"role": "user", "content": json.dumps(context)}
        ])
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            # Fallback: If parsing fails, check if we have a clear next action from patterns
            if matching_sequences and matching_sequences[0]:
                return {
                    'type': matching_sequences[0][0]['type'],
                    'path': matching_sequences[0][0]['target'],
                    'description': 'Following successful pattern'
                }
            return None

from typing import List, Dict, Any, Optional
from ..environment.git_env import GitEnvironment
from ..data.trajectory_manager import TrajectoryManager

class Agent:
    def __init__(self, env: GitEnvironment, trajectory_manager: TrajectoryManager):
        self.env = env
        self.trajectory_manager = trajectory_manager
        self.current_trajectory: Optional[Dict[str, Any]] = None

    def handle_error(self, error: str, state: Dict[str, Any]) -> bool:
        """Handle an error using learned recovery patterns"""
        # First try getting recovery actions from environment
        recovery_actions = self.env._get_recovery_actions(error)
        
        if not recovery_actions:
            # If no direct pattern, look for similar situations in trajectory history
            similar_trajectories = self.trajectory_manager.retrieve_similar_trajectories(
                current_state=state,
                instruction=f"Fix {error}",
                limit=3
            )
            
            for trajectory in similar_trajectories:
                # Look for successful error recovery in trajectory
                for i, obs in enumerate(trajectory.observations):
                    if isinstance(obs, dict) and obs.get('error') == error:
                        # Found matching error, extract recovery actions
                        recovery_idx = i + 1
                        if recovery_idx < len(trajectory.actions):
                            recovery_actions = trajectory.actions[recovery_idx:]
                            break
                if recovery_actions:
                    break

        if recovery_actions:
            # Try the recovery actions
            success = True
            for action in recovery_actions:
                result = self.execute_action(action)
                if not isinstance(result, dict) or result.get('status') != 'success':
                    success = False
                    break
            
            # Update pattern success rate
            self.env.add_recovery_pattern(error, recovery_actions, success)
            
            if success:
                # Store successful recovery for future learning
                self.trajectory_manager.store_error_recovery(
                    error,
                    type('Trajectory', (), {
                        'instruction': f"Fix {error}",
                        'actions': recovery_actions,
                        'observations': [{'status': 'success'}],
                        'final_state': self.env.get_state()
                    })
                )
            
            return success
            
        return False

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action and handle any errors"""
        try:
            # Record action in current trajectory
            if self.current_trajectory:
                self.current_trajectory['actions'].append(action)
            
            # Execute the action
            result = self.env.execute(action)
            
            # Handle errors if needed
            if isinstance(result, dict) and result.get('status') == 'error':
                error = result.get('error', '')
                if error and self.handle_error(error, self.env.get_state()):
                    # Error was handled successfully, update result
                    result = {'status': 'success'}
            
            # Record observation
            if self.current_trajectory:
                self.current_trajectory['observations'].append(result)
            
            return result
            
        except Exception as e:
            # Unexpected errors also trigger recovery
            if self.handle_error(str(e), self.env.get_state()):
                return {'status': 'success'}
            return {'status': 'error', 'error': str(e)}

    def execute_task(self, instruction: str) -> Dict[str, Any]:
        """Execute a task with error recovery and pattern learning"""
        self.current_trajectory = {
            'instruction': instruction,
            'actions': [],
            'observations': [],
            'final_state': None
        }
        
        try:
            # Execute task actions
            state = self.env.get_state()
            
            # Look for similar successful trajectories
            similar = self.trajectory_manager.retrieve_similar_trajectories(
                current_state=state,
                instruction=instruction,
                limit=1
            )
            
            if similar:
                # Found similar successful trajectory, try to replicate it
                trajectory = similar[0]
                for action in trajectory.actions:
                    result = self.execute_action(action)
                    if isinstance(result, dict) and result.get('status') != 'success':
                        break
            else:
                # No similar trajectory, generate new plan
                plan = self._generate_plan(instruction, state)
                for action in plan:
                    result = self.execute_action(action)
                    if isinstance(result, dict) and result.get('status') != 'success':
                        break
            
            # Record final state
            final_state = self.env.get_state()
            self.current_trajectory['final_state'] = final_state
            
            # Store trajectory if successful
            if all(isinstance(obs, dict) and obs.get('status') == 'success' 
                  for obs in self.current_trajectory['observations']):
                self.trajectory_manager.store_trajectory(
                    type('Trajectory', (), self.current_trajectory)()
                )
            
            return {'status': 'success', 'state': final_state}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
        finally:
            self.current_trajectory = None

    def _generate_plan(self, instruction: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a plan of actions based on instruction and current state"""
        # First check for matching patterns in successful trajectories
        similar = self.trajectory_manager.retrieve_similar_trajectories(
            current_state=state,
            instruction=instruction,
            limit=5
        )
        
        if similar:
            # Analyze successful patterns to construct plan
            pattern_actions = []
            for trajectory in similar:
                if all(isinstance(obs, dict) and obs.get('status') == 'success' 
                      for obs in trajectory.observations):
                    pattern_actions.extend(trajectory.actions)
            
            if pattern_actions:
                # Group similar actions and keep most common sequence
                action_sequences = []
                current_sequence = []
                
                for action in pattern_actions:
                    if not current_sequence:
                        current_sequence.append(action)
                    elif action['type'] == current_sequence[-1]['type']:
                        current_sequence.append(action)
                    else:
                        action_sequences.append(current_sequence)
                        current_sequence = [action]
                
                if current_sequence:
                    action_sequences.append(current_sequence)
                
                # Take most common action from each sequence group
                plan = []
                for sequence in action_sequences:
                    most_common = max(sequence, key=sequence.count)
                    if most_common not in plan:  # Avoid duplicates
                        plan.append(most_common)
                
                return plan
        
        # If no good patterns found, fallback to learner-generated plan
        return self.learner.generate_plan(instruction, state)

    def analyze_trajectory(self, trajectory: 'Trajectory') -> Dict[str, Any]:
        """Analyze a trajectory for patterns and quality metrics"""
        analysis = {
            'success_rate': 0.0,
            'completion_rate': 0.0,
            'error_rate': 0.0,
            'patterns': [],
            'risks': []
        }
        
        if not trajectory.actions:
            return analysis
            
        # Calculate success rate
        successes = sum(1 for obs in trajectory.observations 
                       if isinstance(obs, dict) and obs.get('status') == 'success')
        analysis['success_rate'] = successes / len(trajectory.observations)
        
        # Calculate completion rate based on goal alignment
        analysis['completion_rate'] = self._check_goal_alignment(trajectory)
        
        # Calculate error rate
        errors = sum(1 for obs in trajectory.observations
                    if isinstance(obs, dict) and obs.get('error'))
        analysis['error_rate'] = errors / len(trajectory.observations)
        
        # Identify patterns
        patterns = self._analyze_action_patterns(trajectory)
        analysis['patterns'] = patterns.get('patterns', [])
        
        # Identify risks
        if analysis['error_rate'] > 0.3:
            analysis['risks'].append('high_error_rate')
        if self._detect_action_cycle(trajectory):
            analysis['risks'].append('action_cycle_detected')
            
        return analysis

    def _analyze_action_patterns(self, trajectory: 'Trajectory') -> Dict[str, Any]:
        """Extract patterns from action sequence"""
        patterns = {
            'patterns': [],
            'frequent_actions': defaultdict(int),
            'transitions': defaultdict(list)
        }
        
        if not trajectory.actions:
            return patterns
            
        # Count action frequencies
        for action in trajectory.actions:
            action_type = action.get('type', '')
            patterns['frequent_actions'][action_type] += 1
            
        # Analyze action transitions (bigrams)
        for i in range(len(trajectory.actions) - 1):
            current = trajectory.actions[i].get('type', '')
            next_action = trajectory.actions[i + 1].get('type', '')
            patterns['transitions'][current].append(next_action)
            
        # Extract common sequences (using existing code from earlier)
        common_sequences = self._extract_common_sequences(trajectory.actions)
        patterns['patterns'] = [
            {
                'sequence': seq,
                'frequency': freq
            }
            for seq, freq in common_sequences.items()
        ]
        
        return patterns

    def _extract_common_sequences(self, actions: List[Dict[str, Any]], 
                                min_length: int = 2, max_length: int = 4) -> Dict[str, int]:
        """Extract common action sequences with their frequencies"""
        sequences = defaultdict(int)
        
        for length in range(min_length, min(max_length + 1, len(actions) + 1)):
            for i in range(len(actions) - length + 1):
                sequence = tuple(
                    action.get('type', '') 
                    for action in actions[i:i + length]
                )
                sequences[sequence] += 1
                
        return {
            seq: freq for seq, freq in sequences.items()
            if freq > 1  # Only keep repeated sequences
        }

    def save_patterns(self) -> None:
        """Save learned patterns to disk"""
        pattern_dir = Path(self.env.repo_path) / '.ai_agent' / 'patterns'
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        # Save recovery patterns
        recovery_path = pattern_dir / 'recovery_patterns.json'
        recovery_patterns = {
            error: {
                'actions': [a.to_dict() for a in pattern.sequence.steps],
                'success_rate': pattern.success_rate
            }
            for error, pattern in self.env.error_patterns.items()
        }
        with open(recovery_path, 'w') as f:
            json.dump(recovery_patterns, f)
            
        # Save action patterns (from trajectory analysis)
        action_path = pattern_dir / 'action_patterns.json'
        action_patterns = self.trajectory_manager.get_action_patterns()
        with open(action_path, 'w') as f:
            json.dump(action_patterns, f)

    def load_patterns(self) -> None:
        """Load previously saved patterns"""
        pattern_dir = Path(self.env.repo_path) / '.ai_agent' / 'patterns'
        
        if not pattern_dir.exists():
            return
            
        # Load recovery patterns
        recovery_path = pattern_dir / 'recovery_patterns.json'
        if recovery_path.exists():
            with open(recovery_path, 'r') as f:
                recovery_patterns = json.load(f)
                for error, pattern in recovery_patterns.items():
                    self.env.add_recovery_pattern(error, pattern['actions'])
                    
        # Load action patterns
        action_path = pattern_dir / 'action_patterns.json'
        if action_path.exists():
            with open(action_path, 'r') as f:
                action_patterns = json.load(f)
                self.trajectory_manager.load_action_patterns(action_patterns)

    def train_from_examples(self, examples_path: str) -> int:
        """Train agent from example trajectories"""
        examples_dir = Path(examples_path)
        if not examples_dir.exists():
            return 0
            
        loaded = 0
        for filepath in examples_dir.glob('*.json'):
            try:
                with open(filepath, 'r') as f:
                    example = json.load(f)
                    trajectory = type('Trajectory', (), example)()
                    self.trajectory_manager.store_trajectory(trajectory)
                    loaded += 1
            except:
                continue
                
        return loaded

    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor agent's performance metrics"""
        metrics = {
            'overall_success_rate': 0.0,
            'error_recovery_rate': 0.0,
            'pattern_usage': {},
            'common_errors': defaultdict(int),
            'execution_times': [],
            'state_coverage': set()
        }
        
        # Get all trajectories
        trajectories = self.trajectory_manager.load_trajectories()
        if not trajectories:
            return metrics
            
        # Calculate success rates
        successful = 0
        recoveries = 0
        recovery_attempts = 0
        
        for trajectory in trajectories:
            # Overall success
            if all(isinstance(obs, dict) and obs.get('status') == 'success'
                   for obs in trajectory.observations):
                successful += 1
                
            # Error recovery success
            for obs in trajectory.observations:
                if isinstance(obs, dict) and obs.get('error'):
                    recovery_attempts += 1
                    if any(isinstance(later_obs, dict) and 
                          later_obs.get('status') == 'success'
                          for later_obs in trajectory.observations[trajectory.observations.index(obs):]):
                        recoveries += 1
                    metrics['common_errors'][obs.get('error')] += 1
                    
            # Pattern usage
            patterns = self._analyze_action_patterns(trajectory)
            for pattern in patterns['patterns']:
                pattern_key = str(pattern['sequence'])
                metrics['pattern_usage'][pattern_key] = \
                    metrics['pattern_usage'].get(pattern_key, 0) + pattern['frequency']
                    
            # Execution time if available
            if hasattr(trajectory, 'start_time') and hasattr(trajectory, 'end_time'):
                metrics['execution_times'].append(
                    trajectory.end_time - trajectory.start_time
                )
                
            # State coverage
            if trajectory.final_state:
                metrics['state_coverage'].update(
                    f"{k}:{v}" for k, v in trajectory.final_state.items()
                    if isinstance(v, (str, int, bool))
                )
                
        # Calculate rates
        total = len(trajectories)
        metrics['overall_success_rate'] = successful / total if total > 0 else 0.0
        metrics['error_recovery_rate'] = \
            recoveries / recovery_attempts if recovery_attempts > 0 else 1.0
            
        # Convert state coverage to size
        metrics['state_coverage'] = len(metrics['state_coverage'])
        
        # Calculate execution time statistics if available
        if metrics['execution_times']:
            metrics['avg_execution_time'] = sum(metrics['execution_times']) / len(metrics['execution_times'])
            metrics['max_execution_time'] = max(metrics['execution_times'])
            metrics['min_execution_time'] = min(metrics['execution_times'])
            
        return metrics

    def evaluate_action(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Evaluate the quality of an action in the current state"""
        score = 0.0
        weights = {
            'pattern_match': 0.4,
            'state_impact': 0.3,
            'risk_assessment': 0.3
        }
        
        # Check if action follows learned patterns
        pattern_score = self._evaluate_pattern_match(action, state)
        
        # Assess potential state impact
        impact_score = self._evaluate_state_impact(action, state)
        
        # Assess risks
        risk_score = self._evaluate_risk(action, state)
        
        # Combine scores
        score = (
            weights['pattern_match'] * pattern_score +
            weights['state_impact'] * impact_score +
            weights['risk_assessment'] * risk_score
        )
        
        return min(1.0, max(0.0, score))

    def _evaluate_pattern_match(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Evaluate how well an action matches learned patterns"""
        score = 0.0
        
        # Check for direct pattern matches
        similar = self.trajectory_manager.retrieve_similar_trajectories(
            current_state=state,
            instruction=action.get('description', ''),
            limit=3
        )
        
        if similar:
            for trajectory in similar:
                if any(self._actions_match(a, action) for a in trajectory.actions):
                    score += 0.3
                    if all(isinstance(obs, dict) and obs.get('status') == 'success'
                           for obs in trajectory.observations):
                        score += 0.2
                        
        # Check transition patterns
        if self.current_trajectory and self.current_trajectory['actions']:
            last_action = self.current_trajectory['actions'][-1]
            if (last_action.get('type'), action.get('type')) in self._get_common_transitions():
                score += 0.2
                
        return min(1.0, score)

    def _evaluate_state_impact(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Evaluate potential impact of action on repository state"""
        score = 0.0
        
        # Check if action addresses current state issues
        if action.get('type') == 'fix_permissions' and \
           any('permission' in str(issue).lower() 
               for issue in state.get('warnings', [])):
            score += 0.4
            
        if action.get('type') == 'resolve_conflict' and \
           state.get('git_status', {}).get('merge_conflicts', []):
            score += 0.4
            
        # Check for positive state transitions
        if action.get('type') == 'edit_file' and \
           action.get('path') in state.get('git_status', {}).get('modified', []):
            score += 0.3
            
        if action.get('type') == 'git_commit' and \
           state.get('git_status', {}).get('staged', []):
            score += 0.3
            
        return min(1.0, score)

    def _evaluate_risk(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Evaluate potential risks of an action"""
        score = 1.0  # Start with max score and subtract for risks
        
        # Check for risky patterns
        if action.get('type') in ['delete_file', 'git_reset', 'git_clean']:
            score -= 0.3
            
        # Check for broad impacts
        if action.get('type') == 'git_checkout' and \
           state.get('git_status', {}).get('modified', []):
            score -= 0.2
            
        # Check for conflicts with current state
        if action.get('type') == 'edit_file':
            path = action.get('path', '')
            if path in state.get('git_status', {}).get('merge_conflicts', []):
                score -= 0.3
                
        return max(0.0, score)

    def _actions_match(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Check if two actions are effectively the same"""
        if action1.get('type') != action2.get('type'):
            return False
            
        # For file operations, check paths
        if action1.get('type') in ['edit_file', 'create_file', 'delete_file']:
            return action1.get('path') == action2.get('path')
            
        # For git operations, check commands
        if action1.get('type').startswith('git_'):
            return True  # Consider same type of git operations as matching
            
        return False

    def _get_common_transitions(self) -> Set[Tuple[str, str]]:
        """Get set of common action type transitions"""
        transitions = {
            ('create_file', 'edit_file'),
            ('edit_file', 'git_commit'),
            ('git_checkout', 'edit_file'),
            ('resolve_conflict', 'git_commit'),
            ('fix_permissions', 'edit_file')
        }
        return transitions

    def _check_goal_alignment(self, trajectory: 'Trajectory') -> float:
        """Check how well the actions align with the intended goal"""
        if not trajectory.instruction or not trajectory.actions:
            return 0.0
            
        instruction = trajectory.instruction.lower()
        actions = trajectory.actions
        
        # Define expected action patterns for common instructions
        patterns = {
            'create': {'create_file', 'edit_file'},
            'update': {'edit_file'},
            'delete': {'delete_file'},
            'fix': {'edit_file', 'fix_permissions', 'resolve_conflict'},
            'test': {'run_tests'},
            'commit': {'git_commit'},
            'merge': {'git_merge', 'resolve_conflict'}
        }
        
        # Check for pattern matches
        score = 0.0
        action_types = {a.get('type') for a in actions}
        
        for keyword, expected_types in patterns.items():
            if keyword in instruction:
                overlap = expected_types & action_types
                if overlap:
                    score += len(overlap) / len(expected_types)
                    
        # Check for file target matches
        target_files = {word for word in instruction.split() if '.' in word}
        action_files = {a.get('path', '').split('/')[-1] for a in actions}
        if target_files and target_files & action_files:
            score += 0.3
            
        # Check for success signals
        if any(isinstance(obs, dict) and obs.get('status') == 'success'
               for obs in trajectory.observations[-2:]):  # Last 2 observations
            score += 0.2
            
        return min(1.0, score)

    def assess_pattern_quality(self, pattern: Dict[str, Any], min_frequency: int = 3) -> float:
        """Assess the quality of an action pattern"""
        if not pattern or 'sequence' not in pattern:
            return 0.0
            
        score = 0.0
        sequence = pattern['sequence']
        frequency = pattern.get('frequency', 0)
        
        # Base quality on frequency
        if frequency >= min_frequency:
            score += 0.3
            
        # Check sequence validity
        if self._is_valid_sequence(sequence):
            score += 0.3
            
        # Check historical success
        success_rate = self._get_sequence_success_rate(sequence)
        score += 0.4 * success_rate
        
        return score
        
    def _is_valid_sequence(self, sequence: Tuple[str, ...]) -> bool:
        """Check if an action sequence is valid"""
        if not sequence:
            return False
            
        # Define valid action transitions
        valid_transitions = {
            'create_file': {'edit_file', 'git_commit'},
            'edit_file': {'git_commit', 'run_tests'},
            'git_commit': {'git_checkout', 'git_merge', 'edit_file'},
            'resolve_conflict': {'git_commit'},
            'fix_permissions': {'edit_file', 'git_commit'},
            'run_tests': {'git_commit', 'edit_file'}
        }
        
        # Check each transition
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_action = sequence[i + 1]
            if current in valid_transitions:
                if next_action not in valid_transitions[current]:
                    return False
            else:
                return False
                
        return True
        
    def _get_sequence_success_rate(self, sequence: Tuple[str, ...]) -> float:
        """Get historical success rate of an action sequence"""
        trajectories = [t for t in self.trajectory_manager.load_trajectories()
                       if len(t.actions) >= len(sequence)]
                       
        if not trajectories:
            return 0.0
            
        matches = 0
        successes = 0
        
        for trajectory in trajectories:
            # Look for sequence matches
            for i in range(len(trajectory.actions) - len(sequence) + 1):
                current = tuple(a.get('type', '') for a in 
                              trajectory.actions[i:i + len(sequence)])
                if current == sequence:
                    matches += 1
                    # Check if the sequence led to success
                    relevant_obs = trajectory.observations[i:i + len(sequence)]
                    if all(isinstance(obs, dict) and obs.get('status') == 'success'
                           for obs in relevant_obs):
                        successes += 1
                        
        return successes / matches if matches > 0 else 0.0

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate a quality report for the agent's performance"""
        report = {
            'patterns': {
                'total': 0,
                'high_quality': 0,
                'success_rates': {},
                'risk_levels': {}
            },
            'error_handling': {
                'recovery_patterns': len(self.env.error_patterns),
                'success_rate': 0.0
            },
            'coverage': {
                'action_types': set(),
                'file_types': set(),
                'state_transitions': 0
            }
        }
        
        # Analyze patterns
        patterns = self.trajectory_manager.get_action_patterns()
        report['patterns']['total'] = len(patterns)
        
        for pattern in patterns:
            quality = self.assess_pattern_quality(pattern)
            if quality > 0.7:
                report['patterns']['high_quality'] += 1
                
            # Track success rates
            sequence = str(pattern['sequence'])
            success_rate = self._get_sequence_success_rate(pattern['sequence'])
            report['patterns']['success_rates'][sequence] = success_rate
            
            # Assess risk levels
            risk_level = 'low'
            if any(action in str(pattern['sequence']) 
                   for action in ['delete', 'reset', 'clean']):
                risk_level = 'high'
            elif any(action in str(pattern['sequence'])
                     for action in ['merge', 'checkout']):
                risk_level = 'medium'
            report['patterns']['risk_levels'][sequence] = risk_level
            
        # Analyze error handling
        successful_recoveries = 0
        total_recoveries = 0
        for pattern in self.env.error_patterns.values():
            if pattern.success_rate > 0.5:
                successful_recoveries += 1
            total_recoveries += 1
            
        if total_recoveries > 0:
            report['error_handling']['success_rate'] = \
                successful_recoveries / total_recoveries
                
        # Analyze coverage
        trajectories = self.trajectory_manager.load_trajectories()
        for trajectory in trajectories:
            for action in trajectory.actions:
                report['coverage']['action_types'].add(action.get('type', ''))
                if 'path' in action:
                    ext = Path(action['path']).suffix
                    if ext:
                        report['coverage']['file_types'].add(ext)
                        
            if trajectory.final_state:
                report['coverage']['state_transitions'] += 1
                
        # Convert sets to lists for JSON serialization
        report['coverage']['action_types'] = list(report['coverage']['action_types'])
        report['coverage']['file_types'] = list(report['coverage']['file_types'])
        
        return report

    def _detect_action_cycle(self, trajectory: 'Trajectory') -> bool:
        """Detect if trajectory contains repeating action cycles"""
        if not hasattr(trajectory, 'actions') or len(trajectory.actions) < 4:
            return False
            
        # Look for repeating sequences of 2-3 actions
        for seq_len in range(2, 4):
            for i in range(len(trajectory.actions) - seq_len * 2 + 1):
                seq1 = trajectory.actions[i:i + seq_len]
                seq2 = trajectory.actions[i + seq_len:i + seq_len * 2]
                
                if self._sequences_match(seq1, seq2):
                    return True
                    
        return False
        
    def _sequences_match(self, seq1: List[Dict], seq2: List[Dict]) -> bool:
        """Check if two action sequences match"""
        if len(seq1) != len(seq2):
            return False
            
        for a1, a2 in zip(seq1, seq2):
            if a1.get('type') != a2.get('type'):
                return False
            if a1.get('path') != a2.get('path'):
                return False
                
        return True