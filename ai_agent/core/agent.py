from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
from .learner import SelfLearner, TaskExample
from ..environment.git_env import GitEnvironment
from ..data.trajectory_manager import TrajectoryManager, Trajectory

class CodingAgent:
    def __init__(self, repo_path: str, storage_path: str, bm25_top_k: int = 50, api_key: Optional[str] = None):
        """Initialize the coding agent
        
        Args:
            repo_path: Path to Git repository
            storage_path: Path to store trajectories
            bm25_top_k: Number of candidates to retrieve in first-stage BM25 retrieval
            api_key: Optional OpenAI API key for testing
        """
        self.environment = GitEnvironment(repo_path)
        self.learner = SelfLearner(api_key=api_key)
        self.trajectory_manager = TrajectoryManager(storage_path, learner=self.learner)
        self.bm25_top_k = bm25_top_k
        
    def set_learner(self, learner):
        """Set a new learner instance and update trajectory manager"""
        self.learner = learner
        self.trajectory_manager = TrajectoryManager(self.trajectory_manager.storage_path, learner=learner)
        
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
        
        # Track failed actions to avoid cycles
        failed_actions = set()
        
        while not self._task_complete(trajectory):
            try:
                # Plan next action
                next_action = self._plan_next_action(
                    current_state,
                    trajectory,
                    similar_trajectories
                )
                
                if next_action is None:
                    break
                    
                # Skip if this action previously failed
                action_key = f"{next_action.get('type')}:{next_action.get('path')}"
                if action_key in failed_actions:
                    continue
                    
                # Execute action with retry logic
                for attempt in range(max_retries):
                    try:
                        observation = self.environment.execute_action(next_action)
                        trajectory.actions.append(next_action)
                        trajectory.observations.append(observation)
                        current_state = self.environment.get_state()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(retry_delay)
                        
            except Exception as e:
                # Record the failed action
                failed_actions.add(action_key)
                
                # Try to recover
                recovery_action = self._handle_action_error(e, trajectory, current_state)
                if recovery_action:
                    try:
                        observation = self.environment.execute_action(recovery_action)
                        trajectory.actions.append(recovery_action)
                        trajectory.observations.append(observation)
                        current_state = self.environment.get_state()
                        continue
                    except:
                        pass
                        
                # Record failure if recovery wasn't possible
                trajectory.observations.append({
                    'error': str(e),
                    'status': 'failed',
                    'state': current_state
                })
                
                # Check if we should continue
                if len(failed_actions) > len(trajectory.actions) * 0.5:
                    # Too many failures, stop execution
                    break
        
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