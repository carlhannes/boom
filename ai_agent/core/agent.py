from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from .learner import SelfLearner
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
        # Pass the learner instance to TrajectoryManager
        self.trajectory_manager = TrajectoryManager(storage_path, learner=self.learner)
        self.bm25_top_k = bm25_top_k
        
    def set_learner(self, learner):
        """Set a new learner instance and update trajectory manager"""
        self.learner = learner
        self.trajectory_manager = TrajectoryManager(self.trajectory_manager.storage_path, learner=learner)

    def execute_task(self, instruction: str) -> Trajectory:
        """Execute a coding task based on the instruction"""
        # Get current environment state
        current_state = self.environment.get_state()
        
        # Get similar trajectories using hybrid retrieval
        similar_trajectories = self.trajectory_manager.retrieve_similar_trajectories(
            current_state=current_state,
            instruction=instruction,
            limit=5,  # Number of final examples to use
            bm25_top_k=self.bm25_top_k  # First-stage retrieval size
        )
        
        # If we have a very close match, reuse its actions
        if similar_trajectories and similar_trajectories[0].instruction == instruction:
            return similar_trajectories[0]
            
        # Initialize trajectory
        trajectory = Trajectory(
            instruction=instruction,
            actions=[],
            observations=[],
            final_state={}
        )
        
        while not self._task_complete(trajectory):
            # Plan next action using the LLM with retrieved examples
            next_action = self._plan_next_action(
                current_state,
                trajectory,
                similar_trajectories
            )
            
            if next_action is None:
                break
                
            # Execute action and record results
            try:
                observation = self.environment.execute_action(next_action)
                trajectory.actions.append(next_action)
                trajectory.observations.append(observation)
                
                # Update current state
                current_state = self.environment.get_state()
            except Exception as e:
                # Record failed action attempt
                trajectory.observations.append({
                    'error': str(e),
                    'status': 'failed'
                })
        
        trajectory.final_state = current_state
        
        # Do backward construction to align instruction with actual steps
        refined_instruction = self.learner.backward_construct({
            'instruction': instruction,
            'actions': trajectory.actions,
            'observations': trajectory.observations,
            'similar_trajectories': [t.to_dict() for t in similar_trajectories]
        })
        trajectory.instruction = refined_instruction
        
        # Store trajectory for future reference
        self.trajectory_manager.store_trajectory(trajectory)
        
        return trajectory

    def _task_complete(self, trajectory: Trajectory) -> bool:
        """Determine if the current task is complete"""
        if not trajectory.observations:
            return False
            
        last_obs = trajectory.observations[-1]
        
        # Check for explicit completion signal
        if isinstance(last_obs, dict):
            if last_obs.get('status') == 'complete':
                return True
            if last_obs.get('error'):
                return True  # Stop on errors
                
        # Could add more sophisticated completion detection here
        return len(trajectory.actions) >= 10  # Limit action sequence length

    def _plan_next_action(self,
                         current_state: Dict[str, Any],
                         trajectory: Trajectory,
                         examples: List[Trajectory]) -> Optional[Dict[str, Any]]:
        """Plan the next action based on current state and similar examples"""
        # Use retrieved examples to guide action planning
        example_data = []
        for ex in examples:
            if ex.actions:  # Only use examples with successful actions
                example_data.append({
                    'instruction': ex.instruction,
                    'actions': ex.actions,
                    'observations': ex.observations
                })

        # For tests, use mock implementation if available
        if hasattr(self.learner, 'chat_completion'):
            return self.learner.chat_completion([
                {"role": "system", "content": "You are a coding agent that plans actions based on the current state and similar examples."},
                {"role": "user", "content": (
                    f"Current task: {trajectory.instruction}\n"
                    f"Current state: {json.dumps(current_state, indent=2)}\n"
                    f"Similar examples: {json.dumps(example_data, indent=2)}\n"
                    f"Current trajectory: {json.dumps(trajectory.to_dict(), indent=2)}"
                )}
            ])

        # Real implementation using OpenAI
        response = self.learner.client.chat.completions.create(
            model=self.learner.model,
            messages=[
                {"role": "system", "content": (
                    "You are a coding agent that plans actions based on the current "
                    "state and similar examples. Plan the next action to take."
                )},
                {"role": "user", "content": (
                    f"Current task: {trajectory.instruction}\n"
                    f"Current state: {json.dumps(current_state, indent=2)}\n"
                    f"Similar examples: {json.dumps(example_data, indent=2)}\n"
                    f"Current trajectory: {json.dumps(trajectory.to_dict(), indent=2)}\n"
                    "\nWhat should be the next action?"
                )}
            ]
        )

        try:
            # Parse action from response
            action_text = response.choices[0].message.content
            action = json.loads(action_text)
            return action
        except:
            return None