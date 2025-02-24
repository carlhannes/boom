"""Example usage of the AI coding agent"""
from pathlib import Path
from ai_agent.core.agent import CodingAgent
from ai_agent.core.learner import SelfLearner
from ai_agent.data.trajectory_manager import TrajectoryManager

def main():
    # Initialize paths
    repo_path = Path.cwd()
    storage_path = Path.home() / '.ai-agent' / 'trajectories'
    
    # Initialize components
    agent = CodingAgent(str(repo_path), str(storage_path))
    learner = SelfLearner()
    
    # Example: Generate tasks from repository documentation
    docs = [
        "Implement input validation for user data",
        "Add error handling to the database connections",
        "Create unit tests for the authentication module"
    ]
    
    # Generate and store example tasks
    tasks = learner.generate_tasks_from_docs(docs)
    for task in tasks:
        print(f"\nExecuting task: {task.instruction}")
        trajectory = agent.execute_task(task.instruction)
        print(f"Actions taken: {len(trajectory.actions)}")
        print(f"Final instruction after backward construction: {trajectory.instruction}")

if __name__ == "__main__":
    main()