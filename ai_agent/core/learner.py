from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer

@dataclass
class TaskExample:
    """Represents a generated task example with its instruction and context"""
    instruction: str
    context: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class SelfLearner:
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, client=None, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize SelfLearner with optional API key for testing
        
        Args:
            model: The LLM model to use for task generation and planning
            api_key: Optional OpenAI API key for testing
            client: Optional pre-configured OpenAI client
            embedding_model: Model to use for embeddings, defaults to all-MiniLM-L6-v2
        """
        self.model = model
        if client is not None:
            self.client = client
        else:
            try:
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                # For testing without API key
                if api_key == "mock-key":
                    self.client = None
                else:
                    raise e
                    
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            if api_key == "mock-key":
                self.embedding_model = None
            else:
                raise e

    def generate_tasks_from_docs(self, docs: List[str]) -> List[TaskExample]:
        """Generate tasks by analyzing documentation"""
        tasks = []
        for doc in docs:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a coding agent that generates realistic coding tasks "
                        "from documentation. Generate specific, actionable tasks that "
                        "could be performed in a codebase."
                    )},
                    {"role": "user", "content": f"Generate 3 specific coding tasks based on this documentation:\n\n{doc}"}
                ]
            )
            
            # Parse tasks from response
            task_text = response.choices[0].message.content
            for line in task_text.split('\n'):
                if line.strip():
                    tasks.append(TaskExample(
                        instruction=line.strip(),
                        context={"source_doc": doc}
                    ))
        
        return tasks
    
    def backward_construct(self, trajectory: Dict[str, Any]) -> str:
        """
        Given a trajectory of actions and observations, construct a precise
        instruction that matches what actually happened
        """
        # Construct prompt with the full trajectory
        actions_text = "\n".join(
            f"Action {i+1}: {action['type']} - {action.get('description', '')}"
            for i, action in enumerate(trajectory['actions'])
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a coding agent that writes precise instructions "
                    "based on actual sequences of actions taken in a codebase. "
                    "Write a single, specific instruction that accurately describes "
                    "the sequence of actions, focusing on what was actually done."
                )},
                {"role": "user", "content": (
                    f"Original instruction: {trajectory['instruction']}\n\n"
                    f"Actual actions taken:\n{actions_text}\n\n"
                    "Write a single instruction that precisely describes what "
                    "these actions accomplished:"
                )}
            ]
        )
        
        return response.choices[0].message.content.strip()
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text using Sentence Transformers model"""
        if self.embedding_model is None:
            # For testing scenarios
            return np.ones(384)  # Default embedding dimension for all-MiniLM-L6-v2
            
        # Get embeddings using Sentence Transformers
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def retrieve_similar_trajectories(self,
                                   query: str,
                                   trajectories: List[Dict[str, Any]],
                                   k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectories using embedding similarity
        """
        if not trajectories:
            return []
            
        query_embedding = self.compute_embedding(query)
        
        # Compute similarities
        similarities = []
        for traj in trajectories:
            if 'embedding' not in traj:
                traj['embedding'] = self.compute_embedding(traj['instruction'])
            
            similarity = np.dot(query_embedding, traj['embedding'])
            similarities.append((similarity, traj))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [traj for _, traj in similarities[:k]]