from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np
import os
from openai import OpenAI, OpenAIError
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
            except OpenAIError as e:
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

    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text using Sentence Transformers model"""
        if self.embedding_model is None:
            # For testing scenarios
            return np.ones(384)  # Default embedding dimension for all-MiniLM-L6-v2
            
        # Get embeddings using Sentence Transformers
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def generate_tasks_from_docs(self, docs: List[str]) -> List[TaskExample]:
        """Generate tasks by analyzing documentation"""
        if not self.client:
            # For testing
            return [TaskExample(instruction=doc, context={}) for doc in docs]
            
        examples = []
        for doc in docs:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert at generating coding tasks from documentation. "
                        "Extract concrete, actionable tasks that can be performed."
                    )},
                    {"role": "user", "content": f"Generate a specific coding task from this documentation:\n{doc}"}
                ]
            )
            
            instruction = response.choices[0].message.content.strip()
            examples.append(TaskExample(instruction=instruction, context={"doc": doc}))
            
        return examples

    def backward_construct(self, trajectory: Dict[str, Any]) -> str:
        """Construct a clear instruction from the completed trajectory
        
        This helps improve future retrievals by generating high-quality task descriptions
        that capture what was actually done, rather than just the initial instruction.
        """
        if not self.client:
            # For testing
            return trajectory.get('instruction', '')
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are an expert at describing coding changes. Given a sequence "
                    "of actions and their results, write a clear instruction that "
                    "describes what was accomplished."
                )},
                {"role": "user", "content": (
                    "Here are the actions that were performed:\n"
                    f"{json.dumps(trajectory.get('actions', []), indent=2)}\n\n"
                    "Write a single instruction that precisely describes what "
                    "these actions accomplished:"
                )}
            ]
        )
        
        return response.choices[0].message.content.strip()

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