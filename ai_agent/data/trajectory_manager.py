from typing import List, Dict, Any, Optional, Tuple
import json
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from ..core.learner import SelfLearner

class Trajectory:
    def __init__(self, 
                instruction: str,
                actions: List[Dict[str, Any]],
                observations: List[Dict[str, Any]],
                final_state: Dict[str, Any]):
        self.instruction = instruction
        self.actions = actions
        self.observations = observations
        self.final_state = final_state
        self._embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for storage"""
        return {
            'instruction': self.instruction,
            'actions': self.actions,
            'observations': self.observations,
            'final_state': self.final_state
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """Create trajectory from dictionary"""
        return cls(
            instruction=data['instruction'],
            actions=data['actions'],
            observations=data['observations'],
            final_state=data['final_state']
        )

def simple_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing, preserving code-specific terms"""
    # Split on whitespace but preserve dots for file extensions
    tokens = []
    for word in text.split():
        # If word contains a dot (likely a file extension), keep it whole
        if '.' in word:
            tokens.append(word.lower().strip('!,.?;:'))
        else:
            # Otherwise split and lowercase
            clean_word = word.lower().strip('!,.?;:')
            if clean_word:
                tokens.append(clean_word)
    return tokens

class TrajectoryManager:
    def __init__(self, storage_path: str, learner: Optional[SelfLearner] = None):
        """Initialize TrajectoryManager with optional learner for testing"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.learner = learner or SelfLearner()
        self.bm25_index = None
        self.trajectories = []
        self.rebuild_index()
    
    def rebuild_index(self):
        """Build BM25 index from stored trajectories"""
        self.trajectories = self.load_trajectories()
        if not self.trajectories:
            return
            
        # Prepare documents for BM25
        tokenized_docs = []
        for traj in self.trajectories:
            tokens = simple_tokenize(traj.instruction)
            tokenized_docs.append(tokens)
            
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def store_trajectory(self, trajectory: Trajectory):
        """Store a trajectory for later retrieval"""
        # Create embedding for similarity matching
        if trajectory._embedding is None:
            trajectory._embedding = self.learner.compute_embedding(
                trajectory.instruction
            )
            
        # Store trajectory data with embedding
        data = trajectory.to_dict()
        data['embedding'] = trajectory._embedding.tolist()
        
        # Use a hash of instruction + timestamp as filename
        filename = f"{hash(trajectory.instruction)}_{int(time.time())}.json"
        filepath = self.storage_path / filename
        
        with filepath.open('w') as f:
            json.dump(data, f, indent=2)
            
        # Rebuild index to include new trajectory
        self.rebuild_index()
        
    def load_trajectories(self) -> List[Trajectory]:
        """Load all stored trajectories"""
        trajectories = []
        for file in self.storage_path.glob("*.json"):
            with file.open() as f:
                data = json.load(f)
                traj = Trajectory.from_dict(data)
                if 'embedding' in data:
                    traj._embedding = np.array(data['embedding'])
                trajectories.append(traj)
        return trajectories

    def should_apply_rerank(self, bm25_scores: np.ndarray, query: str) -> bool:
        """
        Determine if re-ranking should be applied based on BM25 scores and query characteristics
        """
        if not len(bm25_scores):
            return False
            
        # 1. Check BM25 score confidence
        top_score = max(bm25_scores)
        runner_up = sorted(bm25_scores)[-2] if len(bm25_scores) > 1 else 0
        score_gap = top_score - runner_up
        
        # If top score is very high and gap is significant, BM25 found strong match
        if top_score > 0.8 and score_gap > 0.3:
            return False
            
        # 2. Query characteristics
        tokens = simple_tokenize(query)
        
        # Short keyword queries often do well with just BM25
        if len(tokens) <= 2:
            return False
            
        # Check for code-specific patterns that do well with BM25
        code_patterns = {'.py', '.js', 'def ', 'class ', 'function', 'error:', 'exception:'}
        if any(pattern in query.lower() for pattern in code_patterns):
            return False
            
        # 3. Check for natural language / conceptual queries
        nl_indicators = {'how', 'what', 'why', 'when', 'implement', 'create', 'design', 'pattern'}
        if any(word in tokens for word in nl_indicators):
            return True
            
        # By default, use re-ranking for queries that don't match above cases
        return True

    def retrieve_similar_trajectories(self,
                                   current_state: Dict[str, Any],
                                   instruction: str,
                                   limit: int = 5,
                                   bm25_top_k: int = 50) -> List[Trajectory]:
        """
        Two-stage retrieval with conditional re-ranking:
        1. Get top-k candidates using BM25
        2. Optionally re-rank using embedding similarity and state similarity
        """
        if not self.trajectories:
            return []
        
        # Stage 1: BM25 retrieval
        query_tokens = simple_tokenize(instruction)
        if not self.bm25_index:
            self.rebuild_index()
            
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        top_k_indices = np.argsort(bm25_scores)[-bm25_top_k:]
        
        # Decide whether to apply re-ranking
        if not self.should_apply_rerank(bm25_scores, instruction):
            # Return top BM25 results directly
            return [self.trajectories[i] for i in top_k_indices[-limit:]]
        
        # Stage 2: Re-ranking with embeddings and state similarity
        query_embedding = self.learner.compute_embedding(instruction)
        scored_trajectories: List[Tuple[float, Trajectory]] = []
        
        for idx in top_k_indices:
            traj = self.trajectories[idx]
            
            # Combine BM25 score (normalized), embedding similarity, and state similarity
            bm25_score = bm25_scores[idx] / max(bm25_scores)
            
            # Embedding similarity
            if not hasattr(traj, '_embedding') or traj._embedding is None:
                traj._embedding = self.learner.compute_embedding(traj.instruction)
            embedding_sim = np.dot(query_embedding, traj._embedding)
            
            # State similarity
            state_sim = self._compute_state_similarity(current_state, traj.final_state)
            
            # Combined score (weighted sum)
            final_score = (0.3 * bm25_score + 
                         0.4 * embedding_sim + 
                         0.3 * state_sim)
            
            scored_trajectories.append((final_score, traj))
        
        # Sort by final score and return top results
        scored_trajectories.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_trajectories[:limit]]

    def _compute_state_similarity(self,
                                state1: Dict[str, Any],
                                state2: Dict[str, Any]) -> float:
        """
        Compute similarity between two environment states
        Basic implementation - could be enhanced with more sophisticated matching
        """
        score = 0.0
        
        # Compare files
        files1 = set(state1.get('files', []))
        files2 = set(state2.get('files', []))
        file_overlap = len(files1 & files2) / max(len(files1 | files2), 1)
        score += 0.5 * file_overlap
        
        # Compare git status
        status1 = state1.get('git_status', {})
        status2 = state2.get('git_status', {})
        
        for key in ['modified', 'staged', 'untracked']:
            s1 = set(status1.get(key, []))
            s2 = set(status2.get(key, []))
            if s1 or s2:
                overlap = len(s1 & s2) / max(len(s1 | s2), 1)
                score += 0.1 * overlap
                
        return score