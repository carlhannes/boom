from typing import List, Dict, Any, Optional, Tuple
import json
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from ..core.learner import SelfLearner

class TrajectoryQualityMetrics:
    """Metrics for assessing trajectory quality"""
    def __init__(self, trajectory: 'Trajectory'):
        self.success_rate = self._calculate_success_rate(trajectory)
        self.completion = self._calculate_completion(trajectory)
        self.complexity = self._calculate_complexity(trajectory)
        
    def _calculate_success_rate(self, trajectory: 'Trajectory') -> float:
        """Calculate the success rate of actions in the trajectory"""
        if not trajectory.observations:
            return 0.0
        
        successes = sum(
            1 for obs in trajectory.observations
            if isinstance(obs, dict) and obs.get('status') == 'success'
        )
        return successes / len(trajectory.observations)
        
    def _calculate_completion(self, trajectory: 'Trajectory') -> float:
        """Assess how complete the trajectory is"""
        if not trajectory.actions:
            return 0.0
            
        # Check for explicit completion marker
        if trajectory.observations and isinstance(trajectory.observations[-1], dict):
            if trajectory.observations[-1].get('status') == 'complete':
                return 1.0
                
        # Otherwise estimate based on action outcomes
        return self.success_rate
        
    def _calculate_complexity(self, trajectory: 'Trajectory') -> float:
        """Assess trajectory complexity (0-1 scale)"""
        if not trajectory.actions:
            return 0.0
            
        # Base complexity on action variety and count
        unique_action_types = len({a.get('type') for a in trajectory.actions})
        action_count = len(trajectory.actions)
        
        # Normalize complexity score
        return min(1.0, (unique_action_types * 0.4 + action_count * 0.1))
        
    def get_overall_score(self) -> float:
        """Calculate overall quality score (0-1)"""
        return (
            self.success_rate * 0.4 +    # Weight success most heavily
            self.completion * 0.4 +      # Completion is equally important
            self.complexity * 0.2        # Complexity matters but less so
        )

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
        self.quality: Optional[TrajectoryQualityMetrics] = None

    def compute_quality_metrics(self) -> TrajectoryQualityMetrics:
        """Compute quality metrics for this trajectory"""
        if self.quality is None:
            self.quality = TrajectoryQualityMetrics(self)
        return self.quality

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for storage"""
        data = {
            'instruction': self.instruction,
            'actions': self.actions,
            'observations': self.observations,
            'final_state': self.final_state
        }
        if self.quality:
            data['quality'] = {
                'success_rate': self.quality.success_rate,
                'completion': self.quality.completion,
                'complexity': self.quality.complexity,
                'overall_score': self.quality.get_overall_score()
            }
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """Create trajectory from dictionary"""
        traj = cls(
            instruction=data['instruction'],
            actions=data['actions'],
            observations=data['observations'],
            final_state=data['final_state']
        )
        if 'quality' in data:
            traj.quality = TrajectoryQualityMetrics(traj)
        return traj

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
        self.min_quality_threshold = 0.5  # Minimum quality score to keep trajectory
        self.rebuild_index()
    
    def rebuild_index(self):
        """Build BM25 index from stored trajectories"""
        self.trajectories = self.load_trajectories()
        
        # Filter low-quality trajectories
        filtered_trajectories = []
        for traj in self.trajectories:
            metrics = traj.compute_quality_metrics()
            if metrics.get_overall_score() >= self.min_quality_threshold:
                filtered_trajectories.append(traj)
        
        self.trajectories = filtered_trajectories
        
        if not self.trajectories:
            return
            
        # Prepare documents for BM25
        tokenized_docs = []
        for traj in self.trajectories:
            tokens = simple_tokenize(traj.instruction)
            tokenized_docs.append(tokens)
            
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def store_trajectory(self, trajectory: Trajectory):
        """Store a trajectory for later retrieval if it meets quality standards"""
        # Compute quality metrics
        metrics = trajectory.compute_quality_metrics()
        
        # Only store if quality meets threshold
        if metrics.get_overall_score() >= self.min_quality_threshold:
            # Create embedding for similarity matching
            if trajectory._embedding is None:
                trajectory._embedding = self.learner.compute_embedding(
                    trajectory.instruction
                )
                
            # Store trajectory data with embedding and quality metrics
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

    def _compute_context_similarity(self, query_context: Dict[str, Any], traj_context: Dict[str, Any]) -> float:
        """
        Compute similarity between technical contexts and repository states
        """
        score = 0.0
        weights = {
            'frameworks': 0.3,
            'languages': 0.3,
            'file_types': 0.2,
            'patterns': 0.2
        }
        
        for key, weight in weights.items():
            query_set = set(query_context.get(key, []))
            traj_set = set(traj_context.get(key, []))
            if query_set or traj_set:
                overlap = len(query_set & traj_set) / max(len(query_set | traj_set), 1)
                score += weight * overlap
                
        return score

    def _boost_by_quality(self, base_score: float, trajectory: Trajectory) -> float:
        """
        Boost ranking score based on trajectory quality metrics
        """
        if not trajectory.quality:
            trajectory.quality = TrajectoryQualityMetrics(trajectory)
            
        quality_score = trajectory.quality.get_overall_score()
        
        # Exponential boost for high-quality trajectories
        quality_boost = 1.0 + (quality_score ** 2) * 0.5
        return base_score * quality_boost

    def _extract_action_patterns(self) -> Dict[str, SequencePattern]:
        """Extract common successful action patterns from stored trajectories"""
        sequences = []
        for trajectory in self.trajectories:
            sequence = ActionSequence.from_trajectory(trajectory)
            if sequence.success_rate > 0.8:  # Only consider highly successful trajectories
                sequences.append(sequence)
                
        # Group sequences by semantic type
        type_groups = {}
        for seq in sequences:
            if seq.semantic_type not in type_groups:
                type_groups[seq.semantic_type] = []
            type_groups[seq.semantic_type].append(seq)
        
        # Create patterns for each type
        patterns = {}
        for sem_type, type_sequences in type_groups.items():
            if len(type_sequences) >= 2:  # Need at least 2 sequences to form a pattern
                patterns[sem_type] = SequencePattern(type_sequences)
                
        return patterns

    def _find_matching_patterns(self, query: str, current_state: Dict[str, Any]) -> List[ActionSequence]:
        """Find action patterns that match the query and current state"""
        patterns = self._extract_action_patterns()
        
        # Convert query to probable semantic types
        query_lower = query.lower()
        likely_types = set()
        
        if any(word in query_lower for word in ['create', 'add', 'new']):
            likely_types.add('create')
        if any(word in query_lower for word in ['edit', 'update', 'modify', 'change']):
            likely_types.add('modify')
        if any(word in query_lower for word in ['test', 'verify', 'check']):
            likely_types.add('test')
        if any(word in query_lower for word in ['fix', 'repair', 'resolve']):
            likely_types.add('fix')
        
        matching_sequences = []
        
        # Check each pattern type that matches query semantics
        for sem_type in likely_types:
            if sem_type in patterns:
                pattern = patterns[sem_type]
                
                # Get the best example from each matching pattern
                example = pattern.get_best_example()
                
                # Verify state compatibility
                if self._compute_state_similarity(
                    current_state,
                    example.steps[0].state_before
                ) > 0.6:  # Reasonable state match threshold
                    matching_sequences.append(example)
        
        return matching_sequences

    def retrieve_similar_trajectories(self,
                                   current_state: Dict[str, Any],
                                   instruction: str,
                                   limit: int = 5,
                                   bm25_top_k: int = 50) -> List[Trajectory]:
        """
        Enhanced retrieval with pattern matching and context awareness
        """
        if not self.trajectories:
            return []
        
        # Stage 1: Pattern-based matching
        pattern_matches = self._find_matching_patterns(instruction, current_state)
        
        # Convert pattern matches back to trajectories
        pattern_trajectories = []
        for sequence in pattern_matches:
            # Find the original trajectory this sequence came from
            for traj in self.trajectories:
                if all(
                    action == step.action
                    for action, step in zip(traj.actions, sequence.steps)
                ):
                    pattern_trajectories.append(traj)
                    break
        
        # Stage 2: Standard retrieval for remaining slots
        remaining_slots = max(0, limit - len(pattern_trajectories))
        if remaining_slots > 0:
            query_tokens = simple_tokenize(instruction)
            if not self.bm25_index:
                self.rebuild_index()
                
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            max_score = max(bm25_scores) if any(bm25_scores) else 1.0
            top_k_indices = np.argsort(bm25_scores)[-bm25_top_k:][::-1]
            
            # Extract technical context
            query_context = {
                'frameworks': current_state.get('frameworks', []),
                'languages': current_state.get('languages', []),
                'file_types': {Path(f).suffix[1:] for f in current_state.get('files', []) if Path(f).suffix},
                'patterns': current_state.get('patterns', [])
            }
            
            if not self.should_apply_rerank(bm25_scores, instruction):
                # BM25-only with quality boosting
                scored_trajectories = [
                    (self._boost_by_quality(bm25_scores[i] / max_score, self.trajectories[i]), self.trajectories[i])
                    for i in top_k_indices
                    if self.trajectories[i] not in pattern_trajectories
                ]
                scored_trajectories.sort(key=lambda x: x[0], reverse=True)
                standard_matches = [t for _, t in scored_trajectories[:remaining_slots]]
            else:
                # Enhanced reranking
                query_embedding = self.learner.compute_embedding(instruction)
                scored_trajectories = []
                
                for idx in top_k_indices:
                    traj = self.trajectories[idx]
                    
                    # Skip if already in pattern matches
                    if traj in pattern_trajectories:
                        continue
                    
                    # Calculate scores
                    bm25_score = bm25_scores[idx] / max_score
                    
                    if not hasattr(traj, '_embedding') or traj._embedding is None:
                        traj._embedding = self.learner.compute_embedding(traj.instruction)
                    embedding_sim = np.dot(query_embedding, traj._embedding)
                    
                    state_sim = self._compute_state_similarity(current_state, traj.final_state)
                    context_sim = self._compute_context_similarity(
                        query_context,
                        {
                            'frameworks': traj.final_state.get('frameworks', []),
                            'languages': traj.final_state.get('languages', []),
                            'file_types': {Path(f).suffix[1:] for f in traj.final_state.get('files', []) if Path(f).suffix},
                            'patterns': traj.final_state.get('patterns', [])
                        }
                    )
                    
                    base_score = (
                        0.25 * bm25_score +
                        0.30 * embedding_sim +
                        0.25 * state_sim +
                        0.20 * context_sim
                    )
                    
                    final_score = self._boost_by_quality(base_score, traj)
                    scored_trajectories.append((final_score, traj))
                
                scored_trajectories.sort(key=lambda x: x[0], reverse=True)
                standard_matches = [t for _, t in scored_trajectories[:remaining_slots]]
            
        # Combine pattern matches with standard retrieval
        return pattern_trajectories + standard_matches

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