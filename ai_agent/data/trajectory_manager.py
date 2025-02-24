from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import json
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

# Only for type hints
if TYPE_CHECKING:
    from ..core.learner import SelfLearner
from .quality_metrics import QualityMetrics, QualityScore
from .state_analyzer import StateChangeAnalyzer, StateChange
from .sequence import SequencePattern, ActionSequence

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
        self.quality_metrics = QualityMetrics()
        self.trajectory_scores: Dict[int, QualityScore] = {}
        self.state_analyzer = StateChangeAnalyzer()
        
        # Load existing trajectories
        stored = self.load_trajectories()
        if stored:
            self.trajectories.extend(stored)
            self.rebuild_index()  # Only build index if we have trajectories

    def rebuild_index(self):
        """Build BM25 index from stored trajectories"""
        if not hasattr(self, 'trajectories'):
            self.trajectories = []
        
        stored = self.load_trajectories()
        if stored:
            self.trajectories.extend(stored)
        
        # Filter low-quality trajectories
        filtered_trajectories = []
        for traj in self.trajectories:
            metrics = traj.compute_quality_metrics()
            if metrics.get_overall_score() >= self.min_quality_threshold:
                filtered_trajectories.append(traj)
        
        self.trajectories = filtered_trajectories
        
        # Build BM25 index
        tokenized_docs = []
        if self.trajectories:
            # Prepare documents for BM25
            for traj in self.trajectories:
                tokens = simple_tokenize(traj.instruction)
                tokenized_docs.append(tokens)
        else:
            # Add a dummy document to avoid division by zero
            tokenized_docs.append(['dummy'])
            
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def store_trajectory(self, trajectory: Trajectory) -> bool:
        """Store trajectory with state change analysis"""
        # Analyze state changes
        changes = []
        for i in range(len(trajectory.observations) - 1):
            state_before = trajectory.observations[i].get('state_before', {})
            state_after = trajectory.observations[i + 1].get('state_after', {})
            
            if state_before and state_after:
                changes.extend(
                    self.state_analyzer.analyze_change(state_before, state_after)
                )
        
        # Compute quality with state change impact
        score = self.quality_metrics.compute_trajectory_quality(
            trajectory,
            trajectory.instruction
        )
        
        # Adjust quality score based on state changes
        if changes:
            avg_impact = sum(c.impact for c in changes) / len(changes)
            adjusted_score = QualityScore(
                success_rate=score.success_rate,
                consistency=score.consistency,
                efficiency=score.efficiency * (1 - 0.2 * avg_impact),  # Penalize high-impact changes
                relevance=score.relevance,
                safety=score.safety * (1 - 0.3 * avg_impact)  # Reduce safety score for high-impact changes
            )
        else:
            adjusted_score = score
        
        # Check if trajectory should be stored
        if self.quality_metrics.should_filter_trajectory(adjusted_score):
            return False
            
        # Update patterns with state changes
        self.state_analyzer._update_patterns(changes)
        
        # Store trajectory with enhanced metadata
        trajectory_id = len(self.trajectories)
        self.trajectories.append(trajectory)
        self.trajectory_scores[trajectory_id] = adjusted_score
        
        # Store state changes for future reference
        if not hasattr(trajectory, 'state_changes'):
            trajectory.state_changes = changes
        
        # Rebuild index if needed
        if self.bm25_index is not None:
            self.rebuild_index()
            
        return True

    def maintain_quality(self, min_score: float = 0.7) -> int:
        """
        Maintain trajectory library quality by removing low-quality examples
        Returns number of trajectories removed
        """
        initial_count = len(self.trajectories)
        
        # Re-score all trajectories
        self.trajectory_scores = {}
        high_quality_trajectories = []
        
        for trajectory in self.trajectories:
            score = self.quality_metrics.compute_trajectory_quality(
                trajectory,
                trajectory.instruction
            )
            
            if score.total_score >= min_score:
                trajectory_id = len(high_quality_trajectories)
                high_quality_trajectories.append(trajectory)
                self.trajectory_scores[trajectory_id] = score
        
        # Update trajectory list
        self.trajectories = high_quality_trajectories
        
        # Rebuild index after filtering
        if self.bm25_index is not None:
            self.rebuild_index()
        
        return initial_count - len(self.trajectories)

    def get_trajectory_quality(self, trajectory_id: int) -> Optional[QualityScore]:
        """Get quality score for a trajectory"""
        return self.trajectory_scores.get(trajectory_id)

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
                                   min_quality: float = 0.7,
                                   bm25_top_k: int = 50) -> List[Trajectory]:
        """Enhanced retrieval with hybrid matching including state patterns and BM25"""
        if not self.trajectories:
            return []
            
        # First get candidates using text similarity
        query_tokens = simple_tokenize(instruction)
        if not self.bm25_index:
            self.rebuild_index()
            
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        max_score = max(bm25_scores) if any(bm25_scores) else 1.0
        top_k_indices = np.argsort(bm25_scores)[-bm25_top_k:][::-1]
        
        # Get current changes if available
        current_changes = []
        if hasattr(current_state, 'previous_state'):
            current_changes = self.state_analyzer.analyze_change(
                current_state.get('previous_state', {}),
                current_state
            )
        
        scored_candidates = []
        
        for idx in top_k_indices:
            traj = self.trajectories[idx]
            
            # Calculate different similarity components
            bm25_score = bm25_scores[idx] / max_score
            
            # Get trajectory's state changes
            trajectory_changes = getattr(traj, 'state_changes', [])
            
            # Calculate state pattern similarity if we have current changes
            state_similarity = 0.0
            if current_changes and trajectory_changes:
                similar_patterns = self.state_analyzer.get_similar_changes(
                    current_changes,
                    threshold=0.6
                )
                if similar_patterns:
                    max_similarity = max(
                        self._calculate_change_similarity(current_changes, pattern)
                        for pattern in similar_patterns
                    )
                    state_similarity = max_similarity
            
            # Get environment context match
            context_sim = self._compute_context_match(
                current_state,
                traj.final_state
            )
            
            # Calculate action applicability score
            action_score = self._compute_action_applicability(
                current_state,
                traj.actions
            )
            
            # Calculate semantic similarity if embeddings available
            semantic_sim = 0.0
            if hasattr(traj, '_embedding') and traj._embedding is not None:
                query_embedding = self.learner.compute_embedding(instruction)
                semantic_sim = np.dot(query_embedding, traj._embedding)
            
            # Get quality score
            quality_score = self.get_trajectory_quality(
                self.trajectories.index(traj)
            )
            
            if quality_score and quality_score.total_score >= min_quality:
                # Combine all scores with weights
                final_score = (
                    0.25 * bm25_score +           # Text similarity
                    0.20 * state_similarity +      # State change patterns
                    0.20 * context_sim +          # Environment context
                    0.15 * action_score +         # Action applicability  
                    0.10 * semantic_sim +         # Semantic similarity
                    0.10 * quality_score.total_score  # Trajectory quality
                )
                
                scored_candidates.append((final_score, traj))
        
        # Sort by final score and return top results
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_candidates[:limit]]

    def _calculate_change_similarity(self,
                                  changes1: List[StateChange],
                                  changes2: List[StateChange]) -> float:
        """Calculate similarity between two sets of state changes"""
        if not changes1 or not changes2:
            return 0.0
            
        # Compare change types and impacts
        matches = 0
        total = max(len(changes1), len(changes2))
        
        for c1 in changes1:
            for c2 in changes2:
                # Type match (0.7 weight)
                type_match = 0.7 if c1.type == c2.type else 0.0
                
                # Impact similarity (0.3 weight)
                impact_diff = abs(c1.impact - c2.impact)
                impact_sim = 0.3 * (1.0 - min(1.0, impact_diff))
                
                # Combined similarity for this pair
                if type_match + impact_sim > 0.5:  # Only count if reasonably similar
                    matches += 1
                    break  # Move to next c1
                    
        return matches / total

    def _compute_context_match(self,
                             current_state: Dict[str, Any],
                             trajectory_state: Dict[str, Any]) -> float:
        """
        Compute similarity between technical contexts and repository states
        Considers frameworks, languages, and patterns
        """
        if not current_state or not trajectory_state:
            return 0.0
            
        score = 0.0
        total_weight = 0.0
        
        # Compare frameworks
        frameworks1 = set(current_state.get('frameworks', []))
        frameworks2 = set(trajectory_state.get('frameworks', []))
        if frameworks1 or frameworks2:
            score += 0.3 * len(frameworks1 & frameworks2) / max(len(frameworks1 | frameworks2), 1)
            total_weight += 0.3
            
        # Compare languages
        langs1 = set(current_state.get('languages', []))
        langs2 = set(trajectory_state.get('languages', []))
        if langs1 or langs2:
            score += 0.3 * len(langs1 & langs2) / max(len(langs1 | langs2), 1)
            total_weight += 0.3
            
        # Compare file patterns
        patterns1 = set(current_state.get('patterns', []))
        patterns2 = set(trajectory_state.get('patterns', []))
        if patterns1 or patterns2:
            score += 0.2 * len(patterns1 & patterns2) / max(len(patterns1 | patterns2), 1)
            total_weight += 0.2
            
        # Compare directory structure patterns
        dirs1 = set(str(Path(f).parent) for f in current_state.get('files', []))
        dirs2 = set(str(Path(f).parent) for f in trajectory_state.get('files', []))
        if dirs1 or dirs2:
            score += 0.2 * len(dirs1 & dirs2) / max(len(dirs1 | dirs2), 1)
            total_weight += 0.2
            
        return score / total_weight if total_weight > 0 else 0.0

    def _compute_action_applicability(self,
                                   current_state: Dict[str, Any],
                                   actions: List[Dict[str, Any]]) -> float:
        """
        Compute how applicable a trajectory's actions are to current state
        """
        if not actions:
            return 0.0
            
        applicable_count = 0
        current_files = set(current_state.get('files', []))
        
        for action in actions:
            action_type = action.get('type', '')
            target_file = action.get('file')
            
            if action_type in ['create_file', 'add_file']:
                # Creating new files is always applicable
                applicable_count += 1
            elif target_file:
                # Check if target file exists for file operations
                if target_file in current_files:
                    applicable_count += 1
            else:
                # General actions without file targets are considered applicable
                applicable_count += 1
                
        return applicable_count / len(actions)

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

    def extract_error_patterns(self) -> Dict[str, List[ActionSequence]]:
        """Extract successful error recovery patterns from trajectories"""
        error_patterns = {}
        
        for trajectory in self.trajectories:
            # Look for error recovery sequences
            for i, obs in enumerate(trajectory.observations):
                if isinstance(obs, dict) and 'error' in obs:
                    error_type = obs.get('error', '')
                    if not error_type:
                        continue
                        
                    # Look for successful recovery in subsequent actions
                    if i + 1 < len(trajectory.observations):
                        recovery_actions = trajectory.actions[i+1:]
                        recovery_obs = trajectory.observations[i+1:]
                        
                        # Check if recovery was successful
                        if any(isinstance(o, dict) and o.get('status') == 'success' 
                              for o in recovery_obs):
                            # Create sequence from recovery actions
                            sequence = ActionSequence.from_trajectory(
                                trajectory.__class__(
                                    instruction=f"Fix {error_type}",
                                    actions=recovery_actions,
                                    observations=recovery_obs,
                                    final_state=trajectory.final_state
                                )
                            )
                            
                            if error_type not in error_patterns:
                                error_patterns[error_type] = []
                            error_patterns[error_type].append(sequence)
        
        return error_patterns
    
    def update_environment_patterns(self, git_env: 'GitEnvironment') -> None:
        """Update GitEnvironment with learned error recovery patterns"""
        error_patterns = self.extract_error_patterns()
        
        for error_type, sequences in error_patterns.items():
            for sequence in sequences:
                # Convert sequence back to actions
                actions = [step.action for step in sequence.steps]
                git_env.add_recovery_pattern(error_type, actions, True)

    def store_error_recovery(self, error: str, recovery_trajectory: 'Trajectory') -> None:
        """Store a successful error recovery trajectory"""
        if recovery_trajectory.compute_quality_metrics().success_rate >= 0.8:
            self.store_trajectory(recovery_trajectory)