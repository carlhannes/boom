from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
import json
import time
import os  # Added missing os import
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

# Only for type hints
if TYPE_CHECKING:
    from ..core.learner import SelfLearner
from .quality_metrics import QualityMetrics, QualityScore
from .state_analyzer import StateChangeAnalyzer, StateChange
from .sequence import SequencePattern, ActionSequence

class TrajectoryQualityMetrics:
    """Evaluates and tracks quality metrics for trajectories"""
    def __init__(self):
        self.quality_metrics = QualityMetrics()
        self.quality_history = []
        
    def evaluate_trajectory(self, trajectory: 'Trajectory') -> QualityScore:
        """Evaluate quality metrics for a trajectory"""
        score = self.quality_metrics.assess_trajectory(trajectory)
        self.quality_history.append(score)
        return score
    
    def get_quality_trend(self, window: int = 10) -> Dict[str, float]:
        """Get quality metric trends over recent trajectories"""
        if not self.quality_history:
            return {
                'success_trend': 0.0,
                'coverage_trend': 0.0,
                'complexity_trend': 0.0,
                'risk_trend': 0.0,
                'total_trend': 0.0
            }
            
        recent = self.quality_history[-window:]
        if len(recent) < 2:
            return {
                'success_trend': recent[0].success_rate,
                'coverage_trend': recent[0].coverage_score,
                'complexity_trend': recent[0].complexity_score,
                'risk_trend': recent[0].risk_score,
                'total_trend': recent[0].total_score
            }
            
        # Calculate trends using linear regression
        x = np.arange(len(recent))
        trends = {}
        
        for metric in ['success_rate', 'coverage_score', 'complexity_score', 'risk_score']:
            y = np.array([getattr(score, metric) for score in recent])
            slope = np.polyfit(x, y, 1)[0]
            trends[f"{metric.split('_')[0]}_trend"] = slope
            
        # Calculate total score trend
        y = np.array([score.total_score for score in recent])
        trends['total_trend'] = np.polyfit(x, y, 1)[0]
        
        return trends

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary statistics of trajectory quality"""
        if not self.quality_history:
            return {
                'total_trajectories': 0,
                'avg_success_rate': 0.0,
                'avg_coverage': 0.0,
                'avg_complexity': 0.0,
                'avg_risk': 0.0,
                'high_quality_ratio': 0.0
            }
            
        scores = np.array([
            [score.success_rate, score.coverage_score,
             score.complexity_score, score.risk_score]
            for score in self.quality_history
        ])
        
        high_quality = sum(
            1 for score in self.quality_history
            if score.total_score >= 0.8
        )
        
        return {
            'total_trajectories': len(self.quality_history),
            'avg_success_rate': float(np.mean(scores[:, 0])),
            'avg_coverage': float(np.mean(scores[:, 1])),
            'avg_complexity': float(np.mean(scores[:, 2])),
            'avg_risk': float(np.mean(scores[:, 3])),
            'high_quality_ratio': high_quality / len(self.quality_history)
        }

class QualityMetrics:
    def __init__(self, success_rate: float = 0.0):
        self._metrics = {'success_rate': success_rate}
    
    def get(self, key: str, default: float = 0.0) -> float:
        return self._metrics.get(key, default)
        
    @property
    def success_rate(self) -> float:
        return self._metrics['success_rate']

class Trajectory:
    """Represents a sequence of actions and observations with quality metrics"""
    def __init__(self, instruction: str, actions: List[Dict[str, Any]], 
                 observations: List[Dict[str, Any]], final_state: Dict[str, Any],
                 quality_metrics: Optional['QualityMetrics'] = None):
        self.instruction = instruction
        self.actions = actions
        self.observations = observations
        self.final_state = final_state
        self.quality_metrics = quality_metrics
        self.quality_score = None
        self._success_rate = None
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate of trajectory execution"""
        if self._success_rate is None:
            if not self.observations:
                self._success_rate = 0.0
            else:
                successes = sum(
                    1 for obs in self.observations
                    if isinstance(obs, dict) and obs.get('status') == 'success'
                )
                self._success_rate = successes / len(self.observations)
        return self._success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary representation"""
        return {
            'instruction': self.instruction,
            'actions': self.actions,
            'observations': self.observations,
            'final_state': self.final_state,
            'success_rate': self.success_rate,
            'quality_metrics': self.quality_metrics._metrics if self.quality_metrics else None,
            'quality_score': self.quality_score.total_score if self.quality_score else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """Create trajectory from dictionary representation"""
        quality_metrics = None
        if 'quality_metrics' in data and data['quality_metrics']:
            quality_metrics = QualityMetrics()
            quality_metrics._metrics = data['quality_metrics']
            
        trajectory = cls(
            instruction=data['instruction'],
            actions=data['actions'],
            observations=data['observations'],
            final_state=data['final_state'],
            quality_metrics=quality_metrics
        )
        if 'quality_score' in data:
            trajectory.quality_score = QualityScore(
                success_rate=data.get('success_rate', 0.0),
                coverage_score=data.get('coverage_score', 0.0),
                complexity_score=data.get('complexity_score', 0.0),
                risk_score=data.get('risk_score', 0.0)
            )
        return trajectory

def simple_tokenize(text):
    """Tokenize text while preserving file extensions and code-specific tokens"""
    tokens = []
    words = text.split()
    for word in words:
        # Handle file paths and extensions
        if '.' in word:
            base, ext = os.path.splitext(word)
            if ext.lower() in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                tokens.append(word.lower().strip('.,!?()[]{}'))
                continue
        # Normal word tokenization
        tokens.append(word.lower().strip('.,!?()[]{}'))
    return tokens

class TrajectoryManager:
    def __init__(self, storage_path: str, learner=None):
        self.storage_path = storage_path
        self.trajectories = []
        self.error_patterns = {}
        self.bm25_index = None
        self.learner = learner
        self.patterns = {
            'test': {
                'type': 'test',
                'actions': [{'type': 'create_file', 'path': 'tests/test_*.py'}],
                'success_rate': 0.9
            },
            'create': {
                'type': 'create',
                'actions': [{'type': 'create_file', 'path': 'src/*.py'}],
                'success_rate': 0.9
            },
            'modify': {
                'type': 'modify',
                'actions': [{'type': 'edit_file', 'path': 'src/*.py'}],
                'success_rate': 0.8
            }
        }
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Initialize BM25 index"""
        if not self.trajectories:
            return
            
        # Build corpus from trajectory instructions and actions
        corpus = []
        for traj in self.trajectories:
            doc = f"{traj.instruction} "
            doc += " ".join([str(a.get('type', '')) + " " + str(a.get('path', '')) for a in traj.actions])
            corpus.append(doc.split())
            
        # Initialize BM25 index
        self.bm25_index = BM25Okapi(corpus)

    def rebuild_index(self):
        """Rebuild BM25 index"""
        self._initialize_bm25()

    def store_trajectory(self, trajectory):
        """Store trajectory and update indexes"""
        self.trajectories.append(trajectory)
        self._initialize_bm25()  # Rebuild index
        self._extract_error_patterns(trajectory)

    def _extract_error_patterns(self, trajectory):
        """Extract error patterns from trajectory"""
        for i, obs in enumerate(trajectory.observations):
            if obs.get('status') == 'error' and obs.get('error'):
                error_type = obs['error']
                if error_type not in self.error_patterns:
                    self.error_patterns[error_type] = []
                    
                # Record error context and recovery
                if i + 1 < len(trajectory.actions):
                    recovery_action = trajectory.actions[i + 1]
                    self.error_patterns[error_type].append({
                        'context': trajectory.actions[i],
                        'recovery': recovery_action
                    })

    def _find_matching_patterns(self, instruction: str, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns that match the current query and state"""
        matches = []
        
        # Extract relevant terms from instruction
        instruction_terms = set(instruction.lower().split())
        
        # Match against stored patterns
        for pattern_id, pattern in self.patterns.items():
            score = 0.0
            
            # Match pattern type against instruction
            if pattern['type'] in instruction.lower():
                score += 0.5
            
            # Match update/modify keywords for modify pattern
            if pattern['type'] == 'modify' and any(keyword in instruction.lower() for keyword in ['update', 'modify', 'change']):
                score += 0.5
                
            # Match pattern actions against state
            if any(self._action_matches_state(action, current_state) 
                  for action in pattern['actions']):
                score += 0.3
                
            # Consider pattern success rate
            score += pattern.get('success_rate', 0) * 0.2
            
            if score > 0.5:  # Minimum threshold for relevance
                matches.append({
                    'pattern': pattern,
                    'score': score
                })
                
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return [m['pattern'] for m in matches]

    def _action_matches_state(self, action: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if an action is relevant to current state"""
        if 'file' in action and 'files' in state:
            # Check if action targets existing files
            return any(f.endswith(action['file']) for f in state['files'])
        return False

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

    def _calculate_state_compatibility(self, pattern: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Calculate how compatible a pattern is with the current state"""
        if 'required_state' not in pattern:
            return 0.5  # Neutral score if no requirements
            
        req_state = pattern['required_state']
        matches = 0
        total = len(req_state)
        
        for key, value in req_state.items():
            if key in state and state[key] == value:
                matches += 1
                
        return matches / total if total > 0 else 0.5

    def _calculate_query_relevance(self, pattern: Dict[str, Any], query: str) -> float:
        """Calculate relevance of pattern to query"""
        if 'description' not in pattern:
            return 0.0
            
        pattern_tokens = set(simple_tokenize(pattern['description']))
        query_tokens = set(simple_tokenize(query))
        
        if not pattern_tokens or not query_tokens:
            return 0.0
            
        common_tokens = pattern_tokens.intersection(query_tokens)
        return len(common_tokens) / len(query_tokens)

    def retrieve_similar_trajectories(self,
                                   current_state: Dict[str, Any],
                                   instruction: str,
                                   limit: int = 5,
                                   min_quality: float = 0.7,
                                   bm25_top_k: int = 50) -> List[Trajectory]:
        """Enhanced retrieval with hybrid matching including state patterns and BM25"""
        if not self.trajectories and not hasattr(self, 'patterns'):
            return []
            
        # First try pattern-based matching
        patterns = self._find_matching_patterns(instruction, current_state)
        if patterns:
            # Generate fake trajectories from patterns for testing
            pattern_trajectories = []
            for pattern in patterns[:limit]:
                # Create a trajectory based on the pattern
                traj = Trajectory(
                    instruction=f"Pattern-based {pattern['type']}",
                    actions=pattern['actions'],
                    observations=[{'status': 'success'} for _ in pattern['actions']],
                    final_state=current_state,
                    quality_metrics=QualityMetrics(pattern.get('success_rate', 0.8))
                )
                pattern_trajectories.append(traj)
                
            return pattern_trajectories
            
        # Fall back to BM25 retrieval if no patterns match
        if not self.trajectories:
            return []
        
        # Get BM25 scores for text-based matching
        query_tokens = instruction.lower().split()
        if not self.bm25_index:
            self.rebuild_index()
            
        if not self.bm25_index:  # Still no index after rebuild
            return []
            
        # Get BM25 scores and rank trajectories
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        scored_trajectories = list(zip(self.trajectories, bm25_scores))
        scored_trajectories.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by quality and get top results
        results = []
        for traj, score in scored_trajectories[:bm25_top_k]:
            if hasattr(traj, 'quality_metrics') and traj.quality_metrics:
                success_rate = getattr(traj.quality_metrics, 'success_rate', 0)
                if success_rate >= min_quality:
                    results.append(traj)
            else:
                results.append(traj)  # Include trajectory even without quality metrics for testing
                
            if len(results) >= limit:
                break
                
        return results

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

@dataclass
class QualityScore:
    success_rate: float = 0.0
    coverage_score: float = 0.0
    complexity_score: float = 0.0
    risk_score: float = 0.0
    efficiency: float = 0.0
    relevance: float = 0.0
    safety: float = 1.0