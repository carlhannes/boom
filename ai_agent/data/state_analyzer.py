from typing import Dict, Any, Optional, Set, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib
import json
from collections import defaultdict
import time
from datetime import datetime

@dataclass
class StateChange:
    action_type: str
    target: str
    old_state: Optional[Dict[str, Any]]
    new_state: Optional[Dict[str, Any]]
    impact: float
    
    @property
    def type(self):
        """Backward compatibility for tests"""
        return self.action_type
        
    @property
    def path(self):
        """Backward compatibility for tests"""
        return self.target

class StateChangeAnalyzer:
    MAX_HISTORY_SIZE = 100  # Maximum number of states to keep in history
    
    def __init__(self):
        self.pattern_history = []
        self.state_history = []
        self.safe_paths = set()
        self.patterns = []
        self.impact_weights = {
            'file_modified': 0.5,
            'file_created': 0.6,
            'file_deleted': 0.9,
            'file_moved': 0.4
        }

    def is_safe_path(self, path):
        """Check if a path has been safely modified in the past"""
        return path in self.safe_paths
        
    def get_recent_states(self, count):
        """Get the most recent states"""
        return self.state_history[-count:] if self.state_history else []

    def _compute_pattern_similarity(self, pattern1: List[StateChange], 
                                  pattern2: List[StateChange]) -> float:
        """Calculate similarity between two change patterns"""
        if not pattern1 or not pattern2:
            return 1.0 if not pattern1 and not pattern2 else 0.0
            
        # Compare action types and targets
        similar_actions = 0
        total_comparisons = max(len(pattern1), len(pattern2))
        
        for i in range(min(len(pattern1), len(pattern2))):
            if pattern1[i].action_type == pattern2[i].action_type:
                similar_actions += 0.7
                if pattern1[i].target == pattern2[i].target:
                    similar_actions += 0.3
                    
        return similar_actions / total_comparisons

    def compute_total_impact(self, changes: List[StateChange]) -> float:
        """Compute total impact score for a set of changes"""
        if not changes:
            return 0.0
            
        # Normalize the total impact to be between 0 and 1
        total_impact = sum(change.impact for change in changes) 
        return min(1.0, total_impact)

    def _update_history(self, state):
        """Update state history with new state"""
        self.state_history.append(state)
        if len(self.state_history) > self.MAX_HISTORY_SIZE:
            self.state_history.pop(0)

    def analyze_changes(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> List[StateChange]:
        """Analyze changes between states and return StateChange objects"""
        changes = []
        
        # Analyze file changes
        old_files = set(old_state.get('files', []))
        new_files = set(new_state.get('files', []))
        
        # Added files
        for file in new_files - old_files:
            changes.append(StateChange(
                action_type='file_created',
                target=file,
                old_state=None,
                new_state={'exists': True},
                impact=self.impact_weights.get('file_created', 0.6)
            ))
            
        # Removed files
        for file in old_files - new_files:
            changes.append(StateChange(
                action_type='file_deleted',
                target=file,
                old_state={'exists': True},
                new_state=None,
                impact=self.impact_weights.get('file_deleted', 0.8)  # Higher impact for deletions
            ))
            
        # Modified files (those that exist in both states)
        for file in old_files & new_files:
            old_content = old_state.get('file_contents', {}).get(file)
            new_content = new_state.get('file_contents', {}).get(file)
            if old_content != new_content:
                changes.append(StateChange(
                    action_type='file_modified',
                    target=file,
                    old_state={'content': old_content},
                    new_state={'content': new_content},
                    impact=self.impact_weights.get('file_modified', 0.3)
                ))
        
        self._update_patterns(changes)
        return changes

    def _update_patterns(self, changes: List[StateChange]) -> None:
        """Update pattern database with new changes"""
        if len(changes) > 1:
            self.patterns.append(changes)
            
        # Update safe paths for modified files
        for change in changes:
            if change.action_type in ['file_modified']:
                self.safe_paths.add(change.target)

    def _compute_file_impact(self, file_path: str) -> float:
        """Calculate impact score for changes to a given file"""
        # Higher impact for critical files
        critical_patterns = ['config.', 'settings.', 'security.', 'auth.', '.env']
        base_impact = 0.5  # Default impact
        
        path = Path(file_path)
        file_name = path.name.lower()
        
        # Check for critical files - high impact
        for pattern in critical_patterns:
            if pattern in file_name:
                return min(1.0, base_impact * 1.8)  # Significantly higher impact
        
        # Medium impact for documentation
        if file_name.endswith(('.md', '.txt', '.rst')):
            return base_impact * 0.8  # Medium impact
                
        # Lower impact for test and log files
        if 'test' in str(path).lower() or file_name.endswith(('.log', '.test')):
            return base_impact * 0.5  # Lower impact
        elif any(dir_name in str(path).lower() for dir_name in ['core', 'main', 'auth']):
            return base_impact * 1.3  # Higher impact for core functionality
            
        return base_impact
            
    def _compute_git_impact(self, git_before: Dict, git_after: Dict) -> float:
        """Calculate impact of Git state changes"""
        impact = 0.0
        
        # Check for branch changes
        if git_before.get('branch') != git_after.get('branch'):
            impact += 0.4
        
        # Check for commit changes
        if git_before.get('commit') != git_after.get('commit'):
            impact += 0.3
            
        # Check for staging changes
        before_staged = set(git_before.get('staged', []))
        after_staged = set(git_after.get('staged', []))
        staged_changes = len(before_staged ^ after_staged)
        impact += min(0.3, staged_changes * 0.1)
            
        return min(1.0, impact)

    def get_similar_changes(self, changes: List[StateChange], threshold: float = 0.7) -> List[List[StateChange]]:
        """Find similar change patterns in history"""
        if not changes or not self.patterns:
            return []
            
        similar_patterns = []
        for pattern in self.patterns:
            similarity = self._compute_pattern_similarity(changes, pattern)
            if similarity >= threshold:
                similar_patterns.append(pattern)
                
        return similar_patterns

    def analyze_change(self, state_before: Dict[str, Any], state_after: Dict[str, Any]) -> List[StateChange]:
        """Alias for analyze_changes to fix test compatibility"""
        return self.analyze_changes(state_before, state_after)