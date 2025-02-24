from typing import Dict, Any, Optional, Set, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib
import json
from collections import defaultdict
import time

@dataclass
class StateChange:
    """Represents a change in repository state"""
    type: str  # Type of change (file, git, dependency, etc)
    path: str  # Affected path or resource
    before: Any  # State before change
    after: Any  # State after change
    impact: float  # Impact score of change (0-1)

class StateChangeAnalyzer:
    """Analyzes and tracks state changes in the repository"""
    
    def __init__(self):
        self.state_history: List[Dict[str, Any]] = []
        self.change_patterns = defaultdict(list)
        self.safe_paths: Set[str] = set()
        
    def analyze_change(self, 
                      state_before: Dict[str, Any],
                      state_after: Dict[str, Any]) -> List[StateChange]:
        """Analyze changes between two states"""
        changes = []
        
        # Analyze file changes
        files_before = set(state_before.get('files', []))
        files_after = set(state_after.get('files', []))
        
        # New files
        for file in files_after - files_before:
            changes.append(StateChange(
                type='file_created',
                path=file,
                before=None,
                after={'exists': True},
                impact=self._compute_file_impact(file)
            ))
        
        # Deleted files
        for file in files_before - files_after:
            changes.append(StateChange(
                type='file_deleted',
                path=file,
                before={'exists': True},
                after=None,
                impact=self._compute_file_impact(file)
            ))
        
        # Modified files
        for file in files_before & files_after:
            if self._file_content_changed(state_before, state_after, file):
                changes.append(StateChange(
                    type='file_modified',
                    path=file,
                    before=self._get_file_state(state_before, file),
                    after=self._get_file_state(state_after, file),
                    impact=self._compute_file_impact(file)
                ))
        
        # Analyze git changes
        git_before = state_before.get('git_status', {})
        git_after = state_after.get('git_status', {})
        
        if git_before != git_after:
            changes.append(StateChange(
                type='git_state',
                path='repository',
                before=git_before,
                after=git_after,
                impact=self._compute_git_impact(git_before, git_after)
            ))
        
        # Record state and update patterns
        self._update_history(state_after)
        self._update_patterns(changes)
        
        return changes
    
    def _compute_file_impact(self, file_path: str) -> float:
        """Compute impact score for file changes"""
        # Higher impact for certain file types
        high_impact = {'.py', '.js', '.java', 'requirements.txt', 'package.json'}
        medium_impact = {'.md', '.txt', '.json', '.yml', '.yaml'}
        
        ext = Path(file_path).suffix
        
        if any(file_path.endswith(p) for p in high_impact):
            base_impact = 0.8
        elif any(file_path.endswith(p) for p in medium_impact):
            base_impact = 0.5
        else:
            base_impact = 0.3
            
        # Adjust based on path safety
        if file_path in self.safe_paths:
            base_impact *= 0.8
            
        return min(1.0, base_impact)
    
    def _compute_git_impact(self, 
                          before: Dict[str, Any], 
                          after: Dict[str, Any]) -> float:
        """Compute impact score for git state changes"""
        impact = 0.0
        
        # Check for commits
        if after.get('commit') != before.get('commit'):
            impact += 0.3
            
        # Check for branch changes
        if after.get('branch') != before.get('branch'):
            impact += 0.4
            
        # Check for staged/unstaged changes
        staged_before = set(before.get('staged', []))
        staged_after = set(after.get('staged', []))
        if staged_before != staged_after:
            impact += 0.2
            
        unstaged_before = set(before.get('unstaged', []))
        unstaged_after = set(after.get('unstaged', []))
        if unstaged_before != unstaged_after:
            impact += 0.1
            
        return min(1.0, impact)
    
    def _update_history(self, state: Dict[str, Any]) -> None:
        """Update state history"""
        self.state_history.append(state)
        if len(self.state_history) > 100:  # Keep last 100 states
            self.state_history.pop(0)
    
    def _update_patterns(self, changes: List[StateChange]) -> None:
        """Update change patterns"""
        if not changes:
            return
            
        # Create pattern signature
        pattern = tuple((c.type, c.impact > 0.7) for c in changes)
        self.change_patterns[pattern].append({
            'changes': changes,
            'timestamp': time.time()
        })
        
        # Update safe paths based on successful patterns
        for change in changes:
            if (change.type in ('file_modified', 'file_created') and 
                len(self.change_patterns[pattern]) >= 3):
                self.safe_paths.add(change.path)
    
    def get_similar_changes(self, 
                          changes: List[StateChange], 
                          threshold: float = 0.7) -> List[List[StateChange]]:
        """Find similar change patterns from history"""
        if not changes:
            return []
            
        pattern = tuple((c.type, c.impact > 0.7) for c in changes)
        similar = []
        
        for p, instances in self.change_patterns.items():
            if self._pattern_similarity(pattern, p) >= threshold:
                similar.extend(inst['changes'] for inst in instances)
                
        return similar
    
    def _pattern_similarity(self, pattern1: tuple, pattern2: tuple) -> float:
        """Compute similarity between change patterns"""
        if not pattern1 or not pattern2:
            return 0.0
            
        # Use longest common subsequence
        matcher = difflib.SequenceMatcher(None, pattern1, pattern2)
        return matcher.ratio()
    
    def _file_content_changed(self,
                            state_before: Dict[str, Any],
                            state_after: Dict[str, Any],
                            file_path: str) -> bool:
        """Check if file content changed between states"""
        before_hash = state_before.get('file_hashes', {}).get(file_path)
        after_hash = state_after.get('file_hashes', {}).get(file_path)
        return before_hash != after_hash
    
    def _get_file_state(self, state: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Get file state information"""
        return {
            'hash': state.get('file_hashes', {}).get(file_path),
            'size': state.get('file_sizes', {}).get(file_path),
            'last_modified': state.get('file_times', {}).get(file_path)
        }