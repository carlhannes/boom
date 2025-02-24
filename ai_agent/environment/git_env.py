from typing import Dict, Any, List, Optional, Deque
from pathlib import Path
import time
import git
from collections import deque
import shutil
import os

class StateSnapshot:
    """Represents a point-in-time snapshot of repository state"""
    def __init__(self, files: Dict[str, bytes], git_status: Dict[str, List[str]], branch: str):
        self.files = files
        self.git_status = git_status
        self.branch = branch
        self.timestamp = time.time()

class ErrorRecoveryPattern:
    """Pattern for successful error recovery strategies"""
    def __init__(self, error_type: str, actions: List[Dict[str, Any]], success_rate: float):
        self.error_type = error_type
        self.recovery_actions = actions
        self.success_rate = success_rate
        
    @classmethod
    def from_trajectory(cls, trajectory: Dict[str, Any]) -> Optional['ErrorRecoveryPattern']:
        """Extract error recovery pattern from a trajectory if it contains successful recovery"""
        error_actions = []
        current_error = None
        success_count = 0
        total_recoveries = 0
        
        for i, (action, obs) in enumerate(zip(trajectory['actions'], trajectory['observations'])):
            if isinstance(obs, dict) and obs.get('error'):
                current_error = obs['error']
                recovery_sequence = []
                # Look ahead for recovery actions
                for next_action, next_obs in zip(trajectory['actions'][i+1:], trajectory['observations'][i+1:]):
                    recovery_sequence.append(next_action)
                    if isinstance(next_obs, dict):
                        if next_obs.get('status') == 'success':
                            error_actions.append((current_error, recovery_sequence))
                            success_count += 1
                            break
                        if next_obs.get('error'):
                            break
                total_recoveries += 1
                
        if error_actions:
            # Group by error type and find most successful pattern
            error_patterns = {}
            for error, actions in error_actions:
                error_type = cls._categorize_error(error)
                if error_type not in error_patterns:
                    error_patterns[error_type] = []
                error_patterns[error_type].append(actions)
            
            # Get most common pattern for most frequent error type
            if error_patterns:
                error_type, patterns = max(error_patterns.items(), key=lambda x: len(x[1]))
                most_common = max(set(tuple(p) for p in patterns), key=lambda x: patterns.count(x))
                success_rate = success_count / total_recoveries
                return cls(error_type, list(most_common), success_rate)
        
        return None
    
    @staticmethod
    def _categorize_error(error_msg: str) -> str:
        """Categorize error message into general error type"""
        error_msg = error_msg.lower()
        if 'permission' in error_msg:
            return 'permission_error'
        if 'not found' in error_msg:
            return 'not_found_error'
        if 'conflict' in error_msg:
            return 'merge_conflict'
        if 'locked' in error_msg or 'busy' in error_msg:
            return 'resource_locked'
        return 'unknown_error'
    
    def matches_error(self, error: str) -> bool:
        """Check if this pattern matches an error message"""
        return self.error_type == self._categorize_error(error)

class GitEnvironment:
    def __init__(self, repo_path: str, max_history: int = 10):
        """Initialize GitEnvironment with state history tracking
        
        Args:
            repo_path: Path to Git repository
            max_history: Maximum number of state snapshots to keep
        """
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.max_history = max_history
        self.state_history: Deque[StateSnapshot] = deque(maxlen=max_history)
        self.recovery_patterns: List[ErrorRecoveryPattern] = []
        self._take_snapshot()  # Initial state
    
    def add_recovery_pattern(self, trajectory: Dict[str, Any]):
        """Learn new error recovery pattern from a trajectory"""
        pattern = ErrorRecoveryPattern.from_trajectory(trajectory)
        if pattern and pattern.success_rate > 0.7:  # Only keep highly successful patterns
            self.recovery_patterns.append(pattern)
    
    def _get_recovery_actions(self, error: str) -> Optional[List[Dict[str, Any]]]:
        """Get recovery actions for an error based on learned patterns"""
        matching_patterns = [p for p in self.recovery_patterns if p.matches_error(error)]
        if matching_patterns:
            # Use the pattern with highest success rate
            best_pattern = max(matching_patterns, key=lambda p: p.success_rate)
            return best_pattern.recovery_actions
        return None
    
    def _take_snapshot(self) -> None:
        """Take a snapshot of current repository state"""
        # Capture file contents
        files = {}
        for filepath in self._get_files():
            try:
                with open(self.repo_path / filepath, 'rb') as f:
                    files[filepath] = f.read()
            except (IOError, OSError):
                continue
                
        snapshot = StateSnapshot(
            files=files,
            git_status=self._get_git_status(),
            branch=self.repo.active_branch.name
        )
        self.state_history.append(snapshot)

    def _restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore repository to a previous state"""
        # Restore files
        for filepath, content in snapshot.files.items():
            file_path = self.repo_path / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(content)
                
        # Restore git state
        if snapshot.branch != self.repo.active_branch.name:
            try:
                self.repo.git.checkout(snapshot.branch)
            except git.GitCommandError:
                # If branch doesn't exist, create it
                self.repo.git.checkout('-b', snapshot.branch)

    def rollback(self, steps: int = 1) -> bool:
        """Rollback repository state by n steps"""
        if len(self.state_history) <= steps:
            return False
            
        # Remove current state(s)
        for _ in range(steps):
            self.state_history.pop()
            
        # Restore previous state
        if self.state_history:
            self._restore_snapshot(self.state_history[-1])
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        """Get current environment state with enhanced information"""
        current_state = {
            'timestamp': time.time(),
            'files': self._get_files(),
            'git_status': self._get_git_status(),
            'branch': self.repo.active_branch.name,
            'last_commit': self._get_last_commit_info(),
            'merge_conflicts': self._get_merge_conflicts(),
            'frameworks': self._detect_frameworks(),
            'languages': self._detect_languages(),
            'patterns': self._detect_patterns()
        }
        
        # Add state analysis
        current_state.update(self._analyze_state())
        return current_state

    def _detect_frameworks(self) -> List[str]:
        """Detect frameworks used in the repository"""
        frameworks = set()
        files = set(self._get_files())
        
        # Node.js
        if 'package.json' in files:
            frameworks.add('node')
            try:
                with open(self.repo_path / 'package.json') as f:
                    import json
                    pkg = json.load(f)
                    deps = {**pkg.get('dependencies', {}), **pkg.get('devDependencies', {})}
                    if 'react' in deps:
                        frameworks.add('react')
                    if 'vue' in deps:
                        frameworks.add('vue')
                    if 'express' in deps:
                        frameworks.add('express')
            except:
                pass
                
        # Python
        if 'requirements.txt' in files or 'setup.py' in files:
            frameworks.add('python')
            if any(f.endswith('django') for f in files):
                frameworks.add('django')
            if any(f.endswith('flask') for f in files):
                frameworks.add('flask')
                
        return list(frameworks)

    def _detect_languages(self) -> List[str]:
        """Detect programming languages used in the repository"""
        extensions = {Path(f).suffix[1:] for f in self._get_files() if Path(f).suffix}
        languages = set()
        
        ext_to_lang = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'rb': 'ruby',
            'java': 'java',
            'go': 'go',
            'rs': 'rust',
            'cpp': 'c++',
            'h': 'c++',
            'cs': 'c#'
        }
        
        for ext in extensions:
            if ext in ext_to_lang:
                languages.add(ext_to_lang[ext])
                
        return list(languages)

    def _detect_patterns(self) -> List[str]:
        """Detect common development patterns in the repository"""
        patterns = set()
        files = set(self._get_files())
        
        # Testing patterns
        if any('test' in f.lower() for f in files):
            patterns.add('testing')
        if any('spec' in f.lower() for f in files):
            patterns.add('testing')
            
        # CI/CD
        if '.github/workflows' in files or '.gitlab-ci.yml' in files:
            patterns.add('ci_cd')
            
        # Docker
        if 'Dockerfile' in files or 'docker-compose.yml' in files:
            patterns.add('containerized')
            
        # Code quality
        if any(f in files for f in ['.eslintrc', '.pylintrc', 'mypy.ini']):
            patterns.add('linting')
            
        return list(patterns)

    def _analyze_state(self) -> Dict[str, Any]:
        """Analyze current repository state for potential issues"""
        analysis = {
            'warnings': [],
            'suggestions': [],
            'state_changes': self._analyze_state_changes()
        }
        
        # Check for uncommon patterns
        status = self._get_git_status()
        if len(status.get('modified', [])) > 10:
            analysis['warnings'].append('Large number of modified files')
            
        if len(status.get('untracked', [])) > 5:
            analysis['warnings'].append('Many untracked files')
            
        # Check for potential issues
        if self._get_merge_conflicts():
            analysis['warnings'].append('Unresolved merge conflicts')
            
        return analysis

    def _analyze_state_changes(self) -> Dict[str, Any]:
        """Analyze changes between current and previous state"""
        if len(self.state_history) < 2:
            return {}
            
        current = self.state_history[-1]
        previous = self.state_history[-2]
        
        # Compare files
        new_files = set(current.files) - set(previous.files)
        deleted_files = set(previous.files) - set(current.files)
        modified_files = {
            f for f in set(current.files) & set(previous.files)
            if current.files[f] != previous.files[f]
        }
        
        return {
            'new_files': list(new_files),
            'deleted_files': list(deleted_files),
            'modified_files': list(modified_files),
            'branch_changed': current.branch != previous.branch
        }

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action with enhanced error handling and recovery"""
        action_type = action['type']
        result = None
        
        try:
            if action_type == 'edit_file':
                result = self._edit_file(action['path'], action['content'])
            elif action_type == 'git_commit':
                result = self._commit_changes(action['message'])
            elif action_type == 'git_checkout':
                result = self._checkout_branch(action['branch'], action.get('create', False))
            elif action_type == 'run_tests':
                result = self._run_tests(action.get('test_path'))
            elif action_type == 'fix_permissions':
                result = self._fix_permissions(action.get('path'))
            elif action_type == 'resolve_conflict':
                result = self._resolve_conflict(action['path'])
            else:
                raise ValueError(f"Unknown action type: {action_type}")
                
            if result.get('status') == 'success':
                self._take_snapshot()
                
            return {
                **result,
                'state': self.get_state()
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Try learned recovery patterns first
            recovery_actions = self._get_recovery_actions(error_msg)
            if recovery_actions:
                for recovery_action in recovery_actions:
                    try:
                        result = self.execute_action(recovery_action)
                        if result.get('status') == 'success':
                            # Retry original action
                            return self.execute_action(action)
                    except:
                        continue
            
            # Fall back to basic error handling
            return {
                'status': 'failed',
                'error': error_msg,
                'state': self.get_state(),
                'can_retry': self._can_retry_action(action_type, error_msg)
            }

    def _can_retry_action(self, action_type: str, error_msg: str) -> bool:
        """Determine if a failed action can be retried"""
        # Some errors are temporary and can be retried
        retriable_patterns = [
            'locked', 'temporary', 'timeout', 'retry',
            'connection', 'network', 'permission denied'
        ]
        
        if any(pattern in error_msg.lower() for pattern in retriable_patterns):
            return True
            
        # Action-specific retry logic
        if action_type == 'edit_file' and 'file exists' in error_msg.lower():
            return True
        if action_type == 'git_checkout' and 'branch exists' in error_msg.lower():
            return True
            
        return False

    def _fix_permissions(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Fix permission issues on files"""
        files_to_fix = [path] if path else self._get_files()
        fixed = []
        
        for filepath in files_to_fix:
            full_path = self.repo_path / filepath
            try:
                # Make file writable
                os.chmod(full_path, 0o644)
                fixed.append(filepath)
            except:
                continue
                
        return {
            'status': 'success' if fixed else 'failed',
            'fixed_files': fixed
        }

    def _resolve_conflict(self, path: str) -> Dict[str, Any]:
        """Attempt to resolve a merge conflict"""
        try:
            # Check if file is in conflict
            if path not in self._get_merge_conflicts():
                return {'status': 'failed', 'error': 'No merge conflict in file'}
                
            # Read file content
            with open(self.repo_path / path) as f:
                content = f.read()
                
            # Simple resolution: take the current changes
            resolved_content = []
            in_conflict = False
            current_section = []
            
            for line in content.splitlines():
                if line.startswith('<<<<<<<'):
                    in_conflict = True
                    current_section = []
                elif line.startswith('=======') and in_conflict:
                    current_section = []  # Discard "ours" section
                elif line.startswith('>>>>>>>') and in_conflict:
                    in_conflict = False
                elif in_conflict and '=======' not in line:
                    current_section.append(line)
                elif not in_conflict:
                    resolved_content.append(line)
                    
            # Write resolved content
            with open(self.repo_path / path, 'w') as f:
                f.write('\n'.join(resolved_content))
                
            # Stage the resolved file
            self.repo.index.add([path])
            
            return {
                'status': 'success',
                'resolved_file': path
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _get_files(self) -> List[str]:
        """Get list of tracked files in repository"""
        tracked_files = []
        for filepath in self.repo_path.rglob('*'):
            if filepath.is_file() and not self._is_ignored(filepath):
                tracked_files.append(str(filepath.relative_to(self.repo_path)))
        return tracked_files
    
    def _is_ignored(self, filepath: Path) -> bool:
        """Check if file is ignored by git"""
        try:
            return filepath.relative_to(self.repo_path / '.git').exists()
        except ValueError:
            return False
    
    def _get_git_status(self) -> Dict[str, List[str]]:
        """Get Git status including modified, staged, and untracked files"""
        return {
            'modified': [item.a_path for item in self.repo.index.diff(None)],
            'staged': [item.a_path for item in self.repo.index.diff('HEAD')],
            'untracked': self.repo.untracked_files,
            'merge_conflicts': self._get_merge_conflicts()
        }
    
    def _get_merge_conflicts(self) -> List[str]:
        """Get list of files with merge conflicts"""
        if not self.repo.head.is_valid():
            return []
            
        conflicted_files = []
        for item in self.repo.index.diff(None):
            try:
                content = item.a_blob.data_stream.read().decode('utf-8')
                if '<<<<<<< HEAD' in content and '>>>>>>>' in content:
                    conflicted_files.append(item.a_path)
            except (AttributeError, UnicodeDecodeError):
                continue
                
        return conflicted_files
    
    def _get_last_commit_info(self) -> Dict[str, Any]:
        """Get information about the last commit"""
        if not self.repo.head.is_valid():
            return {}
            
        commit = self.repo.head.commit
        return {
            'hash': commit.hexsha,
            'message': commit.message,
            'author': str(commit.author),
            'timestamp': commit.authored_datetime.timestamp()
        }

    def _edit_file(self, path: str, content: str) -> Dict[str, Any]:
        """Edit or create a file"""
        file_path = self.repo_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Record original content for observation
        original = file_path.read_text() if file_path.exists() else None
        
        # Write new content
        file_path.write_text(content)
        
        return {
            'status': 'success',
            'file': str(file_path.relative_to(self.repo_path)),
            'original_content': original,
            'new_content': content
        }
    
    def _commit_changes(self, message: str) -> Dict[str, Any]:
        """Stage and commit changes"""
        # Stage all changes
        self.repo.git.add('--all')
        
        # Get staged files for observation
        staged = [item.a_path for item in self.repo.index.diff('HEAD')]
        
        # Commit
        commit = self.repo.index.commit(message)
        
        return {
            'status': 'success',
            'commit_hash': commit.hexsha,
            'staged_files': staged,
            'message': message
        }
    
    def _checkout_branch(self, branch: str, create: bool = False) -> Dict[str, Any]:
        """Checkout or create a branch"""
        if create:
            new_branch = self.repo.create_head(branch)
            new_branch.checkout()
        else:
            self.repo.git.checkout(branch)
            
        return {
            'status': 'success',
            'branch': branch,
            'created': create
        }
    
    def _run_tests(self, test_path: Optional[str] = None) -> Dict[str, Any]:
        """Run tests in the repository"""
        import pytest
        
        if test_path:
            test_path = str(self.repo_path / test_path)
        else:
            test_path = str(self.repo_path)
            
        # Capture test results
        pytest_output = pytest.main([test_path])
        
        return {
            'status': 'success' if pytest_output == 0 else 'failed',
            'test_path': test_path,
            'exit_code': pytest_output
        }