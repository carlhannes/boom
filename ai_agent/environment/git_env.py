from typing import List, Dict, Any
from pathlib import Path
import time
import git
from typing import Optional
import pytest

class GitEnvironment:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current environment state including files, git status, etc."""
        return {
            'timestamp': time.time(),
            'files': self._get_files(),
            'git_status': self._get_git_status(),
            'branch': self.repo.active_branch.name,
            'last_commit': self._get_last_commit_info()
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

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action in the environment and return observation"""
        action_type = action['type']
        
        if action_type == 'edit_file':
            return self._edit_file(action['path'], action['content'])
        elif action_type == 'git_commit':
            return self._commit_changes(action['message'])
        elif action_type == 'git_checkout':
            return self._checkout_branch(action['branch'], action.get('create', False))
        elif action_type == 'run_tests':
            return self._run_tests(action.get('test_path'))
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
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