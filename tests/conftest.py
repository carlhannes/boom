import pytest
from pathlib import Path
from git import Repo
import shutil

@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary Git repository for testing"""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Initialize repository
    repo = Repo.init(str(repo_path))
    
    # Create initial commit
    readme = repo_path / "README.md"
    readme.write_text("# Test Repository")
    repo.index.add(["README.md"])
    repo.index.commit("Initial commit")
    
    yield repo
    
    # Cleanup
    shutil.rmtree(repo_path, ignore_errors=True)

@pytest.fixture
def test_storage(tmp_path):
    """Create temporary storage path for test data"""
    storage_path = tmp_path / "test_storage"
    storage_path.mkdir()
    return storage_path