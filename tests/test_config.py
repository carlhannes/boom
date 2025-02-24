import pytest
from pathlib import Path
from ai_agent.core.config import ConfigManager, AgentConfig

def test_config_creation(tmp_path):
    # Setup test repository path
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Initialize config manager
    config_manager = ConfigManager(str(repo_path))
    
    # Verify default config was created
    config = config_manager.get_config()
    assert isinstance(config, AgentConfig)
    assert config.min_trajectory_quality == 0.7
    assert config.storage_path == str(repo_path / '.ai_agent' / 'storage')
    
    # Verify config file was created
    config_path = repo_path / '.ai_agent' / 'config.json'
    assert config_path.exists()

def test_config_updates(tmp_path):
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    config_manager = ConfigManager(str(repo_path))
    
    # Update specific values
    updates = {
        'min_trajectory_quality': 0.8,
        'enable_continuous_learning': False,
        'max_similar_trajectories': 10
    }
    
    config_manager.update_config(updates)
    
    # Verify updates were applied
    config = config_manager.get_config()
    assert config.min_trajectory_quality == 0.8
    assert not config.enable_continuous_learning
    assert config.max_similar_trajectories == 10
    
    # Verify other values remained default
    assert config.min_pattern_success_rate == 0.8
    assert config.model_name == "gpt-4"

def test_config_validation(tmp_path):
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    config_manager = ConfigManager(str(repo_path))
    
    # Test invalid values
    invalid_config = AgentConfig(
        min_trajectory_quality=1.5,  # Should be <= 1
        max_trajectories=-10  # Should be > 0
    )
    
    assert not ConfigManager.validate_config(invalid_config)
    
    # Test valid values
    valid_config = AgentConfig(
        min_trajectory_quality=0.9,
        max_trajectories=100
    )
    
    assert ConfigManager.validate_config(valid_config)

def test_config_persistence(tmp_path):
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Create and update config
    config_manager1 = ConfigManager(str(repo_path))
    config_manager1.update_config({'min_trajectory_quality': 0.9})
    
    # Create new manager instance and verify persistence
    config_manager2 = ConfigManager(str(repo_path))
    config = config_manager2.get_config()
    assert config.min_trajectory_quality == 0.9