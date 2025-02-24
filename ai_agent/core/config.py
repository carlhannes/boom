from typing import Dict, Any, Optional
import json
from pathlib import Path
import os
from dataclasses import dataclass, asdict

@dataclass
class AgentConfig:
    """Configuration options for the AI agent"""
    
    # Quality thresholds
    min_trajectory_quality: float = 0.7
    min_pattern_success_rate: float = 0.8
    
    # Learning parameters
    max_trajectories: int = 1000
    enable_continuous_learning: bool = True
    
    # Retrieval settings
    max_similar_trajectories: int = 5
    state_match_threshold: float = 0.6
    
    # Safety settings
    allow_risky_actions: bool = False
    require_confirmation: bool = True
    
    # Model settings
    model_name: str = "gpt-4"
    embedding_batch_size: int = 32
    
    # Storage settings
    storage_path: Optional[str] = None

class ConfigManager:
    """Manages agent configuration per repository"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.config_path = Path(repo_path) / '.ai_agent' / 'config.json'
        self.config = self._load_config()
    
    def _load_config(self) -> AgentConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config_dict = json.load(f)
                return AgentConfig(**config_dict)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self._create_default_config()
        return self._create_default_config()
    
    def _create_default_config(self) -> AgentConfig:
        """Create and save default configuration"""
        config = AgentConfig(
            storage_path=str(Path(self.repo_path) / '.ai_agent' / 'storage')
        )
        self.save_config(config)
        return config
    
    def save_config(self, config: AgentConfig) -> None:
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        self.config = config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update specific configuration values"""
        current = asdict(self.config)
        current.update(updates)
        new_config = AgentConfig(**current)
        self.save_config(new_config)
    
    def get_config(self) -> AgentConfig:
        """Get current configuration"""
        return self.config
    
    @staticmethod
    def validate_config(config: AgentConfig) -> bool:
        """Validate configuration values"""
        try:
            assert 0 <= config.min_trajectory_quality <= 1
            assert 0 <= config.min_pattern_success_rate <= 1
            assert config.max_trajectories > 0
            assert config.max_similar_trajectories > 0
            assert 0 <= config.state_match_threshold <= 1
            return True
        except AssertionError:
            return False