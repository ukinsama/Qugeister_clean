"""
Configuration management for Qugeister system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container for Qugeister system"""
    
    game: Dict[str, Any]
    quantum: Dict[str, Any] 
    network: Dict[str, Any]
    training: Dict[str, Any]
    analysis: Dict[str, Any]
    web: Dict[str, Any]
    paths: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        return cls(
            game=config_dict.get('game', {}),
            quantum=config_dict.get('quantum', {}),
            network=config_dict.get('network', {}),
            training=config_dict.get('training', {}),
            analysis=config_dict.get('analysis', {}),
            web=config_dict.get('web', {}),
            paths=config_dict.get('paths', {})
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.__dict__
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file"""
    
    if config_path is None:
        # Default config locations
        possible_paths = [
            Path("config/default.yaml"),
            Path("../config/default.yaml"),
            Path(__file__).parent.parent.parent.parent / "config" / "default.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError("Could not find default configuration file")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)