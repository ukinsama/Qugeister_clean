"""
Configuration management for Qugeister system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumConfig:
    """Quantum circuit configuration"""

    n_qubits: int = 4
    n_layers: int = 2
    device: str = "lightning.qubit"
    embedding: str = "angle"
    entanglement: str = "linear"
    shots: int = 1000


@dataclass
class TrainingConfig:
    """Training configuration"""

    learning_rate: float = 0.001
    batch_size: int = 32
    episodes: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000


@dataclass
class NetworkConfig:
    """Neural network configuration"""

    input_dim: int = 252
    output_dim: int = 36
    hidden_layers: list[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout_rate: float = 0.2
    activation: str = "relu"


@dataclass
class GameConfig:
    """Game configuration"""

    board_size: int = 6
    num_pieces: int = 8
    time_limit: Optional[float] = None
    enable_escapes: bool = True


@dataclass
class Config:
    """Configuration container for Qugeister system"""

    game: GameConfig = field(default_factory=GameConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: Dict[str, Any] = field(default_factory=dict)
    web: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary with proper type conversion"""

        # Convert nested dictionaries to dataclass instances
        game_config = GameConfig(**config_dict.get("game", {}))
        quantum_config = QuantumConfig(**config_dict.get("quantum", {}))
        network_config = NetworkConfig(**config_dict.get("network", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            game=game_config,
            quantum=quantum_config,
            network=network_config,
            training=training_config,
            analysis=config_dict.get("analysis", {}),
            web=config_dict.get("web", {}),
            paths=config_dict.get("paths", {}),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation (e.g., 'quantum.n_qubits')"""
        keys = key.split(".")
        value = self.__dict__

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration values"""
        for key, value in updates.items():
            if "." in key:
                # Handle nested updates
                keys = key.split(".")
                obj = self
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                setattr(obj, keys[-1], value)
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "__dict__"):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file with enhanced error handling"""

    if config_path is None:
        # Default config locations
        possible_paths = [
            Path("config/default.yaml"),
            Path("../config/default.yaml"),
            Path(__file__).parent.parent.parent.parent / "config" / "default.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            logger.warning("Could not find default configuration file, using defaults")
            return Config()  # Return default configuration

    try:
        config_path = Path(config_path)
        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        return Config.from_dict(config_dict)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
