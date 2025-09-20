"""
Qugeister - Quantum Geister AI System

A quantum-enhanced AI system for playing the Geister board game,
featuring quantum neural networks and advanced reinforcement learning.

Enhanced with proper error handling, type safety, and modular configuration.
"""

from typing import TYPE_CHECKING

# Core modules
from .core import GeisterEngine, GameState
from .utils import setup_logging, load_config, Config
from .utils.config import QuantumConfig, TrainingConfig, NetworkConfig, GameConfig

# Optional imports for type checking
if TYPE_CHECKING:
    from .quantum import (
        FastQuantumTrainer,
        FastQuantumNeuralNetwork,
        FastQuantumCircuit,
    )
    from .analysis import QValueAnalyzer

__version__ = "1.0.0"
__author__ = "Qugeister Development Team"
__description__ = "Quantum Geister AI Competition System"

# Core exports (always available)
__all__ = [
    "GeisterEngine",
    "GameState",
    "Config",
    "QuantumConfig",
    "TrainingConfig",
    "NetworkConfig",
    "GameConfig",
    "setup_logging",
    "load_config",
]


# Lazy imports for optional modules
def __getattr__(name: str):
    """Lazy import for optional modules"""
    if name == "FastQuantumTrainer":
        from .quantum import FastQuantumTrainer

        return FastQuantumTrainer
    elif name == "FastQuantumNeuralNetwork":
        from .quantum import FastQuantumNeuralNetwork

        return FastQuantumNeuralNetwork
    elif name == "FastQuantumCircuit":
        from .quantum import FastQuantumCircuit

        return FastQuantumCircuit
    elif name == "QValueAnalyzer":
        from .analysis import QValueAnalyzer

        return QValueAnalyzer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
