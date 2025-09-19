"""
Qugeister - Quantum Geister AI System

A quantum-enhanced AI system for playing the Geister board game,
featuring quantum neural networks and advanced reinforcement learning.
"""

from .core import GeisterEngine, GameState
from .quantum import FastQuantumTrainer, FastQuantumNeuralNetwork, FastQuantumCircuit
from .analysis import QValueAnalyzer
from .utils import setup_logging, load_config

__version__ = "1.0.0"
__author__ = "Qugeister Development Team"
__description__ = "Quantum Geister AI Competition System"

__all__ = [
    "GeisterEngine",
    "GameState", 
    "FastQuantumTrainer",
    "FastQuantumNeuralNetwork",
    "FastQuantumCircuit",
    "QValueAnalyzer",
    "setup_logging",
    "load_config",
]