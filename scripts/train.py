#!/usr/bin/env python3
"""
Training script for Qugeister quantum AI models.

This script provides a convenient interface for training quantum neural networks
with various configurations and hyperparameters.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qugeister.utils.config import load_config
from qugeister.utils.logging import setup_logging
from qugeister.quantum.quantum_trainer import FastQuantumTrainer, FastQuantumNeuralNetwork, train_fast_quantum


def main():
    parser = argparse.ArgumentParser(description="Train Qugeister quantum AI")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--output", type=str, default="models/trained_model.pth", help="Output model path")
    parser.add_argument("--config", type=Path, help="Configuration file")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    logger = setup_logging("INFO")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training with {args.episodes} episodes, {args.qubits} qubits")
    
    # Train model
    model, rewards = train_fast_quantum(
        episodes=args.episodes,
        n_qubits=args.qubits
    )
    
    logger.info(f"Training completed. Model saved to: {output_path}")


if __name__ == "__main__":
    main()