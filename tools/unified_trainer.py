#!/usr/bin/env python3
"""
Unified Quantum AI Trainer
Simplified training system that reads JSON configs and outputs PTH models
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ASCII-only encoding for Windows compatibility
EXPERIMENT_SEED = 42
random.seed(EXPERIMENT_SEED)
np.random.seed(EXPERIMENT_SEED)
torch.manual_seed(EXPERIMENT_SEED)
torch.backends.cudnn.deterministic = True

print("=" * 80)
print("        Unified Quantum AI Trainer")
print("        JSON Config -> PTH Models")
print("=" * 80)

class GeisterEnvironment:
    """Simplified Geister game environment"""
    def __init__(self, max_turns=180):
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """Reset game to initial state"""
        self.board = np.zeros((6, 6), dtype=int)
        self.turn_count = 0
        self.game_over = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Get current game state as 252D vector"""
        # Flatten board and add game info
        state = np.zeros(252)
        state[:36] = self.board.flatten()
        state[36] = self.turn_count / self.max_turns
        state[37] = 1.0 if not self.game_over else 0.0
        return state

    def step(self, action):
        """Execute action and return new state, reward, done"""
        self.turn_count += 1

        # Simple random game logic for demo
        reward = random.uniform(-1, 1)

        # Game ends conditions
        if self.turn_count >= self.max_turns:
            self.game_over = True
            self.winner = random.choice([1, 2, 0])  # P1, P2, Draw
        elif random.random() < 0.01:  # 1% chance of early end
            self.game_over = True
            self.winner = random.choice([1, 2])

        return self.get_state(), reward, self.game_over

    def get_valid_actions(self):
        """Get list of valid actions"""
        return list(range(36))  # Simplified action space

class QuantumLayer(nn.Module):
    """Quantum processing layer using PennyLane"""
    def __init__(self, n_qubits, n_layers, embedding_type='amplitude'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding_type = embedding_type

        # Create quantum device
        self.device = qml.device('default.qubit', wires=n_qubits)

        # Define quantum circuit
        @qml.qnode(self.device, interface='torch')
        def circuit(inputs, weights):
            # Embedding
            if embedding_type == 'amplitude':
                qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            else:  # angle embedding
                qml.AngleEmbedding(inputs, wires=range(n_qubits))

            # Parametrized layers
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)

                # Entanglement
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])  # Circular

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

    def forward(self, x):
        # Ensure input fits quantum dimension
        if x.shape[-1] > 2**self.n_qubits:
            x = x[..., :2**self.n_qubits]
        elif x.shape[-1] < 2**self.n_qubits:
            padding = torch.zeros(*x.shape[:-1], 2**self.n_qubits - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)

        # Process batch
        if x.dim() == 1:
            x = x.unsqueeze(0)

        outputs = []
        for i in range(x.shape[0]):
            result = self.circuit(x[i].float(), self.weights)
            outputs.append(torch.tensor(result, dtype=torch.float32))

        return torch.stack(outputs)

class CQCNN(nn.Module):
    """Classical-Quantum CNN for Geister AI"""
    def __init__(self, config):
        super().__init__()

        # Extract config
        quantum_config = config['module_02_quantum']['config']
        qmap_config = config['module_05_qmap']['config']

        self.n_qubits = quantum_config['n_qubits']
        self.n_layers = quantum_config['n_layers']
        self.state_dim = qmap_config['state_dim']
        self.action_dim = qmap_config['action_dim']

        # Classical frontend
        self.frontend = nn.Sequential(
            nn.Linear(self.state_dim, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.20),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Dropout(0.15),
            nn.Linear(60, 28),
            nn.ReLU(),
            nn.Linear(28, 2**self.n_qubits)
        )

        # Quantum layer
        self.quantum = QuantumLayer(
            self.n_qubits,
            self.n_layers,
            quantum_config['embedding_type']
        )

        # Classical backend
        quantum_out_dim = self.n_qubits
        self.backend = nn.Sequential(
            nn.Linear(quantum_out_dim, 56),
            nn.ReLU(),
            nn.BatchNorm1d(56),
            nn.Dropout(0.25),
            nn.Linear(56, 112),
            nn.ReLU(),
            nn.BatchNorm1d(112),
            nn.Dropout(0.20),
            nn.Linear(112, 56),
            nn.ReLU(),
            nn.Linear(56, self.action_dim)
        )

    def forward(self, x):
        x = self.frontend(x.float())
        x = self.quantum(x)
        x = self.backend(x)
        return x

class DQNAgent:
    """DQN Agent for reinforcement learning"""
    def __init__(self, config):
        self.config = config
        self.action_config = config['module_06_action']['config']
        self.hyperparams = config['training_config']['hyperparameters']

        # Networks
        self.q_network = CQCNN(config)
        self.target_network = CQCNN(config)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.hyperparams['learningRate'])

        # Training parameters
        self.epsilon = self.hyperparams['epsilon']
        self.epsilon_decay = self.hyperparams['epsilonDecay']
        self.gamma = self.hyperparams['gamma']
        self.batch_size = self.hyperparams['batchSize']

        # Experience replay
        self.memory = deque(maxlen=self.hyperparams['replayBufferSize'])
        self.update_target_every = self.hyperparams['targetUpdateFreq']
        self.steps = 0

        # Copy weights to target
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, valid_actions):
        """Select action using epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state).float().unsqueeze(0))
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor([e[0] for e in batch]).float()
        actions = torch.tensor([e[1] for e in batch]).long()
        rewards = torch.tensor([e[2] for e in batch]).float()
        next_states = torch.tensor([e[3] for e in batch]).float()
        dones = torch.tensor([e[4] for e in batch]).bool()

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss using Huber Loss (more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()

        return loss.item()

class UnifiedTrainer:
    """Main training coordinator"""
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        self.setup_experiment()

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"Configuration loaded: {self.config_path}")

            # Handle both new structured format and legacy format
            if 'metadata' in self.config:
                # New structured format
                print(f"Experiment: {self.config['metadata']['experiment_name']}")
            elif 'learning_config' in self.config:
                # Legacy format - convert to expected structure
                print(f"Legacy format detected - converting...")
                self.config = self.convert_legacy_config(self.config)
                print(f"Experiment: {self.config['metadata']['experiment_name']}")
            else:
                print(f"Unknown config format")
                sys.exit(1)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            sys.exit(1)

    def convert_legacy_config(self, legacy_config):
        """Convert legacy config format to new structured format"""
        # Extract key information from legacy format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create new structured config
        new_config = {
            "metadata": {
                "experiment_name": f"Legacy_CQCNN_DQN_{timestamp}",
                "philosophy": "Legacy configuration converted to new format",
                "architecture_type": "CQCNN",
                "generation_timestamp": datetime.now().isoformat(),
                "version": "1.0_legacy"
            },
            "module_02_quantum": {
                "config": {
                    "n_qubits": legacy_config.get("quantum", {}).get("n_qubits", 4),
                    "n_layers": legacy_config.get("quantum", {}).get("n_layers", 1),
                    "embedding_type": legacy_config.get("quantum", {}).get("embedding_type", "amplitude")
                }
            },
            "module_05_qmap": {
                "config": {
                    "state_dim": 252,
                    "action_dim": 36
                }
            },
            "training_config": {
                "hyperparameters": {
                    "batchSize": legacy_config.get("hyperparameters", {}).get("batchSize", 128),
                    "epochs": legacy_config.get("hyperparameters", {}).get("epochs", 1000),
                    "learningRate": legacy_config.get("hyperparameters", {}).get("learningRate", 0.002),
                    "optimizer": "adam",
                    "epsilon": 0.1,
                    "epsilonDecay": 0.999,
                    "gamma": 0.95,
                    "replayBufferSize": 10000,
                    "targetUpdateFreq": 100
                },
                "learning_schedule": {
                    "method": "reinforcement",
                    "algorithm": "dqn_5d",
                    "total_episodes": 5000,
                    "evaluation_frequency": 100
                }
            }
        }

        return new_config

    def setup_experiment(self):
        """Initialize environment and agents"""
        # Environment
        self.env = GeisterEnvironment()

        # Agents
        self.agent_1 = DQNAgent(self.config)
        self.agent_2 = DQNAgent(self.config)

        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

        # Extract training schedule
        self.total_episodes = self.config['training_config']['learning_schedule']['total_episodes']
        self.eval_freq = self.config['training_config']['learning_schedule']['evaluation_frequency']

        print(f"Training setup complete - Target: {self.total_episodes} episodes")

    def train(self):
        """Main training loop"""
        start_time = time.time()

        print(f"\nStarting training: {self.total_episodes} episodes")
        print("-" * 60)

        for episode in range(self.total_episodes):
            episode_reward, episode_length, avg_loss = self.run_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if avg_loss > 0:
                self.losses.append(avg_loss)

            # Progress report
            if (episode + 1) % self.eval_freq == 0:
                self.print_progress(episode + 1, start_time)

        # Save final models
        self.save_models()

        # Final report
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total Episodes: {self.total_episodes}")
        print(f"Total Time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print(f"Average Episode Time: {total_time/self.total_episodes:.3f}s")
        if self.losses:
            print(f"Final Loss: {np.mean(self.losses[-10:]):.4f}")
        print("=" * 60)

    def run_episode(self):
        """Run single training episode"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        losses = []

        current_player = 1
        agents = {1: self.agent_1, 2: self.agent_2}

        while not self.env.game_over and steps < self.env.max_turns:
            agent = agents[current_player]

            # Select action
            valid_actions = self.env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            # Environment step
            next_state, reward, done = self.env.step(action)

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)

            total_reward += reward
            state = next_state
            steps += 1

            # Switch players
            current_player = 2 if current_player == 1 else 1

        avg_loss = np.mean(losses) if losses else 0.0
        return total_reward, steps, avg_loss

    def print_progress(self, episode, start_time):
        """Print training progress"""
        recent_rewards = self.episode_rewards[-self.eval_freq:]
        recent_lengths = self.episode_lengths[-self.eval_freq:]
        recent_losses = self.losses[-50:] if self.losses else [0]

        elapsed = time.time() - start_time
        progress = episode / self.total_episodes * 100

        print(f"Episode {episode:6d} ({progress:5.1f}%) | "
              f"Reward: {np.mean(recent_rewards):6.2f} | "
              f"Length: {np.mean(recent_lengths):5.1f} | "
              f"Loss: {np.mean(recent_losses):6.4f} | "
              f"Eps: {self.agent_1.epsilon:.3f} | "
              f"Time: {elapsed:.0f}s")

    def save_models(self):
        """Save trained models to files"""
        # Create model filename from config
        experiment_name = self.config['metadata']['experiment_name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        p1_path = f"models/{experiment_name}_p1_{timestamp}.pth"
        p2_path = f"models/{experiment_name}_p2_{timestamp}.pth"

        torch.save(self.agent_1.q_network.state_dict(), p1_path)
        torch.save(self.agent_2.q_network.state_dict(), p2_path)

        print(f"\nModels saved:")
        print(f"  Player 1: {p1_path}")
        print(f"  Player 2: {p2_path}")

        # Save config copy
        config_copy_path = f"configs/{experiment_name}_{timestamp}.json"
        with open(config_copy_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        print(f"  Config:   {config_copy_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python unified_trainer.py <config.json>")
        print("Example: python unified_trainer.py configs/cqcnn_config_2025-09-26.json")
        sys.exit(1)

    config_path = sys.argv[1]

    trainer = UnifiedTrainer(config_path)
    trainer.train()

if __name__ == "__main__":
    main()