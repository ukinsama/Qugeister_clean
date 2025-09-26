#!/usr/bin/env python3
"""
Simple Unified Trainer - Fixed Version
Simplified trainer with boundary learning focus
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("        Simple Unified Trainer - Fixed Version")
print("        Boundary Learning Focus")
print("=" * 70)

class QuantumLayer(nn.Module):
    """Quantum processing layer"""
    def __init__(self, n_qubits=4, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        try:
            dev = qml.device('lightning.qubit', wires=n_qubits)
        except:
            dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev)
        def circuit(inputs, weights):
            # Amplitude encoding
            for i in range(min(len(inputs), n_qubits)):
                qml.RY(inputs[i] * np.pi / 2, wires=i)

            # Variational layers
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)

                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        outputs = []
        for i in range(x.shape[0]):
            result = self.circuit(x[i].float()[:self.n_qubits], self.weights)
            outputs.append(torch.tensor(result, dtype=torch.float32))

        return torch.stack(outputs)

class SimpleCQCNN(nn.Module):
    """Simplified CQCNN without BatchNorm issues"""
    def __init__(self, n_qubits=4, n_layers=2, state_dim=252, action_dim=36):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Frontend without BatchNorm
        self.frontend = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_qubits)
        )

        # Quantum layer
        self.quantum = QuantumLayer(self.n_qubits, n_layers)

        # Backend without BatchNorm
        self.backend = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )

    def forward(self, x):
        x = self.frontend(x.float())
        x = self.quantum(x)
        x = self.backend(x)
        return x

class SimpleEnvironment:
    """Simplified Geister environment for training"""
    def __init__(self, max_turns=60):
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """Reset environment"""
        self.board = np.zeros((6, 6), dtype=int)
        self.turn = 0
        self.current_player = 0  # 0 = player A, 1 = player B
        self.game_over = False
        self.winner = None

        # Initial setup
        self.board[4, 1:5] = 1  # Player A pieces
        self.board[1, 1:5] = -1  # Player B pieces

        return self.get_state()

    def get_state(self):
        """Get 252D state vector"""
        state = np.zeros(252)

        # Board state (36 dimensions)
        state[:36] = self.board.flatten()

        # Game info
        state[36] = self.turn / self.max_turns
        state[37] = self.current_player
        state[38] = 1.0 if not self.game_over else 0.0

        # Additional features
        state[39] = np.sum(self.board == 1) / 4.0  # Player A pieces
        state[40] = np.sum(self.board == -1) / 4.0  # Player B pieces

        return state

    def get_valid_actions(self):
        """Get valid actions for current player"""
        player_value = 1 if self.current_player == 0 else -1
        valid_actions = []

        # Find player pieces
        pieces = np.where(self.board == player_value)
        for i in range(len(pieces[0])):
            piece_row, piece_col = pieces[0][i], pieces[1][i]

            # Try all 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = piece_row + dr, piece_col + dc

                # Check bounds
                if 0 <= new_row < 6 and 0 <= new_col < 6:
                    # Check if move is valid (empty or opponent piece)
                    target = self.board[new_row, new_col]
                    if target == 0 or target != player_value:
                        # Convert to action index
                        action = piece_row * 6 + piece_col
                        if action < 36:
                            valid_actions.append(action)

        return valid_actions if valid_actions else [0]  # Fallback

    def step(self, action):
        """Execute action"""
        if self.game_over:
            return self.get_state(), 0, True

        player_value = 1 if self.current_player == 0 else -1

        # Basic action validation
        if not isinstance(action, int) or action < 0 or action >= 36:
            # Invalid action penalty
            self.current_player = 1 - self.current_player
            self.turn += 1
            if self.turn >= self.max_turns:
                self.game_over = True
                self.winner = 'Draw'
            return self.get_state(), -2.0, self.game_over

        # Find pieces and try to make move
        pieces = np.where(self.board == player_value)
        if len(pieces[0]) == 0:
            # No pieces left
            self.game_over = True
            self.winner = 1 - self.current_player
            return self.get_state(), -10.0, True

        # Try to execute move (simplified)
        piece_idx = action % len(pieces[0])
        if piece_idx < len(pieces[0]):
            piece_row, piece_col = pieces[0][piece_idx], pieces[1][piece_idx]

            # Try random direction for simplicity
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            direction_idx = (action // len(pieces[0])) % 4
            dr, dc = directions[direction_idx]
            new_row, new_col = piece_row + dr, piece_col + dc

            reward = 0.1  # Base reward for valid move

            # Check bounds
            if 0 <= new_row < 6 and 0 <= new_col < 6:
                target = self.board[new_row, new_col]

                # Can't move to own piece
                if target == player_value:
                    reward = -1.0
                else:
                    # Make the move
                    if target != 0:  # Capture
                        reward = 5.0

                    self.board[piece_row, piece_col] = 0
                    self.board[new_row, new_col] = player_value

                    # Check win conditions
                    if (player_value == 1 and new_row == 0) or \
                       (player_value == -1 and new_row == 5):
                        # Escape win
                        self.game_over = True
                        self.winner = self.current_player
                        reward = 50.0
            else:
                # Out of bounds
                reward = -2.0

        # Switch player
        self.current_player = 1 - self.current_player
        self.turn += 1

        # Check game end
        if self.turn >= self.max_turns:
            self.game_over = True
            self.winner = 'Draw'
            reward -= 5.0  # Draw penalty

        return self.get_state(), reward, self.game_over

class SimpleTrainer:
    """Simple trainer with DQN"""
    def __init__(self, n_qubits=4, n_layers=2, episodes=5000):
        self.episodes = episodes
        self.env = SimpleEnvironment()

        # Networks
        self.q_network = SimpleCQCNN(n_qubits, n_layers)
        self.target_network = SimpleCQCNN(n_qubits, n_layers)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Training config
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # DQN config
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.batch_size = 32
        self.memory = []
        self.memory_size = 5000
        self.target_update = 100

    def select_action(self, state, valid_actions):
        """Select action with epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor)

            # Mask invalid actions
            masked_q = q_values.clone()
            for i in range(36):
                if i not in valid_actions:
                    masked_q[0, i] = float('-inf')

            return masked_q.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Single training step"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.tensor([self.memory[i][0] for i in batch], dtype=torch.float32)
        actions = torch.tensor([self.memory[i][1] for i in batch], dtype=torch.long)
        rewards = torch.tensor([self.memory[i][2] for i in batch], dtype=torch.float32)
        next_states = torch.tensor([self.memory[i][3] for i in batch], dtype=torch.float32)
        dones = torch.tensor([self.memory[i][4] for i in batch], dtype=torch.bool)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)

        # Loss and optimization
        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """Main training loop"""
        print(f"Starting training: {self.episodes} episodes")
        print(f"Network: {self.q_network.n_qubits}Q{self.q_network.quantum.n_layers}L")
        print("-" * 50)

        total_rewards = []

        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            losses = []

            while not self.env.game_over and episode_length < 50:
                valid_actions = self.env.get_valid_actions()
                action = self.select_action(state, valid_actions)

                next_state, reward, done = self.env.step(action)

                self.store_transition(state, action, reward, next_state, done)

                # Training step
                loss = self.train_step()
                if loss > 0:
                    losses.append(loss)

                state = next_state
                episode_reward += reward
                episode_length += 1

            total_rewards.append(episode_reward)

            # Update target network
            if episode % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Progress reporting
            if episode % 200 == 0:
                avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                avg_loss = np.mean(losses) if losses else 0.0
                print(f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | "
                      f"Avg100: {avg_reward:6.1f} | Loss: {avg_loss:.4f} | "
                      f"ε: {self.epsilon:.3f}")

        print("\nTraining completed!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"experiments/models/simple_trainer_{timestamp}.pth"
        os.makedirs("experiments/models", exist_ok=True)

        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'config': {
                'n_qubits': self.q_network.n_qubits,
                'n_layers': self.q_network.quantum.n_layers,
                'episodes': self.episodes
            },
            'final_epsilon': self.epsilon,
            'avg_reward': np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
        }, model_path)

        print(f"Model saved: {model_path}")
        return model_path

def main():
    if len(sys.argv) > 1:
        episodes = int(sys.argv[1])
    else:
        episodes = 5000

    print(f"Episodes: {episodes}")

    trainer = SimpleTrainer(n_qubits=6, n_layers=2, episodes=episodes)
    model_path = trainer.train()

    print(f"\nExperiment complete!")
    print(f"Saved model: {model_path}")

if __name__ == "__main__":
    main()