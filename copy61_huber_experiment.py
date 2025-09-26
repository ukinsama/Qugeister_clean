#!/usr/bin/env python3
"""
Copy 6.1 Huber Loss Experiment
Based on successful Copy 6.1 results, using Huber Loss instead of MSE
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

print("=" * 80)
print("        Copy 6.1 Huber Loss Progressive Learning Experiment")
print("             12K Episodes - 4-Phase Optimized System with Huber Loss")
print("=" * 80)

class QuantumLayer(nn.Module):
    """4-qubit 2-layer quantum processor"""
    def __init__(self, n_qubits=4, n_layers=2):
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
                # Circular entanglement
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits-1, 0])

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

class Copy61CQCNN(nn.Module):
    """Copy 6.1 Enhanced CQCNN Architecture with 55,200 parameters"""
    def __init__(self):
        super().__init__()
        self.n_qubits = 4
        self.state_dim = 252
        self.action_dim = 36

        # Enhanced frontend (matching Copy 6.1 architecture)
        self.frontend = nn.Sequential(
            nn.Linear(252, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.20),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Dropout(0.15),
            nn.Linear(60, 28),
            nn.ReLU(),
            nn.Linear(28, 4)  # To quantum layer
        )

        # Quantum processing
        self.quantum = QuantumLayer(4, 2)

        # Enhanced backend (matching Copy 6.1 architecture)
        self.backend = nn.Sequential(
            nn.Linear(4, 56),
            nn.ReLU(),
            nn.BatchNorm1d(56),
            nn.Dropout(0.25),
            nn.Linear(56, 112),
            nn.ReLU(),
            nn.BatchNorm1d(112),
            nn.Dropout(0.20),
            nn.Linear(112, 56),
            nn.ReLU(),
            nn.Linear(56, 36)
        )

        print(f"Copy 6.1 Huber Enhanced CQCNN: 252D -> 4Q2L -> 36D")
        # Calculate parameters
        total_params = sum(p.numel() for p in self.parameters())
        classical_params = total_params - 16  # 4*2*2 quantum weights
        print(f"Parameters: Classical={classical_params:,}, Quantum=16, Total={total_params:,}")

    def forward(self, x):
        # Handle batch normalization for single samples
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # If batch size is 1, temporarily duplicate for BatchNorm
        single_sample = x.shape[0] == 1
        if single_sample:
            x = x.repeat(2, 1)

        x = self.frontend(x.float())
        x = self.quantum(x)
        x = self.backend(x)

        # Return single sample if needed
        if single_sample:
            x = x[:1]

        return x

class Copy61HuberEnvironment:
    """Copy 6.1 Environment with Huber Loss optimization"""
    def __init__(self, max_turns=180):
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """Reset to initial state"""
        self.board = np.zeros((6, 6), dtype=int)
        self.turn = 0
        self.current_player = 0
        self.game_over = False
        self.winner = None

        # Initial piece placement (Copy 6.1 style)
        self.board[4, 1:5] = 1  # Player A pieces
        self.board[1, 1:5] = -1  # Player B pieces

        self.move_history = []
        return self.get_state()

    def get_state(self):
        """Get 252D state vector (Copy 6.1 format)"""
        state = np.zeros(252)

        # Board representation (36 dims)
        state[:36] = self.board.flatten()

        # Game state features
        state[36] = self.turn / self.max_turns
        state[37] = self.current_player
        state[38] = 1.0 if not self.game_over else 0.0

        # Enhanced features for Copy 6.1
        state[39] = np.sum(self.board == 1) / 4.0
        state[40] = np.sum(self.board == -1) / 4.0

        # Position analysis
        if self.current_player == 0:
            my_pieces = np.where(self.board == 1)
            enemy_pieces = np.where(self.board == -1)
        else:
            my_pieces = np.where(self.board == -1)
            enemy_pieces = np.where(self.board == 1)

        # Advanced tactical features
        if len(my_pieces[0]) > 0:
            state[41] = np.mean(my_pieces[0]) / 5.0  # Average row position
            state[42] = np.mean(my_pieces[1]) / 5.0  # Average column position

        if len(enemy_pieces[0]) > 0:
            state[43] = np.mean(enemy_pieces[0]) / 5.0
            state[44] = np.mean(enemy_pieces[1]) / 5.0

        # Mobility and threat analysis
        state[45] = len(self.get_valid_actions()) / 16.0  # Normalized mobility

        return state

    def get_valid_actions(self):
        """Get valid actions for current player"""
        player_value = 1 if self.current_player == 0 else -1
        valid_actions = []

        pieces = np.where(self.board == player_value)
        for i in range(len(pieces[0])):
            piece_row, piece_col = pieces[0][i], pieces[1][i]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = piece_row + dr, piece_col + dc

                if 0 <= new_row < 6 and 0 <= new_col < 6:
                    target = self.board[new_row, new_col]
                    if target == 0 or target != player_value:
                        action = piece_row * 6 + piece_col
                        if action < 36:
                            valid_actions.append(action)

        return valid_actions if valid_actions else [0]

    def step(self, action):
        """Execute action with enhanced reward system"""
        if self.game_over:
            return self.get_state(), 0, True

        player_value = 1 if self.current_player == 0 else -1
        reward = 0

        # Invalid action penalty
        if not isinstance(action, int) or action < 0 or action >= 36:
            reward = -2.0
        else:
            pieces = np.where(self.board == player_value)
            if len(pieces[0]) == 0:
                self.game_over = True
                self.winner = 1 - self.current_player
                return self.get_state(), -50.0, True

            # Execute move
            piece_idx = action % len(pieces[0])
            if piece_idx < len(pieces[0]):
                piece_row, piece_col = pieces[0][piece_idx], pieces[1][piece_idx]

                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                direction_idx = (action // len(pieces[0])) % 4
                dr, dc = directions[direction_idx]
                new_row, new_col = piece_row + dr, piece_col + dc

                if 0 <= new_row < 6 and 0 <= new_col < 6:
                    target = self.board[new_row, new_col]

                    if target == player_value:
                        reward = -1.0  # Can't capture own piece
                    else:
                        if target != 0:  # Capture
                            reward = 8.0
                        else:
                            reward = 0.2  # Valid move

                        self.board[piece_row, piece_col] = 0
                        self.board[new_row, new_col] = player_value

                        # Win conditions
                        if (player_value == 1 and new_row == 0) or \
                           (player_value == -1 and new_row == 5):
                            self.game_over = True
                            self.winner = self.current_player
                            reward = 100.0
                else:
                    reward = -3.0  # Out of bounds

        # Record move
        self.move_history.append((self.current_player, action, reward))

        # Switch player
        self.current_player = 1 - self.current_player
        self.turn += 1

        # Game end conditions
        if self.turn >= self.max_turns:
            self.game_over = True
            self.winner = 'Draw'
            reward -= 5.0

        return self.get_state(), reward, self.game_over

class Copy61HuberTrainer:
    """Copy 6.1 Trainer with Huber Loss"""
    def __init__(self, episodes=12000):
        self.episodes = episodes
        self.env = Copy61HuberEnvironment()

        # Copy 6.1 architecture
        self.agent_1 = Copy61CQCNN()
        self.agent_2 = Copy61CQCNN()
        self.target_1 = Copy61CQCNN()
        self.target_2 = Copy61CQCNN()

        # Load target networks
        self.target_1.load_state_dict(self.agent_1.state_dict())
        self.target_2.load_state_dict(self.agent_2.state_dict())

        # HUBER LOSS (SmoothL1Loss) - Main change from Copy 6.1
        self.criterion = nn.SmoothL1Loss()  # This is Huber Loss

        # Optimizers
        self.optimizer_1 = optim.Adam(self.agent_1.parameters(), lr=0.0005)
        self.optimizer_2 = optim.Adam(self.agent_2.parameters(), lr=0.0005)

        # Copy 6.1 progressive learning parameters
        self.phase_config = {
            'phase_1_initialization': {'episodes': 300, 'epsilon': 0.5},
            'phase_2_exploration': {'episodes': 1800, 'epsilon': 0.4},
            'phase_3_consolidation': {'episodes': 1500, 'epsilon': 0.3},
            'phase_4_mastery': {'episodes': 8400, 'epsilon': 0.1}
        }

        # DQN parameters
        self.gamma = 0.95
        self.batch_size = 32
        self.memory_1 = []
        self.memory_2 = []
        self.memory_size = 8000
        self.target_update = 150

        # Statistics
        self.stats = {
            'p1_wins': 0, 'p2_wins': 0, 'draws': 0,
            'total_turns': 0, 'emergency_interventions': 0,
            'perturbations': 0
        }

    def get_current_phase(self, episode):
        """Determine current learning phase"""
        if episode < 300:
            return 'phase_1_initialization', 0.5
        elif episode < 2100:
            return 'phase_2_exploration', 0.4
        elif episode < 3600:
            return 'phase_3_consolidation', 0.3
        else:
            return 'phase_4_mastery', max(0.05, 0.1 * (0.99 ** (episode - 3600)))

    def select_action(self, agent, state, valid_actions, epsilon):
        """Enhanced action selection with epsilon-greedy"""
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = agent(state_tensor)

            # Mask invalid actions
            masked_q = q_values.clone()
            for i in range(36):
                if i not in valid_actions:
                    masked_q[0, i] = float('-inf')

            return masked_q.argmax().item()

    def store_transition(self, memory, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        if len(memory) >= self.memory_size:
            memory.pop(0)
        memory.append((state, action, reward, next_state, done))

    def train_agent(self, agent, target, optimizer, memory):
        """Train agent with Huber Loss"""
        if len(memory) < self.batch_size:
            return 0.0

        # Sample batch
        batch_indices = np.random.choice(len(memory), self.batch_size, replace=False)
        states = torch.tensor([memory[i][0] for i in batch_indices], dtype=torch.float32)
        actions = torch.tensor([memory[i][1] for i in batch_indices], dtype=torch.long)
        rewards = torch.tensor([memory[i][2] for i in batch_indices], dtype=torch.float32)
        next_states = torch.tensor([memory[i][3] for i in batch_indices], dtype=torch.float32)
        dones = torch.tensor([memory[i][4] for i in batch_indices], dtype=torch.bool)

        # Current Q values
        current_q = agent(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = target(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)

        # HUBER LOSS CALCULATION
        loss = self.criterion(current_q.squeeze(), target_q)

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self):
        """Main training loop with Copy 6.1 progressive learning"""
        print(f"Starting Copy 6.1 Huber Loss Progressive Learning")
        print(f"Target: {self.episodes:,} episodes")
        print()

        start_time = datetime.now()

        for episode in range(self.episodes):
            state = self.env.reset()
            episode_length = 0
            losses = []

            # Get current phase and epsilon
            phase, epsilon = self.get_current_phase(episode)

            while not self.env.game_over and episode_length < 200:
                if self.env.current_player == 0:
                    agent, target, optimizer, memory = self.agent_1, self.target_1, self.optimizer_1, self.memory_1
                else:
                    agent, target, optimizer, memory = self.agent_2, self.target_2, self.optimizer_2, self.memory_2

                valid_actions = self.env.get_valid_actions()
                action = self.select_action(agent, state, valid_actions, epsilon)

                next_state, reward, done = self.env.step(action)

                self.store_transition(memory, state, action, reward, next_state, done)

                # Train with Huber Loss
                loss = self.train_agent(agent, target, optimizer, memory)
                if loss > 0:
                    losses.append(loss)

                state = next_state
                episode_length += 1

            # Update statistics
            if self.env.winner == 0:
                self.stats['p1_wins'] += 1
            elif self.env.winner == 1:
                self.stats['p2_wins'] += 1
            else:
                self.stats['draws'] += 1

            self.stats['total_turns'] += episode_length

            # Update target networks
            if episode % self.target_update == 0:
                self.target_1.load_state_dict(self.agent_1.state_dict())
                self.target_2.load_state_dict(self.agent_2.state_dict())

            # Progress reporting
            if episode % 300 == 0 and episode > 0:
                elapsed = (datetime.now() - start_time).total_seconds()

                # Calculate recent stats
                recent_window = min(300, episode)
                p1_rate = self.stats['p1_wins'] / (episode + 1)
                p2_rate = self.stats['p2_wins'] / (episode + 1)
                draw_rate = self.stats['draws'] / (episode + 1)
                avg_turns = self.stats['total_turns'] / (episode + 1)
                avg_loss = np.mean(losses) if losses else 0.0
                balance = min(p1_rate, p2_rate) / max(p1_rate, p2_rate) if max(p1_rate, p2_rate) > 0 else 0.0

                print(f"Episode {episode:5d} ({episode/self.episodes*100:5.1f}%) | "
                      f"Phase: {phase.split('_')[1][0]} | "
                      f"P1={p1_rate:.3f} P2={p2_rate:.3f} D={draw_rate:.3f} | "
                      f"Balance={balance:.3f} | Turns={avg_turns:.1f} | "
                      f"ε={epsilon:.4f} | HuberLoss={avg_loss:.4f} | "
                      f"Time={elapsed:.0f}s")

        # Final report
        total_time = (datetime.now() - start_time).total_seconds()

        print()
        print("=" * 80)
        print("      Copy 6.1 Huber Loss Progressive Learning - Final Report")
        print("=" * 80)
        print()
        print(f"Total Episodes: {self.episodes:,}")
        print(f"Training Time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        print()
        print("Game Results:")
        print(f"  Player 1: {self.stats['p1_wins']:,} ({self.stats['p1_wins']/self.episodes*100:.1f}%)")
        print(f"  Player 2: {self.stats['p2_wins']:,} ({self.stats['p2_wins']/self.episodes*100:.1f}%)")
        print(f"  Draws: {self.stats['draws']:,} ({self.stats['draws']/self.episodes*100:.1f}%)")
        print()
        print(f"Average Game Length: {self.stats['total_turns']/self.episodes:.1f} turns")

        # Calculate balance
        p1_rate = self.stats['p1_wins'] / self.episodes
        p2_rate = self.stats['p2_wins'] / self.episodes
        final_balance = min(p1_rate, p2_rate) / max(p1_rate, p2_rate) if max(p1_rate, p2_rate) > 0 else 0.0
        print(f"Final Balance Score: {final_balance:.4f}")

        if final_balance >= 0.85:
            print("Final Grade: EXCELLENT")
        elif final_balance >= 0.70:
            print("Final Grade: GOOD")
        else:
            print("Final Grade: NEEDS IMPROVEMENT")

        print()
        print("HUBER LOSS EXPERIMENT - Successfully used SmoothL1Loss throughout training")
        print("=" * 80)

        # Save models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("experiments/models", exist_ok=True)

        model1_path = f"experiments/models/copy61_huber_p1_{timestamp}.pth"
        model2_path = f"experiments/models/copy61_huber_p2_{timestamp}.pth"

        torch.save({
            'model_state_dict': self.agent_1.state_dict(),
            'config': {'architecture': 'Copy61_Huber', 'loss': 'SmoothL1Loss'},
            'stats': self.stats,
            'final_balance': final_balance
        }, model1_path)

        torch.save({
            'model_state_dict': self.agent_2.state_dict(),
            'config': {'architecture': 'Copy61_Huber', 'loss': 'SmoothL1Loss'},
            'stats': self.stats,
            'final_balance': final_balance
        }, model2_path)

        print(f"Models saved:")
        print(f"  Player 1: {model1_path}")
        print(f"  Player 2: {model2_path}")

        return model1_path, model2_path

def main():
    trainer = Copy61HuberTrainer(episodes=12000)
    model1, model2 = trainer.train()

    print()
    print("Copy 6.1 Huber Loss Experiment Complete!")
    print(f"Player 1 Model: {model1}")
    print(f"Player 2 Model: {model2}")

if __name__ == "__main__":
    main()