#!/usr/bin/env python3
"""
Model Battle System
Load PTH models and run battles between them
"""

import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("        Model Battle System")
print("        PTH Model vs PTH Model")
print("=" * 60)

class GeisterEnvironment:
    """Simplified Geister game environment for battles"""
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
        return list(range(36))

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
                    qml.CNOT(wires=[n_qubits - 1, 0])

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
    def __init__(self, n_qubits=4, n_layers=1, state_dim=252, action_dim=36, embedding_type='amplitude'):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Classical frontend - same as Copy 6.1
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
        self.quantum = QuantumLayer(self.n_qubits, self.n_layers, embedding_type)

        # Classical backend - same as Copy 6.1
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

class ModelPlayer:
    """AI Player using trained PTH model"""
    def __init__(self, model_path, player_id):
        self.model_path = model_path
        self.player_id = player_id
        self.load_model()

    def load_model(self):
        """Load trained model from PTH file"""
        try:
            # Create model architecture (using Copy 6.1 defaults)
            self.network = CQCNN(n_qubits=4, n_layers=1, state_dim=252, action_dim=36)

            # Load weights
            state_dict = torch.load(self.model_path, map_location='cpu')
            self.network.load_state_dict(state_dict, strict=False)
            self.network.eval()  # Set to evaluation mode

            print(f"Player {self.player_id} loaded: {self.model_path}")
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")
            print(f"Using random model for Player {self.player_id}")
            self.network = CQCNN(n_qubits=4, n_layers=1, state_dim=252, action_dim=36)

    def select_action(self, state, valid_actions):
        """Select action using trained model"""
        try:
            with torch.no_grad():
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                q_values = self.network(state_tensor)

                # Filter to valid actions only
                valid_q_values = {action: q_values[0][action].item() for action in valid_actions}
                return max(valid_q_values, key=valid_q_values.get)
        except:
            # Fallback to random if model fails
            return random.choice(valid_actions)

class BattleArena:
    """Main battle coordination system"""
    def __init__(self, model1_path, model2_path):
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.setup_battle()

    def setup_battle(self):
        """Initialize battle environment and players"""
        self.env = GeisterEnvironment()
        self.player1 = ModelPlayer(self.model1_path, 1)
        self.player2 = ModelPlayer(self.model2_path, 2)

        # Battle statistics
        self.battle_results = []
        self.wins_p1 = 0
        self.wins_p2 = 0
        self.draws = 0

    def run_single_battle(self):
        """Run a single battle between the two models"""
        state = self.env.reset()
        current_player = 1
        players = {1: self.player1, 2: self.player2}

        moves = 0
        max_moves = self.env.max_turns

        while not self.env.game_over and moves < max_moves:
            player = players[current_player]
            valid_actions = self.env.get_valid_actions()
            action = player.select_action(state, valid_actions)

            state, reward, done = self.env.step(action)
            moves += 1

            # Switch players
            current_player = 2 if current_player == 1 else 1

        # Determine winner
        if self.env.winner == 1:
            self.wins_p1 += 1
            winner = "Player 1"
        elif self.env.winner == 2:
            self.wins_p2 += 1
            winner = "Player 2"
        else:
            self.draws += 1
            winner = "Draw"

        return winner, moves

    def run_tournament(self, num_battles=100):
        """Run multiple battles and show results"""
        print(f"\nStarting tournament: {num_battles} battles")
        print(f"Model 1: {Path(self.model1_path).name}")
        print(f"Model 2: {Path(self.model2_path).name}")
        print("-" * 60)

        start_time = time.time()

        for battle in range(num_battles):
            winner, moves = self.run_single_battle()

            # Show first few battles in detail
            if battle < 5:
                print(f"Battle {battle+1:3d}: {winner:8s} wins in {moves:3d} moves")
            elif (battle + 1) % 20 == 0:
                print(f"Progress: {battle+1}/{num_battles} battles completed")

        elapsed_time = time.time() - start_time

        # Final results
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS")
        print("=" * 60)
        print(f"Total Battles: {num_battles}")
        print(f"Player 1: {self.wins_p1} wins ({100*self.wins_p1/num_battles:.1f}%)")
        print(f"Player 2: {self.wins_p2} wins ({100*self.wins_p2/num_battles:.1f}%)")
        print(f"Draws: {self.draws} ({100*self.draws/num_battles:.1f}%)")
        print(f"Battle Time: {elapsed_time:.1f}s ({elapsed_time/num_battles:.3f}s per battle)")

        # Determine overall winner
        if self.wins_p1 > self.wins_p2:
            print(f"\nWINNER: Player 1 ({Path(self.model1_path).name})")
        elif self.wins_p2 > self.wins_p1:
            print(f"\nWINNER: Player 2 ({Path(self.model2_path).name})")
        else:
            print(f"\nRESULT: TIE")

        # Balance analysis
        balance = 1.0 - abs(self.wins_p1 - self.wins_p2) / num_battles
        print(f"Balance Score: {balance:.3f}")
        print("=" * 60)

def main():
    if len(sys.argv) != 3:
        print("Usage: python model_battle.py <model1.pth> <model2.pth>")
        print("Example: python model_battle.py models/model1_p1.pth models/model2_p1.pth")
        sys.exit(1)

    model1_path = sys.argv[1]
    model2_path = sys.argv[2]

    # Check if files exist
    if not os.path.exists(model1_path):
        print(f"Model 1 not found: {model1_path}")
        sys.exit(1)
    if not os.path.exists(model2_path):
        print(f"Model 2 not found: {model2_path}")
        sys.exit(1)

    # Run battle
    arena = BattleArena(model1_path, model2_path)
    arena.run_tournament(100)

if __name__ == "__main__":
    main()