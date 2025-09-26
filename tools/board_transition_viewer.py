#!/usr/bin/env python3
"""
Board Transition Viewer for Quantum AI Models
Visualizes move-by-move board states during model battles
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("        Board Transition Viewer - Model Battle Analysis")
print("        Quantum AI vs Quantum AI - Move by Move")
print("=" * 70)

class SimpleGeisterBoard:
    """Simple Geister board representation for visualization"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial Geister setup"""
        self.board = np.zeros((6, 6), dtype=str)
        self.turn = 0
        self.current_player = 'A'
        self.game_over = False
        self.winner = None

        # Initial piece placement (simplified)
        # Player A (bottom) - 4 pieces on rows 4,5
        self.board[4, 1:5] = ['A1', 'A2', 'A3', 'A4']
        # Player B (top) - 4 pieces on rows 0,1
        self.board[1, 1:5] = ['B1', 'B2', 'B3', 'B4']

        self.history = []
        self.save_state()

    def save_state(self):
        """Save current board state to history"""
        state = {
            'turn': self.turn,
            'player': self.current_player,
            'board': self.board.copy(),
            'game_over': self.game_over,
            'winner': self.winner
        }
        self.history.append(state)

    def display_board(self, turn_info=""):
        """Display current board state"""
        print(f"\n=== Turn {self.turn} - Player {self.current_player} {turn_info} ===")
        print("  0 1 2 3 4 5")
        for i in range(6):
            row_str = f"{i} "
            for j in range(6):
                cell = self.board[i, j]
                if cell == '':
                    row_str += ". "
                else:
                    row_str += f"{cell[0]} "  # Show just A or B
            print(row_str)
        print()

    def make_random_move(self):
        """Make a random valid move for current player"""
        # Find pieces of current player
        pieces = []
        for i in range(6):
            for j in range(6):
                if self.board[i, j].startswith(self.current_player):
                    pieces.append((i, j))

        if not pieces:
            self.game_over = True
            self.winner = 'B' if self.current_player == 'A' else 'A'
            return

        # Try random moves
        for _ in range(50):  # Max attempts
            piece_pos = pieces[np.random.choice(len(pieces))]
            i, j = piece_pos

            # Try random direction
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            di, dj = directions[np.random.choice(len(directions))]
            ni, nj = i + di, j + dj

            # Check bounds
            if 0 <= ni < 6 and 0 <= nj < 6:
                # Check if target is empty or opponent piece
                target = self.board[ni, nj]
                if target == '' or not target.startswith(self.current_player):
                    # Make move
                    piece = self.board[i, j]
                    self.board[i, j] = ''

                    if target != '':  # Capture
                        print(f"    {piece} captures {target} at ({ni},{nj})")

                    self.board[ni, nj] = piece

                    # Check win conditions (simplified)
                    # Escape: reaching opponent's starting row
                    if (self.current_player == 'A' and ni <= 1) or \
                       (self.current_player == 'B' and ni >= 4):
                        self.game_over = True
                        self.winner = self.current_player
                        print(f"    {piece} escapes! {self.current_player} wins!")

                    break

        # Switch player
        self.current_player = 'B' if self.current_player == 'A' else 'A'
        self.turn += 1

        # Game limit
        if self.turn >= 50:
            self.game_over = True
            self.winner = 'Draw'

        self.save_state()

class QuantumLayer(nn.Module):
    """Quantum processing layer"""
    def __init__(self, n_qubits=4, n_layers=1, embedding_type='amplitude'):
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

class CQCNN(nn.Module):
    """Classical-Quantum CNN for battle visualization"""
    def __init__(self, n_qubits=4, n_layers=1, state_dim=252, action_dim=36):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Classical preprocessing
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
            nn.Linear(28, self.n_qubits)
        )

        # Quantum layer
        self.quantum = QuantumLayer(self.n_qubits, n_layers)

        # Classical postprocessing
        self.backend = nn.Sequential(
            nn.Linear(self.n_qubits, 56),
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

class ModelBattleViewer:
    """Battle viewer with board transition display"""
    def __init__(self, model1_path, model2_path):
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.load_models()

    def load_models(self):
        """Load both models"""
        print(f"Loading Model A: {self.model1_path}")
        self.model_a = CQCNN(n_qubits=4, n_layers=1)
        try:
            state_dict = torch.load(self.model1_path, map_location='cpu')
            self.model_a.load_state_dict(state_dict, strict=False)
            print("  Model A loaded successfully")
        except Exception as e:
            print(f"  Model A load failed: {e}")
            print("  Using random weights for Model A")

        print(f"Loading Model B: {self.model2_path}")
        self.model_b = CQCNN(n_qubits=4, n_layers=1)
        try:
            state_dict = torch.load(self.model2_path, map_location='cpu')
            self.model_b.load_state_dict(state_dict, strict=False)
            print("  Model B loaded successfully")
        except Exception as e:
            print(f"  Model B load failed: {e}")
            print("  Using random weights for Model B")

        self.model_a.eval()
        self.model_b.eval()

    def get_model_decision(self, model, state, player):
        """Get model's action decision with analysis"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor)
            action_probs = torch.softmax(q_values, dim=1)
            best_action = torch.argmax(q_values, dim=1).item()
            confidence = action_probs[0, best_action].item()

        return {
            'action': best_action,
            'confidence': confidence,
            'q_values': q_values[0].numpy(),
            'top3_actions': torch.topk(q_values, 3, dim=1)[1][0].numpy()
        }

    def create_state_vector(self, board, turn, current_player):
        """Convert board to 252D state vector"""
        state = np.zeros(252)

        # Board representation (36 dimensions for 6x6 board)
        for i in range(6):
            for j in range(6):
                idx = i * 6 + j
                cell = board.board[i, j]
                if cell.startswith('A'):
                    state[idx] = 1.0 if current_player == 'A' else -1.0
                elif cell.startswith('B'):
                    state[idx] = 1.0 if current_player == 'B' else -1.0

        # Game state info
        state[36] = turn / 50.0  # Normalized turn
        state[37] = 1.0 if current_player == 'A' else 0.0
        state[38] = 1.0 if not board.game_over else 0.0

        # Add some strategic features
        a_pieces = np.sum([1 for i in range(6) for j in range(6) if board.board[i,j].startswith('A')])
        b_pieces = np.sum([1 for i in range(6) for j in range(6) if board.board[i,j].startswith('B')])
        state[39] = a_pieces / 4.0
        state[40] = b_pieces / 4.0

        return state

    def run_battle_with_visualization(self, max_turns=30):
        """Run battle with move-by-move visualization"""
        print(f"\nStarting Model Battle with Board Visualization")
        print(f"Model A: {Path(self.model1_path).name}")
        print(f"Model B: {Path(self.model2_path).name}")
        print("=" * 70)

        board = SimpleGeisterBoard()
        board.display_board("(Initial Setup)")

        decisions_log = []

        while not board.game_over and board.turn < max_turns:
            current_model = self.model_a if board.current_player == 'A' else self.model_b
            state = self.create_state_vector(board, board.turn, board.current_player)

            # Get model decision
            decision = self.get_model_decision(current_model, state, board.current_player)
            decisions_log.append({
                'turn': board.turn,
                'player': board.current_player,
                'decision': decision,
                'board_before': board.board.copy()
            })

            # Display model analysis
            print(f"Player {board.current_player} Analysis:")
            print(f"  Best Action: {decision['action']} (confidence: {decision['confidence']:.3f})")
            print(f"  Top 3 Actions: {decision['top3_actions']}")
            print(f"  Q-Value Range: [{decision['q_values'].min():.3f}, {decision['q_values'].max():.3f}]")

            # Make move (simplified random move for visualization)
            board.make_random_move()

            # Display resulting board
            board.display_board()

            # Pause for readability
            input("Press Enter to continue to next move...")

        # Final results
        print("=" * 70)
        print(f"Game Over! Winner: {board.winner}")
        print(f"Total Turns: {board.turn}")
        print("=" * 70)

        # Save battle log
        self.save_battle_log(decisions_log, board)

        return board.winner, board.turn, decisions_log

    def save_battle_log(self, decisions_log, board):
        """Save detailed battle log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"experiments/results/battle_log_{timestamp}.json"

        battle_data = {
            'timestamp': timestamp,
            'model_a': str(self.model1_path),
            'model_b': str(self.model2_path),
            'winner': board.winner,
            'total_turns': board.turn,
            'decisions': []
        }

        for decision in decisions_log:
            battle_data['decisions'].append({
                'turn': decision['turn'],
                'player': decision['player'],
                'action': decision['decision']['action'],
                'confidence': decision['decision']['confidence'],
                'top3_actions': decision['decision']['top3_actions'].tolist(),
                'q_value_stats': {
                    'min': float(decision['decision']['q_values'].min()),
                    'max': float(decision['decision']['q_values'].max()),
                    'mean': float(decision['decision']['q_values'].mean())
                }
            })

        os.makedirs("experiments/results", exist_ok=True)
        with open(log_file, 'w') as f:
            json.dump(battle_data, f, indent=2)

        print(f"Battle log saved: {log_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python board_transition_viewer.py <model1.pth> <model2.pth>")
        print("\nAvailable models:")
        models_dir = Path("experiments/models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                print(f"  {model_file}")
        return

    model1_path = sys.argv[1]
    model2_path = sys.argv[2]

    if not Path(model1_path).exists():
        print(f"Model 1 not found: {model1_path}")
        return

    if not Path(model2_path).exists():
        print(f"Model 2 not found: {model2_path}")
        return

    viewer = ModelBattleViewer(model1_path, model2_path)
    winner, turns, log = viewer.run_battle_with_visualization()

    print(f"\nBattle Summary:")
    print(f"Winner: {winner}")
    print(f"Duration: {turns} turns")
    print(f"Decisions logged: {len(log)}")

if __name__ == "__main__":
    main()