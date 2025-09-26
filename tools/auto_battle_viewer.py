#!/usr/bin/env python3
"""
Automatic Battle Viewer - No user input required
Shows complete model battle with board transitions
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
print("        Automatic Battle Viewer - Model Analysis")
print("        Quantum AI Model vs Model - Complete Battle")
print("=" * 70)

class SimpleGeisterBoard:
    """Simplified Geister board for visualization"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to initial state"""
        self.board = np.full((6, 6), '.', dtype=str)
        self.turn = 0
        self.current_player = 'A'
        self.game_over = False
        self.winner = None

        # Place initial pieces - Geister standard 8 pieces per player
        # Player A: 8 pieces on rows 4,5 (good=uppercase, bad=lowercase)
        self.board[4, 1:5] = ['A', 'A', 'a', 'a']  # Player A row 4: 2 good(A), 2 bad(a)
        self.board[5, 1:5] = ['A', 'A', 'a', 'a']  # Player A row 5: 2 good(A), 2 bad(a)
        # Player B: 8 pieces on rows 0,1 (good=uppercase, bad=lowercase)
        self.board[0, 1:5] = ['B', 'B', 'b', 'b']  # Player B row 0: 2 good(B), 2 bad(b)
        self.board[1, 1:5] = ['B', 'B', 'b', 'b']  # Player B row 1: 2 good(B), 2 bad(b)

        self.move_history = []
        self.save_state()

    def save_state(self):
        """Save current state"""
        state = {
            'turn': self.turn,
            'player': self.current_player,
            'board': self.board.copy(),
            'game_over': self.game_over,
            'winner': self.winner
        }
        self.move_history.append(state)

    def display_board(self, move_info=""):
        """Display board with move info"""
        print(f"\n--- Turn {self.turn} - Player {self.current_player} {move_info} ---")
        print("  0 1 2 3 4 5")
        for i in range(6):
            row_str = f"{i} "
            for j in range(6):
                row_str += f"{self.board[i, j]} "
            print(row_str)

    def make_move(self, action):
        """Make a move based on action (simplified)"""
        # Find pieces of current player
        pieces = []
        for i in range(6):
            for j in range(6):
                cell = self.board[i, j]
                if cell.upper() == self.current_player:  # Both uppercase and lowercase belong to player
                    pieces.append((i, j))

        if not pieces:
            self.game_over = True
            self.winner = 'B' if self.current_player == 'A' else 'A'
            return f"No pieces left for {self.current_player}"

        # Use action to select piece and direction
        piece_idx = action % len(pieces)
        direction_idx = (action // len(pieces)) % 4

        i, j = pieces[piece_idx]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        di, dj = directions[direction_idx]
        ni, nj = i + di, j + dj

        move_desc = f"Action {action}: "

        # Check bounds
        if not (0 <= ni < 6 and 0 <= nj < 6):
            move_desc += f"Out of bounds ({i},{j}) -> ({ni},{nj})"
            return move_desc

        target = self.board[ni, nj]

        # Can't move to own piece
        if target.upper() == self.current_player:
            move_desc += f"Blocked by own piece at ({ni},{nj})"
            return move_desc

        # Make the move
        if target == '.':
            move_desc += f"Move ({i},{j}) -> ({ni},{nj})"
        else:
            move_desc += f"Capture {target} at ({i},{j}) -> ({ni},{nj})"

        self.board[i, j] = '.'
        self.board[ni, nj] = piece

        # Check win conditions
        # Escape: good piece (uppercase) reaches opponent's back row
        if piece.isupper():  # Good piece (uppercase)
            if (self.current_player == 'A' and ni == 0) or \
               (self.current_player == 'B' and ni == 5):
                self.game_over = True
                self.winner = self.current_player
                move_desc += f" -> GOOD PIECE ESCAPE WIN!"

        # Capture win: if opponent has no good pieces left
        opponent = 'B' if self.current_player == 'A' else 'A'
        opponent_good_pieces = sum(1 for i in range(6) for j in range(6)
                                 if self.board[i,j] == opponent)  # Uppercase = good pieces
        if opponent_good_pieces == 0:
            self.game_over = True
            self.winner = self.current_player
            move_desc += f" -> ALL OPPONENT GOOD PIECES CAPTURED!"

        # Switch player
        self.current_player = 'B' if self.current_player == 'A' else 'A'
        self.turn += 1

        # Game limit
        if self.turn >= 40:
            self.game_over = True
            self.winner = 'Draw'
            move_desc += " -> Draw (turn limit)"

        self.save_state()
        return move_desc

class QuantumLayer(nn.Module):
    """Quantum layer for models"""
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
            for i in range(min(len(inputs), n_qubits)):
                qml.RY(inputs[i] * np.pi / 2, wires=i)

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
    """Classical-Quantum CNN"""
    def __init__(self, n_qubits=4, n_layers=1, state_dim=252, action_dim=36):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = state_dim
        self.action_dim = action_dim

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

        self.quantum = QuantumLayer(self.n_qubits, n_layers)

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

class AutoBattleViewer:
    """Automatic battle viewer"""
    def __init__(self, model1_path, model2_path):
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.load_models()

    def load_models(self):
        """Load both models"""
        print(f"Loading Model A: {Path(self.model1_path).name}")
        self.model_a = CQCNN(n_qubits=4, n_layers=1)
        try:
            state_dict = torch.load(self.model1_path, map_location='cpu')
            self.model_a.load_state_dict(state_dict, strict=False)
            print("  Model A loaded successfully")
        except Exception as e:
            print(f"  Model A load error: {e}")
            print("  Using random weights")

        print(f"Loading Model B: {Path(self.model2_path).name}")
        self.model_b = CQCNN(n_qubits=4, n_layers=1)
        try:
            state_dict = torch.load(self.model2_path, map_location='cpu')
            self.model_b.load_state_dict(state_dict, strict=False)
            print("  Model B loaded successfully")
        except Exception as e:
            print(f"  Model B load error: {e}")
            print("  Using random weights")

        self.model_a.eval()
        self.model_b.eval()

    def create_state_vector(self, board):
        """Convert board to state vector"""
        state = np.zeros(252)

        # Board encoding (first 36 dimensions)
        for i in range(6):
            for j in range(6):
                idx = i * 6 + j
                cell = board.board[i, j]
                if cell.upper() == 'A':
                    state[idx] = 1.0 if board.current_player == 'A' else -1.0
                elif cell.upper() == 'B':
                    state[idx] = 1.0 if board.current_player == 'B' else -1.0
                # else remains 0 for empty

        # Game state features
        state[36] = board.turn / 40.0  # Normalized turn
        state[37] = 1.0 if board.current_player == 'A' else 0.0
        state[38] = 1.0 if not board.game_over else 0.0

        # Piece counts
        a_count = sum(1 for i in range(6) for j in range(6) if board.board[i,j].upper() == 'A')
        b_count = sum(1 for i in range(6) for j in range(6) if board.board[i,j].upper() == 'B')
        state[39] = a_count / 8.0  # Now 8 pieces per player
        state[40] = b_count / 8.0

        return state

    def get_model_action(self, model, state, player_name):
        """Get model's action and analysis"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = model(state_tensor).squeeze()
            action_probs = torch.softmax(q_values, dim=0)
            best_action = torch.argmax(q_values).item()
            confidence = action_probs[best_action].item()

        return {
            'action': best_action,
            'confidence': confidence,
            'q_values': q_values.numpy(),
            'best_q': q_values[best_action].item()
        }

    def run_automatic_battle(self, max_turns=25):
        """Run complete battle automatically"""
        print(f"\nStarting Automatic Model Battle")
        print(f"Model A: {Path(self.model1_path).name}")
        print(f"Model B: {Path(self.model2_path).name}")
        print("=" * 70)

        board = SimpleGeisterBoard()
        board.display_board("(Initial Setup)")

        battle_log = []

        while not board.game_over and board.turn < max_turns:
            current_model = self.model_a if board.current_player == 'A' else self.model_b
            state = self.create_state_vector(board)

            # Get model decision
            decision = self.get_model_action(current_model, state, board.current_player)

            # Log decision
            battle_log.append({
                'turn': board.turn,
                'player': board.current_player,
                'action': decision['action'],
                'confidence': decision['confidence'],
                'best_q': decision['best_q'],
                'board_before': board.board.copy()
            })

            # Display model thinking
            print(f"\nPlayer {board.current_player} Analysis:")
            print(f"  Selected Action: {decision['action']}")
            print(f"  Confidence: {decision['confidence']:.3f}")
            print(f"  Q-Value: {decision['best_q']:.3f}")
            print(f"  Q-Range: [{decision['q_values'].min():.3f}, {decision['q_values'].max():.3f}]")

            # Execute move
            move_result = board.make_move(decision['action'])
            print(f"  Move Result: {move_result}")

            # Show resulting board
            board.display_board()

        # Final summary
        print("\n" + "=" * 70)
        print("BATTLE COMPLETE")
        print("=" * 70)
        print(f"Winner: {board.winner}")
        print(f"Total Turns: {board.turn}")
        print(f"Final Board State:")
        board.display_board("(Final)")

        # Save battle report
        self.save_battle_report(battle_log, board)

        return board.winner, board.turn, battle_log

    def save_battle_report(self, battle_log, board):
        """Save complete battle report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"experiments/results/auto_battle_{timestamp}.json"

        report = {
            'timestamp': timestamp,
            'model_a_path': str(self.model1_path),
            'model_b_path': str(self.model2_path),
            'winner': board.winner,
            'total_turns': board.turn,
            'battle_log': [],
            'final_board': board.board.tolist()
        }

        for entry in battle_log:
            report['battle_log'].append({
                'turn': entry['turn'],
                'player': entry['player'],
                'action': entry['action'],
                'confidence': float(entry['confidence']),
                'best_q_value': float(entry['best_q']),
                'board_before': entry['board_before'].tolist()
            })

        os.makedirs("experiments/results", exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nBattle report saved: {report_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python auto_battle_viewer.py <model1.pth> <model2.pth>")
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

    viewer = AutoBattleViewer(model1_path, model2_path)
    winner, turns, log = viewer.run_automatic_battle()

    print(f"\n" + "=" * 50)
    print(f"FINAL SUMMARY")
    print(f"=" * 50)
    print(f"Winner: {winner}")
    print(f"Battle Duration: {turns} turns")
    print(f"Decisions Logged: {len(log)}")

    # Decision quality analysis
    a_decisions = [entry for entry in log if entry['player'] == 'A']
    b_decisions = [entry for entry in log if entry['player'] == 'B']

    if a_decisions:
        avg_conf_a = np.mean([d['confidence'] for d in a_decisions])
        avg_q_a = np.mean([d['best_q'] for d in a_decisions])
        print(f"Model A Avg Confidence: {avg_conf_a:.3f}, Avg Q-Value: {avg_q_a:.3f}")

    if b_decisions:
        avg_conf_b = np.mean([d['confidence'] for d in b_decisions])
        avg_q_b = np.mean([d['best_q'] for d in b_decisions])
        print(f"Model B Avg Confidence: {avg_conf_b:.3f}, Avg Q-Value: {avg_q_b:.3f}")

if __name__ == "__main__":
    main()