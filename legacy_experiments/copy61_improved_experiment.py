#!/usr/bin/env python3
"""
Copy 6.1 Improved Progressive Learning Experiment
12K Episodes with Optimized 4-Phase Learning System

Key Improvements from Copy 6 Extended:
1. Reduced training duration: 20K -> 12K episodes
2. Refined epsilon strategy: min 0.05 (vs 0.024)
3. Improved 4-phase system (vs 5-phase)
4. Enhanced monitoring with Loss display
5. Better emergency response thresholds
6. Optimized perturbation parameters
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
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Random seed setup for reproducibility
EXPERIMENT_SEED = 42
random.seed(EXPERIMENT_SEED)
np.random.seed(EXPERIMENT_SEED)
torch.manual_seed(EXPERIMENT_SEED)
torch.backends.cudnn.deterministic = True

print("=" * 80)
print("        Copy 6.1 Improved Progressive Learning Experiment")
print("             12K Episodes - 4-Phase Optimized System")
print("=" * 80)

# Load configuration
config_path = "quantum_copy61_improved_config_2025-09-26.json"

try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"Configuration loaded: {config_path}")
except FileNotFoundError:
    print(f"Configuration file not found: {config_path}")
    sys.exit(1)

print(f"Experiment: {config['metadata']['experiment_name']}")
print(f"Philosophy: {config['metadata']['philosophy']}")
print(f"Improvements: {config['metadata']['major_improvements']}")
print(f"Target Episodes: {config['training_schedule']['total_episodes']:,}")

# Configuration expansion
N_QUBITS = config['quantum_architecture']['n_qubits']
N_LAYERS = config['quantum_architecture']['n_layers']
STATE_DIM = config['quantum_architecture']['state_dimension']
ACTION_DIM = 36
MAX_TURNS = config['game_mechanics']['max_turns']
BATCH_SIZE = config['hyperparameters']['batch_size']
BASE_LEARNING_RATE = config['hyperparameters']['base_learning_rate']
MAX_EPISODES = config['training_schedule']['total_episodes']
BUFFER_SIZE = config['hyperparameters']['memory_size']
GAMMA = config['hyperparameters']['gamma']

LEARNING_PHASES = config['learning_phases']
QUANTUM_PERTURBATION = config['dynamic_quantum_perturbation']
BALANCE_DETECTION = config['advanced_balance_detection']
EMERGENCY_RESPONSE = config['graduated_emergency_response']

print(f"Quantum Architecture: {N_QUBITS}Q{N_LAYERS}L, State: {STATE_DIM}D")

class Copy61ImprovedGeisterEnvironment:
    """Copy 6.1: Optimized 252D environment for 12K episodes"""

    def __init__(self):
        self.board_size = 6
        self.max_turns = MAX_TURNS
        self.stalemate_turns = config['game_mechanics']['stalemate_detection_turns']
        self.repetition_limit = config['game_mechanics']['repetition_limit']
        self.reset()
        print(f"Copy 6.1 Environment: {STATE_DIM}D state space, {self.max_turns} turns max")

    def reset(self):
        self.board = np.zeros((6, 6), dtype=int)
        self.turn = 0
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.position_history = []

        # Player 1 pieces (bottom)
        p1_positions = [(4, 1), (4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4)]
        p1_pieces = [1, 1, 1, 1, -1, -1, -1, -1]
        random.shuffle(p1_pieces)
        for pos, piece in zip(p1_positions, p1_pieces):
            self.board[pos[0]][pos[1]] = piece

        # Player 2 pieces (top)
        p2_positions = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)]
        p2_pieces = [2, 2, 2, 2, -2, -2, -2, -2]
        random.shuffle(p2_pieces)
        for pos, piece in zip(p2_positions, p2_pieces):
            self.board[pos[0]][pos[1]] = piece

        return self.get_state()

    def get_state(self):
        """252D state vector - proven optimal"""
        state = np.zeros(STATE_DIM)
        for i in range(6):
            for j in range(6):
                base_idx = (i * 6 + j) * 7
                value = self.board[i][j]
                if value == 0:
                    state[base_idx] = 1
                elif value == 1:
                    state[base_idx + 1] = 1
                elif value == -1:
                    state[base_idx + 2] = 1
                elif value == 2:
                    state[base_idx + 3] = 1
                elif value == -2:
                    state[base_idx + 4] = 1
                state[base_idx + 5] = 1 if self.current_player == 1 else 0
                state[base_idx + 6] = min(self.turn / self.max_turns, 1.0)
        return state

    def get_valid_moves(self, player=None):
        """Valid moves in 36D action space"""
        if player is None:
            player = self.current_player
        valid_moves = []
        piece_values = [1, -1] if player == 1 else [2, -2]
        move_index = 0

        for from_i in range(6):
            for from_j in range(6):
                if self.board[from_i][from_j] in piece_values:
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if move_index >= ACTION_DIM:
                            break
                        to_i, to_j = from_i + di, from_j + dj
                        if 0 <= to_i < 6 and 0 <= to_j < 6:
                            target = self.board[to_i][to_j]
                            if (target == 0 or
                                (player == 1 and abs(target) == 2) or
                                (player == 2 and abs(target) == 1)):
                                valid_moves.append((move_index, "move", (from_i, from_j), (to_i, to_j)))
                        move_index += 1
                if move_index >= ACTION_DIM:
                    break
            if move_index >= ACTION_DIM:
                break
        return valid_moves

    def make_move(self, move):
        """Enhanced move execution with refined reward"""
        move_index, direction, from_pos, to_pos = move
        from_i, from_j = from_pos
        to_i, to_j = to_pos

        piece = self.board[from_i][from_j]
        target = self.board[to_i][to_j]

        self.position_history.append(self.board.copy())
        self.board[from_i][from_j] = 0
        self.board[to_i][to_j] = piece

        # Copy 6.1 refined reward system
        reward = 0.0
        if target != 0:
            reward += 1.2 if target > 0 else -0.2  # Slightly less aggressive
        if abs(piece) == 1 and piece > 0:
            if ((self.current_player == 1 and to_i < from_i) or
                (self.current_player == 2 and to_i > from_i)):
                reward += 0.08  # Slightly reduced advancement reward

        done = self._check_win_condition(piece, to_pos, target)
        self.turn += 1

        if self.turn >= self.max_turns or self._check_stalemate():
            self.game_over = True
            self.winner = None
            done = True

        if not done:
            self.current_player = 2 if self.current_player == 1 else 1

        return self.get_state(), reward, done, {
            'captured': target, 'winner': self.winner, 'turn': self.turn
        }

    def _check_win_condition(self, piece, to_pos, captured):
        """Standard Geister win conditions"""
        to_i, to_j = to_pos

        # Escape victory
        if abs(piece) == 1 and piece > 0:
            if ((self.current_player == 1 and to_i == 0 and to_j in [0, 5]) or
                (self.current_player == 2 and to_i == 5 and to_j in [0, 5])):
                self.game_over = True
                self.winner = self.current_player
                return True

        # Capture all good pieces victory
        if captured is not None and captured > 0:
            opponent_good_count = 0
            search_value = 2 if self.current_player == 1 else 1
            for i in range(6):
                for j in range(6):
                    if self.board[i][j] == search_value:
                        opponent_good_count += 1
            if opponent_good_count == 0:
                self.game_over = True
                self.winner = self.current_player
                return True
        return False

    def _check_stalemate(self):
        """Enhanced stalemate detection"""
        if len(self.position_history) < self.stalemate_turns:
            return False
        recent_positions = self.position_history[-self.stalemate_turns:]
        current_position = self.board.copy()
        repetition_count = sum(1 for pos in recent_positions
                             if np.array_equal(pos, current_position))
        return repetition_count >= self.repetition_limit


class Copy61ImprovedCQCNN(nn.Module):
    """Copy 6.1: Improved CQCNN with optimized architecture"""

    def __init__(self):
        super().__init__()
        print(f"Copy 6.1 Enhanced CQCNN: {STATE_DIM}D -> {N_QUBITS}Q{N_LAYERS}L -> {ACTION_DIM}D")

        # Optimized frontend
        self.frontend = nn.Sequential(
            nn.Linear(STATE_DIM, 120),
            nn.ReLU(),
            nn.BatchNorm1d(120),
            nn.Dropout(0.20),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Linear(60, 28),
            nn.Tanh(),
            nn.Linear(28, N_QUBITS),
            nn.Tanh()
        )

        # Quantum circuit
        self.dev = qml.device('default.qubit', wires=N_QUBITS)
        self.quantum_params = nn.Parameter(
            torch.empty(N_LAYERS, N_QUBITS, 2).uniform_(-0.1, 0.1)
        )
        self.quantum_node = qml.QNode(self._quantum_circuit, self.dev, interface='torch')

        # Optimized backend
        self.backend = nn.Sequential(
            nn.Linear(N_QUBITS, 56),
            nn.ReLU(),
            nn.BatchNorm1d(56),
            nn.Dropout(0.12),
            nn.Linear(56, 112),
            nn.ReLU(),
            nn.BatchNorm1d(112),
            nn.Linear(112, 56),
            nn.ReLU(),
            nn.Linear(56, ACTION_DIM)
        )

        self._initialize_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        quantum_params = self.quantum_params.numel()
        print(f"Parameters: Classical={total_params-quantum_params:,}, "
              f"Quantum={quantum_params}, Total={total_params:,}")

    def _initialize_weights(self):
        """Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _quantum_circuit(self, features, params):
        """4Q2L quantum circuit with circular entanglement"""
        # Amplitude encoding
        for i in range(N_QUBITS):
            qml.RY(features[i] * np.pi, wires=i)

        # Variational layers
        for layer in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
            # Circular entanglement
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    def forward(self, state):
        batch_size = state.shape[0]
        state_flat = state.view(batch_size, -1)
        frontend_out = self.frontend(state_flat)

        # Quantum processing
        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.quantum_node(frontend_out[i], self.quantum_params)
            quantum_outputs.append(torch.stack(q_out).float())

        quantum_out = torch.stack(quantum_outputs)
        output = self.backend(quantum_out)
        return output


class ImprovedLearningPhaseManager:
    """4-phase optimized learning management for 12K episodes"""

    def __init__(self, learning_phases):
        self.phases = learning_phases
        self.phase_names = list(learning_phases.keys())
        self.current_phase_idx = 0
        self.episode_count = 0
        self.phase_start_episode = 0

        print("\n=== Copy 6.1 Improved Learning Phase Manager ===")
        total_fixed = 0
        for i, (name, config) in enumerate(learning_phases.items()):
            episodes = config['episodes']
            if episodes != "remaining":
                total_fixed += episodes
            print(f"Phase {i+1}: {name} - {episodes} episodes")
        remaining = MAX_EPISODES - total_fixed
        print(f"Phase 4 (Mastery) actual episodes: {remaining}")

    def get_current_phase(self):
        if self.current_phase_idx >= len(self.phase_names):
            return self.phase_names[-1], self.phases[self.phase_names[-1]]
        phase_name = self.phase_names[self.current_phase_idx]
        return phase_name, self.phases[phase_name]

    def advance_episode(self):
        self.episode_count += 1
        phase_name, phase_config = self.get_current_phase()
        episodes_in_phase = self.episode_count - self.phase_start_episode

        if (phase_config['episodes'] != "remaining" and
            episodes_in_phase >= phase_config['episodes']):
            self._transition_to_next_phase()

    def _transition_to_next_phase(self):
        if self.current_phase_idx < len(self.phase_names) - 1:
            old_phase = self.phase_names[self.current_phase_idx]
            self.current_phase_idx += 1
            self.phase_start_episode = self.episode_count
            new_phase = self.phase_names[self.current_phase_idx]
            print(f"\nPHASE PHASE TRANSITION: {old_phase} -> {new_phase} at episode {self.episode_count}")

    def get_phase_epsilon(self):
        phase_name, phase_config = self.get_current_phase()
        episodes_in_phase = self.episode_count - self.phase_start_episode

        epsilon_start = phase_config['epsilon_start']
        epsilon_min = phase_config['epsilon_min']
        epsilon_decay = phase_config['epsilon_decay']

        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episodes_in_phase))
        return epsilon, phase_name

    def get_phase_learning_rate(self):
        phase_name, phase_config = self.get_current_phase()
        return phase_config.get('learning_rate', BASE_LEARNING_RATE)

    def get_phase_batch_multiplier(self):
        phase_name, phase_config = self.get_current_phase()
        return phase_config.get('batch_size_multiplier', 1.0)


class ImprovedAdvancedBalanceMonitor:
    """Improved balance detection for optimized learning"""

    def __init__(self, balance_config):
        self.config = balance_config
        self.timeframes = balance_config['multi_timeframe_analysis']
        self.intervention_triggers = balance_config['intervention_triggers']
        self.game_history = []
        self.balance_history = []
        print(f"Copy 6.1 Balance Monitor: {self.timeframes}")

    def add_game_result(self, result):
        self.game_history.append(result)

    def analyze_balance(self, episode):
        if len(self.game_history) < 50:
            return {
                'status': 'insufficient_data',
                'draw_rate': 0.0,
                'balance': 1.0,
                'intervention_needed': None
            }

        analysis = {}
        for timeframe_name, window_size in self.timeframes.items():
            window_size = min(window_size, len(self.game_history))
            recent_games = self.game_history[-window_size:]

            p1_wins = sum(1 for g in recent_games if g.get('winner') == 1)
            p2_wins = sum(1 for g in recent_games if g.get('winner') == 2)
            draws = sum(1 for g in recent_games if g.get('winner') is None)

            draw_rate = draws / len(recent_games)
            balance = (min(p1_wins, p2_wins) / max(p1_wins, p2_wins)
                      if max(p1_wins, p2_wins) > 0 else 1.0)

            analysis[timeframe_name] = {
                'draw_rate': draw_rate, 'balance': balance,
                'p1_wins': p1_wins, 'p2_wins': p2_wins,
                'draws': draws, 'total_games': len(recent_games)
            }

        intervention_needed = self._determine_intervention(analysis)
        primary_analysis = analysis['short_term']

        result = {
            'status': 'analyzed',
            'draw_rate': primary_analysis['draw_rate'],
            'balance': primary_analysis['balance'],
            'intervention_needed': intervention_needed,
            'episode': episode
        }

        self.balance_history.append(result)
        return result

    def _determine_intervention(self, analysis):
        short_term = analysis['short_term']

        immediate = self.intervention_triggers['immediate']
        if (short_term['draw_rate'] >= immediate['draw_rate_threshold'] or
            short_term['balance'] <= immediate['balance_threshold']):
            return {
                'type': 'immediate',
                'reason': 'critical_metrics',
                'draw_rate': short_term['draw_rate'],
                'balance': short_term['balance']
            }
        return None


class ImprovedDynamicQuantumPerturbation:
    """Improved quantum perturbation system"""

    def __init__(self, perturbation_config, phase_manager):
        self.config = perturbation_config
        self.phase_manager = phase_manager
        self.last_perturbation_episode = -1
        self.perturbation_history = []
        print(f"Copy 6.1 Quantum Perturbation: Base freq={self.config['adaptive_frequency']['base_frequency']}")

    def should_apply_perturbation(self, episode, draw_rate):
        freq_config = self.config['adaptive_frequency']
        base_freq = freq_config['base_frequency']

        if freq_config['draw_rate_scaling']:
            frequency = base_freq * (1 - draw_rate * 0.4)  # Improved scaling
            frequency = max(freq_config['min_frequency'],
                          min(freq_config['max_frequency'], frequency))
        else:
            frequency = base_freq

        episodes_since_last = episode - self.last_perturbation_episode
        return episodes_since_last >= int(frequency)

    def apply_perturbation(self, model_1, model_2, episode, draw_rate):
        if not self.should_apply_perturbation(episode, draw_rate):
            return False

        # Calculate amplitude
        amp_config = self.config['intelligent_amplitude']
        base_amplitude = amp_config['base_amplitude']

        # Phase-based multiplier
        phase_name, _ = self.phase_manager.get_current_phase()
        phase_key_map = {
            'phase_1_initialization': 'initialization',
            'phase_2_exploration': 'exploration',
            'phase_3_consolidation': 'consolidation',
            'phase_4_mastery': 'mastery'
        }

        phase_key = phase_key_map.get(phase_name, 'mastery')
        phase_multiplier = amp_config['phase_multipliers'].get(phase_key, 1.0)

        # Improved draw rate scaling
        if amp_config.get('draw_rate_scaling', False):
            draw_multiplier = 1 + draw_rate * 1.5  # Less aggressive
        else:
            draw_multiplier = 1.0

        amplitude = base_amplitude * phase_multiplier * draw_multiplier

        # Apply selective perturbation
        with torch.no_grad():
            self._apply_selective_perturbation(model_1, model_2, amplitude)

        self.perturbation_history.append({
            'episode': episode,
            'amplitude': amplitude,
            'draw_rate': draw_rate,
            'phase': phase_name
        })

        self.last_perturbation_episode = episode
        return True

    def _apply_selective_perturbation(self, model_1, model_2, amplitude):
        perturbation_ratio = self.config['perturbation_targeting']['perturbation_ratio']
        total_params = model_1.quantum_params.numel()
        num_perturb = int(total_params * perturbation_ratio)
        param_indices = torch.randperm(total_params)[:num_perturb]

        flat_params_1 = model_1.quantum_params.view(-1)
        flat_params_2 = model_2.quantum_params.view(-1)
        perturbation = torch.randn(num_perturb) * amplitude

        flat_params_1[param_indices] += perturbation
        flat_params_2[param_indices] += perturbation * 0.8


class Copy61ImprovedProgressiveExperiment:
    """Copy 6.1: 12K Episodes Improved Progressive Learning"""

    def __init__(self):
        self.env = Copy61ImprovedGeisterEnvironment()

        # Models
        self.model_1 = Copy61ImprovedCQCNN()
        self.model_2 = Copy61ImprovedCQCNN()

        # Player 2 diversity
        with torch.no_grad():
            for param in self.model_2.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        # Learning systems
        self.phase_manager = ImprovedLearningPhaseManager(LEARNING_PHASES)
        self.balance_monitor = ImprovedAdvancedBalanceMonitor(BALANCE_DETECTION)
        self.quantum_perturb = ImprovedDynamicQuantumPerturbation(
            QUANTUM_PERTURBATION, self.phase_manager
        )

        # Optimizers
        self.optimizer_1 = optim.Adam(self.model_1.parameters(), lr=BASE_LEARNING_RATE)
        self.optimizer_2 = optim.Adam(self.model_2.parameters(), lr=BASE_LEARNING_RATE)

        # Learning components
        self.criterion = nn.MSELoss()
        self.replay_buffer_1 = deque(maxlen=BUFFER_SIZE)
        self.replay_buffer_2 = deque(maxlen=BUFFER_SIZE)

        # Tracking
        self.game_results = []
        self.epsilon_history = []
        self.losses_1 = []
        self.losses_2 = []
        self.emergency_responses = []

        # Loss monitoring
        self.recent_losses_1 = deque(maxlen=50)
        self.recent_losses_2 = deque(maxlen=50)

        self.start_time = time.time()

        print(f"\n=== Copy 6.1 Improved Experiment Initialized ===")
        print(f"Target Episodes: {MAX_EPISODES:,}")
        print(f"4-Phase Optimized Progressive Learning Active")
        print(f"Enhanced Monitoring with Loss Display")
        print(f"Refined Emergency Response System Ready")

    def select_action(self, model, state, valid_moves, epsilon):
        """Action selection with epsilon-greedy"""
        if random.random() < epsilon:
            return random.choice(valid_moves)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).float()
            q_values = model(state_tensor).squeeze(0)
        if was_training:
            model.train()

        best_score = float('-inf')
        best_move = valid_moves[0]

        for move in valid_moves:
            move_index, direction, from_pos, to_pos = move
            if move_index < len(q_values):
                score = q_values[move_index].item()
                if score > best_score:
                    best_score = score
                    best_move = move
        return best_move

    def play_game(self, episode):
        """Game execution with improved phase-aware epsilon"""
        state = self.env.reset()
        done = False

        game_states_1 = []
        game_states_2 = []

        # Improved epsilon calculation
        base_epsilon, current_phase = self.phase_manager.get_phase_epsilon()

        # Balance analysis and emergency response
        balance_analysis = self.balance_monitor.analyze_balance(episode)
        emergency_boost = 0.0

        if balance_analysis['intervention_needed']:
            intervention = balance_analysis['intervention_needed']
            if intervention['type'] == 'immediate':
                # Apply improved graduated emergency response
                draw_rate = balance_analysis['draw_rate']
                for level_name, level_config in EMERGENCY_RESPONSE['alert_levels'].items():
                    if draw_rate >= level_config['trigger_draw_rate']:
                        emergency_boost = level_config['epsilon_adjustment']
                        self.emergency_responses.append({
                            'episode': episode,
                            'level': level_name,
                            'draw_rate': draw_rate,
                            'epsilon_adjustment': emergency_boost
                        })
                        break

        epsilon = min(0.4, base_epsilon + emergency_boost)  # Improved max cap

        while not done:
            valid_moves = self.env.get_valid_moves()
            if not valid_moves:
                break

            current_state = state.copy()
            current_player = self.env.current_player

            if current_player == 1:
                chosen_move = self.select_action(self.model_1, current_state,
                                               valid_moves, epsilon)
            else:
                chosen_move = self.select_action(self.model_2, current_state,
                                               valid_moves, epsilon)

            next_state, reward, done, info = self.env.make_move(chosen_move)

            move_index = chosen_move[0]
            experience = (current_state, move_index, reward, next_state, done)

            if current_player == 1:
                game_states_1.append(experience)
            else:
                game_states_2.append(experience)

            state = next_state

        # Game result tracking
        result = {
            'episode': episode,
            'winner': self.env.winner,
            'turns': self.env.turn,
            'phase': current_phase,
            'epsilon_used': epsilon,
            'base_epsilon': base_epsilon,
            'emergency_boost': emergency_boost
        }

        # Final reward distribution
        final_reward_1 = (2.0 if self.env.winner == 1 else
                         (-1.0 if self.env.winner == 2 else 0.0))
        final_reward_2 = (2.0 if self.env.winner == 2 else
                         (-1.0 if self.env.winner == 1 else 0.0))

        # Add to replay buffers
        for state, action, reward, next_state, done in game_states_1:
            final_exp = (state, action, reward + final_reward_1, next_state, done)
            self.replay_buffer_1.append(final_exp)

        for state, action, reward, next_state, done in game_states_2:
            final_exp = (state, action, reward + final_reward_2, next_state, done)
            self.replay_buffer_2.append(final_exp)

        return result

    def train_model(self, model, optimizer, replay_buffer, losses_list, recent_losses):
        """Enhanced model training with loss monitoring"""
        batch_multiplier = self.phase_manager.get_phase_batch_multiplier()
        effective_batch_size = int(BATCH_SIZE * batch_multiplier)

        if len(replay_buffer) < effective_batch_size:
            return None

        batch = random.sample(replay_buffer, effective_batch_size)

        states = torch.FloatTensor([exp[0] for exp in batch]).float()
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch]).float()
        next_states = torch.FloatTensor([exp[3] for exp in batch]).float()
        dones = torch.BoolTensor([exp[4] for exp in batch])

        current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = model(next_states).max(1)[0]
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)

        loss = self.criterion(current_q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = loss.item()
        losses_list.append(loss_value)
        recent_losses.append(loss_value)

        return loss_value

    def _update_learning_rate(self):
        """Update learning rate based on current phase"""
        lr = self.phase_manager.get_phase_learning_rate()
        for param_group in self.optimizer_1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_2.param_groups:
            param_group['lr'] = lr

    def run_improved_progressive_experiment(self):
        """Main improved progressive learning experiment"""
        print(f"\nStarting Copy 6.1 Improved Progressive Learning")
        print(f"Target: {MAX_EPISODES:,} episodes")

        for episode in range(MAX_EPISODES):
            # Phase management
            self.phase_manager.advance_episode()
            self._update_learning_rate()

            # Game execution
            result = self.play_game(episode)
            self.game_results.append(result)
            self.epsilon_history.append(result['epsilon_used'])

            # Balance monitoring
            self.balance_monitor.add_game_result(result)

            # Dynamic quantum perturbation
            if len(self.game_results) >= 100:
                recent_results = self.game_results[-100:]
                draws = sum(1 for r in recent_results if r['winner'] is None)
                draw_rate = draws / len(recent_results)

                self.quantum_perturb.apply_perturbation(
                    self.model_1, self.model_2, episode, draw_rate
                )

            # Training with loss monitoring
            if episode % 3 == 0:
                loss_1 = self.train_model(self.model_1, self.optimizer_1,
                                        self.replay_buffer_1, self.losses_1, self.recent_losses_1)
                loss_2 = self.train_model(self.model_2, self.optimizer_2,
                                        self.replay_buffer_2, self.losses_2, self.recent_losses_2)

            # Progress reporting
            if (episode + 1) % config['training_schedule']['progress_reporting_frequency'] == 0:
                self._print_progress(episode)

            # Early stopping check
            if (episode >= config['training_schedule']['early_stopping']['min_episodes'] and
                (episode + 1) % 1000 == 0):
                if self._check_convergence():
                    print(f"\nEarly convergence at episode {episode+1}!")
                    break

        total_time = time.time() - self.start_time

        print(f"\nOK Copy 6.1 Improved Experiment Complete!")
        print(f"Episodes: {len(self.game_results):,}")
        print(f"Time: {total_time:.1f}s ({total_time/3600:.2f}h)")

        return len(self.game_results), total_time

    def _check_convergence(self):
        """Enhanced convergence detection"""
        if len(self.game_results) < 1000:
            return False

        recent_results = self.game_results[-1000:]
        p1_wins = sum(1 for r in recent_results if r['winner'] == 1)
        p2_wins = sum(1 for r in recent_results if r['winner'] == 2)
        draws = sum(1 for r in recent_results if r['winner'] is None)

        draw_rate = draws / len(recent_results)
        balance = (min(p1_wins, p2_wins) / max(p1_wins, p2_wins)
                  if max(p1_wins, p2_wins) > 0 else 1.0)

        return balance >= 0.85 and draw_rate < 0.65

    def _print_progress(self, episode):
        """Enhanced progress reporting with loss display"""
        window = config['training_schedule']['progress_reporting_frequency']
        recent_results = self.game_results[-window:]

        wins_1 = sum(1 for r in recent_results if r['winner'] == 1)
        wins_2 = sum(1 for r in recent_results if r['winner'] == 2)
        draws = sum(1 for r in recent_results if r['winner'] is None)

        win_rate_1 = wins_1 / window
        win_rate_2 = wins_2 / window
        draw_rate = draws / window

        balance = (min(wins_1, wins_2) / max(wins_1, wins_2)
                  if max(wins_1, wins_2) > 0 else 1.0)
        avg_turns = np.mean([r['turns'] for r in recent_results])

        current_epsilon = self.epsilon_history[-1] if self.epsilon_history else 0
        current_phase = recent_results[-1]['phase'] if recent_results else "Unknown"
        elapsed_time = time.time() - self.start_time

        # Recent interventions and perturbations
        recent_interventions = len([r for r in self.emergency_responses
                                  if r['episode'] >= episode - window])
        recent_perturbations = len([p for p in self.quantum_perturb.perturbation_history
                                  if p['episode'] >= episode - window])

        # Loss information
        avg_loss_1 = np.mean(list(self.recent_losses_1)) if self.recent_losses_1 else 0.0
        avg_loss_2 = np.mean(list(self.recent_losses_2)) if self.recent_losses_2 else 0.0

        progress = (episode + 1) / MAX_EPISODES * 100

        phase_display = (current_phase.split('_')[1] if '_' in current_phase
                        else current_phase[:4])

        print(f"Episode {episode+1:6d} ({progress:5.1f}%) | "
              f"Phase: {phase_display} | "
              f"P1={win_rate_1:.3f} P2={win_rate_2:.3f} D={draw_rate:.3f} | "
              f"Balance={balance:.3f} | "
              f"Turns={avg_turns:.1f} | "
              f"eps={current_epsilon:.4f} | "
              f"Loss={avg_loss_1:.4f}/{avg_loss_2:.4f} | "
              f"Emg={recent_interventions} Pert={recent_perturbations} | "
              f"Time={elapsed_time:.0f}s")

    def generate_final_report(self):
        """Generate improved final report"""
        total_games = len(self.game_results)
        total_p1_wins = sum(1 for r in self.game_results if r['winner'] == 1)
        total_p2_wins = sum(1 for r in self.game_results if r['winner'] == 2)
        total_draws = sum(1 for r in self.game_results if r['winner'] is None)

        # Final period analysis
        final_period = min(1500, total_games)
        final_results = self.game_results[-final_period:]
        final_p1 = sum(1 for r in final_results if r['winner'] == 1)
        final_p2 = sum(1 for r in final_results if r['winner'] == 2)
        final_draws = sum(1 for r in final_results if r['winner'] is None)

        final_balance = (min(final_p1, final_p2) / max(final_p1, final_p2)
                        if max(final_p1, final_p2) > 0 else 1.0)
        final_draw_rate = final_draws / final_period

        experiment_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("      Copy 6.1 Improved Progressive Learning - Final Report")
        print("=" * 80)

        print(f"\n[OVERVIEW] Experiment Overview")
        print(f"  Total Episodes: {total_games:,}")
        print(f"  Experiment Time: {experiment_time:.1f}s ({experiment_time/3600:.2f}h)")
        print(f"  Average Episode Time: {experiment_time/total_games:.3f}s")

        print(f"\n[RESULTS] Game Results")
        print(f"  Player 1: {total_p1_wins:,} ({total_p1_wins/total_games*100:.1f}%)")
        print(f"  Player 2: {total_p2_wins:,} ({total_p2_wins/total_games*100:.1f}%)")
        print(f"  Draws: {total_draws:,} ({total_draws/total_games*100:.1f}%)")

        print(f"\n[ANALYSIS] Final Period Analysis (Last {final_period} games)")
        print(f"  Player 1: {final_p1} ({final_p1/final_period*100:.1f}%)")
        print(f"  Player 2: {final_p2} ({final_p2/final_period*100:.1f}%)")
        print(f"  Draws: {final_draws} ({final_draw_rate*100:.1f}%)")
        print(f"  Final Balance: {final_balance:.4f}")
        print(f"  Target Achievement (>=0.85): {'YES' if final_balance >= 0.85 else 'PARTIAL' if final_balance >= 0.75 else 'NO'}")
        print(f"  Draw Control: {'EXCELLENT' if final_draw_rate < 0.55 else 'GOOD' if final_draw_rate < 0.65 else 'OK' if final_draw_rate < 0.75 else 'POOR'}")

        # System analysis
        emergency_count = len(self.emergency_responses)
        perturbation_count = len(self.quantum_perturb.perturbation_history)

        print(f"\n[EMERGENCY] Emergency Response System")
        print(f"  Total Interventions: {emergency_count}")
        print(f"  Intervention Rate: {emergency_count/total_games*100:.2f}%")

        print(f"\n[QUANTUM] Quantum Perturbation System")
        print(f"  Total Perturbations: {perturbation_count}")
        print(f"  Perturbation Rate: {perturbation_count/total_games*100:.2f}%")

        # Learning analysis
        if self.losses_1 and self.losses_2:
            avg_final_loss_1 = np.mean(self.losses_1[-100:]) if len(self.losses_1) >= 100 else 0
            avg_final_loss_2 = np.mean(self.losses_2[-100:]) if len(self.losses_2) >= 100 else 0

            print(f"\n[LEARNING] Learning Analysis")
            print(f"  Final Loss Player 1: {avg_final_loss_1:.4f}")
            print(f"  Final Loss Player 2: {avg_final_loss_2:.4f}")
            print(f"  Total Training Steps: {len(self.losses_1) + len(self.losses_2):,}")

        # Performance grading
        success_metrics = [
            final_balance >= 0.85,
            final_draw_rate < 0.65,
            emergency_count < total_games * 0.04,  # Tighter threshold
            total_games >= MAX_EPISODES * 0.8,
            experiment_time/3600 < 4
        ]
        success_rate = sum(success_metrics) / len(success_metrics)

        if success_rate >= 0.9:
            grade = "OUTSTANDING"
        elif success_rate >= 0.8:
            grade = "EXCELLENT"
        elif success_rate >= 0.7:
            grade = "OK VERY GOOD"
        elif success_rate >= 0.6:
            grade = "OK GOOD"
        else:
            grade = "WARNING NEEDS IMPROVEMENT"

        print(f"\nFinal Grade: {grade}")
        print(f"Success Rate: {success_rate*100:.0f}% ({sum(success_metrics)}/{len(success_metrics)})")

        # Comparison with Copy 6 Extended
        print(f"\nImprovements over Copy 6 Extended")
        print(f"  Training Time: 3.66h -> {experiment_time/3600:.2f}h ({'OK Improved' if experiment_time < 13200 else 'WARNING Similar'})")
        print(f"  Final Balance: 0.7898 -> {final_balance:.4f} ({'OK Improved' if final_balance > 0.79 else 'NO Declined'})")
        print(f"  Draw Control: 68.5% -> {final_draw_rate*100:.1f}% ({'OK Improved' if final_draw_rate < 0.685 else 'NO Declined'})")
        print(f"  Emergency Rate: 5.02% -> {emergency_count/total_games*100:.2f}% ({'OK Improved' if emergency_count/total_games < 0.05 else 'NO Declined'})")

        print("=" * 80)

        # Save trained models
        print(f"\nSaving trained models...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save Player 1 model
            p1_model_path = f"copy61_p1_model_{timestamp}.pth"
            torch.save(self.agent_1.state_dict(), p1_model_path)
            print(f"Player 1 model saved: {p1_model_path}")

            # Save Player 2 model
            p2_model_path = f"copy61_p2_model_{timestamp}.pth"
            torch.save(self.agent_2.state_dict(), p2_model_path)
            print(f"Player 2 model saved: {p2_model_path}")

            # Save experiment results
            results_path = f"copy61_results_{timestamp}.json"
            results_data = {
                "metadata": {
                    "experiment_name": "Copy 6.1 Improved Progressive Learning",
                    "total_episodes": total_games,
                    "experiment_time_hours": experiment_time / 3600,
                    "timestamp": timestamp
                },
                "final_metrics": {
                    "total_p1_wins": total_p1_wins,
                    "total_p2_wins": total_p2_wins,
                    "total_draws": total_draws,
                    "final_balance": final_balance,
                    "final_draw_rate": final_draw_rate,
                    "success_rate": success_rate
                },
                "training_history": {
                    "emergency_responses": len(self.emergency_responses),
                    "quantum_perturbations": len(self.quantum_perturb.perturbation_history),
                    "final_losses": {
                        "player_1": avg_final_loss_1 if self.losses_1 else 0,
                        "player_2": avg_final_loss_2 if self.losses_2 else 0
                    }
                }
            }

            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Experiment results saved: {results_path}")

            print(f"\nCopy 6.1 models ready for battle!")
            print(f"Use {p1_model_path} and {p2_model_path} for battle viewing")

        except Exception as e:
            print(f"Error saving models: {e}")


def main():
    """Main execution function"""
    try:
        # Initialize and run experiment
        experiment = Copy61ImprovedProgressiveExperiment()
        total_episodes, total_time = experiment.run_improved_progressive_experiment()

        # Generate final report
        experiment.generate_final_report()

        print(f"\nCopy 6.1 Improved Complete!")
        print(f"Successfully trained for {total_episodes:,} episodes in {total_time/3600:.2f} hours")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nExperiment failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()