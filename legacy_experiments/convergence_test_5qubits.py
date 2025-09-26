#!/usr/bin/env python3
"""
5量子ビット収束テスト
4量子ビットとの比較用
"""

import sys
import os
import json
import time
import random
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    import pennylane as qml
except ImportError:
    print("Error: PennyLane required - pip install pennylane")
    sys.exit(1)

class QuantumAI5Qubits(nn.Module):
    """5量子ビット量子AIモデル"""

    def __init__(self, config):
        super().__init__()

        # 設定から量子パラメータを取得
        quantum_config = config['module_config']['quantum']
        qmap_config = config['module_config']['qmap']

        self.n_qubits = quantum_config['n_qubits']
        self.n_layers = quantum_config['n_layers']
        self.action_dim = qmap_config['action_dim']
        self.state_dim = qmap_config['state_dim']

        print(f"Building 5-Qubit QuantumAI...")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  State dim: {self.state_dim}")

        # 量子デバイス
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # 状態エンコーダー（5量子ビット用に調整）
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_qubits),
            nn.Tanh()
        )

        # 量子パラメータ（5量子ビット * 2回転 * 1レイヤー）
        param_count = self.n_layers * self.n_qubits * 2  # RY, RZ per qubit per layer
        self.quantum_params = nn.Parameter(torch.randn(param_count) * 0.1)

        # デコーダー（5量子ビット出力を5次元行動に変換）
        self.decoder = nn.Sequential(
            nn.Linear(self.n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_dim)
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"5-Qubit Model built - Parameters: {total_params}")

    def quantum_circuit(self, inputs, params):
        """5量子ビット量子回路"""
        param_idx = 0

        # データエンコーディング
        for i in range(self.n_qubits):
            qml.RY(inputs[i] * np.pi, wires=i)

        # パラメータ化された層
        for layer in range(self.n_layers):
            # 回転ゲート
            for i in range(self.n_qubits):
                qml.RY(params[param_idx], wires=i)
                param_idx += 1
                qml.RZ(params[param_idx], wires=i)
                param_idx += 1

            # 線形エンタングルメント（5量子ビット）
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # 全量子ビットの期待値を測定
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, state):
        """フォワードパス"""
        batch_size = state.size(0)

        # エンコード
        encoded = self.encoder(state)

        # 量子処理
        quantum_outputs = []
        for i in range(batch_size):
            @qml.qnode(self.dev, interface='torch')
            def circuit(inputs, params):
                return self.quantum_circuit(inputs, params)

            q_out = circuit(encoded[i], self.quantum_params)
            quantum_outputs.append(torch.stack(q_out))

        quantum_tensor = torch.stack(quantum_outputs).float()

        # デコード
        q_values = self.decoder(quantum_tensor)

        return q_values

class ConvergenceDetector:
    """収束検出クラス"""

    def __init__(self, patience=80, min_delta=0.01, warmup_episodes=150):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_episodes = warmup_episodes
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []

    def check_convergence(self, current_loss, episode):
        """収束判定"""
        if episode < self.warmup_episodes:
            return False

        self.loss_history.append(current_loss)

        # 最近15回の平均で安定性をチェック
        if len(self.loss_history) >= 15:
            recent_avg = np.mean(self.loss_history[-15:])
            if recent_avg < self.best_loss - self.min_delta:
                self.best_loss = recent_avg
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        return self.patience_counter >= self.patience

def calculate_reward(player, game_over, winner, reward_strategy):
    """報酬計算"""
    if game_over:
        if winner == player:
            return 100 if reward_strategy == 'balanced' else 120
        elif winner and winner != player:
            return -100
        else:
            return 0
    else:
        return 2 if reward_strategy == 'balanced' else 1

def run_convergence_test_5qubits(model, config, max_episodes=3000):
    """5量子ビット収束テスト実行"""
    print("=" * 60)
    print("5-QUBIT QUANTUM AI CONVERGENCE TEST")
    print("=" * 60)
    print(f"Max episodes: {max_episodes:,}")
    print(f"Qubits: {config['module_config']['quantum']['n_qubits']}")
    print(f"Convergence detection: Enabled")
    print("=" * 60)

    # ハイパーパラメータ
    hyperparams = config['hyperparameters']
    learning_rate = hyperparams['learningRate']
    optimizer_type = hyperparams['optimizer']
    batch_size = hyperparams['batchSize']

    # オプティマイザ設定
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    loss_fn = nn.MSELoss()

    # 経験リプレイバッファー
    replay_buffer = deque(maxlen=hyperparams['replayBufferSize'])

    # 収束検出器
    convergence_detector = ConvergenceDetector()

    # 学習統計
    episode_rewards = []
    all_losses = []
    convergence_data = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'epsilon': [],
        'learning_speed': []
    }

    start_time = time.time()
    episode = 0
    converged = False

    # epsilon-greedy設定
    epsilon_start = 0.9
    epsilon_end = 0.01
    epsilon_decay = 0.998

    reward_strategy = config['module_config']['reward']['strategy']
    action_dim = config['module_config']['qmap']['action_dim']
    state_dim = config['module_config']['qmap']['state_dim']

    print(f"5-Qubit training started...")

    while episode < max_episodes and not converged:
        episode_start = time.time()

        # 初期状態（ランダム）
        state = torch.randn(1, state_dim)
        episode_reward = 0
        step_count = 0
        max_steps = 40
        losses = []

        # epsilon値計算
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        while step_count < max_steps:
            # 行動選択
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax().item()

            # 環境ステップ（シミュレーション）
            next_state = torch.randn(1, state_dim)

            # 報酬計算（シミュレーション）
            game_over = random.random() < 0.1  # 10%の確率でゲーム終了
            if game_over:
                winner = 'A' if random.random() < 0.55 else 'B'  # 55%の勝率
                reward = calculate_reward('A', game_over, winner, reward_strategy)
            else:
                reward = calculate_reward('A', game_over, None, reward_strategy)

            episode_reward += reward

            # 経験を保存
            replay_buffer.append((state.clone(), action, reward, next_state.clone(), game_over))

            # 学習
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)

                states = torch.cat([item[0] for item in batch])
                actions = torch.LongTensor([item[1] for item in batch])
                rewards = torch.FloatTensor([item[2] for item in batch])

                optimizer.zero_grad()
                q_values = model(states)
                q_values_action = q_values.gather(1, actions.unsqueeze(1)).squeeze()

                loss = loss_fn(q_values_action, rewards)
                loss.backward()

                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(loss.item())

            state = next_state
            step_count += 1

            if game_over:
                break

        episode_rewards.append(episode_reward)
        all_losses.extend(losses)

        # エピソード時間を記録
        episode_time = time.time() - episode_start

        # 収束チェック
        if losses and episode >= 30:
            current_loss = np.mean(losses)
            converged = convergence_detector.check_convergence(current_loss, episode)

        # 統計収集とプログレス表示
        if episode % 50 == 0:
            recent_rewards = episode_rewards[-50:] if episode_rewards else [0]
            recent_losses = all_losses[-100:] if all_losses else [0]

            convergence_data['episodes'].append(episode)
            convergence_data['rewards'].append(np.mean(recent_rewards))
            convergence_data['losses'].append(np.mean(recent_losses) if recent_losses else 0)
            convergence_data['epsilon'].append(epsilon)
            convergence_data['learning_speed'].append(1.0 / episode_time if episode_time > 0 else 0)

            # 進捗表示
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            elapsed = time.time() - start_time

            print(f"Episode {episode:4d}: "
                  f"Reward={avg_reward:6.2f}, "
                  f"Loss={avg_loss:7.4f}, "
                  f"eps={epsilon:.3f}, "
                  f"Time={elapsed/60:.1f}m, "
                  f"Speed={1.0/episode_time:.1f}eps/s")

            # 収束進捗
            if episode >= convergence_detector.warmup_episodes:
                progress = (convergence_detector.patience - convergence_detector.patience_counter) / convergence_detector.patience * 100
                print(f"         5-Qubit Convergence progress: {progress:.1f}% (waiting: {convergence_detector.patience_counter}/{convergence_detector.patience})")

        episode += 1

    # 結果
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    if converged:
        print(f"5-QUBIT CONVERGED: Episode {episode}")
    else:
        print(f"5-QUBIT MAX EPISODES REACHED: Episode {episode}")

    print(f"5-Qubit Final Statistics:")
    print(f"   Total episodes: {episode}")
    print(f"   Training time: {total_time/60:.1f}min")
    print(f"   Training speed: {episode/total_time:.1f} eps/s")
    print(f"   Final avg reward: {np.mean(episode_rewards[-50:]):.2f}")
    if all_losses:
        print(f"   Final loss: {np.mean(all_losses[-50:]):.4f}")
    print(f"   Converged: {'YES' if converged else 'NO'}")

    return {
        'converged': converged,
        'total_episodes': episode,
        'total_time': total_time,
        'rewards': episode_rewards,
        'losses': all_losses,
        'convergence_data': convergence_data,
        'qubits': 5
    }

def main():
    # 5量子ビット設定ファイルを読み込み
    config_file = project_root / "quantum_geister_config_5qubits.json"

    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        return

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"5-Qubit WebUI config loaded successfully")
        print(f"Timestamp: {config['learning_config']['timestamp']}")
        print(f"Method: {config['learning_config']['method']}")
        print(f"Qubits: {config['module_config']['quantum']['n_qubits']}")
        print(f"Layers: {config['module_config']['quantum']['n_layers']}")
        print(f"Action dim: {config['module_config']['qmap']['action_dim']}")

    except Exception as e:
        print(f"Config loading error: {e}")
        return

    # 5量子ビットモデルを作成
    model = QuantumAI5Qubits(config)

    # テスト
    print(f"\nRunning 5-qubit test...")
    test_state = torch.randn(1, config['module_config']['qmap']['state_dim'])
    test_output = model(test_state)
    print(f"5-Qubit test successful - output shape: {test_output.shape}")

    # 収束テスト実行
    print("Starting 5-qubit convergence test...")
    results = run_convergence_test_5qubits(model, config, max_episodes=3000)

    # 最終レポート
    print(f"\n{'='*60}")
    print(f"5-QUBIT QUANTUM AI CONVERGENCE TEST REPORT")
    print(f"{'='*60}")

    print(f"\n5-Qubit AI Configuration:")
    print(f"   Qubits: {config['module_config']['quantum']['n_qubits']}")
    print(f"   Layers: {config['module_config']['quantum']['n_layers']}")
    print(f"   Embedding: {config['module_config']['quantum']['embedding_type']}")
    print(f"   Entanglement: {config['module_config']['quantum']['entanglement']}")
    print(f"   Reward strategy: {config['module_config']['reward']['strategy']}")
    print(f"   Learning rate: {config['hyperparameters']['learningRate']}")
    print(f"   Batch size: {config['hyperparameters']['batchSize']}")
    print(f"   Optimizer: {config['hyperparameters']['optimizer']}")

    print(f"\n5-Qubit Training Results:")
    print(f"   Converged: {'YES' if results['converged'] else 'NO'}")
    print(f"   Episodes required: {results['total_episodes']:,}")
    print(f"   Training time: {results['total_time']/60:.1f}min")
    print(f"   Training speed: {results['total_episodes']/results['total_time']:.1f} eps/s")

    print(f"\n5-Qubit Convergence Analysis:")
    if results['converged']:
        print(f"   SUCCESS: 5-Qubit AI converged in {results['total_episodes']:,} episodes!")
        print(f"   Training time needed: ~{results['total_time']/60:.1f}min")

        # 収束速度の評価
        if results['total_episodes'] < 500:
            print(f"   VERY FAST convergence (<500 episodes)")
        elif results['total_episodes'] < 1000:
            print(f"   FAST convergence (<1000 episodes)")
        elif results['total_episodes'] < 1500:
            print(f"   STANDARD convergence speed")
        else:
            print(f"   SLOW convergence (room for improvement)")

    else:
        print(f"   WARNING: Did not converge in {results['total_episodes']:,} episodes")
        print(f"   Might need longer training or hyperparameter tuning")

    # 4量子ビットとの比較予告
    print(f"\nComparison with 4-Qubit:")
    print(f"   4-Qubit converged in: 266 episodes")
    print(f"   5-Qubit converged in: {results['total_episodes'] if results['converged'] else 'N/A'} episodes")
    if results['converged']:
        ratio = results['total_episodes'] / 266
        print(f"   Convergence ratio (5Q/4Q): {ratio:.2f}x")
        if ratio > 1.5:
            print(f"   5-Qubit is SIGNIFICANTLY SLOWER than 4-Qubit")
        elif ratio > 1.1:
            print(f"   5-Qubit is SLOWER than 4-Qubit")
        elif ratio < 0.9:
            print(f"   5-Qubit is FASTER than 4-Qubit")
        else:
            print(f"   5-Qubit and 4-Qubit have SIMILAR convergence speed")

    print(f"\n5-Qubit Experiment Data:")
    print(f"   Reward data: {len(results['rewards'])} episodes")
    print(f"   Loss data: {len(results['losses'])} steps")
    print(f"   Convergence data points: {len(results['convergence_data']['episodes'])}")

    print(f"\n{'='*60}")
    print(f"5-QUBIT QUANTUM AI CONVERGENCE TEST COMPLETE")
    print(f"{'='*60}")

    # 結果を保存
    result_file = project_root / "convergence_results_5qubits.json"
    with open(result_file, 'w') as f:
        # NumPy arrays をリストに変換
        save_results = {
            'converged': results['converged'],
            'total_episodes': results['total_episodes'],
            'total_time': results['total_time'],
            'qubits': results['qubits'],
            'final_reward': float(np.mean(results['rewards'][-50:])) if results['rewards'] else 0.0,
            'final_loss': float(np.mean(results['losses'][-50:])) if results['losses'] else 0.0
        }
        json.dump(save_results, f, indent=2)

    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()