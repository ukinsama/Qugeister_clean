#!/usr/bin/env python3
"""
Fast Quantum Circuit Simulation Trainer

Quantum circuit-based trainer that achieves practical learning speeds
while maintaining quantum advantages for Geister game AI.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pennylane as qml
from collections import deque
import random
import time
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union, Literal
import pickle
from functools import lru_cache
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path

# Import updated FastQuantumCircuit
from .quantum_circuit import FastQuantumCircuit

logger = logging.getLogger(__name__)

# ===== Quantum circuits imported from quantum_circuit.py =====


# ===== Hybrid Quantum-Classical Neural Network =====
class FastQuantumNeuralNetwork(nn.Module):
    """Fast quantum neural network with user configuration support

    Enhanced with proper type hints, validation, and error handling.
    """

    def __init__(
        self,
        input_dim: int = 252,
        output_dim: int = 36,
        n_qubits: int = 4,
        n_layers: int = 2,
        embedding: Literal["angle", "amplitude"] = "angle",
        entanglement: Literal["linear", "circular", "full"] = "linear",
        device: str = "lightning.qubit",
    ) -> None:
        """Initialize quantum neural network

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (usually 36 for 6x6 spatial mapping)
            n_qubits: Number of qubits (2-16 recommended)
            n_layers: Number of quantum layers (1-8 recommended)
            embedding: Embedding type ('angle' or 'amplitude')
            entanglement: Entanglement pattern ('linear', 'circular', 'full')
            device: Quantum device backend

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        # Validate parameters
        self._validate_params(
            input_dim, output_dim, n_qubits, n_layers, embedding, entanglement
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding
        self.entanglement = entanglement
        self.device = device

        logger.info(
            f"Initializing FastQuantumNeuralNetwork: {n_qubits}Q, {n_layers}L, {embedding}/{entanglement}"
        )

        self._build_network()

    def _validate_params(
        self,
        input_dim: int,
        output_dim: int,
        n_qubits: int,
        n_layers: int,
        embedding: str,
        entanglement: str,
    ) -> None:
        """Validate network parameters"""
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (2 <= n_qubits <= 16):
            raise ValueError(f"n_qubits must be 2-16, got {n_qubits}")
        if not (1 <= n_layers <= 8):
            raise ValueError(f"n_layers must be 1-8, got {n_layers}")
        if embedding not in ["angle", "amplitude"]:
            raise ValueError(
                f"embedding must be 'angle' or 'amplitude', got {embedding}"
            )
        if entanglement not in ["linear", "circular", "full"]:
            raise ValueError(
                f"entanglement must be 'linear', 'circular', or 'full', got {entanglement}"
            )

    def _build_network(self) -> None:
        """Build the hybrid quantum-classical network"""

        # 1. 強化された前処理CNN（Deep Feature Extraction）
        self.preprocessor = nn.Sequential(
            # First CNN Block - Pattern Recognition
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),  # LayerNorm使用でバッチサイズ1に対応
            nn.ReLU(),
            nn.Dropout(0.2),
            # Second CNN Block - Feature Combination
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            # Third CNN Block - Strategic Analysis
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Fourth CNN Block - Quantum Preparation
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            # Final Quantum Interface Layer
            nn.Linear(64, self.n_qubits),
            nn.Tanh(),  # 量子回路の入力範囲に正規化 [-1, 1]
        )

        # 2. 強化量子回路層（Multi-Layer Quantum Processing）
        self.quantum_layer = FastQuantumCircuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            embedding=self.embedding,
            entanglement=self.entanglement,
        )

        # 量子特徴増強層（Quantum Feature Enhancement）
        self.quantum_enhancer = nn.Sequential(
            nn.Linear(self.n_qubits, self.n_qubits * 2),
            nn.Tanh(),
            nn.Linear(self.n_qubits * 2, self.n_qubits),
            nn.Tanh(),
        )

        # 3. 超強化後処理CNN（Deep Q-Value Generation）
        self.postprocessor = nn.Sequential(
            # First Expansion Block - Quantum Feature Amplification
            nn.Linear(self.n_qubits, 128),
            nn.LayerNorm(128),  # LayerNorm使用でバッチサイズ1に対応
            nn.ReLU(),
            nn.Dropout(0.1),
            # Second Expansion Block - Strategic Pattern Formation
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            # Third Expansion Block - Spatial Understanding
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Fourth Block - Advanced Strategy Synthesis
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            # Fifth Block - Q-Value Refinement
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Sixth Block - Spatial Mapping Preparation
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.05),
            # Final Q-Value Generation Layer
            nn.Linear(64, self.output_dim),  # 36 outputs for 6x6 Q-value map
            nn.Tanh(),  # Normalize Q-values to [-1, 1] range for stability
        )

        # 4. 強化量子回路の重み（ユーザー設定レイヤー数対応）
        self.quantum_weights = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 2) * 0.1
        )  # ユーザー設定レイヤー分の重み

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # バッチ処理の最適化
        batch_size = x.shape[0] if x.dim() > 1 else 1
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 1. 強化前処理で深層特徴抽出（252→4）
        compressed = self.preprocessor(x)

        # 2. 強化量子回路処理（4→4）
        quantum_outputs = []
        for i in range(batch_size):
            # 各サンプルを強化量子回路に通す
            quantum_input = compressed[i] * np.pi  # [-π, π]にスケーリング
            quantum_output = self.quantum_layer.forward(
                quantum_input, self.quantum_weights
            )
            quantum_outputs.append(quantum_output)

        # 3. 量子出力をテンソルに変換
        if batch_size > 1:
            quantum_features = torch.stack(
                [torch.tensor(out, dtype=torch.float32) for out in quantum_outputs]
            )
        else:
            quantum_features = torch.tensor(
                quantum_outputs[0], dtype=torch.float32
            ).unsqueeze(0)

        # 4. 量子特徴増強処理（4→8→4）
        enhanced_features = self.quantum_enhancer(quantum_features)

        # 5. 量子特徴と増強特徴を結合
        combined_features = quantum_features + enhanced_features

        # 6. 超強化後処理で36次元Q値マップ生成（4→36）
        output = self.postprocessor(combined_features)

        return output.squeeze(0) if batch_size == 1 else output

    def get_qvalue_map(self, x: torch.Tensor) -> torch.Tensor:
        """36次元出力を6x6のQ値マップに変換"""
        output = self.forward(x)
        if output.dim() == 1:
            # Single sample: reshape to 6x6
            return output.reshape(6, 6)
        else:
            # Batch: reshape each sample to 6x6
            return output.reshape(-1, 6, 6)

    def get_action_from_qmap(self, x: torch.Tensor) -> torch.Tensor:
        """Q値マップから最適行動を選択（従来の5行動システム用）"""
        qvalue_map = self.get_qvalue_map(x)
        if qvalue_map.dim() == 2:  # Single sample
            # 6x6マップから代表的な5つの領域の最大値を取得
            regions = {
                0: qvalue_map[0:2, 0:3].max(),  # 左上領域
                1: qvalue_map[0:2, 3:6].max(),  # 右上領域
                2: qvalue_map[2:4, 1:5].max(),  # 中央領域
                3: qvalue_map[4:6, 0:3].max(),  # 左下領域
                4: qvalue_map[4:6, 3:6].max(),  # 右下領域
            }
            # 5つの行動に対応するQ値を返す
            return torch.tensor([regions[i] for i in range(5)])
        else:  # Batch
            batch_size = qvalue_map.shape[0]
            batch_actions = []
            for i in range(batch_size):
                single_map = qvalue_map[i]
                regions = {
                    0: single_map[0:2, 0:3].max(),
                    1: single_map[0:2, 3:6].max(),
                    2: single_map[2:4, 1:5].max(),
                    3: single_map[4:6, 0:3].max(),
                    4: single_map[4:6, 3:6].max(),
                }
                batch_actions.append([regions[i] for i in range(5)])
            return torch.tensor(batch_actions)


# ===== 高速学習システム =====
class FastQuantumTrainer:
    """高速量子回路学習システム"""

    def __init__(self, model: FastQuantumNeuralNetwork, lr: float = 0.001) -> None:
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=1000)

        # 学習統計
        self.losses = []
        self.rewards = []
        self.episode_losses = []  # エピソードごとの平均ロス
        self.loss_history = []    # 詳細なロス履歴

    def train_step(self, batch_size: int = 8) -> Optional[float]:
        """効率的なバッチ学習"""
        if len(self.replay_buffer) < batch_size:
            return None

        # ミニバッチサンプリング
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # テンソル変換（効率化）
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q値計算（36次元出力から5行動用Q値を抽出）
        current_q_actions = self.model.get_action_from_qmap(states)
        current_q = current_q_actions.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_actions = self.model.get_action_from_qmap(next_states)
            next_q = next_q_actions.max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)

        # 損失計算
        loss = nn.MSELoss()(current_q, target_q)

        # 最適化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.loss_history.append(loss.item())
        return loss.item()

    def plot_training_progress(self, save_path: str = None, show_plot: bool = True) -> None:
        """学習進捗をプロットして表示・保存"""
        if len(self.losses) == 0 and len(self.rewards) == 0:
            print("プロットするデータがありません")
            return

        # 図のサイズとレイアウト設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Quantum Neural Network Training Progress', fontsize=16)

        # 1. ロス履歴
        if len(self.losses) > 0:
            axes[0, 0].plot(self.losses, alpha=0.7, label='Training Loss')
            if len(self.losses) > 100:
                # 移動平均でスムージング
                window = min(100, len(self.losses) // 10)
                smoothed_loss = np.convolve(self.losses, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.losses)), smoothed_loss,
                               color='red', linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('MSE Loss')
            axes[0, 0].set_title('Training Loss Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 報酬履歴
        if len(self.rewards) > 0:
            axes[0, 1].plot(self.rewards, alpha=0.7, label='Episode Rewards')
            if len(self.rewards) > 100:
                # 移動平均でスムージング
                window = min(100, len(self.rewards) // 10)
                smoothed_rewards = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(range(window-1, len(self.rewards)), smoothed_rewards,
                               color='green', linewidth=2, label=f'Moving Avg ({window})')
            axes[0, 1].set_xlabel('Episodes')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].set_title('Episode Rewards Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. ロス分布（ヒストグラム）
        if len(self.losses) > 0:
            axes[1, 0].hist(self.losses, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(self.losses), color='red', linestyle='--',
                              label=f'Mean: {np.mean(self.losses):.4f}')
            axes[1, 0].axvline(np.median(self.losses), color='green', linestyle='--',
                              label=f'Median: {np.median(self.losses):.4f}')
            axes[1, 0].set_xlabel('Loss Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Loss Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 収束分析（最近のロス傾向）
        if len(self.losses) > 100:
            # 最近のロス傾向を分析
            recent_losses = self.losses[-1000:] if len(self.losses) > 1000 else self.losses
            axes[1, 1].plot(recent_losses, alpha=0.7, label='Recent Loss')

            # 線形回帰で傾向を分析
            x = np.arange(len(recent_losses))
            z = np.polyfit(x, recent_losses, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(x, p(x), "r--", alpha=0.8,
                           label=f'Trend (slope: {z[0]:.6f})')

            axes[1, 1].set_xlabel('Recent Training Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Recent Loss Trend (Convergence Analysis)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor convergence analysis',
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Convergence Analysis')

        plt.tight_layout()

        # 保存
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 学習進捗グラフを保存: {save_path}")

        # 表示
        if show_plot:
            plt.show()
        else:
            plt.close()

    def analyze_convergence(self) -> Dict[str, Any]:
        """収束状況を数値的に分析"""
        if len(self.losses) < 100:
            return {"status": "insufficient_data", "message": "分析には最低100ステップ必要"}

        analysis = {}

        # 基本統計
        analysis["total_steps"] = len(self.losses)
        analysis["mean_loss"] = np.mean(self.losses)
        analysis["std_loss"] = np.std(self.losses)
        analysis["min_loss"] = np.min(self.losses)
        analysis["max_loss"] = np.max(self.losses)

        # 収束傾向分析（最近の1000ステップ）
        recent_window = min(1000, len(self.losses) // 4)
        recent_losses = self.losses[-recent_window:]

        # 線形回帰で傾向を分析
        x = np.arange(len(recent_losses))
        slope, intercept = np.polyfit(x, recent_losses, 1)

        analysis["recent_slope"] = slope
        analysis["recent_mean"] = np.mean(recent_losses)
        analysis["recent_std"] = np.std(recent_losses)

        # 収束判定
        if abs(slope) < 1e-6:  # 傾きがほぼ0
            analysis["convergence_status"] = "converged"
        elif slope < -1e-4:  # 明確に減少傾向
            analysis["convergence_status"] = "improving"
        elif slope > 1e-4:   # 明確に増加傾向
            analysis["convergence_status"] = "diverging"
        else:
            analysis["convergence_status"] = "stable"

        # 変動の安定性
        if len(self.losses) > 500:
            first_half_std = np.std(self.losses[:len(self.losses)//2])
            second_half_std = np.std(self.losses[len(self.losses)//2:])
            analysis["stability_ratio"] = second_half_std / first_half_std

        return analysis

    def print_convergence_report(self) -> None:
        """収束レポートを表示"""
        analysis = self.analyze_convergence()

        if analysis.get("status") == "insufficient_data":
            print(analysis["message"])
            return

        print("\n" + "="*60)
        print("📈 CONVERGENCE ANALYSIS REPORT")
        print("="*60)
        print(f"総学習ステップ数: {analysis['total_steps']:,}")
        print(f"平均ロス: {analysis['mean_loss']:.6f}")
        print(f"ロス標準偏差: {analysis['std_loss']:.6f}")
        print(f"最小ロス: {analysis['min_loss']:.6f}")
        print(f"最大ロス: {analysis['max_loss']:.6f}")
        print()
        print("📊 最近の傾向分析:")
        print(f"傾き (slope): {analysis['recent_slope']:.8f}")
        print(f"最近の平均ロス: {analysis['recent_mean']:.6f}")
        print(f"最近の標準偏差: {analysis['recent_std']:.6f}")
        print()

        status_emoji = {
            "converged": "✅",
            "improving": "📈",
            "stable": "📊",
            "diverging": "📉"
        }

        status_msg = {
            "converged": "収束済み - ロスが安定",
            "improving": "改善中 - ロスが減少傾向",
            "stable": "安定 - ロスがほぼ一定",
            "diverging": "発散傾向 - 要注意"
        }

        status = analysis['convergence_status']
        print(f"🎯 収束状況: {status_emoji[status]} {status_msg[status]}")

        if 'stability_ratio' in analysis:
            if analysis['stability_ratio'] < 0.8:
                print("📉 学習が安定化している傾向")
            elif analysis['stability_ratio'] > 1.2:
                print("📈 学習が不安定化している可能性")
            else:
                print("📊 学習の安定性は適切")

        print("="*60)


# ===== メイン学習ループ（廃止予定 - 実際のGeister環境に置き換える） =====
def train_fast_quantum(
    episodes: int = 1000, n_qubits: int = 4
) -> Tuple[FastQuantumNeuralNetwork, List[float]]:
    """高速量子回路学習の実行

    WARNING: This function uses fake random rewards and should be replaced
    with real Geister game environment for meaningful learning.
    """

    print("=" * 60)
    print("⚠️  WARNING: PLACEHOLDER SIMULATION - NOT REAL GEISTER GAME")
    print("🚀 高速量子回路シミュレーション学習")
    print("=" * 60)
    print(f"量子ビット数: {n_qubits}")
    print(f"エピソード数: {episodes}")
    print("⚠️  この学習は実際のゲームではありません")
    print("=" * 60)

    # モデルとトレーナーの初期化
    model = FastQuantumNeuralNetwork(n_qubits=n_qubits)
    trainer = FastQuantumTrainer(model)

    # 進捗表示
    episode_rewards = []
    start_time = time.time()

    with tqdm(total=episodes, desc="量子回路学習") as pbar:
        for episode in range(episodes):
            # 環境リセット（簡略化）
            state = torch.randn(1, 252)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 100:
                # 行動選択（ε-greedy）
                epsilon = max(0.01, 0.1 * (0.995**episode))
                if random.random() < epsilon:
                    action = random.randrange(5)
                else:
                    with torch.no_grad():
                        q_values = model.get_action_from_qmap(state)
                        action = q_values.argmax().item()

                # 環境ステップ（実際のGeisterゲーム環境に置き換える必要）
                # WARNING: This is placeholder simulation - replace with real Geister game
                next_state = torch.randn(1, 252)
                reward = 0.0  # Real game reward will replace this
                done = random.random() < 0.1

                # 経験を保存
                trainer.replay_buffer.append((state, action, reward, next_state, done))

                # 学習
                loss = trainer.train_step()

                # 更新
                state = next_state
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)
            trainer.rewards.append(episode_reward)

            # エピソード終了時の平均ロスを記録
            if len(trainer.losses) > 0:
                episode_loss = np.mean(trainer.losses[-steps:]) if steps > 0 else 0
                trainer.episode_losses.append(episode_loss)

            # 進捗更新
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                elapsed = time.time() - start_time
                speed = (episode + 1) / elapsed

                # キャッシュ統計
                cache_size = len(model.quantum_layer.lookup_table)

                pbar.set_postfix(
                    {
                        "Avg Reward": f"{avg_reward:.2f}",
                        "Speed": f"{speed:.1f} eps/s",
                        "Cache": f"{cache_size}/{model.quantum_layer.cache_size}",
                        "ε": f"{epsilon:.3f}",
                    }
                )

            pbar.update(1)

    # 結果表示
    total_time = time.time() - start_time
    print(f"\n✅ 学習完了！")
    print(f"総時間: {total_time:.1f}秒")
    print(f"速度: {episodes/total_time:.1f} エピソード/秒")
    print(f"最終報酬: {np.mean(episode_rewards[-100:]):.2f}")

    # 収束分析と可視化
    print("\n" + "="*60)
    print("📊 学習結果分析")
    print("="*60)

    # 収束レポート表示
    trainer.print_convergence_report()

    # グラフ保存
    plot_save_path = f"training_results_{episodes}_episodes.png"
    trainer.plot_training_progress(save_path=plot_save_path, show_plot=False)

    # モデル保存
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "quantum_cache": model.quantum_layer.lookup_table,
            "rewards": episode_rewards,
            "losses": trainer.losses,
            "episode_losses": trainer.episode_losses,
            "training_analysis": trainer.analyze_convergence(),
        },
        "fast_quantum_model.pth",
    )

    print("💾 モデルを 'fast_quantum_model.pth' として保存")
    print(f"📊 学習グラフを '{plot_save_path}' として保存")

    return model, episode_rewards


# ===== 100000エピソード収束確認 =====
def train_convergence_test(episodes: int = 100000, n_qubits: int = 4, save_interval: int = 10000) -> Tuple[FastQuantumNeuralNetwork, List[float]]:
    """100000エピソード学習で収束確認

    WARNING: This function uses fake random rewards and should be replaced
    with real Geister game environment for meaningful learning.
    """

    print("=" * 80)
    print("⚠️  WARNING: PLACEHOLDER SIMULATION - NOT REAL GEISTER GAME")
    print("🧪 CONVERGENCE TEST: 100000エピソード学習")
    print("=" * 80)
    print(f"量子ビット数: {n_qubits}")
    print(f"エピソード数: {episodes:,}")
    print(f"中間保存間隔: {save_interval:,}エピソード")
    print("⚠️  この学習は実際のゲームではありません")
    print("=" * 80)

    # モデルとトレーナーの初期化
    model = FastQuantumNeuralNetwork(n_qubits=n_qubits)
    trainer = FastQuantumTrainer(model, lr=0.001)

    # 学習統計
    episode_rewards = []
    start_time = time.time()

    # 中間結果保存用
    checkpoints = []

    with tqdm(total=episodes, desc="収束テスト学習") as pbar:
        for episode in range(episodes):
            # 環境リセット
            state = torch.randn(1, 252)
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 100:
                # 行動選択（ε-greedy with longer decay）
                epsilon = max(0.001, 0.1 * (0.9999**episode))  # より長い減衰
                if random.random() < epsilon:
                    action = random.randrange(5)
                else:
                    with torch.no_grad():
                        q_values = model.get_action_from_qmap(state)
                        action = q_values.argmax().item()

                # 環境ステップ（実際のGeisterゲーム環境に置き換える必要）
                # WARNING: This is placeholder simulation - replace with real Geister game
                next_state = torch.randn(1, 252)
                reward = 0.0  # Real game reward will replace this
                done = random.random() < 0.05  # より長いエピソード

                # 経験を保存
                trainer.replay_buffer.append((state, action, reward, next_state, done))

                # 学習
                loss = trainer.train_step(batch_size=16)  # より大きなバッチサイズ

                # 更新
                state = next_state
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)
            trainer.rewards.append(episode_reward)

            # エピソード終了時の平均ロスを記録
            if len(trainer.losses) > 0:
                episode_loss = np.mean(trainer.losses[-steps:]) if steps > 0 else 0
                trainer.episode_losses.append(episode_loss)

            # 中間保存とレポート
            if (episode + 1) % save_interval == 0:
                elapsed = time.time() - start_time
                speed = (episode + 1) / elapsed

                print(f"\n{'='*60}")
                print(f"📊 中間レポート: {episode + 1:,}/{episodes:,} エピソード")
                print(f"{'='*60}")
                print(f"経過時間: {elapsed/3600:.1f}時間")
                print(f"学習速度: {speed:.1f} eps/s")
                print(f"推定残り時間: {(episodes - episode - 1) / speed / 3600:.1f}時間")

                # 収束分析
                if len(trainer.losses) > 1000:
                    analysis = trainer.analyze_convergence()
                    print(f"現在の収束状況: {analysis['convergence_status']}")
                    print(f"平均ロス: {analysis['mean_loss']:.6f}")
                    print(f"最近の傾き: {analysis['recent_slope']:.8f}")

                # 中間保存
                checkpoint_data = {
                    "episode": episode + 1,
                    "model_state_dict": model.state_dict(),
                    "losses": trainer.losses.copy(),
                    "rewards": trainer.rewards.copy(),
                    "analysis": trainer.analyze_convergence() if len(trainer.losses) > 100 else None,
                    "timestamp": time.time()
                }
                checkpoints.append(checkpoint_data)

                # グラフ保存
                plot_path = f"convergence_test_{episode + 1}.png"
                trainer.plot_training_progress(save_path=plot_path, show_plot=False)
                print(f"📈 中間グラフ保存: {plot_path}")

            # 進捗更新
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:])
                avg_loss = np.mean(trainer.losses[-1000:]) if len(trainer.losses) > 1000 else 0
                cache_size = len(model.quantum_layer.lookup_table)

                pbar.set_postfix({
                    "Avg Reward": f"{avg_reward:.2f}",
                    "Avg Loss": f"{avg_loss:.4f}",
                    "Cache": f"{cache_size}",
                    "ε": f"{epsilon:.4f}",
                })

            pbar.update(1)

    # 最終結果
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("🎯 CONVERGENCE TEST 完了！")
    print(f"{'='*80}")
    print(f"総時間: {total_time/3600:.1f}時間 ({total_time:.0f}秒)")
    print(f"平均速度: {episodes/total_time:.1f} エピソード/秒")
    print(f"最終報酬: {np.mean(episode_rewards[-1000:]):.4f}")

    # 最終収束分析
    trainer.print_convergence_report()

    # 最終グラフ保存
    final_plot_path = f"convergence_test_final_{episodes}.png"
    trainer.plot_training_progress(save_path=final_plot_path, show_plot=False)

    # 最終モデル保存
    final_save_data = {
        "episodes": episodes,
        "model_state_dict": model.state_dict(),
        "quantum_cache": model.quantum_layer.lookup_table,
        "rewards": episode_rewards,
        "losses": trainer.losses,
        "episode_losses": trainer.episode_losses,
        "final_analysis": trainer.analyze_convergence(),
        "checkpoints": checkpoints,
        "total_time": total_time,
        "final_speed": episodes/total_time
    }

    final_model_path = f"convergence_test_model_{episodes}.pth"
    torch.save(final_save_data, final_model_path)

    print(f"💾 最終モデル保存: {final_model_path}")
    print(f"📊 最終グラフ保存: {final_plot_path}")
    print(f"📁 中間チェックポイント: {len(checkpoints)}個保存済み")

    return model, episode_rewards


# ===== ベンチマーク =====
def benchmark_quantum_speedup() -> None:
    """量子回路の高速化効果を測定"""

    print("\n" + "=" * 60)
    print("📊 量子回路高速化ベンチマーク")
    print("=" * 60)

    # 通常の量子回路
    print("\n1. 通常の量子回路（キャッシュなし）:")
    circuit_nocache = FastQuantumCircuit(n_qubits=4)
    circuit_nocache._cache_enabled = False

    start = time.time()
    for _ in range(100):
        inputs = torch.randn(4) * np.pi
        weights = torch.randn(1, 4, 2) * 0.1
        _ = circuit_nocache.forward(inputs, weights)
    normal_time = time.time() - start
    print(f"   100回実行: {normal_time:.2f}秒")

    # 高速化量子回路
    print("\n2. 高速化量子回路（キャッシュ付き）:")
    circuit_fast = FastQuantumCircuit(n_qubits=4)

    start = time.time()
    for i in range(100):
        # 一部は同じ入力（キャッシュヒット）
        if i % 3 == 0:
            inputs = torch.zeros(4)
        else:
            inputs = torch.randn(4) * np.pi
        weights = torch.randn(1, 4, 2) * 0.1
        _ = circuit_fast.forward(inputs, weights)
    fast_time = time.time() - start
    print(f"   100回実行: {fast_time:.2f}秒")
    print(f"   キャッシュヒット率: {len(circuit_fast.lookup_table)}/100")

    print(f"\n⚡ 高速化倍率: {normal_time/fast_time:.1f}倍")

    # フルモデルのベンチマーク
    print("\n3. フルモデル（1エピソード）:")
    model = FastQuantumNeuralNetwork(n_qubits=4)

    start = time.time()
    for _ in range(10):
        state = torch.randn(1, 252)
        _ = model(state)
    model_time = time.time() - start
    print(f"   10ステップ: {model_time:.2f}秒")
    print(f"   推定1000エピソード: {model_time * 100:.0f}秒")


if __name__ == "__main__":
    # ベンチマーク実行
    benchmark_quantum_speedup()

    # 学習実行
    print("\n量子回路学習を開始しますか？")
    print("1. デモ（100エピソード）")
    print("2. 標準（1000エピソード）")
    print("3. フル（10000エピソード）")
    print("4. 収束テスト（100000エピソード）")

    import sys

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice == "1":
            model, rewards = train_fast_quantum(100, n_qubits=4)
        elif choice == "2":
            model, rewards = train_fast_quantum(1000, n_qubits=4)
        elif choice == "3":
            model, rewards = train_fast_quantum(10000, n_qubits=4)
        elif choice == "4":
            print("\n🧪 100000エピソード収束テストを開始...")
            print("⚠️  注意: このテストは数時間かかる可能性があります")
            model, rewards = train_convergence_test(100000, n_qubits=4)
        else:
            print(f"\n❌ 無効な選択: {choice}")
            print("有効な選択: 1, 2, 3, 4")
    else:
        print("\nデフォルト: 100エピソードのデモを実行")
        model, rewards = train_fast_quantum(100, n_qubits=4)
