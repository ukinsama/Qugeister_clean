#!/usr/bin/env python3
"""
収束最適化対策
大規模量子ビットでの収束困難対策
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

class AdaptiveConvergenceDetector:
    """適応的収束検出クラス"""

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # 量子ビット数に応じて条件を動的調整
        if n_qubits <= 6:
            # 標準的な厳しい条件
            self.patience = 150
            self.min_delta = 0.005
            self.warmup_episodes = 300
            self.stability_window = 25
        elif n_qubits <= 9:
            # やや緩和
            self.patience = 200
            self.min_delta = 0.01
            self.warmup_episodes = 400
            self.stability_window = 30
        else:  # 10-12 qubits
            # 大幅緩和（収束困難を想定）
            self.patience = 300
            self.min_delta = 0.02
            self.warmup_episodes = 500
            self.stability_window = 40

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.reward_history = []

        print(f"AdaptiveConvergenceDetector for {n_qubits}Q{n_layers}L:")
        print(f"  Patience: {self.patience}")
        print(f"  Min delta: {self.min_delta}")
        print(f"  Warmup: {self.warmup_episodes}")
        print(f"  Stability window: {self.stability_window}")

    def check_convergence(self, current_loss: float, current_reward: float, episode: int) -> bool:
        """適応的収束判定"""
        if episode < self.warmup_episodes:
            return False

        self.loss_history.append(current_loss)
        self.reward_history.append(current_reward)

        # 複数指標による収束判定
        if len(self.loss_history) >= self.stability_window:
            # 1. 損失ベース判定（従来）
            recent_loss = np.mean(self.loss_history[-self.stability_window:])
            loss_improved = recent_loss < self.best_loss - self.min_delta

            # 2. 報酬ベース判定（追加）
            if len(self.reward_history) >= self.stability_window * 2:
                recent_reward = np.mean(self.reward_history[-self.stability_window:])
                old_reward = np.mean(self.reward_history[-self.stability_window*2:-self.stability_window])
                reward_improved = recent_reward > old_reward + 1.0  # 報酬1.0以上改善

                # どちらかが改善していれば継続
                if loss_improved or reward_improved:
                    self.best_loss = recent_loss
                    self.patience_counter = 0
                    return False
                else:
                    self.patience_counter += 1
            else:
                if loss_improved:
                    self.best_loss = recent_loss
                    self.patience_counter = 0
                    return False
                else:
                    self.patience_counter += 1

        return self.patience_counter >= self.patience

    def get_status(self) -> str:
        """収束状態を取得"""
        if len(self.loss_history) < self.stability_window:
            return "Insufficient data"

        progress = (self.patience - self.patience_counter) / self.patience * 100
        return f"Progress: {progress:.1f}% (waiting: {self.patience_counter}/{self.patience})"

class GradientStabilizer:
    """勾配安定化クラス"""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        # 量子ビット数に応じて勾配クリッピング調整
        if n_qubits <= 6:
            self.max_norm = 1.0
        elif n_qubits <= 9:
            self.max_norm = 0.5  # より強い制約
        else:
            self.max_norm = 0.1  # 非常に強い制約

    def clip_gradients(self, model, optimizer):
        """勾配クリッピング"""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

        # 大規模量子ビットでの追加安定化
        if self.n_qubits >= 10:
            # 勾配の分散を制限
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        grad_std = param.grad.std()
                        if grad_std > 0.1:
                            param.grad *= 0.1 / grad_std

class AdaptiveHyperparameters:
    """適応的ハイパーパラメータ"""

    @staticmethod
    def get_learning_rate(n_qubits: int, base_lr: float = 0.001) -> float:
        """量子ビット数に応じた学習率"""
        if n_qubits <= 6:
            return base_lr
        elif n_qubits <= 9:
            return base_lr * 0.5  # 50%削減
        else:
            return base_lr * 0.1  # 90%削減

    @staticmethod
    def get_batch_size(n_qubits: int, base_batch: int = 8) -> int:
        """量子ビット数に応じたバッチサイズ"""
        if n_qubits <= 6:
            return base_batch
        elif n_qubits <= 9:
            return max(4, base_batch // 2)
        else:
            return max(2, base_batch // 4)

    @staticmethod
    def get_epsilon_decay(n_qubits: int, base_decay: float = 0.998) -> float:
        """量子ビット数に応じたepsilon減衰率"""
        if n_qubits <= 6:
            return base_decay
        elif n_qubits <= 9:
            return base_decay ** 0.5  # より遅い減衰
        else:
            return base_decay ** 0.1  # 非常に遅い減衰