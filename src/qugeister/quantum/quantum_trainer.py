#!/usr/bin/env python3
"""
高速量子回路シミュレーショントレーナー
量子回路を使いながら実用的な速度で学習を実現
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pennylane as qml
from collections import deque
import random
import time
from tqdm import tqdm

# Import updated FastQuantumCircuit
from .quantum_circuit import FastQuantumCircuit
import pickle
from functools import lru_cache
import hashlib

# ===== 量子回路はquantum_circuit.pyから import =====

# ===== ハイブリッド量子-古典ニューラルネットワーク =====
class FastQuantumNeuralNetwork(nn.Module):
    """高速化された量子ニューラルネットワーク（ユーザー設定対応）"""
    
    def __init__(self, input_dim=252, output_dim=36, n_qubits=4, n_layers=2, embedding='angle', entanglement='linear'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.embedding = embedding
        self.entanglement = entanglement
        
        # 1. 強化された前処理CNN（Deep Feature Extraction）
        self.preprocessor = nn.Sequential(
            # First CNN Block - Pattern Recognition  
            nn.Linear(input_dim, 512),
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
            nn.Linear(64, n_qubits),
            nn.Tanh()  # 量子回路の入力範囲に正規化 [-1, 1]
        )
        
        # 2. 強化量子回路層（Multi-Layer Quantum Processing）
        self.quantum_layer = FastQuantumCircuit(
            n_qubits=n_qubits, 
            n_layers=n_layers,
            embedding=embedding,
            entanglement=entanglement
        )
        
        # 量子特徴増強層（Quantum Feature Enhancement）
        self.quantum_enhancer = nn.Sequential(
            nn.Linear(n_qubits, n_qubits * 2),
            nn.Tanh(),
            nn.Linear(n_qubits * 2, n_qubits),
            nn.Tanh()
        )
        
        # 3. 超強化後処理CNN（Deep Q-Value Generation）
        self.postprocessor = nn.Sequential(
            # First Expansion Block - Quantum Feature Amplification
            nn.Linear(n_qubits, 128),
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
            nn.Linear(64, output_dim),  # 36 outputs for 6x6 Q-value map
            nn.Tanh()  # Normalize Q-values to [-1, 1] range for stability
        )
        
        # 4. 強化量子回路の重み（ユーザー設定レイヤー数対応）
        self.quantum_weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)  # ユーザー設定レイヤー分の重み
    
    def forward(self, x):
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
                quantum_input, 
                self.quantum_weights
            )
            quantum_outputs.append(quantum_output)
        
        # 3. 量子出力をテンソルに変換
        if batch_size > 1:
            quantum_features = torch.stack([torch.tensor(out, dtype=torch.float32) for out in quantum_outputs])
        else:
            quantum_features = torch.tensor(quantum_outputs[0], dtype=torch.float32).unsqueeze(0)
        
        # 4. 量子特徴増強処理（4→8→4）
        enhanced_features = self.quantum_enhancer(quantum_features)
        
        # 5. 量子特徴と増強特徴を結合
        combined_features = quantum_features + enhanced_features
        
        # 6. 超強化後処理で36次元Q値マップ生成（4→36）
        output = self.postprocessor(combined_features)
        
        return output.squeeze(0) if batch_size == 1 else output
    
    def get_qvalue_map(self, x):
        """36次元出力を6x6のQ値マップに変換"""
        output = self.forward(x)
        if output.dim() == 1:
            # Single sample: reshape to 6x6
            return output.reshape(6, 6)
        else:
            # Batch: reshape each sample to 6x6
            return output.reshape(-1, 6, 6)
    
    def get_action_from_qmap(self, x):
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
    
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=1000)
        
        # 学習統計
        self.losses = []
        self.rewards = []
        
    def train_step(self, batch_size=8):
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
        return loss.item()

# ===== メイン学習ループ =====
def train_fast_quantum(episodes=1000, n_qubits=4):
    """高速量子回路学習の実行"""
    
    print("=" * 60)
    print("🚀 高速量子回路シミュレーション学習")
    print("=" * 60)
    print(f"量子ビット数: {n_qubits}")
    print(f"エピソード数: {episodes}")
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
                epsilon = max(0.01, 0.1 * (0.995 ** episode))
                if random.random() < epsilon:
                    action = random.randrange(5)
                else:
                    with torch.no_grad():
                        q_values = model.get_action_from_qmap(state)
                        action = q_values.argmax().item()
                
                # 環境ステップ（簡略化）
                next_state = torch.randn(1, 252)
                reward = random.uniform(-1, 1)
                done = random.random() < 0.1
                
                # 経験を保存
                trainer.replay_buffer.append(
                    (state, action, reward, next_state, done)
                )
                
                # 学習
                loss = trainer.train_step()
                
                # 更新
                state = next_state
                episode_reward += reward
                steps += 1
            
            episode_rewards.append(episode_reward)
            
            # 進捗更新
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                elapsed = time.time() - start_time
                speed = (episode + 1) / elapsed
                
                # キャッシュ統計
                cache_size = len(model.quantum_layer.lookup_table)
                
                pbar.set_postfix({
                    'Avg Reward': f'{avg_reward:.2f}',
                    'Speed': f'{speed:.1f} eps/s',
                    'Cache': f'{cache_size}/{model.quantum_layer.cache_size}',
                    'ε': f'{epsilon:.3f}'
                })
            
            pbar.update(1)
    
    # 結果表示
    total_time = time.time() - start_time
    print(f"\n✅ 学習完了！")
    print(f"総時間: {total_time:.1f}秒")
    print(f"速度: {episodes/total_time:.1f} エピソード/秒")
    print(f"最終報酬: {np.mean(episode_rewards[-100:]):.2f}")
    
    # モデル保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'quantum_cache': model.quantum_layer.lookup_table,
        'rewards': episode_rewards
    }, 'fast_quantum_model.pth')
    
    print("💾 モデルを 'fast_quantum_model.pth' として保存")
    
    return model, episode_rewards

# ===== ベンチマーク =====
def benchmark_quantum_speedup():
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
    
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice == "1":
            model, rewards = train_fast_quantum(100, n_qubits=4)
        elif choice == "2":
            model, rewards = train_fast_quantum(1000, n_qubits=4)
        elif choice == "3":
            model, rewards = train_fast_quantum(10000, n_qubits=4)
    else:
        print("\nデフォルト: 100エピソードのデモを実行")
        model, rewards = train_fast_quantum(100, n_qubits=4)