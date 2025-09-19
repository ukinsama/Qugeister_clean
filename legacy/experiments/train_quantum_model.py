#!/usr/bin/env python3
"""
Quantum AI Model Training Script
test19.pyの量子回路設定を使用してガイスター量子AIを学習
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qugeister.core.game_engine import GeisterEngine
from qugeister.quantum.quantum_trainer import FastQuantumNeuralNetwork

class QuantumTrainer:
    """量子CNNトレーナー"""
    
    def __init__(self, config=None):
        # test19.pyの設定を読み込み (OpenQASMコメントから抽出)
        self.config = config or {
            'qubits': 8,
            'layers': 3,
            'entanglement': 'linear',
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        print(f"🔧 量子CNN設定: {self.config['qubits']}qubits, {self.config['layers']}layers, {self.config['entanglement']}")
        
        # モデル初期化
        self.model = FastQuantumNeuralNetwork(
            input_dim=252,
            output_dim=36,
            n_qubits=self.config['qubits'],
            n_layers=self.config['layers'],
            embedding='angle',
            entanglement=self.config['entanglement']
        )
        
        # オプティマイザー
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # 学習統計
        self.training_stats = {
            'losses': [],
            'accuracies': [],
            'q_values': []
        }
        
    def generate_training_data(self, num_samples=1000):
        """ゲームデータから学習データを生成"""
        print(f"🎮 {num_samples}個の学習データを生成中...")
        
        X, y = [], []
        
        for i in range(num_samples):
            # ランダムなゲーム状態を生成
            game = GeisterEngine()
            
            # ランダムに数手進める
            for _ in range(random.randint(0, 20)):
                if game.game_over:
                    break
                legal_moves = game.get_legal_moves(game.current_player)
                if legal_moves:
                    move = random.choice(legal_moves)
                    game.make_move(move[0], move[1])
            
            # ゲーム状態をベクトル化
            state = game.get_game_state('A')
            state_vector = torch.tensor(state.to_vector(), dtype=torch.float32)
            
            # ターゲットQ値を生成（簡単な評価関数）
            target_q = self._evaluate_position(game, 'A')
            
            X.append(state_vector)
            y.append(target_q)
            
            if (i + 1) % 100 == 0:
                print(f"  データ生成進捗: {i+1}/{num_samples}")
        
        return torch.stack(X), torch.stack(y)
    
    def _evaluate_position(self, game, player):
        """簡単な位置評価関数"""
        # 36次元のQ値を生成
        q_values = torch.zeros(36)
        
        # 自分の駒数に基づく基本評価
        player_pieces = len(game.player_a_pieces if player == 'A' else game.player_b_pieces)
        enemy_pieces = len(game.player_b_pieces if player == 'A' else game.player_a_pieces)
        
        base_value = (player_pieces - enemy_pieces) * 0.1
        
        # ランダムノイズを追加して多様性を確保
        for i in range(36):
            q_values[i] = base_value + random.uniform(-0.3, 0.3)
        
        # 脱出可能性を考慮
        if game.can_escape(player):
            q_values[-1] = base_value + 0.5  # 最後の要素を脱出Q値として使用
        
        return q_values
    
    def train_epoch(self, X, y, batch_size=32):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(X) // batch_size
        
        # データをシャッフル
        indices = torch.randperm(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            # 順伝播
            self.optimizer.zero_grad()
            predictions = torch.stack([self.model(x) for x in batch_X])
            
            # 損失計算
            loss = self.criterion(predictions, batch_y)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def evaluate_model(self, X, y):
        """モデル評価"""
        self.model.eval()
        with torch.no_grad():
            predictions = torch.stack([self.model(x) for x in X[:100]])  # 100サンプルで評価
            targets = y[:100]
            
            # MSE精度
            mse = nn.MSELoss()(predictions, targets).item()
            
            # 相関係数（近似精度）
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1].item()
            
            return {
                'mse': mse,
                'correlation': correlation if not torch.isnan(torch.tensor(correlation)) else 0.0
            }
    
    def train(self, epochs=None, save_path="models/quantum_model_test19.pth"):
        """メイン学習ループ"""
        epochs = epochs or self.config['epochs']
        
        print(f"🚀 量子CNN学習開始: {epochs}エポック")
        print(f"📊 モデルパラメータ: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 学習データ生成
        X_train, y_train = self.generate_training_data(2000)
        X_val, y_val = self.generate_training_data(500)
        
        print(f"📈 学習データ: {len(X_train)}, 検証データ: {len(X_val)}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 学習
            train_loss = self.train_epoch(X_train, y_train, self.config['batch_size'])
            
            # 評価
            if epoch % 10 == 0:
                eval_metrics = self.evaluate_model(X_val, y_val)
                
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Loss={train_loss:.4f}, "
                      f"Val_MSE={eval_metrics['mse']:.4f}, "
                      f"Correlation={eval_metrics['correlation']:.3f}")
                
                # 統計保存
                self.training_stats['losses'].append(train_loss)
                self.training_stats['accuracies'].append(eval_metrics['correlation'])
                
                # ベストモデル保存
                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save_model(save_path, epoch, train_loss)
        
        print(f"✅ 学習完了! ベスト損失: {best_loss:.4f}")
        return self.training_stats
    
    def save_model(self, path, epoch, loss):
        """モデル保存"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'stats': self.training_stats
        }, path)
        
        print(f"💾 モデル保存: {path}")
    
    def test_model(self):
        """学習済みモデルのテスト"""
        print("🧪 学習済みモデルテスト...")
        
        # テストゲーム作成
        game = GeisterEngine()
        state = game.get_game_state('A')
        state_vector = torch.tensor(state.to_vector(), dtype=torch.float32)
        
        # 推論実行
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_vector)
            action_q = self.model.get_action_from_qmap(state_vector)
        
        print(f"📊 Q値出力範囲: [{q_values.min():.3f}, {q_values.max():.3f}]")
        print(f"🎯 行動Q値: {action_q.tolist()}")
        print(f"✅ 推論成功: {state_vector.shape} → {q_values.shape}")

def parse_test19_config():
    """test19.pyから設定を解析"""
    try:
        with open('test19.py', 'r') as f:
            content = f.read()
        
        config = {'qubits': 8, 'layers': 3, 'entanglement': 'linear'}
        
        # コメントから設定抽出
        for line in content.split('\n'):
            if 'Qubits:' in line:
                config['qubits'] = int(line.split('Qubits:')[1].strip())
            elif 'Layers:' in line:
                config['layers'] = int(line.split('Layers:')[1].strip())
            elif 'Entanglement:' in line:
                config['entanglement'] = line.split('Entanglement:')[1].strip()
        
        return config
    except:
        return {'qubits': 8, 'layers': 3, 'entanglement': 'linear'}

def main():
    """メイン実行"""
    print("🌌 Quantum Geister AI Training")
    print("=" * 50)
    
    # test19.pyの設定を読み込み
    config = parse_test19_config()
    config.update({
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 0.001
    })
    
    print(f"📋 学習設定: {config}")
    
    # トレーナー初期化
    trainer = QuantumTrainer(config)
    
    # 学習実行
    stats = trainer.train()
    
    # テスト実行
    trainer.test_model()
    
    # 統計保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = f"learning/trained_models/test19_{timestamp}/training_stats.json"
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📈 学習統計保存: {stats_path}")
    print("🎮 学習完了!")

if __name__ == "__main__":
    main()