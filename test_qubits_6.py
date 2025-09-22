# ============================================
# Quantum Battle System - 6 Qubits Experiment
# ============================================
# 量子ビット数: 6
# レイヤー数: 1 (固定)
# エポック数: 1000
# ============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path

# 修正されたゲームエンジンのインポート
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

# ===== 学習方法の設定 =====
learning_config = {
    'method': 'reinforcement',
    'algorithm': 'dqn',
}

# ===== モジュール設定 =====
module_config = {
    # モジュール1: 初期配置（修正されたエンジン対応 - プレイヤーA下側配置）
    'placement': {
        'type': 'standard',
        'player_a_bottom': True,    # プレイヤーAは下側（y=4,5）に配置
        'player_b_top': True,       # プレイヤーBは上側（y=0,1）に配置
        'my_pieces_only': np.array([
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ]),
        'escape_positions': {
            'player_a': [(0, 0), (5, 0)],  # A脱出口：相手陣地（上側角）
            'player_b': [(0, 5), (5, 5)]   # B脱出口：相手陣地（下側角）
        },
        'coordinate_system': 'corrected',  # 修正された座標系
        'opponent_unknown': True,          # 相手の駒配置は完全に未知
        'rule_compliant': True            # ガイスタールール準拠
    },
    
    # モジュール2: 敵駒推定（CQCNN） - 6量子ビット
    'quantum': {
        'n_qubits': 6,
        'n_layers': 1,
        'embedding_type': 'angle',
        'entanglement': 'linear',
        'total_params': 18  # 6 qubits * 1 layer * 3 params
    },
    
    # モジュール3: 報酬関数（修正されたルール対応）
    'reward': {
        'strategy': 'balanced',
        'capture_good_reward': 10,
        'capture_bad_penalty': -5,
        'escape_reward': 50,
        'captured_good_penalty': -20,
        'captured_bad_reward': 10,
        'win_condition_rewards': {
            'escape_win': 100,
            'eliminate_all_good': 100,
            'eliminate_all_bad': 100
        },
        'position_rewards': {
            'advance_toward_escape': 2,
            'center_control': 1,
            'opponent_territory': 3
        }
    },
    
    # モジュール4: Q値計算（修正されたエンジン対応）
    'qmap': {
        'method': 'dqn',
        'state_dim': 252,  # 6x6x7チャンネル
        'action_dim': 5,         # 4方向 + 脱出
        'selected_channels': 7,
        'state_channels': {
            'my_good_pieces': 1,
            'my_bad_pieces': 1,
            'opponent_pieces': 1,
            'known_opponent_good': 1,
            'known_opponent_bad': 1,
            'legal_moves': 1,
            'escape_positions': 1,
        },
        'legal_moves_only': True,
        'game_engine': 'DebugGeisterGame',
        'move_validation': True
    },
    
    # モジュール5: 行動選択
    'action': {
        'strategy': 'epsilon',
        'epsilon': 0.1,
        'temperature': None
    }
}

# ===== ハイパーパラメータ =====
hyperparameters = {
    # 基本学習設定
    'learning_rate': 0.001,
    'batch_size': 8,
    'epochs': 1000,  # 1000エポック固定
    'validation_split': 0.2,
    
    # 最適化設定
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'dropout_rate': 0.2,
    'l2_regularization': 0.0001,
    
    # 強化学習パラメータ
    'epsilon': 0.1,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'gamma': 0.99,
    'replay_buffer_size': 1000,
    'target_update_freq': 100,
}

# ===== モデル構築（6量子ビット版） =====
class QuantumBattleAI_6Qubits(nn.Module):
    def __init__(self, name="QuantumBattleAI_6Q", player_id="B"):
        super().__init__()
        self.name = name
        self.player_id = player_id
        self.config = module_config
        
        # CQCNNエンコーダ
        self.quantum_encoder = self._build_quantum_encoder()
        
        # 古典的デコーダ（6量子ビット対応）
        self.classical_decoder = nn.Sequential(
            nn.Linear(self.config['quantum']['n_qubits'], 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 4方向 + 脱出アクション
        )
    
    def get_move(self, game_state, legal_moves):
        """修正されたエンジン用の手選択（BaseAI抽象メソッド実装）"""
        if not legal_moves:
            return None
        
        # 脱出可能手を最優先（move形式を安全にチェック）
        escape_moves = []
        for move in legal_moves:
            try:
                if hasattr(move, '__len__') and len(move) > 1 and (move[1] == "ESCAPE" or (isinstance(move[1], str) and move[1] == "ESCAPE")):
                    escape_moves.append(move)
            except:
                continue
        if escape_moves and self.config['reward']['strategy'] == 'escape':
            return escape_moves[0]
        
        # 量子ネットワークで評価
        state_tensor = self._encode_game_state(game_state)
        q_values = self.forward(state_tensor)
        
        # バッチ次元を削除して1次元テンソルにする
        if len(q_values.shape) > 1:
            q_values = q_values.squeeze(0)
        
        # 合法手のみから選択
        best_move = None
        best_value = float('-inf')
        
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            if move_idx < len(q_values):
                q_val = float(q_values[move_idx].item()) if hasattr(q_values[move_idx], 'item') else float(q_values[move_idx])
                if q_val > best_value:
                    best_value = q_val
                    best_move = move
        
        return best_move if best_move else legal_moves[0]
    
    def choose_action(self, game_state):
        """human_vs_ai_battle.py用のアクション選択"""
        legal_moves = game_state.get_legal_moves(self.player_id)
        return self.get_move(game_state, legal_moves)
    
    def _build_quantum_encoder(self):
        # 量子回路の構築（6量子ビット）
        from pennylane import numpy as qnp
        import pennylane as qml
        
        n_qubits = self.config['quantum']['n_qubits']
        n_layers = self.config['quantum']['n_layers']
        
        # lightning.qubitが使えない場合はdefault.qubitを使用
        try:
            dev = qml.device('lightning.qubit', wires=n_qubits)
        except (ImportError, Exception):
            dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            # データエンコーディング
            for i in range(n_qubits):
                if self.config['quantum']['embedding_type'] == 'angle':
                    qml.RY(inputs[i], wires=i)
                else:
                    qml.RX(inputs[i], wires=i)
            
            # パラメータ化量子回路
            for l in range(n_layers):
                # 回転層
                for i in range(n_qubits):
                    qml.RY(weights[l][i][0], wires=i)
                    qml.RZ(weights[l][i][1], wires=i)
                
                # エンタングルメント層
                if self.config['quantum']['entanglement'] == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                elif self.config['quantum']['entanglement'] == 'full':
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            qml.CZ(wires=[i, j])
            
            # 測定
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return quantum_circuit
    
    def _encode_game_state(self, game_state):
        """ゲーム状態を7チャンネルテンソルにエンコード"""
        import torch
        import numpy as np
        
        # 6x6x7のテンソルを初期化
        state_tensor = torch.zeros(7, 6, 6)
        
        # チャンネル0: 自分の善玉位置
        my_pieces = game_state.player_a_pieces if self.player_id == "A" else game_state.player_b_pieces
        for (x, y), piece_type in my_pieces.items():
            if piece_type == "good":
                state_tensor[0, y, x] = 1
        
        # チャンネル1: 自分の悪玉位置
        for (x, y), piece_type in my_pieces.items():
            if piece_type == "bad":
                state_tensor[1, y, x] = 1
        
        # チャンネル2: 相手の駒位置（種類不明）
        opponent_pieces = game_state.player_b_pieces if self.player_id == "A" else game_state.player_a_pieces
        for (x, y), piece_type in opponent_pieces.items():
            state_tensor[2, y, x] = 1
        
        # チャンネル3&4: 確認済み相手善玉・悪玉（簡易実装）
        
        # チャンネル5: 移動可能位置
        legal_moves = game_state.get_legal_moves(self.player_id)
        for move in legal_moves:
            if len(move) == 2 and isinstance(move[1], tuple):
                from_pos, to_pos = move
                x, y = to_pos
                state_tensor[5, y, x] = 1
        
        # チャンネル6: 脱出可能位置
        escape_positions = self.config['placement']['escape_positions']
        if self.player_id in escape_positions:
            for x, y in escape_positions[self.player_id]:
                state_tensor[6, y, x] = 1
        
        # バッチ次元を追加して返す
        flattened = state_tensor.flatten()  # 252次元に平坦化
        return flattened.unsqueeze(0)  # [1, 252] バッチサイズ1
    
    def _move_to_index(self, move):
        """移動を行動インデックスに変換"""
        if len(move) == 2 and move[1] == "ESCAPE":
            return 4  # 脱出アクション
        
        # 4方向移動を0-3にマップ
        if len(move) == 2:
            from_pos, to_pos = move
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if dx == 0 and dy == -1: return 0  # 上
            elif dx == 1 and dy == 0: return 1  # 右
            elif dx == 0 and dy == 1: return 2  # 下
            elif dx == -1 and dy == 0: return 3  # 左
        
        return 0  # デフォルト
    
    def _distribute_channels_to_qubits(self, x):
        """252次元テンソルを6量子ビット用に変換（CNN風シンプル設計）"""
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        n_qubits = self.config['quantum']['n_qubits']
        batch_size = x.size(0)
        
        # 252次元を7チャンネル × 6×6に再構成
        channels = x.view(batch_size, 7, 6, 6)  # [batch_size, 7, 6, 6]
        
        # Global Average Poolingで各チャンネルから1つの値を抽出
        channel_features = F.adaptive_avg_pool2d(channels, (1, 1))  # [batch_size, 7, 1, 1]
        channel_features = channel_features.view(batch_size, 7)  # [batch_size, 7]
        
        # 6量子ビット用に最初の6チャンネルを選択
        quantum_input = channel_features[:, :6]  # [batch_size, 6]
        
        # 量子回路用に正規化（0-π範囲）
        quantum_input = torch.tanh(quantum_input) * np.pi / 2
        
        return quantum_input
    
    def forward(self, x):
        # 7チャンネル均等分散量子エンコーディング（6量子ビット版）
        batch_size = x.size(0)
        weights = torch.randn(self.config['quantum']['n_layers'], self.config['quantum']['n_qubits'], 2) * 0.1
        quantum_input = self._distribute_channels_to_qubits(x)
        
        # バッチ処理対応
        quantum_features_list = []
        for i in range(batch_size):
            qf = self.quantum_encoder(quantum_input[i], weights)
            quantum_features_list.append(torch.tensor(qf, dtype=torch.float32))
        
        quantum_features = torch.stack(quantum_features_list)
        
        # 古典的処理
        output = self.classical_decoder(quantum_features)
        
        return output

# ===== 学習プロセス =====
def train_model(model, config, hyperparameters):
    # オプティマイザ設定
    if hyperparameters['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['l2_regularization']
        )
    elif hyperparameters['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparameters['learning_rate'],
            momentum=0.9,
            weight_decay=hyperparameters['l2_regularization']
        )
    
    # スケジューラ設定
    if hyperparameters['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hyperparameters['epochs']
        )
    elif hyperparameters['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None
    
    # 強化学習ループ
    from collections import deque
    import random
    
    # リプレイバッファ
    replay_buffer = deque(maxlen=hyperparameters['replay_buffer_size'])
    
    # ε-greedy パラメータ
    epsilon = hyperparameters['epsilon']
    
    # 勝率記録
    win_count = 0
    total_games = 0
    
    # ゲーム環境を初期化
    from qugeister import GeisterEngine
    system = GeisterEngine()
    
    for episode in range(hyperparameters['epochs']):
        system.reset_game()
        # 初期状態を作成（252次元ベクトル）
        state = torch.zeros(252)
        total_reward = 0
        done = False
        
        step_count = 0
        max_steps = 50  # 1エピソードの最大ステップ数
        
        while not done and step_count < max_steps:
            # 行動選択
            if random.random() < epsilon:
                action = random.choice(range(5))  # 5個の可能な行動（4方向+脱出）
            else:
                with torch.no_grad():
                    q_values = model(state.unsqueeze(0))  # バッチ次元を追加
                    action = q_values.argmax().item()
            
            # 簡単な環境シミュレーション
            legal_moves = system.get_legal_moves(system.current_player)
            if legal_moves and action < len(legal_moves):
                # 合法手を実行
                from_pos, to_pos = legal_moves[action % len(legal_moves)]
                success = system.make_move(from_pos, to_pos)
                reward = 1.0 if success else -0.1
            else:
                # 非合法手
                reward = -0.5
            
            # 次の状態（ランダム）
            next_state = torch.randn(252) * 0.1
            
            # ゲーム終了条件
            step_count += 1
            done = step_count >= max_steps or system.game_over
            
            # 勝率計算
            if system.game_over:
                total_games += 1
                if system.winner == "A":  # プレイヤーA勝利と仮定
                    win_count += 1
            
            # リプレイバッファに保存
            replay_buffer.append((state.clone(), action, reward, next_state.clone(), done))
            
            state = next_state
            total_reward += reward
            
            # ミニバッチ学習
            if len(replay_buffer) >= hyperparameters['batch_size']:
                batch = random.sample(replay_buffer, hyperparameters['batch_size'])
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Q値の計算と更新
                states_tensor = torch.stack(states)
                next_states_tensor = torch.stack(next_states)
                actions_tensor = torch.tensor(actions).unsqueeze(1)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32)
                
                current_q = model(states_tensor)
                next_q = model(next_states_tensor)
                target_q = rewards_tensor + hyperparameters['gamma'] * next_q.max(1)[0] * (1 - dones_tensor)
                
                loss = nn.MSELoss()(current_q.gather(1, actions_tensor).squeeze(), target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # εの減衰
        epsilon = max(0.01, epsilon * hyperparameters['epsilon_decay'])
        
        # スケジューラ更新
        if scheduler:
            scheduler.step()
        
        # 勝率計算と表示
        win_rate = win_count / max(total_games, 1) * 100
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, ε = {epsilon:.4f}, Win Rate = {win_rate:.1f}% ({win_count}/{total_games})")
    
    final_win_rate = win_count / max(total_games, 1) * 100
    return model, final_win_rate

# ===== メイン実行 =====
if __name__ == "__main__":
    # モデル初期化
    model = QuantumBattleAI_6Qubits()
    
    print("========================================")
    print("Quantum Battle System - 6 Qubits Training")
    print("========================================")
    print(f"学習方法: 強化学習")
    print(f"量子ビット数: {module_config['quantum']['n_qubits']}")
    print(f"レイヤー数: {module_config['quantum']['n_layers']}")
    print(f"エポック数: {hyperparameters['epochs']}")
    print(f"総パラメータ数: {sum(p.numel() for p in model.parameters())}")
    print("========================================")
    
    # 学習実行
    try:
        trained_model, win_rate = train_model(model, module_config, hyperparameters)
        
        # モデル保存
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'module_config': module_config,
            'hyperparameters': hyperparameters,
            'learning_method': learning_config,
            'final_win_rate': win_rate
        }, 'quantum_battle_model_6qubits.pth')
        
        print("\n========================================")
        print(f"Training Complete! Final Win Rate: {win_rate:.1f}%")
        print("Model saved as 'quantum_battle_model_6qubits.pth'")
        print("========================================")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Please check your configuration and try again.")