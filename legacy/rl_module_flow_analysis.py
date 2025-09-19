#!/usr/bin/env python3
"""
強化学習システムの各モジュール間の情報フロー詳細解析
"""

import torch
import numpy as np

def create_module_flow_diagram():
    """モジュール間の情報フローを図解"""
    print("🏗️ 強化学習システムのモジュール構成と情報フロー")
    print("=" * 70)
    
    flow_diagram = """
    
    ┌──────────────────────────────────────────────────────────────┐
    │                     🎮 ゲーム環境（DebugGeisterGame）          │
    │  ・現在の盤面状態                                              │
    │  ・合法手リスト                                                │
    │  ・報酬信号                                                    │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [盤面情報: 6×6マス、駒位置、種類]
    ┌──────────────────────────────────────────────────────────────┐
    │           📊 状態エンコーダ（State Encoder）                    │
    │  入力: 生の盤面データ                                          │
    │  処理: 7〜16チャンネルのテンソル化                             │
    │  出力: 252〜576次元ベクトル                                    │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [状態ベクトル: torch.Tensor(batch, dim)]
    ┌──────────────────────────────────────────────────────────────┐
    │           🧠 量子エンコーダ（Quantum Encoder）                  │
    │  入力: 状態ベクトル                                            │
    │  処理: 量子回路による特徴抽出                                  │
    │  出力: 量子特徴ベクトル（n_qubits次元）                        │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [量子特徴: 6次元]
    ┌──────────────────────────────────────────────────────────────┐
    │           🎯 Q値ネットワーク（Q-Network）                       │
    │  入力: 量子特徴ベクトル                                        │
    │  処理: 3層ニューラルネットワーク                               │
    │  出力: 各行動のQ値（5次元: 4方向+脱出）                        │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [Q値: [上, 右, 下, 左, 脱出]]
    ┌──────────────────────────────────────────────────────────────┐
    │           🎲 行動選択（Action Selector）                        │
    │  入力: Q値ベクトル + 合法手リスト                              │
    │  処理: ε-greedy/Boltzmann選択                                 │
    │  出力: 選択された行動                                          │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [行動: (from_pos, to_pos)]
    ┌──────────────────────────────────────────────────────────────┐
    │           💰 報酬計算（Reward Calculator）                      │
    │  入力: 行動結果 + ゲーム状態変化                               │
    │  処理: 即時報酬 + 形成報酬の計算                               │
    │  出力: 総合報酬値                                              │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [報酬: float]
    ┌──────────────────────────────────────────────────────────────┐
    │           📝 経験バッファ（Experience Replay Buffer）           │
    │  保存: (状態, 行動, 報酬, 次状態, 終了フラグ)                  │
    │  容量: 10000〜50000経験                                        │
    │  サンプリング: ランダム or 優先度付き                          │
    └────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼ [バッチ: 32個の経験]
    ┌──────────────────────────────────────────────────────────────┐
    │           📉 損失計算＆学習（Loss & Optimizer）                 │
    │  入力: バッチ経験データ                                        │
    │  処理: TD Error計算 → MSE/Huber Loss → 勾配降下               │
    │  更新: ネットワーク重み                                        │
    └──────────────────────────────────────────────────────────────┘
    """
    
    print(flow_diagram)

def analyze_each_module_detail():
    """各モジュールの詳細な入出力分析"""
    print("\n📦 各モジュールの詳細な情報処理")
    print("=" * 70)
    
    modules = {
        "1. 状態エンコーダ": {
            "入力": {
                "player_a_pieces": "Dict[(x,y), 'good'/'bad']",
                "player_b_pieces": "Dict[(x,y), 'good'/'bad']",
                "current_player": "'A' or 'B'",
                "turn": "int"
            },
            "処理": """
            # 7チャンネル版の例
            channel_0 = 自分の善玉位置（0/1のマトリクス）
            channel_1 = 自分の悪玉位置
            channel_2 = 相手の駒位置（種類不明）
            channel_3 = 確認済み相手善玉
            channel_4 = 確認済み相手悪玉
            channel_5 = 移動可能位置
            channel_6 = 脱出可能位置
            
            # 12チャンネル版では追加
            channel_7 = 相手の脱出経路
            channel_8 = 攻撃可能位置
            channel_9 = 危険地帯
            channel_10 = 制圧領域
            channel_11 = 脱出阻止位置
            """,
            "出力": "torch.Tensor([batch_size, 252]) # 6×6×7を平坦化"
        },
        
        "2. 量子エンコーダ（CQCNN）": {
            "入力": "torch.Tensor([batch_size, 252])",
            "処理": """
            for i in range(n_qubits=6):
                # データエンコーディング
                qml.RY(input[i], wires=i)  # 角度エンコーディング
                
            for layer in range(n_layers=2):
                # パラメータ化回路
                for i in range(n_qubits):
                    qml.RY(weights[layer][i][0], wires=i)
                    qml.RZ(weights[layer][i][1], wires=i)
                
                # エンタングルメント
                for i in range(n_qubits-1):
                    qml.CNOT(wires=[i, i+1])
                    
            # 測定
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            """,
            "出力": "torch.Tensor([batch_size, 6]) # 量子特徴"
        },
        
        "3. Q値ネットワーク": {
            "入力": "torch.Tensor([batch_size, 6]) # 量子特徴",
            "処理": """
            x = self.fc1(x)      # Linear(6, 128)
            x = ReLU(x)
            x = Dropout(0.2)(x)
            x = self.fc2(x)      # Linear(128, 64)
            x = ReLU(x)
            x = Dropout(0.2)(x)
            x = self.fc3(x)      # Linear(64, 5)
            return x             # [上, 右, 下, 左, 脱出]のQ値
            """,
            "出力": "torch.Tensor([batch_size, 5]) # 各行動のQ値"
        },
        
        "4. 行動選択モジュール": {
            "入力": {
                "q_values": "torch.Tensor([5])",
                "legal_moves": "List[((x1,y1), (x2,y2))]",
                "epsilon": "float (0.1〜1.0)"
            },
            "処理": """
            if random.random() < epsilon:
                # 探索: ランダム選択
                action = random.choice(legal_moves)
            else:
                # 活用: 最大Q値の合法手を選択
                legal_q_values = []
                for move in legal_moves:
                    action_idx = move_to_index(move)
                    legal_q_values.append(q_values[action_idx])
                best_idx = argmax(legal_q_values)
                action = legal_moves[best_idx]
            """,
            "出力": "((from_x, from_y), (to_x, to_y)) or ((x,y), 'ESCAPE')"
        },
        
        "5. 報酬計算モジュール": {
            "入力": {
                "action": "選択された行動",
                "game_state_before": "行動前の状態",
                "game_state_after": "行動後の状態",
                "game_over": "bool",
                "winner": "'A'/'B'/None"
            },
            "処理": """
            base_reward = 0
            
            # 即時報酬
            if captured_good: base_reward += 10
            if captured_bad: base_reward -= 5
            if lost_good: base_reward -= 20
            if lost_bad: base_reward += 10
            if escaped: base_reward += 100
            if game_won: base_reward += 100
            if game_lost: base_reward -= 100
            
            # 形成報酬（Shaping）
            escape_progress = calc_escape_distance()
            base_reward += escape_progress * 0.1
            
            territory_control = calc_territory()
            base_reward += territory_control * 0.05
            
            return base_reward
            """,
            "出力": "float # 総合報酬値"
        },
        
        "6. 経験バッファ": {
            "入力": "(state, action, reward, next_state, done)",
            "処理": """
            # リングバッファとして実装
            buffer = deque(maxlen=10000)
            buffer.append(experience)
            
            # バッチサンプリング
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size=32)
                return batch
            """,
            "出力": "List[Experience] × 32"
        },
        
        "7. 損失計算＆最適化": {
            "入力": "batch = [(s, a, r, s', done) × 32]",
            "処理": """
            # 現在のQ値
            current_q = model(states)  # [32, 5]
            q_selected = current_q.gather(1, actions)  # [32, 1]
            
            # ターゲットQ値
            next_q = model(next_states)  # [32, 5]
            max_next_q = next_q.max(1)[0]  # [32]
            target_q = rewards + gamma * max_next_q * (1 - dones)
            
            # 損失計算
            loss = MSELoss(q_selected, target_q)
            
            # 重み更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """,
            "出力": "更新されたネットワーク重み"
        }
    }
    
    for module_name, details in modules.items():
        print(f"\n{'='*60}")
        print(f"📦 {module_name}")
        print(f"{'='*60}")
        
        print("\n【入力】")
        if isinstance(details["入力"], dict):
            for key, value in details["入力"].items():
                print(f"  • {key}: {value}")
        else:
            print(f"  {details['入力']}")
        
        print("\n【処理】")
        print(details["処理"])
        
        print("\n【出力】")
        print(f"  {details['出力']}")

def show_data_transformation_example():
    """実際のデータ変換例"""
    print("\n\n🔄 具体的なデータ変換の流れ")
    print("=" * 70)
    
    print("""
    【例: プレイヤーAが(2,4)から(2,3)へ移動して相手駒を捕獲】
    
    1️⃣ ゲーム状態
       player_a_pieces = {(2,4): 'good', (3,4): 'bad', ...}
       player_b_pieces = {(2,3): 'good', (1,1): 'bad', ...}
       ↓
    
    2️⃣ 状態エンコード（7チャンネル → 252次元）
       Channel 0 (自分善玉): [[0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,1,0,0,0],  ← (2,4)に善玉
                            [0,0,0,0,0,0]]
       → Flatten → [0,0,0,...,1,0,...] (252次元)
       ↓
    
    3️⃣ 量子エンコード（252次元 → 6次元）
       量子回路で特徴抽出
       → [0.23, -0.45, 0.67, 0.12, -0.89, 0.34]
       ↓
    
    4️⃣ Q値計算（6次元 → 5次元）
       ニューラルネットワーク処理
       → [3.2, 5.1, 2.8, 4.0, 0.5]  # [上, 右, 下, 左, 脱出]
       ↓
    
    5️⃣ 行動選択
       合法手: [((2,4),(2,3)), ((2,4),(3,4)), ...]
       最大Q値の合法手を選択 → ((2,4),(2,3))  # 上へ移動
       ↓
    
    6️⃣ 環境実行
       相手の善玉を捕獲
       報酬 = +10
       ↓
    
    7️⃣ 経験保存
       (state_252d, action_idx=0, reward=10, next_state_252d, done=False)
       ↓
    
    8️⃣ バッチ学習
       32個の経験をサンプル
       TD Error計算 → Loss計算 → 重み更新
    """)

def explain_information_flow_importance():
    """情報フローの重要性"""
    print("\n💡 各モジュール間の情報フローの重要ポイント")
    print("=" * 70)
    
    important_points = {
        "状態表現の次元削減": {
            "問題": "生の盤面データは高次元でスパース",
            "解決": "チャンネル化により意味のある特徴を抽出",
            "効果": "学習効率の大幅向上"
        },
        "量子エンコーダの役割": {
            "問題": "古典的な特徴抽出の限界",
            "解決": "量子もつれによる非線形特徴の捕捉",
            "効果": "複雑な戦略パターンの学習"
        },
        "合法手フィルタリング": {
            "問題": "違法な手を学習してしまう",
            "解決": "Q値計算後に合法手のみ選択",
            "効果": "無駄な学習の削減"
        },
        "経験再生の必要性": {
            "問題": "時間的相関による学習の偏り",
            "解決": "ランダムサンプリングで相関を破壊",
            "効果": "安定した学習"
        },
        "報酬設計の影響": {
            "問題": "スパースな報酬では学習困難",
            "解決": "形成報酬による段階的フィードバック",
            "効果": "学習速度の向上"
        }
    }
    
    for point, details in important_points.items():
        print(f"\n🔍 {point}")
        print(f"  問題: {details['問題']}")
        print(f"  解決: {details['解決']}")
        print(f"  効果: {details['効果']}")

def show_bottlenecks_and_optimizations():
    """ボトルネックと最適化ポイント"""
    print("\n\n⚡ システムのボトルネックと最適化案")
    print("=" * 70)
    
    optimizations = [
        {
            "ボトルネック": "状態エンコード（毎ステップ実行）",
            "現在の計算量": "O(6×6×channels)",
            "最適化案": "差分更新（変更箇所のみ更新）",
            "期待効果": "計算時間80%削減"
        },
        {
            "ボトルネック": "量子回路シミュレーション",
            "現在の計算量": "O(2^n_qubits)",
            "最適化案": "GPU実装 or 実量子デバイス",
            "期待効果": "10〜100倍高速化"
        },
        {
            "ボトルネック": "経験バッファのメモリ使用",
            "現在の計算量": "50000 × 252 × 4 bytes",
            "最適化案": "圧縮表現 or 優先度付きサンプリング",
            "期待効果": "メモリ50%削減"
        }
    ]
    
    for opt in optimizations:
        print(f"\n⚠️ {opt['ボトルネック']}")
        print(f"   現在: {opt['現在の計算量']}")
        print(f"   改善: {opt['最適化案']}")
        print(f"   効果: {opt['期待効果']}")

if __name__ == "__main__":
    create_module_flow_diagram()
    analyze_each_module_detail()
    show_data_transformation_example()
    explain_information_flow_importance()
    show_bottlenecks_and_optimizations()
    
    print("\n" + "=" * 70)
    print("🏆 まとめ: 7つのモジュールが協調して「盤面→行動→学習」を実現")
    print("各モジュールは特定の役割を持ち、適切な形式でデータを受け渡します")
    print("=" * 70)