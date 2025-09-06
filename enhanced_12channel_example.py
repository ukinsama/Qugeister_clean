#!/usr/bin/env python3
"""
12チャンネル拡張版ガイスター状態表現の例
6x6x12 = 432次元
"""

import torch
import numpy as np

def encode_enhanced_game_state(game_state, player_id):
    """ゲーム状態を12チャンネルテンソルにエンコード"""
    # 6x6x12のテンソルを初期化
    state_tensor = torch.zeros(12, 6, 6)
    
    my_pieces = game_state.player_a_pieces if player_id == "A" else game_state.player_b_pieces
    opponent_pieces = game_state.player_b_pieces if player_id == "A" else game_state.player_a_pieces
    
    # === 基本7チャンネル ===
    # チャンネル0: 自分の善玉位置
    for (x, y), piece_type in my_pieces.items():
        if piece_type == "good":
            state_tensor[0, y, x] = 1
    
    # チャンネル1: 自分の悪玉位置
    for (x, y), piece_type in my_pieces.items():
        if piece_type == "bad":
            state_tensor[1, y, x] = 1
    
    # チャンネル2: 相手の駒位置（種類不明）
    for (x, y), piece_type in opponent_pieces.items():
        state_tensor[2, y, x] = 1
    
    # チャンネル3&4: 確認済み相手善玉・悪玉
    # (過去の捕獲情報から推定 - 簡易実装)
    
    # チャンネル5: 移動可能位置
    legal_moves = game_state.get_legal_moves(player_id)
    for move in legal_moves:
        if len(move) == 2 and isinstance(move[1], tuple):
            from_pos, to_pos = move
            x, y = to_pos
            state_tensor[5, y, x] = 1
    
    # チャンネル6: 脱出可能位置
    if player_id == "A":
        escape_positions = [(0, 0), (5, 0)]
    else:
        escape_positions = [(0, 5), (5, 5)]
    
    for x, y in escape_positions:
        state_tensor[6, y, x] = 1
    
    # === 拡張5チャンネル ===
    
    # チャンネル7: 相手の脱出経路
    opponent_escape_positions = [(0, 5), (5, 5)] if player_id == "A" else [(0, 0), (5, 0)]
    for (opp_x, opp_y), piece_type in opponent_pieces.items():
        if piece_type == "good":  # 善玉の場合のみ
            # 脱出口への最短経路を計算
            for esc_x, esc_y in opponent_escape_positions:
                # マンハッタン距離が3以下なら脱出圏内
                if abs(opp_x - esc_x) + abs(opp_y - esc_y) <= 3:
                    state_tensor[7, opp_y, opp_x] = 0.5
                    # 脱出経路上の位置もマーク
                    path_x = opp_x + (1 if esc_x > opp_x else -1 if esc_x < opp_x else 0)
                    path_y = opp_y + (1 if esc_y > opp_y else -1 if esc_y < opp_y else 0)
                    if 0 <= path_x < 6 and 0 <= path_y < 6:
                        state_tensor[7, path_y, path_x] = 0.3
    
    # チャンネル8: 攻撃可能位置（相手駒を取れる位置）
    for (my_x, my_y), my_piece_type in my_pieces.items():
        # 4方向をチェック
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            target_x, target_y = my_x + dx, my_y + dy
            if (target_x, target_y) in opponent_pieces:
                state_tensor[8, target_y, target_x] = 1
    
    # チャンネル9: 危険地帯（自分の駒が取られる可能性）
    for (opp_x, opp_y), opp_piece_type in opponent_pieces.items():
        # 4方向をチェック
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            danger_x, danger_y = opp_x + dx, opp_y + dy
            if 0 <= danger_x < 6 and 0 <= danger_y < 6:
                state_tensor[9, danger_y, danger_x] = 0.7
    
    # チャンネル10: 制圧領域（自分が支配している領域）
    for (my_x, my_y), my_piece_type in my_pieces.items():
        # 周囲2マスを制圧領域とする
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                ctrl_x, ctrl_y = my_x + dx, my_y + dy
                if 0 <= ctrl_x < 6 and 0 <= ctrl_y < 6:
                    distance = abs(dx) + abs(dy)
                    if distance <= 2:
                        state_tensor[10, ctrl_y, ctrl_x] = max(
                            state_tensor[10, ctrl_y, ctrl_x], 
                            1.0 - distance * 0.3
                        )
    
    # チャンネル11: 脱出阻止位置
    for (opp_x, opp_y), opp_piece_type in opponent_pieces.items():
        for esc_x, esc_y in opponent_escape_positions:
            # 脱出経路を阻止できる位置を計算
            if abs(opp_x - esc_x) + abs(opp_y - esc_y) <= 2:
                # 脱出経路の中間点
                mid_x = (opp_x + esc_x) // 2
                mid_y = (opp_y + esc_y) // 2
                if 0 <= mid_x < 6 and 0 <= mid_y < 6:
                    state_tensor[11, mid_y, mid_x] = 1
    
    return state_tensor

def get_enhanced_config():
    """12チャンネル用の設定"""
    return {
        'state_dim': 432,  # 6x6x12
        'state_channels': {
            'my_good_pieces': 1,
            'my_bad_pieces': 1,
            'opponent_pieces': 1,
            'known_opponent_good': 1,
            'known_opponent_bad': 1,
            'legal_moves': 1,
            'escape_positions': 1,
            'opponent_escape_routes': 1,    # 新規
            'attack_positions': 1,          # 新規
            'danger_zones': 1,              # 新規
            'territory_control': 1,         # 新規
            'escape_blocking': 1            # 新規
        },
        'action_dim': 5,
        'enhanced_features': True
    }

if __name__ == "__main__":
    print("🧠 12チャンネル拡張ガイスター状態表現")
    print("=" * 50)
    
    config = get_enhanced_config()
    print(f"状態次元: {config['state_dim']}")
    print(f"チャンネル数: {len(config['state_channels'])}")
    print("\nチャンネル構成:")
    for i, (name, count) in enumerate(config['state_channels'].items()):
        print(f"  {i}: {name}")
    
    print("\n✨ 拡張チャンネルの効果:")
    print("  - 相手の脱出意図をより正確に予測")
    print("  - 攻撃・防御の戦術的判断が向上") 
    print("  - 盤面支配度による長期戦略")
    print("  - 脱出阻止による勝率向上")