#!/usr/bin/env python3
"""
人間 vs AI 対戦システム
既存のAIと対戦してゲームエンジンの動作を確認
"""

import sys
import os
import pygame
from pathlib import Path

# プロジェクトのルートパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.qugeister_competitive.debug_game_viewer import DebugGeisterGame, DebugGUI
from src.qugeister_competitive.ai_base import BaseAI
import random
import numpy as np


class SimpleAI(BaseAI):
    """テスト用シンプルAI"""
    
    def __init__(self, name="SimpleAI", player_id="B"):
        super().__init__(name, player_id)
        
    def get_move(self, game_state, legal_moves):
        """手を選択（BaseAI抽象メソッドの実装）"""
        if legal_moves:
            return random.choice(legal_moves)
        return None
        
    def choose_action(self, game_state):
        """ランダムに行動を選択"""
        legal_moves = game_state.get_legal_moves(self.player_id)
        if legal_moves:
            return random.choice(legal_moves)
        return None


class HumanVsAIGUI(DebugGUI):
    """人間 vs AI 対戦GUI"""
    
    def __init__(self, ai_opponent=None):
        super().__init__()
        self.ai_opponent = ai_opponent or SimpleAI("TestAI", "B")
        self.human_player = "A"
        self.ai_player = "B"
        
        print(f"🤖 AI対戦相手: {self.ai_opponent.name}")
        print(f"👤 あなた: プレイヤー{self.human_player}")
        print(f"🤖 AI: プレイヤー{self.ai_player}")
        
    def handle_ai_turn(self):
        """AIのターン処理"""
        if (self.game.current_player == self.ai_player and 
            not self.game.game_over):
            
            print(f"🤖 {self.ai_opponent.name}が思考中...")
            
            # AIの行動を取得
            try:
                # ゲーム状態をAI用に変換
                game_state_for_ai = self.game
                action = self.ai_opponent.choose_action(game_state_for_ai)
                
                if action:
                    from_pos, to_pos = action
                    if to_pos == "ESCAPE":
                        # 脱出の場合
                        current_pieces = self.game.player_b_pieces
                        if (from_pos in current_pieces and 
                            current_pieces[from_pos] == "good"):
                            del current_pieces[from_pos]
                            self.game.turn += 1
                            self.game.game_over = True
                            self.game.winner = self.ai_player
                            print(f"🎊 {self.ai_opponent.name}が脱出勝利！")
                    else:
                        # 通常の移動
                        success = self.game.make_move(from_pos, to_pos)
                        if success:
                            print(f"🤖 {self.ai_opponent.name}: {from_pos} → {to_pos}")
                        else:
                            print(f"❌ {self.ai_opponent.name}の手が無効: {from_pos} → {to_pos}")
                else:
                    print(f"🤖 {self.ai_opponent.name}: 有効な手がありません")
                    
            except Exception as e:
                print(f"❌ AIエラー: {e}")
                
    def run(self):
        """ゲームメインループ（AI対戦対応）"""
        pygame.init()
        
        # ウィンドウサイズを調整
        window_width = 1000
        window_height = 700
        
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("人間 vs AI ガイスター対戦")
        
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # ボードレイアウト設定
        self.cell_size = 80  
        self.board_width = self.cell_size * 6
        self.board_height = self.cell_size * 6
        self.board_size = self.board_width  # 互換性のため追加
        self.board_x = 50
        self.board_y = 100
        
        clock = pygame.time.Clock()
        running = True
        
        print("🎮 人間 vs AI 対戦開始！")
        print("操作方法:")
        print("  - 左クリック: 駒選択・移動")
        print("  - 右クリック/ESCキー: 脱出実行")
        print("  - Rキー: ゲームリセット")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if (self.game.current_player == self.human_player and 
                        not self.game.game_over):
                        if event.button == 1:  # 左クリック
                            self.handle_click(event.pos)
                        elif event.button == 3:  # 右クリック
                            self.handle_escape()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Rキーでリセット
                        print("🔄 ゲームをリセット")
                        self.game.reset_game()
                        self.selected_piece = None
                        self.legal_moves = []
                    elif (event.key == pygame.K_ESCAPE and 
                          self.game.current_player == self.human_player):
                        self.handle_escape()
            
            # AIのターン処理
            self.handle_ai_turn()
            
            # 描画
            self.screen.fill(self.colors["background"])
            self.draw_title_with_ai_info()
            self.draw_board()
            self.draw_info_panel_with_ai()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        
    def draw_title_with_ai_info(self):
        """タイトル（AI情報付き）"""
        title_text = "人間 vs AI ガイスター対戦"
        title_surf = self.font_large.render(title_text, True, (255, 255, 255))
        title_rect = title_surf.get_rect(center=(self.screen.get_width() // 2, 30))
        self.screen.blit(title_surf, title_rect)
        
        # AI情報
        ai_text = f"対戦相手: {self.ai_opponent.name}"
        ai_surf = self.font.render(ai_text, True, (200, 200, 255))
        ai_rect = ai_surf.get_rect(center=(self.screen.get_width() // 2, 65))
        self.screen.blit(ai_surf, ai_rect)
        
    def draw_info_panel_with_ai(self):
        """情報パネル（AI情報付き）"""
        # 元の情報パネルを描画
        self.draw_info_panel()
        
        # AI情報を追加
        panel_x = self.board_x + self.board_width + 20
        panel_y = self.board_y + 400
        
        # 現在のターン強調
        if not self.game.game_over:
            if self.game.current_player == self.human_player:
                turn_text = "👤 あなたのターン"
                color = (0, 255, 0)
            else:
                turn_text = f"🤖 {self.ai_opponent.name}のターン"
                color = (255, 100, 100)
            
            turn_surf = self.font.render(turn_text, True, color)
            self.screen.blit(turn_surf, (panel_x, panel_y))


def load_existing_ais():
    """既存のAIモデルを読み込み"""
    models = []
    
    # integrated_aisディレクトリから読み込み
    integrated_path = Path("integrated_ais")
    if integrated_path.exists():
        for ai_dir in integrated_path.iterdir():
            if ai_dir.is_dir():
                ai_file = ai_dir / f"{ai_dir.name}_ai.py"
                if ai_file.exists():
                    models.append({
                        'name': ai_dir.name,
                        'path': str(ai_file),
                        'type': 'integrated'
                    })
    
    return models


def load_ai_model(model_info):
    """AIモデルを実際に読み込む"""
    try:
        # integrated_aisから実際のAIクラスを読み込む
        import importlib.util
        spec = importlib.util.spec_from_file_location(model_info['name'], model_info['path'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # AIクラスを探す
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseAI) and attr != BaseAI:
                return attr("B")  # プレイヤーBとして初期化
        
        # 見つからなければSimpleAIを返す
        return SimpleAI(model_info['name'], "B")
        
    except Exception as e:
        print(f"⚠️ {model_info['name']}の読み込みに失敗: {e}")
        return SimpleAI(model_info['name'], "B")


def main():
    """メイン実行関数"""
    print("🎮 人間 vs AI 対戦システム")
    print("=" * 40)
    
    # 既存のAIを探す
    models = load_existing_ais()
    
    # 使用可能なAIリスト
    available_ais = []
    if models:
        available_ais.extend(models)
    available_ais.append({'name': 'SimpleAI', 'type': 'builtin'})
    
    print("利用可能なAIモデル:")
    for i, model in enumerate(available_ais, 1):
        print(f"  {i}. {model['name']}")
    
    try:
        choice = input(f"\n対戦相手を選択 (1-{len(available_ais)}): ")
        choice_idx = int(choice) - 1
        
        if 0 <= choice_idx < len(available_ais):
            selected_model = available_ais[choice_idx]
            print(f"🤖 {selected_model['name']}と対戦します")
            
            if selected_model['name'] == 'SimpleAI':
                ai_opponent = SimpleAI("SimpleAI", "B")
            else:
                # 実際のAIを読み込み
                ai_opponent = load_ai_model(selected_model)
        else:
            print("無効な選択です。SimpleAIを使用します。")
            ai_opponent = SimpleAI("SimpleAI", "B")
            
    except (ValueError, IndexError):
        print("無効な選択です。SimpleAIを使用します。")
        ai_opponent = SimpleAI("SimpleAI", "B")
    
    # 対戦開始
    try:
        game_gui = HumanVsAIGUI(ai_opponent)
        game_gui.run()
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()