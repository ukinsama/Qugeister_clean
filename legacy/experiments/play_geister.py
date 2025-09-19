#!/usr/bin/env python3
"""
ガイスター対戦システム起動スクリプト
"""

import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """メイン関数"""
    print("🎮 ガイスター対戦システム")
    print("=" * 30)
    print("1. 人間 vs 人間（デバッグモード）")
    print("2. 人間 vs AI")
    print("3. AIトーナメント観戦")
    print("0. 終了")
    
    while True:
        try:
            choice = input("\n選択してください (0-3): ")
            
            if choice == "1":
                print("🎮 人間 vs 人間モード起動中...")
                from src.qugeister_competitive.debug_game_viewer import main as debug_main
                debug_main()
                break
                
            elif choice == "2":
                print("🤖 人間 vs AI モード起動中...")
                from human_vs_ai_battle import main as ai_battle_main
                ai_battle_main()
                break
                
            elif choice == "3":
                print("🏆 AIトーナメント起動中...")
                from simple_tournament import main as tournament_main
                tournament_main()
                break
                
            elif choice == "0":
                print("👋 終了します")
                break
                
            else:
                print("❌ 無効な選択です。0-3の数字を入力してください。")
                
        except KeyboardInterrupt:
            print("\n👋 終了します")
            break
        except Exception as e:
            print(f"❌ エラー: {e}")


if __name__ == "__main__":
    main()