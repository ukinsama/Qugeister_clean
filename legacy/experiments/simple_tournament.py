#!/usr/bin/env python3
"""
Simple Tournament - 統合AIによる簡単トーナメント
3stepで作成した学習済みAIによる対戦システム
"""

import os
import sys
from pathlib import Path
import torch
import json
import random
from datetime import datetime

def find_integrated_models():
    """integrated_aisからモデルを発見"""
    models = []
    integrated_dir = Path("integrated_ais")
    
    if integrated_dir.exists():
        for ai_dir in integrated_dir.iterdir():
            if ai_dir.is_dir():
                model_path = ai_dir / "model.pth"
                ai_info_path = ai_dir / "ai_info.json"
                
                if model_path.exists():
                    # AI情報読み込み
                    ai_info = {}
                    if ai_info_path.exists():
                        try:
                            with open(ai_info_path, 'r', encoding='utf-8') as f:
                                ai_info = json.load(f)
                        except:
                            pass
                    
                    models.append({
                        'name': ai_info.get('name', ai_dir.name),
                        'path': str(model_path),
                        'dir': str(ai_dir),
                        'info': ai_info
                    })
    
    # learning/trained_modelsからも検索
    learning_dir = Path("learning/trained_models")
    if learning_dir.exists():
        for model_dir in learning_dir.iterdir():
            if model_dir.is_dir():
                model_path = model_dir / "model.pth"
                if model_path.exists():
                    models.append({
                        'name': model_dir.name,
                        'path': str(model_path),
                        'dir': str(model_dir),
                        'info': {'type': 'trained_model'}
                    })
    
    return models

def simulate_battle(model1, model2):
    """2モデル間の対戦シミュレーション"""
    # 簡単なランダム対戦シミュレーション
    # 実際の実装では、ゲームロジックを使用
    
    model1_wins = 0
    model2_wins = 0
    draws = 0
    
    for game in range(10):  # 10ゲーム対戦
        result = random.choice(['model1', 'model2', 'draw'])
        if result == 'model1':
            model1_wins += 1
        elif result == 'model2':
            model2_wins += 1
        else:
            draws += 1
    
    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'winner': 'model1' if model1_wins > model2_wins else ('model2' if model2_wins > model1_wins else 'draw')
    }

def run_tournament(models):
    """トーナメント実行"""
    if len(models) < 2:
        print("❌ 対戦に必要なモデルが不足しています (最低2個)")
        return
    
    print(f"\n🏆 Simple Tournament 開始")
    print(f"参加AI: {len(models)}個")
    print("=" * 50)
    
    # 参加AI表示
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
    print()
    
    # 総当たり戦
    results = []
    wins = {model['name']: 0 for model in models}
    total_games = {model['name']: 0 for model in models}
    
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            print(f"⚔️ {model1['name']} vs {model2['name']}")
            
            battle_result = simulate_battle(model1, model2)
            results.append({
                'model1': model1['name'],
                'model2': model2['name'],
                'result': battle_result
            })
            
            # 勝利数カウント
            if battle_result['winner'] == 'model1':
                wins[model1['name']] += 1
            elif battle_result['winner'] == 'model2':
                wins[model2['name']] += 1
            
            total_games[model1['name']] += 1
            total_games[model2['name']] += 1
            
            print(f"   結果: {model1['name']} {battle_result['model1_wins']}-{battle_result['model2_wins']} {model2['name']} ({'引分' if battle_result['winner'] == 'draw' else '勝者: ' + (model1['name'] if battle_result['winner'] == 'model1' else model2['name'])})")
            print()
    
    # ランキング生成
    rankings = []
    for model in models:
        name = model['name']
        win_rate = wins[name] / max(1, total_games[name])
        rankings.append({
            'name': name,
            'wins': wins[name],
            'total': total_games[name],
            'win_rate': win_rate
        })
    
    rankings.sort(key=lambda x: x['win_rate'], reverse=True)
    
    # 結果表示
    print("\n🏆 トーナメント結果")
    print("=" * 50)
    for i, rank in enumerate(rankings, 1):
        print(f"{i}位: {rank['name']} (勝率: {rank['win_rate']:.2%}, {rank['wins']}/{rank['total']}勝)")
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("tournament_results")
    results_dir.mkdir(exist_ok=True)
    
    result_data = {
        'timestamp': timestamp,
        'participants': [model['name'] for model in models],
        'battles': results,
        'rankings': rankings
    }
    
    result_file = results_dir / f"simple_tournament_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 結果保存: {result_file}")
    return rankings

def main():
    """メイン実行"""
    print("🎯 Simple Tournament System")
    print("3stepで作成したAI同士の対戦システム")
    print("=" * 50)
    
    # モデル発見
    models = find_integrated_models()
    
    if not models:
        print("❌ 対戦可能なモデルが見つかりません")
        print("💡 次のいずれかを実行してください:")
        print("   1. python qugeister_ai_system/examples/integration_example.py")
        print("   2. python learning/recipe_trainer.py --recipe test02 50")
        return
    
    print(f"✅ {len(models)}個のAIモデルを発見")
    for model in models:
        print(f"   - {model['name']}")
    
    # トーナメント実行
    rankings = run_tournament(models)
    
    if rankings:
        champion = rankings[0]
        print(f"\n🏆 チャンピオン: {champion['name']}")
        print(f"   勝率: {champion['win_rate']:.2%}")

if __name__ == "__main__":
    main()