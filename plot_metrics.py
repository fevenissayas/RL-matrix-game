"""
Generate training performance plots - no external dependencies.
Usage: python3 plot_metrics.py
"""

import csv
import math

def read_csv(filename):
    episodes, rewards, epsilons, rl_wins, h_wins, ties, rolling = [], [], [], [], [], [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            epsilons.append(float(row['epsilon']))
            rl_wins.append(int(row['rl_wins']))
            h_wins.append(int(row['heuristic_wins']))
            ties.append(int(row['ties']))
            rolling.append(float(row['rolling_win_rate']))
    return episodes, rewards, epsilons, rl_wins, h_wins, ties, rolling

def plot_metrics():
    episodes, rewards, epsilons, rl_wins, h_wins, ties, rolling = read_csv('training_metrics.csv')
    
    print("="*60)
    print("TRAINING METRICS VISUALIZATION")
    print("="*60)
    
    print("\n[1] REWARD TREND (last 30 episodes)")
    print("-"*50)
    recent = rewards[-30:] if len(rewards) > 30 else rewards
    mx, mn = max(recent), min(recent)
    rng = mx - mn if mx != mn else 1
    for r in recent:
        bar = int((r - mn) / rng * 40)
        print("|" + "="*bar + " "*(40-bar) + f" {r:.1f}")
    
    print("\n[2] WIN RATE (last 30)")
    print("-"*50)
    recent = rolling[-30:] if len(rolling) > 30 else rolling
    for w in recent:
        bar = int(w * 40)
        print("|" + "#"*bar + " "*(40-bar) + f" {w*100:.1f}%")
    
    print("\n[3] EPSILON DECAY")
    print("-"*50)
    recent = epsilons[-30:] if len(epsilons) > 30 else epsilons
    for e in recent:
        bar = int(e * 40)
        print("|" + "*"*bar + " "*(40-bar) + f" {e:.4f}")
    
    print("\n[4] CUMULATIVE WINS")
    print("-"*50)
    print(f"  RL Wins:      {rl_wins[-1]}")
    print(f"  Heuristic:  {h_wins[-1]}")
    print(f"  Ties:        {ties[-1]}")
    
    total = rl_wins[-1] + h_wins[-1] + ties[-1]
    if total > 0:
        print(f"\n  RL Win Rate: {100*rl_wins[-1]/total:.1f}%")
        print(f"  H Win Rate:  {100*h_wins[-1]/total:.1f}%")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    plot_metrics()