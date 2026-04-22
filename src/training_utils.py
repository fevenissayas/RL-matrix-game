import numpy as np
from typing import List

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

class MetricsTracker:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.epsilon_history: List[float] = []
        
        self.rl_wins: List[int] = []
        self.heuristic_wins: List[int] = []
        self.ties: List[int] = []
        
        self.cumulative_rl_wins = 0
        self.cumulative_heuristic_wins = 0
        self.cumulative_ties = 0
    
    def record(self, episode: int, reward: float, epsilon: float, rl_win: bool, h_win: bool, tie: bool):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.epsilon_history.append(epsilon)
        
        if rl_win:
            self.cumulative_rl_wins += 1
        elif h_win:
            self.cumulative_heuristic_wins += 1
        else:
            self.cumulative_ties += 1
        
        self.rl_wins.append(self.cumulative_rl_wins)
        self.heuristic_wins.append(self.cumulative_heuristic_wins)
        self.ties.append(self.cumulative_ties)
    
    def get_rolling_win_rate(self) -> List[float]:
        window = self.window_size
        if len(self.rl_wins) < window:
            return [0.0] * len(self.rl_wins)
        
        rolling = []
        for i in range(len(self.rl_wins)):
            if i < window:
                rolling.append(self.rl_wins[i] / (i + 1))
            else:
                rolling.append((self.rl_wins[i] - self.rl_wins[i - window]) / window)
        
        return rolling
    
    def plot_ascii(self):
        print("\n" + "="*60)
        print("TRAINING METRICS (ASCII Visualization)")
        print("="*60)
        
        print("\n[1] REWARD vs EPISODE (last 50)")
        if len(self.rewards) > 0:
            r = self.rewards[-50:] if len(self.rewards) > 50 else self.rewards
            max_r, min_r = max(r) if r else 0, min(r) if r else 0
            range_r = max_r - min_r if max_r != min_r else 1
            for val in r:
                bar_len = int(((val - min_r) / range_r) * 40)
                print("|" + "="*bar_len + " "*(40-bar_len) + f"] {val:.1f}")
        
        print("\n[2] WIN RATE (rolling 100)")
        rolling = self.get_rolling_win_rate()
        if rolling:
            recent = rolling[-50:] if len(rolling) > 50 else rolling
            for val in recent:
                bar_len = int(val * 40)
                print("|" + "#"*bar_len + " "*(40-bar_len) + f"] {val*100:.1f}%")
        
        print("\n[3] EPSILON DECAY")
        if self.epsilon_history:
            e = self.epsilon_history[-50:] if len(self.epsilon_history) > 50 else self.epsilon_history
            for val in e:
                bar_len = int(val * 40)
                print("|" + "*"*bar_len + " "*(40-bar_len) + f"] {val:.4f}")
        
        print("\n" + "="*60)
    
    def plot_all(self, save_path: str = "training_performance.png"):
        if not HAS_MATPLOTLIB:
            self.plot_ascii()
            return save_path
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RL Agent Training Performance', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.plot(self.episodes, self.rewards, color='#2E86AB', alpha=0.6, linewidth=0.8)
        if len(self.rewards) > 20:
            smoothed = np.convolve(self.rewards, np.ones(20)/20, mode='valid')
            ax1.plot(range(19, len(self.rewards)), smoothed, color='#E94F37', linewidth=2, label='20-ep moving avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward vs Episode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        rolling_rate = self.get_rolling_win_rate()
        ax2.plot(self.episodes, rolling_rate, color='#44AF69', linewidth=1.5)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate (rolling 100)')
        ax2.set_title('RL Agent Win Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(self.episodes, self.epsilon_history, color='#F18F01', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Epsilon Decay')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        window = max(1, len(self.episodes) // 20)
        binned_penalties = []
        binned_episodes = []
        for i in range(0, len(self.episodes), window):
            binned_episodes.append(self.episodes[i])
            wins_in_window = self.rl_wins[min(i+window, len(self.rl_wins)-1)] - self.rl_wins[i-1] if i > 0 else self.rl_wins[min(i+window, len(self.rl_wins)-1)]
            binned_penalties.append(wins_in_window)
        ax4.bar(binned_episodes, binned_penalties, width=window*0.8, color='#44AF69', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('RL Wins')
        ax4.set_title('RL Wins per Episode Block')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Metrics saved to {save_path}")
        
        return save_path
    
    def save_csv(self, filename: str = "training_metrics.csv"):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'epsilon', 'rl_wins', 'heuristic_wins', 'ties', 'rolling_win_rate'])
            rolling = self.get_rolling_win_rate()
            for i in range(len(self.episodes)):
                writer.writerow([
                    self.episodes[i],
                    self.rewards[i],
                    self.epsilon_history[i],
                    self.rl_wins[i],
                    self.heuristic_wins[i],
                    self.ties[i],
                    rolling[i] if i < len(rolling) else 0
                ])
        print(f"Metrics saved to {filename}")
    
    def print_summary(self):
        total = sum([self.cumulative_rl_wins, self.cumulative_heuristic_wins, self.cumulative_ties])
        if total == 0:
            return
        
        print("\n" + "="*50)
        print("TRAINING METRICS SUMMARY")
        print("="*50)
        print(f"Total Episodes: {len(self.episodes)}")
        print(f"RL Wins: {self.cumulative_rl_wins} ({100*self.cumulative_rl_wins/total:.1f}%)")
        print(f"Heuristic Wins: {self.cumulative_heuristic_wins} ({100*self.cumulative_heuristic_wins/total:.1f}%)")
        print(f"Ties: {self.cumulative_ties} ({100*self.cumulative_ties/total:.1f}%)")
        if self.epsilon_history:
            print(f"Final Epsilon: {self.epsilon_history[-1]:.4f}")
        print("="*50)