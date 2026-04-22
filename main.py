import argparse
import numpy as np
import random
from typing import Dict, Any, List, Tuple
import os

from src.game.matrix_game import MatrixGame
from src.agents.q_learning_agent import QLearningAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.training_utils import MetricsTracker


class GameRunner:
    def __init__(self, args):
        self.args = args
        self.game = MatrixGame()
        self.rl_agent = QLearningAgent(
            alpha=0.1,
            gamma=0.9,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            q_table_file="q_table.pkl" if args.save_qtable else None
        )
        self.heuristic_agent = HeuristicAgent(strategy="adaptive")
        self.stats = {
            'rl_wins': 0,
'heuristic_wins': 0,
            'ties': 0,
            'singular_games': 0,
            'full_games': 0
        }
        self.metrics = MetricsTracker(window_size=100)

    def train(self):
        print(f"Training RL agent for {self.args.episodes} episodes...")
        
        for episode in range(self.args.episodes):
            self.game.reset()
            episode_reward = 0.0
            
            state_keys = {0: None, 1: None}
            actions = {0: None, 1: None}
            
            turn = 0
            while not self.game.game_over:
                current_player = self.game.current_player
                valid_actions = self.game.get_valid_actions()
                game_state = self.game.get_game_state()
                
                if current_player == 0:
                    state_key = self.rl_agent.get_state_key(self.game.matrix)
                    state_keys[current_player] = state_key
                    
                    action_idx, action = self.rl_agent.select_action(
                        state_key,
                        valid_actions,
                        self.game.matrix
                    )
                    actions[current_player] = action_idx
                else:
                    action_idx, action = self.heuristic_agent.select_action(
                        game_state,
                        valid_actions
                    )
                
                self.game.make_move(row=action[0], col=action[1], value=action[2])
                
                if self.game.last_rank_penalty and current_player == 0:
                    self.game.undo_move(1)
                    continue
                
                if current_player == 0:
                    next_state_key = self.rl_agent.get_state_key(self.game.matrix)
                    next_valid_actions = self.game.get_valid_actions()
                    
                    self.rl_agent.update_q_value(
                        state_key,
                        action_idx,
                        self.game.last_reward,
                        next_state_key,
                        next_valid_actions,
                        self.game.game_over
                    )
                    episode_reward += self.game.last_reward
                
                if self.game.game_over:
                    if self.game.winner == 0:
                        self.stats['rl_wins'] += 1
                        if self.game.is_singular():
                            self.stats['singular_games'] += 1
                    elif self.game.winner == 1:
                        self.stats['heuristic_wins'] += 1
                    else:
                        self.stats['ties'] += 1
                    
                    if self.game.is_full():
                        self.stats['full_games'] += 1
            
            self.metrics.record(
                episode=episode,
                reward=episode_reward,
                epsilon=self.rl_agent.epsilon,
                rl_win=(self.game.winner == 0),
                h_win=(self.game.winner == 1),
                tie=(self.game.winner is None)
            )
            
            self.rl_agent.decay_epsilon()
            
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{self.args.episodes} | "
                      f"RL Wins: {self.stats['rl_wins']} | "
                      f"H Wins: {self.stats['heuristic_wins']} | "
                      f"Ties: {self.stats['ties']} | "
                      f"Epsilon: {self.rl_agent.epsilon:.4f}")
        
        if self.args.save_qtable:
            self.rl_agent.save_q_table()
            print(f"\nQ-table saved to q_table.pkl")
        
        self.metrics.plot_ascii()
        self.metrics.save_csv("training_metrics.csv")
        self.metrics.print_summary()
        
        self._print_training_stats()
        
        if self.args.save_qtable:
            self.rl_agent.save_q_table()
            print(f"\nQ-table saved to q_table.pkl")
        
        self._print_training_stats()

    def _print_training_stats(self):
        total = sum(self.stats.values())
        print("\n=== Training Complete ===")
        print(f"Total Games: {total}")
        print(f"RL Wins: {self.stats['rl_wins']} ({100*self.stats['rl_wins']/total:.1f}%)")
        print(f"Heuristic Wins: {self.stats['heuristic_wins']} ({100*self.stats['heuristic_wins']/total:.1f}%)")
        print(f"Ties: {self.stats['ties']} ({100*self.stats['ties']/total:.1f}%)")
        print(f"Singular Games: {self.stats['singular_games']}")
        print(f"Full Board Games: {self.stats['full_games']}")

    def play(self, num_games: int = 100):
        print(f"Playing {num_games} games with trained agent...")
        
        self.game = MatrixGame()
        self.rl_agent = QLearningAgent(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            q_table_file="q_table.pkl" if os.path.exists("q_table.pkl") else None
        )
        
        if not self.rl_agent.q_table:
            print("Warning: No trained Q-table found, using random agent")
        
        for episode in range(num_games):
            self.game.reset()
            
            while not self.game.game_over:
                current_player = self.game.current_player
                valid_actions = self.game.get_valid_actions()
                game_state = self.game.get_game_state()
                
                if current_player == 0:
                    state_key = self.rl_agent.get_state_key(self.game.matrix)
                    
                    _, action = self.rl_agent.get_best_action(
                        state_key,
                        valid_actions
                    )
                else:
                    _, action = self.heuristic_agent.select_action(
                        game_state,
valid_actions
                     )
                
                self.game.make_move(row=action[0], col=action[1], value=action[2])
            
            if self.game.winner == 0:
                self.stats['rl_wins'] += 1
            elif self.game.winner == 1:
                self.stats['heuristic_wins'] += 1
            else:
                self.stats['ties'] += 1
            
            if (episode + 1) % 10 == 0:
                print(f"Games: {episode + 1}/{num_games} | "
                      f"RL: {self.stats['rl_wins']} | "
                      f"H: {self.stats['heuristic_wins']} | "
                      f"Ties: {self.stats['ties']}")
        
        self._print_training_stats()

    def demo(self, num_demos: int = 5):
        print(f"Running {num_demos} demo games...")
        
        self.game = MatrixGame()
        self.rl_agent = QLearningAgent(
            alpha=0.1,
            gamma=0.9,
            epsilon=0.3,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        for demo in range(num_demos):
            print(f"\n{'='*50}")
            print(f"Demo Game {demo + 1}")
            print('='*50)
            
            self.game.reset()
            self.game.print_board()
            
            move_num = 0
            while not self.game.game_over:
                current_player = self.game.current_player
                valid_actions = self.game.get_valid_actions()
                game_state = self.game.get_game_state()
                
                if current_player == 0:
                    state_key = self.rl_agent.get_state_key(self.game.matrix)
                    
                    action_idx, action = self.rl_agent.select_action(
                        state_key,
                        valid_actions,
                        self.game.matrix
                    )
                    player_name = "RL Agent"
                else:
                    action_idx, action = self.heuristic_agent.select_action(
                        game_state,
                        valid_actions
                    )
                    player_name = "Heuristic"
                
                move_num += 1
                print(f"\nMove {move_num} - {player_name}: row={action[0]}, col={action[1]}, value={action[2]}")
                
                self.game.make_move(row=action[0], col=action[1], value=action[2])
                self.game.print_board()
                
                if self.args.verbose:
                    input("Press Enter to continue...")
            
            if self.game.winner == 0:
                print(f"\n*** RL Agent Wins! ***")
            elif self.game.winner == 1:
                print(f"\n*** Heuristic Agent Wins! ***")
            else:
                print(f"\n*** Tie! ***")


def main():
    parser = argparse.ArgumentParser(description="RL Matrix Game")
    parser.add_argument('--train', action='store_true', help='Train the RL agent')
    parser.add_argument('--play', action='store_true', help='Play with trained agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--save-qtable', action='store_true', help='Save Q-table to file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output for demos')
    parser.add_argument('--demo', action='store_true', help='Run demo game')
    
    args = parser.parse_args()
    
    if args.train:
        runner = GameRunner(args)
        runner.train()
    elif args.play:
        runner = GameRunner(args)
        runner.play(args.episodes)
    else:
        args.demo = True
        runner = GameRunner(args)
        runner.demo(3)


if __name__ == "__main__":
    main()