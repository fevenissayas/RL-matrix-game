import numpy as np
import random
from typing import Dict, Tuple, List, Optional
import pickle
import os

class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        q_table_file: Optional[str] = None
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[str, np.ndarray] = {}
        self.actions_taken: Dict[str, int] = {}
        self.q_table_file = q_table_file
        
        if q_table_file and os.path.exists(q_table_file):
            self.load_q_table(q_table_file)

    def get_state_representation(self, matrix: np.ndarray) -> str:
        try:
            matrix_float = matrix.astype(float)
            rank = int(np.linalg.matrix_rank(matrix_float))
        except:
            rank = 0
        
        try:
            if rank < matrix.shape[0]:
                cond_bucket = 10
            else:
                cond = np.linalg.cond(matrix_float)
                if cond < 10:
                    cond_bucket = 0
                elif cond < 100:
                    cond_bucket = 1
                elif cond < 1000:
                    cond_bucket = 2
                elif cond < 10000:
                    cond_bucket = 3
                else:
                    cond_bucket = 4
        except:
            cond_bucket = 10
        
        det = np.linalg.det(matrix.astype(float))
        if abs(det) < 1e-9:
            det_bucket = 0
        elif abs(det) < 10:
            det_bucket = 1
        elif abs(det) < 100:
            det_bucket = 2
        elif abs(det) < 1000:
            det_bucket = 3
        else:
            det_bucket = 4
        
        return f"{rank}:{cond_bucket}:{det_bucket}"

    def get_state_key(self, matrix: np.ndarray, move_count: int = None) -> str:
        return self.get_state_representation(matrix)

    def get_q_values(self, state_key: str, num_actions: int) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(160)
            self.actions_taken[state_key] = 0
        else:
            current_size = len(self.q_table[state_key])
            if current_size < num_actions:
                self.q_table[state_key] = np.pad(self.q_table[state_key], (0, num_actions - current_size))
        return self.q_table[state_key]

    def select_action(
        self,
        state_key: str,
        valid_actions: List[Tuple[int, int, int]],
        game_matrix: np.ndarray
    ) -> Tuple[int, Tuple[int, int, int]]:
        if not valid_actions:
            return -1, (-1, -1, -1)
        
        num_actions = len(valid_actions)
        q_values = self.get_q_values(state_key, num_actions)
        
        exploration = random.random() < self.epsilon
        
        if exploration:
            action_idx = random.randint(0, num_actions - 1)
        else:
            valid_q = q_values[:num_actions]
            max_q = np.max(valid_q)
            if max_q <= 0:
                action_idx = random.randint(0, num_actions - 1)
            else:
                best_indices = [i for i, q in enumerate(valid_q) if q == max_q]
                action_idx = random.choice(best_indices)
        
        return action_idx, valid_actions[action_idx]

    def update_q_value(
        self,
        state_key: str,
        action_idx: int,
        reward: float,
        next_state_key: str,
        next_valid_actions: List[Tuple[int, int, int]],
        game_over: bool
    ):
        q_values = self.get_q_values(state_key, len(self.q_table[state_key]))
        
        current_q = q_values[action_idx]
        
        if game_over:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state_key, len(next_valid_actions))
            max_next_q = np.max(next_q_values)
            target = reward + self.gamma * max_next_q
        
        new_q = current_q + self.alpha * (target - current_q)
        q_values[action_idx] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_action_stats(self, state_key: str) -> int:
        return self.actions_taken.get(state_key, 0)

    def save_q_table(self, filepath: str = None):
        target = filepath or self.q_table_file
        if target:
            with open(target, 'wb') as f:
                pickle.dump(self.q_table, f)

    def load_q_table(self, filepath: str = None):
        target = filepath or self.q_table_file
        if target and os.path.exists(target):
            with open(target, 'rb') as f:
                self.q_table = pickle.load(f)

    def get_best_action(
        self,
        state_key: str,
        valid_actions: List[Tuple[int, int, int]]
    ) -> Tuple[int, Tuple[int, int, int]]:
        q_values = self.get_q_values(state_key, len(valid_actions))
        max_q = np.max(q_values)
        
        if max_q == 0:
            action_idx = random.randint(0, len(valid_actions) - 1)
        else:
            best_indices = [i for i, q in enumerate(q_values) if q == max_q]
            action_idx = random.choice(best_indices)
        
        return action_idx, valid_actions[action_idx]