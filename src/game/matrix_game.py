import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math

class MatrixGame:
    def __init__(self, size: int = 4):
        self.size = size
        self.matrix = np.zeros((size, size), dtype=int)
        self.available_numbers: List[int] = list(range(1, 11))
        self.current_player = 0
        self.move_count = 0
        self.history: List[Dict[str, Any]] = []
        self.history_stack: List[Dict[str, Any]] = []
        self.game_over = False
        self.winner: Optional[int] = None
        self.scores: Dict[int, float] = {0: 0.0, 1: 0.0}
        self.last_reward: float = 0.0
        self.last_rank_penalty: bool = False
        self.init_determinant = 0.0
        self.init_rank = size
        self._initialize_board()

    def _initialize_board(self):
        num_filled = int(0.10 * self.size * self.size)
        num_filled = max(2, min(num_filled, 2))
        
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            self.matrix = np.zeros((self.size, self.size), dtype=int)
            
            rows_used = set()
            cols_used = set()
            filled_count = 0
            
            while filled_count < num_filled:
                row = np.random.randint(0, self.size)
                col = np.random.randint(0, self.size)
                if row not in rows_used and col not in cols_used:
                    rows_used.add(row)
                    cols_used.add(col)
                    value = np.random.choice([i for i in self.available_numbers if i != 0])
                    self.matrix[row, col] = value
                    filled_count += 1
            
            det = abs(self.compute_determinant())
            if det > 1e-3:
                break
            attempts += 1
        
        self.init_determinant = abs(self.compute_determinant())
        self.init_rank = self.compute_rank()
        self._save_state_snapshot()
        self._push_to_history_stack()

    def _save_state_snapshot(self):
        snapshot = {
            'matrix': self.matrix.copy(),
            'move_count': self.move_count,
            'current_player': self.current_player
        }
        self.history.append(snapshot)

    def reset(self):
        self.matrix = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 0
        self.move_count = 0
        self.history = []
        self.history_stack = []
        self.game_over = False
        self.winner = None
        self.scores = {0: 0.0, 1: 0.0}
        self.last_reward = 0.0
        self.last_rank_penalty = False
        self._initialize_board()

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        empty = []
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i, j] == 0:
                    empty.append((i, j))
        return empty

    def is_full(self) -> bool:
        return len(self.get_empty_cells()) == 0

    def compute_determinant(self) -> float:
        try:
            return float(np.linalg.det(self.matrix.astype(float)))
        except:
            return 0.0

    def compute_condition(self) -> float:
        try:
            matrix_float = self.matrix.astype(float)
            if np.linalg.matrix_rank(matrix_float) < self.size:
                return float('inf')
            return np.linalg.cond(matrix_float)
        except:
            return float('inf')

    def check_singular(self, threshold: float = 1e-6) -> bool:
        det = abs(self.compute_determinant())
        return det < threshold

    def check_rank_deficiency(self) -> bool:
        matrix_float = self.matrix.astype(float)
        rank = np.linalg.matrix_rank(matrix_float)
        return rank < self.size

    def compute_rank(self) -> int:
        matrix_float = self.matrix.astype(float)
        return int(np.linalg.matrix_rank(matrix_float))

    def check_zeros_row(self) -> bool:
        for i in range(self.size):
            if np.all(self.matrix[i, :] == 0):
                return True
        return False

    def is_singular(self, threshold: float = 1e-9) -> bool:
        det = self.compute_determinant()
        return math.isclose(det, 0, abs_tol=threshold)

    def is_condition_singular(self, threshold: float = 1e10) -> bool:
        cond = self.compute_condition()
        return cond > threshold or np.isinf(cond)

    def get_singularity_score(self) -> float:
        det = abs(self.compute_determinant())
        if self.is_singular():
            return 1.0
        
        init_det = max(self.init_determinant, 1.0)
        ratio = det / init_det
        return 1.0 - min(ratio, 1.0)

    def _push_to_history_stack(self):
        snapshot = {
            'matrix': self.matrix.copy(),
            'move_count': self.move_count,
            'current_player': self.current_player,
            'scores': self.scores.copy()
        }
        self.history_stack.append(snapshot)

    def undo_move(self, steps: int = 1) -> bool:
        if len(self.history_stack) <= steps:
            return False
        
        for _ in range(steps):
            self.history_stack.pop()
        
        old_state = self.history_stack[-1]
        self.matrix = old_state['matrix'].copy()
        self.move_count = old_state['move_count']
        self.current_player = old_state['current_player']
        self.scores = old_state['scores'].copy()
        self.game_over = False
        self.winner = None
        return True

    def make_move(self, row: int, col: int, value: int) -> Tuple[bool, float]:
        if self.matrix[row, col] != 0:
            return False, 0.0
        
        if value not in self.available_numbers:
            return False, 0.0

        prev_rank = self.compute_rank()
        prev_was_healthy = (prev_rank == self.size)

        self.matrix[row, col] = value
        self.move_count += 1
        
        reward = -1.0
        rank_penalty = False
        
        new_rank = self.compute_rank()
        
        if (self.is_singular() or self.is_condition_singular()) and self.move_count > 4 and prev_was_healthy:
            self.scores[self.current_player] += 50.0
            reward = 50.0
            self.game_over = True
        elif new_rank < prev_rank:
            self.scores[self.current_player] += -10.0
            reward = -10.0
            rank_penalty = True
        
        self.scores[self.current_player] += reward
        self.last_reward = reward
        self.last_rank_penalty = rank_penalty
        
        if self.is_full():
            self.scores[self.current_player] += 100.0
            reward += 100.0
            self.game_over = True
        
        if self.game_over:
            self._determine_winner_by_scores()
        
        self._save_state_snapshot()
        self._push_to_history_stack()
        
        self.current_player = 1 - self.current_player
        
        return True, reward, rank_penalty

    def _determine_winner_by_scores(self):
        if self.scores[0] > self.scores[1]:
            self.winner = 0
        elif self.scores[1] > self.scores[0]:
            self.winner = 1
        else:
            det = abs(self.compute_determinant())
            if det < 1e-3:
                self.winner = self.current_player
            else:
                self.winner = 1 - self.current_player

    def get_state_hash(self) -> str:
        return str(self.matrix.tobytes())

    def get_valid_actions(self) -> List[Tuple[int, int, int]]:
        actions = []
        empty_cells = self.get_empty_cells()
        for row, col in empty_cells:
            for value in self.available_numbers:
                actions.append((row, col, value))
        return actions

    def apply_action_with_reward(self, action: Tuple[int, int, int]) -> Tuple[bool, float]:
        row, col, value = action
        return self.make_move(row, col, value)

    def traceback(self, steps: int = 1):
        if len(self.history) > steps:
            old_state = self.history[-(steps + 1)]
            self.matrix = old_state['matrix'].copy()
            self.move_count = old_state['move_count']
            self.current_player = old_state['current_player']
            self.history = self.history[:-(steps)]
            self.game_over = False
            self.winner = None

    def get_game_state(self) -> Dict[str, Any]:
        return {
            'matrix': self.matrix.copy(),
            'current_player': self.current_player,
            'move_count': self.move_count,
            'determinant': self.compute_determinant(),
            'condition': self.compute_condition(),
            'is_singular': self.check_singular(),
            'is_rank_deficient': self.check_rank_deficiency(),
            'rank': self.compute_rank(),
            'is_full': self.is_full(),
            'game_over': self.game_over,
            'winner': self.winner,
            'singularity_score': self.get_singularity_score(),
            'scores': self.scores.copy()
        }

    def print_board(self):
        print("\nCurrent Matrix:")
        print(self.matrix)
        print(f"\nDeterminant: {self.compute_determinant():.4f}")
        print(f"Condition: {self.compute_condition():.4f}")
        print(f"Rank: {self.compute_rank()}")
        print(f"Singularity Score: {self.get_singularity_score():.4f}")
        print(f"Scores - Player 0: {self.scores[0]:.2f}, Player 1: {self.scores[1]:.2f}")
        print(f"Move: {self.move_count}, Current Player: {self.current_player}")
        if self.check_rank_deficiency():
            print("WARNING: Rank deficiency detected!")
        if self.is_singular():
            print("WARNING: Matrix is singular!")