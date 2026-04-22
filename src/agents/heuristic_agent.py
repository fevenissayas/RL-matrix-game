import numpy as np
import random
from typing import Tuple, List, Dict, Any

class HeuristicAgent:
    def __init__(self, strategy: str = "adaptive"):
        self.strategy = strategy
        self.name = "Heuristic"

    def select_action(
        self,
        game_state: Dict[str, Any],
        valid_actions: List[Tuple[int, int, int]]
    ) -> Tuple[int, Tuple[int, int, int]]:
        if not valid_actions:
            return -1, (-1, -1, -1)
        
        sample = random.sample(valid_actions, min(20, len(valid_actions)))
        matrix = game_state['matrix']
        det = abs(game_state.get('determinant', 0))
        
        if self.strategy == "aggressive":
            idx, action = self._aggressive_play(sample, matrix)
        elif self.strategy == "defensive":
            idx, action = self._defensive_play(sample, matrix)
        else:
            idx, action = self._adaptive_play(sample, matrix, det)
        
        return idx, action

    def _aggressive_play(
        self,
        valid_actions: List[Tuple[int, int, int]],
        matrix: np.ndarray
    ) -> Tuple[int, Tuple[int, int, int]]:
        best_score = float('inf')
        best_action = valid_actions[0]
        best_idx = 0
        
        for idx, (row, col, value) in enumerate(valid_actions):
            test_matrix = matrix.copy()
            test_matrix[row, col] = value
            det_value = abs(np.linalg.det(test_matrix.astype(float)))
            score = det_value
            
            if score < best_score:
                best_score = score
                best_action = (row, col, value)
                best_idx = idx
        
        return best_idx, best_action

    def _defensive_play(
        self,
        valid_actions: List[Tuple[int, int, int]],
        matrix: np.ndarray
    ) -> Tuple[int, Tuple[int, int, int]]:
        candidates = []
        
        for idx, (row, col, value) in enumerate(valid_actions):
            test_matrix = matrix.copy()
            test_matrix[row, col] = value
            test_rank = np.linalg.matrix_rank(test_matrix.astype(float))
            current_rank = np.linalg.matrix_rank(matrix.astype(float))
            
            if test_rank >= current_rank:
                candidates.append((idx, row, col, value))
        
        if candidates:
            idx, row, col, value = random.choice(candidates)
            return idx, (row, col, value)
        
        idx = random.randint(0, len(valid_actions) - 1)
        return idx, valid_actions[idx]

    def _adaptive_play(
        self,
        valid_actions: List[Tuple[int, int, int]],
        matrix: np.ndarray,
        det: float
    ) -> Tuple[int, Tuple[int, int, int]]:
        roll = random.random()
        
        if roll < 0.5:
            return self._aggressive_play(valid_actions, matrix)
        elif roll < 0.8:
            return self._defensive_play(valid_actions, matrix)
        else:
            idx = random.randint(0, len(valid_actions) - 1)
            return idx, valid_actions[idx]

    def get_name(self) -> str:
        return self.name