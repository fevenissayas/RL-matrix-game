# RL Matrix Game

A reinforcement learning game where a Q-Learning agent competes against a Heuristic agent to manipulate a 4x4 matrix.

## The Game

Players take turns placing numbers (1-10) in empty cells. The goal is to make the matrix **singular** (determinant = 0 or rank-deficient).

### Win Conditions
- **Singularity**: det(A) = 0 or cond(A) = ∞ (after move 4)
- **Board Full**: All 16 cells filled
- **Winner**: Agent with highest accumulated score

### Scoring
| Action | Reward |
|--------|--------|
| Make move | -1 |
| Cause singularity | +50 |
| Cause rank deficiency | -10 (traceback) |
| Fill board | +100 |

## Quick Start

```bash
# Train the RL agent
python3 main.py --train --episodes 5000 --save-qtable

# Play with trained agent
python3 main.py --play

# Demo (random moves)
python3 main.py --demo
```

## Commands

| Command | Description |
|---------|------------|
| `--train` | Train the RL agent |
| `--play` | Play with trained agent |
| `--episodes N` | Number of episodes |
| `--save-qtable` | Save Q-table to file |
| `--demo` | Watch demo games |

## Files

```
main.py              - CLI trainer
q_table.pkl          - Trained Q-learning model
training_metrics.csv  - Training progress log
plot_metrics.py     - Visualize metrics
notebook.ipynb      - Jupyter analysis
src/game/          - Game logic
src/agents/        - RL and Heuristic agents
```

## Technical Details

**Q-Learning:**
- α (alpha) = 0.1 - learning rate
- γ (gamma) = 0.9 - discount factor
- ε (epsilon) = 1.0 → 0.01 - exploration (decays 0.995/ep)

**State Abstraction:**
Uses feature binning: `rank:cond_bucket:det_bucket`
- Reduces 11^16 states to ~16 states for fast learning

## Requirements

```
numpy>=1.21.0
```