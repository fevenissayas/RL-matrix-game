#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from src.game.matrix_game import MatrixGame
from src.agents.q_learning_agent import QLearningAgent
from src.agents.heuristic_agent import HeuristicAgent

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

game = MatrixGame()
rl_agent = QLearningAgent(
    alpha=0.1, gamma=0.9, epsilon=0.0,
    epsilon_min=0.0, epsilon_decay=1.0,
    q_table_file="q_table.pkl" if os.path.exists("q_table.pkl") else None
)
heuristic_agent = HeuristicAgent(strategy="adaptive")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    global game
    game = MatrixGame()
    return jsonify({
        'status': 'ok',
        'matrix': game.matrix.tolist(),
        'move_count': game.move_count,
        'current_player': game.current_player
    })

@app.route('/api/state', methods=['GET'])
def state():
    condition_value = game.compute_condition()
    if condition_value > 1e10 or condition_value == float('inf') or condition_value != condition_value:
        condition = "inf"
    else:
        condition = round(condition_value, 4)
    return jsonify({
        'matrix': game.matrix.tolist(),
        'determinant': round(game.compute_determinant(), 4),
        'condition': condition,
        'rank': game.compute_rank(),
        'singularity_score': round(game.get_singularity_score(), 4),
        'is_singular': game.is_singular(),
        'is_full': game.is_full(),
        'game_over': game.game_over,
        'winner': game.winner,
        'scores': game.scores,
        'current_player': game.current_player,
        'move_count': game.move_count
    })

@app.route('/api/move', methods=['POST'])
def move():
    if game.game_over:
        return jsonify({'error': 'Game is over', 'winner': game.winner})
    
    player = game.current_player
    valid_actions = game.get_valid_actions()
    
    if player == 0:
        state_key = rl_agent.get_state_key(game.matrix)
        _, action = rl_agent.select_action(state_key, valid_actions, game.matrix)
    else:
        _, action = heuristic_agent.select_action(game.get_game_state(), valid_actions)
    
    game.make_move(action[0], action[1], action[2])
    
    return jsonify({
        'success': True,
        'matrix': game.matrix.tolist(),
        'move': action,
        'player': player,
        'game_over': game.game_over,
        'winner': game.winner,
        'scores': game.scores,
        'current_player': game.current_player,
        'determinant': round(game.compute_determinant(), 4),
        'rank': game.compute_rank()
    })

@app.route('/api/auto', methods=['POST'])
def auto():
    while not game.game_over:
        valid_actions = game.get_valid_actions()
        player = game.current_player
        
        if player == 0:
            state_key = rl_agent.get_state_key(game.matrix)
            _, action = rl_agent.select_action(state_key, valid_actions, game.matrix)
        else:
            _, action = heuristic_agent.select_action(game.get_game_state(), valid_actions)
        
        game.make_move(action[0], action[1], action[2])
    
    return jsonify({
        'matrix': game.matrix.tolist(),
        'game_over': game.game_over,
        'winner': game.winner,
        'scores': game.scores
    })

if __name__ == '__main__':
    print("=" * 50)
    print("RL Matrix Game Web Interface")
    print("=" * 50)
    print("Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000)