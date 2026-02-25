"""Tests for TicTacToe environment and Q-learning agent."""
import numpy as np
import pytest
from app.ml.rl.tictactoe import TicTacToeEnv, QLearningAgent, train_rl_agent


class TestTicTacToeEnv:
    def test_initial_board(self):
        env = TicTacToeEnv()
        state = env.reset()
        assert state == tuple([0] * 9)

    def test_valid_move(self):
        env = TicTacToeEnv()
        env.reset()
        state, reward, done = env.step(0)
        assert state[0] == 1  # player 1 placed
        assert reward == 0
        assert not done

    def test_invalid_move(self):
        env = TicTacToeEnv()
        env.reset()
        env.step(0)
        with pytest.raises(ValueError):
            env.step(0)  # same cell again

    def test_available_actions(self):
        env = TicTacToeEnv()
        env.reset()
        assert env.available_actions() == list(range(9))
        env.step(4)
        assert 4 not in env.available_actions()
        assert len(env.available_actions()) == 8

    def test_win_detection_row(self):
        env = TicTacToeEnv()
        env.reset()
        env.board = [1, 1, 0, -1, -1, 0, 0, 0, 0]
        env.current_player = 1
        _, reward, done = env.step(2)  # complete row
        assert done
        assert reward == 1.0

    def test_win_detection_diagonal(self):
        env = TicTacToeEnv()
        env.reset()
        env.board = [1, 0, 0, 0, 1, 0, 0, 0, 0]
        env.current_player = 1
        _, reward, done = env.step(8)
        assert done
        assert reward == 1.0

    def test_draw(self):
        env = TicTacToeEnv()
        env.reset()
        # Fill board to draw:  X O X / X O O / O X X
        env.board = [1, -1, 1, 1, -1, -1, -1, 1, 0]
        env.current_player = 1
        _, reward, done = env.step(8)
        assert done
        assert reward == 0.5  # draw


class TestQLearningAgent:
    def test_qtable_updates(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.0)
        state = (0,) * 9
        agent.update(state, 0, 1.0, (1,) + (0,) * 8, True)
        assert agent.q_table[state][0] == 0.5  # alpha * reward

    def test_greedy_action(self):
        agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0)
        state = (0,) * 9
        agent.q_table[state] = {4: 10.0}
        action = agent.choose_action(state, list(range(9)))
        assert action == 4  # should pick best

    def test_training_runs(self):
        """Train briefly and ensure history is populated."""
        history = train_rl_agent(episodes=50, alpha=0.3, gamma=0.9, epsilon=0.3)
        assert "episode_rewards" in history
        assert len(history["episode_rewards"]) == 50
