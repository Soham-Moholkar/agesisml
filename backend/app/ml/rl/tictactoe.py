"""Reinforcement Learning – Q-learning TicTacToe."""
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Optional

from app.core.config import MODELS_DIR, ARTIFACTS_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TicTacToeEnv:
    """Tic-Tac-Toe environment.
    Board: 9 cells. 0=empty, 1=X, 2=O.
    Agent plays as O (2), opponent is X (1).
    """

    def __init__(self):
        self.board = [0] * 9
        self.done = False
        self.winner = None  # None, 1, 2, or 0 (draw)

    def reset(self):
        self.board = [0] * 9
        self.done = False
        self.winner = None
        return tuple(self.board)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action, player):
        """Make a move. Returns (state, reward, done)."""
        if self.board[action] != 0 or self.done:
            return tuple(self.board), -10, True  # illegal move
        self.board[action] = player
        if self._check_win(player):
            self.done = True
            self.winner = player
            reward = 1 if player == 2 else -1  # agent is 2
            return tuple(self.board), reward, True
        if len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0
            return tuple(self.board), 0.5, True  # draw
        return tuple(self.board), 0, False

    def _check_win(self, player):
        b = self.board
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
            [0, 4, 8], [2, 4, 6],              # diagonals
        ]
        return any(all(b[i] == player for i in line) for line in lines)

    def copy(self):
        env = TicTacToeEnv()
        env.board = list(self.board)
        env.done = self.done
        env.winner = self.winner
        return env


class QLearningAgent:
    """Tabular Q-learning agent for Tic-Tac-Toe."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.99995, epsilon_min=0.01):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        best = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return np.random.choice(best)

    def update(self, state, action, reward, next_state, next_available, done):
        old_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            future_q = max([self.get_q(next_state, a) for a in next_available]) if next_available else 0
            target = reward + self.gamma * future_q
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def random_opponent_move(env):
    """Random opponent (player 1 = X)."""
    actions = env.available_actions()
    if not actions:
        return None
    action = np.random.choice(actions)
    return action


def train_rl_agent(episodes=50000, alpha=0.1, gamma=0.9, epsilon=1.0,
                   epsilon_decay=0.99995, epsilon_min=0.01, seed=42):
    """Train Q-learning agent. Returns agent + training stats."""
    np.random.seed(seed)
    agent = QLearningAgent(alpha, gamma, epsilon, epsilon_decay, epsilon_min)
    env = TicTacToeEnv()

    stats = {"wins": 0, "losses": 0, "draws": 0}
    history = []  # (episode, win_rate, draw_rate, loss_rate) every 1000

    for ep in range(episodes):
        state = env.reset()

        # Random who goes first
        agent_first = np.random.random() < 0.5

        if not agent_first:
            opp_action = random_opponent_move(env)
            if opp_action is not None:
                state, _, done = env.step(opp_action, 1)
                if done:
                    stats["losses" if env.winner == 1 else "draws"] += 1
                    if (ep + 1) % 1000 == 0:
                        total = ep + 1
                        history.append({
                            "episode": total,
                            "win_rate": stats["wins"] / total,
                            "draw_rate": stats["draws"] / total,
                            "loss_rate": stats["losses"] / total,
                        })
                    agent.decay_epsilon()
                    continue

        done = False
        while not done:
            # Agent move (player 2 = O)
            avail = env.available_actions()
            if not avail:
                break
            action = agent.choose_action(state, avail)
            next_state, reward, done = env.step(action, 2)

            if done:
                agent.update(state, action, reward, next_state, [], True)
                if env.winner == 2:
                    stats["wins"] += 1
                elif env.winner == 1:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
                break

            # Opponent move
            opp_action = random_opponent_move(env)
            if opp_action is None:
                agent.update(state, action, 0.5, next_state, [], True)
                stats["draws"] += 1
                break
            next_state2, opp_reward, done2 = env.step(opp_action, 1)

            if done2:
                r = -1 if env.winner == 1 else 0.5
                agent.update(state, action, r, next_state2, [], True)
                if env.winner == 1:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
                done = True
            else:
                agent.update(state, action, 0, next_state2, env.available_actions(), False)
                state = next_state2

        agent.decay_epsilon()

        if (ep + 1) % 1000 == 0:
            total = ep + 1
            history.append({
                "episode": total,
                "win_rate": stats["wins"] / total,
                "draw_rate": stats["draws"] / total,
                "loss_rate": stats["losses"] / total,
            })

    return agent, stats, history


def save_rl_agent(run_id: str, agent: QLearningAgent, stats: dict, history: list):
    """Save agent Q-table and training artifacts."""
    model_path = MODELS_DIR / f"{run_id}_rl.pkl"
    with open(str(model_path), "wb") as f:
        pickle.dump({"q_table": agent.q_table, "stats": stats}, f)

    artifact_dir = ARTIFACTS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Training curve
    if history:
        eps = [h["episode"] for h in history]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, [h["win_rate"] for h in history], label="Win Rate", color="green")
        ax.plot(eps, [h["draw_rate"] for h in history], label="Draw Rate", color="blue")
        ax.plot(eps, [h["loss_rate"] for h in history], label="Loss Rate", color="red")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Rate")
        ax.set_title("Q-Learning Training Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(artifact_dir / "training_curve.png"), dpi=100)
        plt.close(fig)

    # Save history JSON
    with open(str(artifact_dir / "training_history.json"), "w") as f:
        json.dump(history, f)

    return str(model_path)


def load_rl_agent(run_id: str):
    """Load a trained Q-learning agent."""
    model_path = MODELS_DIR / f"{run_id}_rl.pkl"
    with open(str(model_path), "rb") as f:
        data = pickle.load(f)
    agent = QLearningAgent()
    agent.q_table = data["q_table"]
    agent.epsilon = 0  # no exploration during play
    return agent


def agent_move(agent: QLearningAgent, board: list):
    """Get agent's move given current board state."""
    state = tuple(board)
    available = [i for i in range(9) if board[i] == 0]
    if not available:
        return None
    return agent.choose_action(state, available, training=False)
