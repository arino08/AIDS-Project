import numpy as np

from .env import SnakeEnv


def evaluate(agent, episodes=50, size=10, seed=12345, policy="greedy"):
    """Play `episodes` games. policy: 'greedy' (trained agent) or 'random'."""
    env = SnakeEnv(size=size, seed=seed)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            a = int(rng.integers(3)) if policy == "random" else agent.act(obs, 0.0)
            obs, _, done, info = env.step(a)
        scores.append(info["score"])
    return np.array(scores)
