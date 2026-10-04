"""Snake environment with a Gym-like API (reset / step) and no external dependencies.

Actions are *relative* to the snake's heading, which makes the policy
rotation-invariant and easier to learn:
    0 = go straight, 1 = turn right, 2 = turn left
"""
from collections import deque

import numpy as np

# (dx, dy) in screen coordinates: y grows downward.
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
N_ACTIONS = 3
OBS_DIM = 14

R_FOOD = 10.0
R_DEATH = -10.0
R_CLOSER = 0.1   # shaping: step towards food
R_FARTHER = -0.1  # shaping: step away from food


class SnakeEnv:
    def __init__(self, size=10, shaping=True, seed=None, starve_factor=100):
        self.size = size
        self.shaping = shaping
        self.starve_factor = starve_factor
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ core
    def reset(self):
        c = self.size // 2
        self.direction = 1  # heading right
        self.snake = deque([(c, c), (c - 1, c), (c - 2, c)])  # head first
        self.body = set(self.snake)
        self.score = 0
        self.steps = 0
        self.since_food = 0
        self.done = False
        self._place_food()
        return self.observe()

    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.body]
        if not free:
            self.food = None
            return
        self.food = free[self.rng.integers(len(free))]

    def step(self, action):
        assert not self.done, "call reset() first"
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        hx, hy = self.snake[0]
        dx, dy = DIRS[self.direction]
        new = (hx + dx, hy + dy)
        self.steps += 1
        self.since_food += 1

        eating = new == self.food
        # The tail vacates its cell this step unless we grow, so it is safe to enter.
        tail = self.snake[-1]
        blocked = self.body if eating else self.body - {tail}
        if not self._inside(new) or new in blocked:
            self.done = True
            return self.observe(), R_DEATH, True, {"score": self.score, "reason": "collision"}

        old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
        self.snake.appendleft(new)
        self.body.add(new)
        if eating:
            self.score += 1
            self.since_food = 0
            reward = R_FOOD
            self._place_food()
            if self.food is None:  # board full: perfect game
                self.done = True
                return self.observe(), reward, True, {"score": self.score, "reason": "win"}
        else:
            self.body.discard(self.snake.pop())
            self.body.add(new)
            reward = 0.0
            if self.shaping:
                new_dist = abs(new[0] - self.food[0]) + abs(new[1] - self.food[1])
                reward = R_CLOSER if new_dist < old_dist else R_FARTHER

        if self.since_food > self.starve_factor * len(self.snake):
            self.done = True
            return self.observe(), R_DEATH, True, {"score": self.score, "reason": "starved"}
        return self.observe(), reward, False, {"score": self.score}

    # ------------------------------------------------------------ observation
    def _inside(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def _open(self, p):
        return self._inside(p) and p not in self.body

    def _reachable(self, start):
        """Number of empty cells reachable from `start` (flood fill)."""
        if not self._open(start):
            return 0
        seen = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRS:
                n = (x + dx, y + dy)
                if n not in seen and self._open(n):
                    seen.add(n)
                    stack.append(n)
        return len(seen)

    def observe(self):
        hx, hy = self.snake[0]
        d = self.direction
        ahead = [(d + k) % 4 for k in (0, 1, 3)]  # straight, right, left
        cells = [(hx + DIRS[a][0], hy + DIRS[a][1]) for a in ahead]
        danger = [0.0 if self._open(c) else 1.0 for c in cells]
        heading = [1.0 if d == i else 0.0 for i in range(4)]
        fx, fy = self.food if self.food else (hx, hy)
        food = [fx < hx, fx > hx, fy < hy, fy > hy]  # left, right, up, down
        free_total = self.size * self.size - len(self.snake)
        space = [self._reachable(c) / max(free_total, 1) for c in cells]
        return np.array(danger + heading + [float(f) for f in food] + space, dtype=np.float32)
