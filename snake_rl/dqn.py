"""Double-DQN agent implemented in pure NumPy (MLP + Adam + replay buffer)."""
import numpy as np


class MLP:
    """Fully connected net: in -> hidden -> hidden -> out, ReLU activations."""

    def __init__(self, sizes, rng):
        self.params = {}
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
            self.params[f"W{i}"] = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
            self.params[f"b{i}"] = np.zeros(b, dtype=np.float32)
        self.n_layers = len(sizes) - 1

    def forward(self, x, cache=False):
        acts = [x]
        for i in range(1, self.n_layers + 1):
            x = x @ self.params[f"W{i}"] + self.params[f"b{i}"]
            if i < self.n_layers:
                x = np.maximum(x, 0)
            acts.append(x)
        return (x, acts) if cache else x

    def backward(self, acts, dout):
        grads = {}
        d = dout
        for i in range(self.n_layers, 0, -1):
            grads[f"W{i}"] = acts[i - 1].T @ d
            grads[f"b{i}"] = d.sum(0)
            if i > 1:
                d = (d @ self.params[f"W{i}"].T) * (acts[i - 1] > 0)
        return grads

    def copy_from(self, other):
        for k, v in other.params.items():
            self.params[k] = v.copy()


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng):
        self.cap, self.rng, self.n, self.i = capacity, rng, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch):
        idx = self.rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden=128, lr=1e-3, gamma=0.95,
                 batch_size=64, buffer_size=100_000, target_update=500, seed=0):
        self.rng = np.random.default_rng(seed)
        sizes = [obs_dim, hidden, hidden, n_actions]
        self.q = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.q)
        self.opt = Adam(self.q.params, lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, self.rng)
        self.n_actions, self.gamma = n_actions, gamma
        self.batch_size, self.target_update = batch_size, target_update
        self.updates = 0

    def act(self, obs, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q.forward(obs[None])[0]))

    def learn(self):
        if self.buffer.n < self.batch_size * 4:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        # Double DQN: online net picks the action, target net evaluates it.
        best = np.argmax(self.q.forward(s2), axis=1)
        q_next = self.target.forward(s2)[np.arange(len(a)), best]
        y = r + self.gamma * (1.0 - d) * q_next

        q, acts = self.q.forward(s, cache=True)
        pred = q[np.arange(len(a)), a]
        err = pred - y
        # Huber loss gradient (delta = 1)
        g = np.clip(err, -1.0, 1.0) / len(a)
        dout = np.zeros_like(q)
        dout[np.arange(len(a)), a] = g
        grads = self.q.backward(acts, dout)
        for k in grads:  # gradient-norm clipping per tensor
            np.clip(grads[k], -5, 5, out=grads[k])
        self.opt.step(self.q.params, grads)

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.copy_from(self.q)
        return float(np.mean(np.where(np.abs(err) <= 1, 0.5 * err ** 2, np.abs(err) - 0.5)))

    # ------------------------------------------------------------------ io
    def save(self, path):
        np.savez(path, **self.q.params)

    def load(self, path):
        data = np.load(path)
        for k in self.q.params:
            self.q.params[k] = data[k].astype(np.float32)
        self.target.copy_from(self.q)
