"""Record the agent's learning progress as an animated GIF.

Plays several checkpoints side by side (untrained -> early -> mid -> final) on the
same food sequence, so the jury can literally watch the model getting better.

    python record.py --run runs/main --out runs/main/evolution.gif
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw

from snake_rl.dqn import DQNAgent
from snake_rl.env import N_ACTIONS, OBS_DIM, SnakeEnv
from snake_rl.render import BG, TEXT, draw_board


def pick_checkpoints(run, k):
    files = sorted(glob.glob(os.path.join(run, "checkpoints", "step_*.npz")))
    idx = sorted({int(round(i)) for i in np.linspace(0, len(files) - 1, k)})
    return [(int(os.path.basename(files[i])[5:-4]), files[i]) for i in idx]


class Panel:
    """One checkpoint playing game after game, looping forever."""

    def __init__(self, label, path, size, seed):
        self.label, self.size, self.seed = label, size, seed
        self.agent = DQNAgent(OBS_DIM, N_ACTIONS)
        self.agent.load(path)
        self.games, self.best, self.hold = 0, 0, 0
        self._new_game()

    def _new_game(self):
        self.env = SnakeEnv(size=self.size, seed=self.seed + self.games)
        self.obs, self.done = self.env.reset(), False
        self.games += 1

    def tick(self):
        if self.done:
            self.hold += 1
            if self.hold > 8:  # freeze on the final board briefly, then restart
                self.hold = 0
                self._new_game()
            return
        self.obs, _, self.done, _ = self.env.step(self.agent.act(self.obs, 0.0))
        self.best = max(self.best, self.env.score)

    def frame(self, cell):
        status = "GAME OVER" if self.done else "playing"
        return draw_board(self.env, cell, self.label,
                          f"score {self.env.score:>2}   best {self.best:>2}   game {self.games}   {status}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/main")
    p.add_argument("--out", default=None)
    p.add_argument("--panels", type=int, default=4)
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--cell", type=int, default=26)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    out = args.out or os.path.join(args.run, "evolution.gif")

    panels = [Panel(f"step {s:,}" if s else "untrained (step 0)", f, args.size, args.seed)
              for s, f in pick_checkpoints(args.run, args.panels)]
    frames = []
    for _ in range(args.frames):
        tiles = [pn.frame(args.cell) for pn in panels]
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * len(tiles) + 6 * (len(tiles) - 1), h), BG)
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * (w + 6), 0))
        frames.append(sheet)
        for pn in panels:
            pn.tick()
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames, {len(panels)} panels)")


if __name__ == "__main__":
    main()
