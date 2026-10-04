"""Train a DQN agent to play Snake.

    python train.py --steps 300000 --out runs/main
"""
import argparse
import csv
import os
import time

import numpy as np

from snake_rl.dqn import DQNAgent
from snake_rl.env import N_ACTIONS, OBS_DIM, SnakeEnv
from snake_rl.evaluate import evaluate


def _stats(scores):
    return float(scores.mean()), float(np.median(scores)), float(scores.min()), float(scores.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--out", default="runs/main")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shaping", action="store_true", help="disable distance reward shaping")
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-frac", type=float, default=0.4, help="fraction of steps over which epsilon decays")
    p.add_argument("--train-every", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=20_000)
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "checkpoints"), exist_ok=True)
    env = SnakeEnv(size=args.size, shaping=not args.no_shaping, seed=args.seed)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, seed=args.seed)

    log = open(os.path.join(args.out, "episodes.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["episode", "step", "score", "reward", "length", "epsilon"])
    evals = open(os.path.join(args.out, "eval.csv"), "w", newline="")
    ew = csv.writer(evals)
    ew.writerow(["step", "mean_score", "median_score", "min_score", "max_score", "mean_loss", "epsilon"])

    ckpt = lambda step: os.path.join(args.out, "checkpoints", f"step_{step:07d}.npz")
    agent.save(ckpt(0))  # untrained network, the "before" picture for the jury
    ew.writerow([0, *[round(x, 3) for x in _stats(evaluate(agent, episodes=30, size=args.size))], 0.0, 1.0])
    decay_steps = int(args.steps * args.eps_decay_frac)
    losses = []
    best, ep, t0 = -1.0, 0, time.time()
    obs, ep_reward, ep_len = env.reset(), 0.0, 0
    recent = []
    for step in range(1, args.steps + 1):
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * step / decay_steps)
        a = agent.act(obs, eps)
        obs2, r, done, info = env.step(a)
        agent.buffer.add(obs, a, r, obs2, float(done))
        obs, ep_reward, ep_len = obs2, ep_reward + r, ep_len + 1
        if step % args.train_every == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        if done:
            ep += 1
            recent.append(info["score"])
            w.writerow([ep, step, info["score"], round(ep_reward, 2), ep_len, round(eps, 3)])
            obs, ep_reward, ep_len = env.reset(), 0.0, 0
        if step % args.eval_every == 0:
            scores = evaluate(agent, episodes=30, size=args.size)
            ew.writerow([step, *[round(x, 3) for x in _stats(scores)],
                         round(float(np.mean(losses)) if losses else 0.0, 4), round(eps, 3)])
            losses.clear()
            agent.save(ckpt(step))
            evals.flush(); log.flush()
            tag = ""
            if scores.mean() >= best:
                best = scores.mean()
                agent.save(os.path.join(args.out, "best.npz"))
                tag = "  * new best"
            print(f"step {step:>8} | eps {eps:.2f} | train avg {np.mean(recent[-100:]):5.2f} | "
                  f"eval mean {scores.mean():5.2f} max {scores.max():2d} | {time.time()-t0:5.0f}s{tag}", flush=True)
    agent.save(os.path.join(args.out, "final.npz"))
    log.close(); evals.close()
    print(f"done. best eval mean score = {best:.2f}")


if __name__ == "__main__":
    main()
