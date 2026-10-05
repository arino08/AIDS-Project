# Snake + Reinforcement Learning (Deep Q-Network)

An AI that teaches itself to play Snake from scratch using **Double DQN**, implemented in pure NumPy
(no PyTorch needed, trains in ~2.5 minutes on a CPU).

| Agent                              | Avg. food per game (100 fresh games, 10×10 board) |
| ---------------------------------- | -------------------------------------------------- |
| Random                             | 0.16                                               |
| DQN, no reward shaping             | 33.1                                               |
| **DQN, with reward shaping** | **35.5** (best game: 61)                     |

## Quick start

```bash
pip install -r requirements.txt
python train.py --steps 200000 --out runs/main                 # train (~2.5 min)
python train.py --steps 200000 --no-shaping --out runs/no_shaping   # ablation
python record.py --run runs/main                                # evolution.gif
python report.py --run runs/main --compare runs/no_shaping      # plots + report.html
python play.py --mode ai --model runs/main/best.npz             # watch the AI (pygame)
python play.py --mode human                                     # play yourself
```

## Showing progress to the jury

Everything is already generated in `runs/main/`:

| File                           | What it shows                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `report/report.html`         | **Start here.** One self-contained page: KPIs, evolution GIF, all plots, results table, checkpoint log. |
| `evolution.gif`              | The same network at steps 0 / 70k / 130k / 200k playing side by side.                                         |
| `report/learning_curve.png`  | Score per training game over time (with vs. without shaping).                                                 |
| `report/eval_progress.png`   | Greedy evaluation at every checkpoint with min–max band.                                                     |
| `report/loss_epsilon.png`    | TD loss and exploration rate.                                                                                 |
| `checkpoints/step_*.npz`     | A saved model every 10k steps, so any stage can be replayed.                                                  |
| `episodes.csv`, `eval.csv` | Raw logs of every game and every evaluation.                                                                  |

## How it works

- **Environment** (`snake_rl/env.py`): 10×10 grid, three relative actions (straight / right / left).
- **State** (14 numbers): danger in 3 directions, heading (one-hot), food direction, and the fraction of free space reachable after each possible move (flood fill) — this last feature stops the snake trapping itself.
- **Reward:** +10 food, −10 collision or starvation, ±0.1 for moving closer to / farther from food (optional shaping).
- **Agent** (`snake_rl/dqn.py`): 14→128→128→3 MLP, experience replay (100k), target network, Double-DQN targets, Huber loss, Adam, ε-greedy decaying 1.0→0.02.

## Notes on the results

- Training-game scores (~21) are lower than evaluation scores (~35) because training keeps 2% random actions.
- Reward shaping made no clear difference here (35.5 vs 33.1 is within run-to-run noise for one seed each); the flood-fill features do most of the work. Running several seeds would be needed for a firm conclusion.

Team Members : 231452, 231453
