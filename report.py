"""Build learning-curve plots and a self-contained HTML progress report for the jury.

    python report.py --run runs/main --compare runs/no_shaping
"""
import argparse
import base64
import csv
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snake_rl.dqn import DQNAgent
from snake_rl.env import N_ACTIONS, OBS_DIM
from snake_rl.evaluate import evaluate

BLUE, ORANGE, GREY, RED = "#2f6fdb", "#e8833a", "#8a8f98", "#d1453b"


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def smooth(y, w=50):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)


def final_scores(run, episodes, size):
    agent = DQNAgent(OBS_DIM, N_ACTIONS)
    agent.load(os.path.join(run, "best.npz"))
    return evaluate(agent, episodes=episodes, size=size, seed=999)


def make_plots(run, cmp_run, out_dir, episodes, size):
    ep, ev = read_csv(f"{run}/episodes.csv"), read_csv(f"{run}/eval.csv")
    cmp_ep = read_csv(f"{cmp_run}/episodes.csv") if cmp_run else None
    cmp_ev = read_csv(f"{cmp_run}/eval.csv") if cmp_run else None
    paths = {}

    # 1. Learning curve (training episodes)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter(ep["step"], ep["score"], s=3, color=BLUE, alpha=0.12)
    ax.plot(ep["step"][49:], smooth(ep["score"]), color=BLUE, lw=2, label="with reward shaping (50-game avg)")
    if cmp_ep is not None:
        ax.plot(cmp_ep["step"][49:], smooth(cmp_ep["score"]), color=ORANGE, lw=2, label="no shaping (50-game avg)")
    ax.legend(frameon=False)
    style(ax, "Learning curve: food eaten per training game", "training steps", "score (food eaten)")
    paths["learning"] = f"{out_dir}/learning_curve.png"; fig.tight_layout(); fig.savefig(paths["learning"], dpi=140); plt.close(fig)

    # 2. Greedy evaluation progress with min-max band
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(ev["step"], ev["min_score"], ev["max_score"], color=BLUE, alpha=0.15, label="min-max over 30 games")
    ax.plot(ev["step"], ev["mean_score"], color=BLUE, lw=2, marker="o", ms=4, label="mean score (shaping)")
    if cmp_ev is not None:
        ax.plot(cmp_ev["step"], cmp_ev["mean_score"], color=ORANGE, lw=2, marker="o", ms=4, label="mean score (no shaping)")
    ax.legend(frameon=False)
    style(ax, "Checkpoint evaluation (no exploration)", "training steps", "score")
    paths["eval"] = f"{out_dir}/eval_progress.png"; fig.tight_layout(); fig.savefig(paths["eval"], dpi=140); plt.close(fig)

    # 3. Loss + epsilon
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(ev["step"][1:], ev["mean_loss"][1:], color=RED, lw=2, label="TD loss")
    ax2 = ax.twinx(); ax2.plot(ev["step"], ev["epsilon"], color=GREY, lw=2, ls="--", label="epsilon (exploration)")
    ax2.set_ylabel("epsilon"); ax2.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    style(ax, "Training internals", "training steps", "Huber TD loss")
    paths["internals"] = f"{out_dir}/loss_epsilon.png"; fig.tight_layout(); fig.savefig(paths["internals"], dpi=140); plt.close(fig)

    # 4. Final comparison, many fresh games
    rand = evaluate(None, episodes=episodes, size=size, seed=999, policy="random")
    results = {"Random agent": rand, "DQN (with shaping)": final_scores(run, episodes, size)}
    if cmp_run:
        results["DQN (no shaping)"] = final_scores(cmp_run, episodes, size)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names, cols = list(results), [GREY, BLUE, ORANGE]
    means = [results[n].mean() for n in names]
    bars = ax.bar(names, means, color=cols[:len(names)], width=0.55)
    ax.errorbar(names, means, yerr=[results[n].std() for n in names], fmt="none", ecolor="#333", capsize=4)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.4, f"{m:.1f}", ha="center", fontweight="bold")
    style(ax, f"Final performance over {episodes} fresh games (mean ± std)", "", "score")
    paths["final"] = f"{out_dir}/final_comparison.png"; fig.tight_layout(); fig.savefig(paths["final"], dpi=140); plt.close(fig)
    return paths, results, ev


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html(run, cmp_run, paths, results, ev, out_path, size):
    img = lambda p, alt: f'<img alt="{alt}" src="data:image/png;base64,{b64(p)}">'
    gif = os.path.join(run, "evolution.gif")
    gif_html = f'<img alt="evolution" src="data:image/gif;base64,{b64(gif)}">' if os.path.exists(gif) else "<p>(run record.py first)</p>"
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{s.mean():.2f}</td><td>{np.median(s):.0f}</td><td>{s.max()}</td><td>{s.min()}</td></tr>"
        for n, s in results.items())
    ev_rows = "".join(
        f"<tr><td>{int(st):,}</td><td>{m:.2f}</td><td>{mx:.0f}</td><td>{l:.3f}</td><td>{e:.2f}</td></tr>"
        for st, m, mx, l, e in zip(ev["step"], ev["mean_score"], ev["max_score"], ev["mean_loss"], ev["epsilon"]))
    rand_m, dqn_m = results["Random agent"].mean(), results["DQN (with shaping)"].mean()
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Snake RL progress report</title>
<style>
:root{{--bg:#fff;--fg:#1f2328;--dim:#59636e;--card:#f6f8fa;--line:#d1d9e0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--card:#161b22;--line:#30363d}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px 64px}}
h1{{margin:.2em 0}} h2{{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.2em}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px}}
.kpi b{{display:block;font-size:1.8em}} .kpi span{{color:var(--dim);font-size:.9em}}
img{{max-width:100%;height:auto;border-radius:8px;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:.92em}} td,th{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}} .p{{color:var(--dim)}}
</style></head><body><main>
<h1>Teaching an AI to play Snake with Reinforcement Learning</h1>
<p class="p">Deep Q-Network (Double DQN, pure NumPy) · {size}×{size} board · reward shaping ablation</p>
<div class="kpis">
<div class="kpi"><b>{rand_m:.2f}</b><span>random agent, avg food/game</span></div>
<div class="kpi"><b>{dqn_m:.1f}</b><span>trained DQN, avg food/game</span></div>
<div class="kpi"><b>{results["DQN (with shaping)"].max()}</b><span>best single game</span></div>
<div class="kpi"><b>{int(ev["step"][-1]):,}</b><span>training steps</span></div>
</div>
<h2>1. Watch the model learn</h2>
<p>Same game seeds, four snapshots of the <em>same</em> network taken at increasing points of training.
The leftmost panel is the untrained network (random weights).</p>
{gif_html}
<h2>2. Learning curve</h2>{img(paths["learning"], "learning curve")}
<p class="p">Each dot is one training game; the line is a 50-game moving average. Early games include random exploration (epsilon-greedy).</p>
<h2>3. Checkpoint evaluation</h2>{img(paths["eval"], "eval")}
<p class="p">At every checkpoint the agent plays 30 games with exploration switched off.</p>
<h2>4. Final result</h2>{img(paths["final"], "final")}
<table><tr><th>Agent</th><th>mean</th><th>median</th><th>max</th><th>min</th></tr>{rows}</table>
<h2>5. Training internals</h2>{img(paths["internals"], "internals")}
<h2>6. Checkpoint log</h2>
<table><tr><th>step</th><th>eval mean score</th><th>eval max</th><th>TD loss</th><th>epsilon</th></tr>{ev_rows}</table>
<h2>How it works</h2>
<ul>
<li><b>State (14 numbers):</b> danger straight/right/left, heading, food direction, and how much free space is reachable in each of the three moves.</li>
<li><b>Actions:</b> go straight, turn right, turn left.</li>
<li><b>Reward:</b> +10 for food, −10 for dying or starving, ±0.1 for moving closer/farther from the food (shaping).</li>
<li><b>Learning:</b> Double DQN with experience replay, a target network, Huber loss and Adam; 128-128 MLP.</li>
</ul></main></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/main")
    p.add_argument("--compare", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--size", type=int, default=10)
    args = p.parse_args()
    out_dir = os.path.join(args.run, "report")
    os.makedirs(out_dir, exist_ok=True)
    paths, results, ev = make_plots(args.run, args.compare, out_dir, args.episodes, args.size)
    build_html(args.run, args.compare, paths, results, ev, os.path.join(out_dir, "report.html"), args.size)
    for n, s in results.items():
        print(f"{n:22s} mean {s.mean():6.2f}  median {np.median(s):4.0f}  max {s.max():3d}")
    print("wrote", os.path.join(out_dir, "report.html"))


if __name__ == "__main__":
    main()
