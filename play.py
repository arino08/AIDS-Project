"""Play Snake yourself, or watch the trained AI play, in a pygame window.

    python play.py --mode ai --model runs/main/best.npz
    python play.py --mode human          # arrow keys; Esc quits
"""
import argparse

import pygame

from snake_rl.dqn import DQNAgent
from snake_rl.env import DIRS, N_ACTIONS, OBS_DIM, SnakeEnv
from snake_rl.render import BG, BODY, FOOD, GRID, HEAD, TEXT

CELL, HEADER = 40, 50


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ai", "human"], default="ai")
    p.add_argument("--model", default="runs/main/best.npz")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    agent = None
    if args.mode == "ai":
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.load(args.model)

    pygame.init()
    screen = pygame.display.set_mode((args.size * CELL, args.size * CELL + HEADER))
    pygame.display.set_caption(f"Snake RL - {args.mode}")
    font, clock = pygame.font.SysFont(None, 28), pygame.time.Clock()
    env = SnakeEnv(size=args.size)
    obs, best, want = env.reset(), 0, None
    keymap = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3}

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key in keymap:
                want = keymap[ev.key]
        if env.done:
            pygame.time.wait(600)
            obs, want = env.reset(), None
        if agent:
            action = agent.act(obs, 0.0)
        else:  # convert an absolute arrow key into a relative turn
            d = env.direction
            action = 1 if want == (d + 1) % 4 else 2 if want == (d - 1) % 4 else 0
        obs, _, _, _ = env.step(action)
        best = max(best, env.score)

        screen.fill(BG)
        for i in range(args.size + 1):
            pygame.draw.line(screen, GRID, (i * CELL, HEADER), (i * CELL, HEADER + args.size * CELL))
            pygame.draw.line(screen, GRID, (0, HEADER + i * CELL), (args.size * CELL, HEADER + i * CELL))
        if env.food:
            fx, fy = env.food
            pygame.draw.circle(screen, FOOD, (fx * CELL + CELL // 2, HEADER + fy * CELL + CELL // 2), CELL // 2 - 6)
        for i, (x, y) in enumerate(env.snake):
            pygame.draw.rect(screen, HEAD if i == 0 else BODY,
                             (x * CELL + 2, HEADER + y * CELL + 2, CELL - 4, CELL - 4), border_radius=8)
        label = "AI" if agent else "You"
        screen.blit(font.render(f"{label}   score {env.score}   best {best}", True, TEXT), (10, 14))
        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
