"""Play Snake yourself, or watch the trained AI play, in a pygame window.

    python play.py --mode ai --model runs/main/best.npz
    python play.py --mode human          # arrow keys; Esc quits
    python play.py --mode human --fps 8  # adjust speed (FPS)
"""
import argparse
import pygame

from snake_rl.dqn import DQNAgent
from snake_rl.env import DIRS, N_ACTIONS, OBS_DIM, SnakeEnv
from snake_rl.render import BG, BODY, FOOD, GRID, HEAD, TEXT, DIM

CELL, HEADER = 40, 50
BTN_COLOR = (46, 160, 67)
BTN_HOVER = (56, 185, 78)
BTN_TEXT = (255, 255, 255)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ai", "human"], default="ai")
    p.add_argument("--model", default="runs/main/best.npz")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--fps", type=int, default=12, help="Game speed in frames per second (default: 12)")
    args = p.parse_args()

    agent = None
    if args.mode == "ai":
        agent = DQNAgent(OBS_DIM, N_ACTIONS)
        agent.load(args.model)

    pygame.init()
    width = args.size * CELL
    height = args.size * CELL + HEADER
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Snake RL - {args.mode.upper()} Mode (Speed: {args.fps} FPS)")
    
    font = pygame.font.SysFont(None, 26)
    font_large = pygame.font.SysFont(None, 34)
    clock = pygame.time.Clock()
    
    is_human = (args.mode == "human")
    env = SnakeEnv(size=args.size)
    
    # In human mode, start with random direction and wait for player to click START
    game_state = "START_SCREEN" if is_human else "PLAYING"
    obs = env.reset(random_dir=is_human)
    best, want = 0, env.direction
    keymap = {pygame.K_UP: 0, pygame.K_RIGHT: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3}

    # Start button dimensions (centered)
    btn_w, btn_h = 160, 46
    btn_rect = pygame.Rect((width - btn_w) // 2, HEADER + (args.size * CELL - btn_h) // 2, btn_w, btn_h)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        btn_hovered = btn_rect.collidepoint(mouse_pos)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if game_state in ("START_SCREEN", "GAME_OVER") and btn_hovered:
                    obs = env.reset(random_dir=is_human)
                    want = env.direction
                    game_state = "PLAYING"
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_SPACE, pygame.K_RETURN) and game_state in ("START_SCREEN", "GAME_OVER"):
                    obs = env.reset(random_dir=is_human)
                    want = env.direction
                    game_state = "PLAYING"
                elif ev.key in keymap and game_state == "PLAYING":
                    want = keymap[ev.key]

        if game_state == "PLAYING":
            if env.done:
                if is_human:
                    game_state = "GAME_OVER"
                else:
                    pygame.time.wait(600)
                    obs = env.reset(random_dir=False)
                    want = None
            else:
                if agent:
                    action = agent.act(obs, 0.0)
                else:
                    # Convert absolute arrow key desired direction into relative turn
                    d = env.direction
                    action = 1 if want == (d + 1) % 4 else 2 if want == (d - 1) % 4 else 0
                obs, _, _, _ = env.step(action)
                best = max(best, env.score)

        # ----------------- Render -----------------
        screen.fill(BG)

        # Draw Grid
        for i in range(args.size + 1):
            pygame.draw.line(screen, GRID, (i * CELL, HEADER), (i * CELL, HEADER + args.size * CELL))
            pygame.draw.line(screen, GRID, (0, HEADER + i * CELL), (args.size * CELL, HEADER + i * CELL))

        # Draw Food
        if env.food:
            fx, fy = env.food
            pygame.draw.circle(screen, FOOD, (fx * CELL + CELL // 2, HEADER + fy * CELL + CELL // 2), CELL // 2 - 6)

        # Draw Snake
        for i, (x, y) in enumerate(env.snake):
            pygame.draw.rect(screen, HEAD if i == 0 else BODY,
                             (x * CELL + 2, HEADER + y * CELL + 2, CELL - 4, CELL - 4), border_radius=8)

        # Header Info
        label = "AI" if agent else "You"
        header_text = f"{label}  |  Score: {env.score}  |  Best: {best}  |  Speed: {args.fps} FPS"
        screen.blit(font.render(header_text, True, TEXT), (10, 14))

        # Start / Game Over overlay for Human mode
        if game_state in ("START_SCREEN", "GAME_OVER"):
            # Dim overlay
            overlay = pygame.Surface((width, args.size * CELL), pygame.SRCALPHA)
            overlay.fill((10, 14, 20, 180))
            screen.blit(overlay, (0, HEADER))

            # Button
            col = BTN_HOVER if btn_hovered else BTN_COLOR
            pygame.draw.rect(screen, col, btn_rect, border_radius=10)
            pygame.draw.rect(screen, (255, 255, 255), btn_rect, width=2, border_radius=10)

            btn_label = "START" if game_state == "START_SCREEN" else "PLAY AGAIN"
            btn_surface = font_large.render(btn_label, True, BTN_TEXT)
            screen.blit(btn_surface, btn_surface.get_rect(center=btn_rect.center))

            # Hint text
            hint = "Click button or press SPACE to start"
            hint_surface = font.render(hint, True, DIM)
            screen.blit(hint_surface, (width // 2 - hint_surface.get_width() // 2, btn_rect.bottom + 16))

            if game_state == "GAME_OVER":
                msg = font_large.render(f"Game Over! Final Score: {env.score}", True, FOOD)
                screen.blit(msg, (width // 2 - msg.get_width() // 2, btn_rect.top - 40))

        pygame.display.flip()
        clock.tick(args.fps)
    pygame.quit()


if __name__ == "__main__":
    main()

