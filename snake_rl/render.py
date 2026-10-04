"""Headless frame rendering with Pillow (no display needed)."""
from PIL import Image, ImageDraw, ImageFont

BG, GRID = (22, 27, 34), (33, 40, 50)
HEAD, BODY, FOOD = (86, 211, 100), (46, 160, 67), (248, 81, 73)
TEXT, DIM = (230, 237, 243), (139, 148, 158)


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_board(env, cell=24, title="", subtitle="", header=44):
    n = env.size
    w = n * cell
    img = Image.new("RGB", (w, w + header), BG)
    d = ImageDraw.Draw(img)
    d.text((8, 5), title, fill=TEXT, font=_font(14))
    d.text((8, 24), subtitle, fill=DIM, font=_font(12))
    for i in range(n + 1):
        d.line([(i * cell, header), (i * cell, header + w)], fill=GRID)
        d.line([(0, header + i * cell), (w, header + i * cell)], fill=GRID)
    if env.food:
        x, y = env.food
        d.ellipse([x * cell + 4, header + y * cell + 4, (x + 1) * cell - 4, header + (y + 1) * cell - 4], fill=FOOD)
    for i, (x, y) in enumerate(env.snake):
        box = [x * cell + 2, header + y * cell + 2, (x + 1) * cell - 2, header + (y + 1) * cell - 2]
        d.rounded_rectangle(box, radius=5, fill=HEAD if i == 0 else BODY)
    return img
