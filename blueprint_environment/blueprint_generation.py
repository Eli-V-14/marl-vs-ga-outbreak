from dataclasses import dataclass
from pathlib import Path
import pygame
import numpy as np
import random

TILES = {'#': 1, '.': 0, 'D': 2}

@dataclass
class Blueprint:
    name: str
    grid: np.ndarray
    weight: float = 1.0

    @property
    def size(self):
        return self.grid.size

    def rotated(self, k):
        return np.rot90(self.grid, k)

def load_blueprint(path):
    lines = Path(path).read_text().splitlines()
    meta = {}
    while lines and lines[0].startswith('@'):
        key, val = lines.pop(0)[1:].split(maxsplit=1)
        meta[key] = val
    rows = [l for l in lines if l.strip()]
    if len({len(r) for r in rows}) != 1:
        raise ValueError(f'{path}: rows have different lengths')
    grid = np.array([[TILES[c] for c in r] for r in rows])
    return Blueprint(Path(path).stem, grid, float(meta.get('weight', 1)))

def stamp(big, small, x, y):
    h, w = small.shape
    if x < 0 or y < 0 or y + h > big.shape[0] or x + w > big.shape[1]:
        raise ValueError('out of bounds')
    big[y:y+h, x:x+w] = small

def legal_positions(blocked, w, h):
    H, W = blocked.shape
    if w > W or h > H:
        return np.array([], int), np.array([], int)
    sat = np.pad(blocked.astype(int).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = (sat[h:, w:] - sat[:H-h+1, w:] - sat[h:, :W-w+1] + sat[:H-h+1, :W-w+1])
    ys, xs = np.nonzero(s == 0)
    return xs, ys

def generate_map(blueprints, n, m, W, H, seed, fill=0):
    rng = random.Random(seed)
    grid = np.full((H, W), fill, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)
    placed = []

    picks = rng.choices(blueprints, weights=[b.weight for b in blueprints], k=n)
    picks.sort(key=lambda b: b.size, reverse=True)

    for bp in picks:
        rots = [0, 1, 2, 3]
        rng.shuffle(rots)
        for k in rots:
            tile = bp.rotated(k)
            h, w = tile.shape
            xs, ys = legal_positions(blocked, w, h)
            if len(xs) == 0:
                continue
            i = rng.randrange(len(xs))
            x, y = int(xs[i]), int(ys[i])
            stamp(grid, tile, x, y)
            blocked[max(0, y-m):min(H, y+h+m), max(0, x-m):min(W, x+w+m)] = True
            placed.append((bp.name, x, y, w, h))
            break
        else:
            print(f"warning: couldn't place {bp.name}")
    return grid, placed

COLORS = {
    -1: (56, 142, 60),
    0: (186, 149, 95),
    1: (117, 117, 117),
    2: (105, 68, 38),
}
DEFAULT = (255, 0, 255)

def grid_to_rgb(grid):
    rgb = np.empty((*grid.shape, 3), dtype=np.uint8)
    rgb[:] = DEFAULT
    for tile_id, color in COLORS.items():
        rgb[grid == tile_id] = color
    return rgb


if __name__ == "__main__":
    blueprints = [load_blueprint(p) for p in sorted((Path(__file__).parent / 'blueprints').glob("*.bp"))]

    seed = 42

    grid, place = generate_map(blueprints, 50, 5, 240, 140, seed, fill=-1)

    rgb = grid_to_rgb(grid)

    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, (1200, 700))

    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    seed += 1

                    grid, place = generate_map(blueprints, 50, 5, 240, 140, seed, fill=-1)

                    rgb = grid_to_rgb(grid)

                    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                    surf = pygame.transform.scale(surf, (1200, 700))

        screen.fill("white")

        screen.blit(surf, (0, 0))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()