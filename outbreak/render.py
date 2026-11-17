"""pygame drawing. The only module in outbreak/ that imports pygame; training code never imports it."""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from outbreak.maps import grid_to_rgb

HUMAN_COLOR = (66, 135, 245)
ZOMBIE_COLOR = (200, 40, 40)
TEXT_COLOR = (255, 255, 255)


class Renderer:
    def __init__(self, world, scale=14):
        self.scale = scale
        map_h, map_w = world.grid.shape
        self.size = (map_w * scale, map_h * scale)
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Outbreak")
        self.font = pygame.font.SysFont("consolas", 16)
        self._map_surf = None
        self._map_grid = None

    def _map_surface(self, grid):
        # rebuild the background only when the map changes (new episode)
        if self._map_grid is not grid:
            surf = pygame.surfarray.make_surface(grid_to_rgb(grid).swapaxes(0, 1))
            self._map_surf = pygame.transform.scale(surf, self.size)
            self._map_grid = grid
        return self._map_surf

    def draw(self, world, status=""):
        self.screen.blit(self._map_surface(world.grid), (0, 0))
        radius = max(2, int(self.scale * 0.4))
        for (x, y), zombie in zip(world.pos, world.is_zombie):
            color = ZOMBIE_COLOR if zombie else HUMAN_COLOR
            pygame.draw.circle(self.screen, color, (int(x * self.scale), int(y * self.scale)), radius)

        hud = f"seed {world.seed}  t={world.t}  humans {world.n_humans_left}  zombies {int(world.is_zombie.sum())}"
        if status:
            hud += f"  [{status}]"
        self.screen.blit(self.font.render(hud, True, TEXT_COLOR, (0, 0, 0)), (6, 6))
        pygame.display.flip()

    def close(self):
        pygame.quit()