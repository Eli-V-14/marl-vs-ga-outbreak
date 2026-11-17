import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
from outbreak.maps import load_blueprints, generate_map, grid_to_rgb

def make_surface(blueprints, seed):
    grid, _ = generate_map(blueprints, 50, 5, 240, 140, seed, fill=-1)
    surf = pygame.surfarray.make_surface(grid_to_rgb(grid).swapaxes(0, 1))
    return pygame.transform.scale(surf, (1200, 700))

def main():
    blueprints = load_blueprints()
    seed = 42
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    clock = pygame.time.Clock()
    surf = make_surface(blueprints, seed)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                seed += 1
                surf = make_surface(blueprints, seed)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()