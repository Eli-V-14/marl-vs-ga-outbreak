"""Watch scripted humans and zombies live.

Controls: SPACE pause/resume, N single step (while paused), R new episode (next seed),
UP/DOWN change simulation speed, ESC or close window to quit.
"""
import pygame

from outbreak.sim import World, Config
from outbreak.policies import scripted_actions
from outbreak.render import Renderer


def main(seed=0, fps=30):
    world = World(Config()).reset(seed)
    renderer = Renderer(world)
    clock = pygame.time.Clock()
    paused, done, steps_per_frame = False, False, 1

    running = True
    while running:
        single_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    single_step = True
                elif event.key == pygame.K_r:
                    seed += 1
                    world.reset(seed)
                    done = False
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame * 2, 64)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame // 2, 1)

        if not done and (not paused or single_step):
            for _ in range(1 if single_step else steps_per_frame):
                info = world.step(scripted_actions(world))
                if info["terminated"] or info["truncated"]:
                    done = True
                    break

        if done:
            status = "all infected - R for next" if world.n_humans_left == 0 else "time up - R for next"
        else:
            status = "paused" if paused else f"x{steps_per_frame}"
        renderer.draw(world, status)
        clock.tick(fps)

    renderer.close()


if __name__ == "__main__":
    main()