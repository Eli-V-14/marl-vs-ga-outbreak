"""Watch a saved episode.

    python -m scripts.replay runs/replays/scripted_seed0.npz

Controls: SPACE pause/resume, LEFT/RIGHT step one frame (while paused),
UP/DOWN playback speed, HOME restart, ESC or close window to quit.
"""
import argparse
import pygame

from outbreak.replay import load_episode
from outbreak.render import Renderer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    ep = load_episode(args.path)
    renderer = Renderer(ep.frame(0))
    clock = pygame.time.Clock()
    t, paused, speed = 0, False, 1
    last = len(ep) - 1

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    t = min(t + 1, last)
                elif event.key == pygame.K_LEFT:
                    t = max(t - 1, 0)
                elif event.key == pygame.K_HOME:
                    t = 0
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2, 64)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed // 2, 1)

        if not paused:
            t = min(t + speed, last)
        status = "end" if t == last else ("paused" if paused else f"x{speed}")
        renderer.draw(ep.frame(t), f"replay {t}/{last}  {status}")
        clock.tick(args.fps)

    renderer.close()


if __name__ == "__main__":
    main()