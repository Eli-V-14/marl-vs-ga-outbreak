import os
import numpy as np
from outbreak.sim import World, Config
from outbreak.policies import greedy_zombies, flee_humans, scripted_actions


def make_world(positions, zombie):
    world = World(Config()).reset(0)
    world.walkable[:] = True
    world.pos = np.array(positions, dtype=np.float64)
    world.is_zombie = np.array(zombie)
    world.turned_at = np.where(world.is_zombie, 0, -1)
    return world


def dist(world, a, b):
    return np.linalg.norm(world.pos[a] - world.pos[b])


def test_greedy_zombie_closes_in():
    world = make_world([[10.5, 10.5], [20.5, 14.5]], zombie=[False, True])
    before = dist(world, 0, 1)
    world.step(greedy_zombies(world))  # human stands still
    assert dist(world, 0, 1) < before


def test_human_flees_when_zombie_visible():
    world = make_world([[10.5, 10.5], [14.5, 10.5]], zombie=[False, True])
    before = dist(world, 0, 1)
    world.step(flee_humans(world))     # zombie stands still
    assert dist(world, 0, 1) > before


def test_human_ignores_zombie_out_of_vision():
    world = make_world([[10.5, 10.5], [40.5, 10.5]], zombie=[False, True])
    assert np.allclose(flee_humans(world, vision=10.0)[0], 0.0)


def test_policies_only_control_their_own_agents():
    world = World(Config()).reset(0)
    assert np.allclose(greedy_zombies(world)[~world.is_zombie], 0.0)
    assert np.allclose(flee_humans(world)[world.is_zombie], 0.0)


def test_scripted_episode_runs_to_the_end():
    world = World(Config(max_steps=500)).reset(1)
    for _ in range(500):
        info = world.step(scripted_actions(world))
        assert world.walkable_at(world.pos).all()
        if info["terminated"] or info["truncated"]:
            break
    assert info["terminated"] or info["truncated"]


def test_renderer_draws_headless():
    os.environ["SDL_VIDEODRIVER"] = "dummy"   # no window needed
    from outbreak.render import Renderer
    world = World(Config()).reset(0)
    renderer = Renderer(world)
    for _ in range(5):
        world.step(scripted_actions(world))
        renderer.draw(world, "test")
    renderer.close()