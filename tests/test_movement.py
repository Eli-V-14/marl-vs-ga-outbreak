import numpy as np
from outbreak.sim import World, Config


def open_world(positions, zombie=None):
    """A world with no walls and agents at the given positions, for controlled tests."""
    world = World(Config()).reset(0)
    world.walkable[:] = True
    world.pos = np.array(positions, dtype=np.float64)
    world.is_zombie = np.array(zombie or [False] * len(positions))
    return world


def test_random_walk_stays_legal():
    world = World(Config(n_humans=50, n_zombies=10))
    rng = np.random.default_rng(0)
    for seed in range(5):
        world.reset(seed)
        for _ in range(1000):
            world.step(rng.uniform(-1, 1, size=world.pos.shape))
            assert world.walkable_at(world.pos).all()


def test_speed_is_capped():
    world = open_world([[10.5, 10.5], [20.5, 20.5]], zombie=[False, True])
    world.step([[5.0, 0.0], [0.0, 5.0]])  # way over length 1
    assert np.isclose(world.pos[0, 0], 10.5 + world.cfg.human_speed)
    assert np.isclose(world.pos[1, 1], 20.5 + world.cfg.zombie_speed)


def test_slides_along_wall():
    world = open_world([[9.5, 5.5]])
    world.walkable[:, 10] = False  # vertical wall at x = 10
    for _ in range(20):
        world.step([[1.0, 1.0]])
    assert world.pos[0, 0] < 10.0      # never entered the wall
    assert world.pos[0, 1] > 5.5 + 2   # kept moving in y


def test_no_corner_cutting():
    world = open_world([[9.9, 4.9]])
    world.walkable[4, 10] = False  # tile to the right
    world.walkable[5, 9] = False   # tile below
    for _ in range(10):
        world.step([[1.0, 1.0]])
    assert np.floor(world.pos[0, 0]) == 9 and np.floor(world.pos[0, 1]) == 4


def test_map_edges():
    world = open_world([[0.1, 0.1], [59.9, 39.9]])
    for _ in range(10):
        world.step([[-1.0, -1.0], [1.0, 1.0]])
    assert world.walkable_at(world.pos).all()