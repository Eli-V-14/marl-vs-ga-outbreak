import numpy as np
from outbreak.sim import World, Config


def test_spawns_on_walkable_tiles():
    world = World(Config(n_humans=80, n_zombies=20, zombie_spawn_min_dist=2.0))
    for seed in range(20):
        world.reset(seed)
        assert world.walkable_at(world.pos).all()


def test_zombies_spawn_away_from_humans():
    cfg = Config()
    world = World(cfg)
    for seed in range(20):
        world.reset(seed)
        humans = world.pos[~world.is_zombie]
        zombies = world.pos[world.is_zombie]
        d = np.linalg.norm(zombies[:, None] - humans[None], axis=2)
        assert d.min() >= cfg.zombie_spawn_min_dist


def test_same_seed_same_world():
    a = World().reset(5)
    b = World().reset(5)
    assert (a.grid == b.grid).all()
    assert np.allclose(a.pos, b.pos)


def test_counts():
    cfg = Config()
    world = World(cfg).reset(0)
    assert world.is_zombie.sum() == cfg.n_zombies
    assert world.n_humans_left == cfg.n_humans