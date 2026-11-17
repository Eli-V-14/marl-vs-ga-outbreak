import numpy as np
from outbreak.sim import World, Config


def make_world(positions, zombie, **cfg):
    """A world with no walls and agents at the given positions, for controlled tests."""
    world = World(Config(**cfg)).reset(0)
    world.walkable[:] = True
    world.pos = np.array(positions, dtype=np.float64)
    world.is_zombie = np.array(zombie)
    world.turned_at = np.where(world.is_zombie, 0, -1)
    return world


def stay(world):
    return np.zeros_like(world.pos)


def test_close_human_is_infected():
    world = make_world([[10.5, 10.5], [11.0, 10.5]], zombie=[False, True])
    info = world.step(stay(world))
    assert world.is_zombie.all()
    assert info["infections"] == [(1, 0)]
    assert world.turned_at[0] == 1


def test_far_human_is_safe():
    world = make_world([[10.5, 10.5], [15.5, 10.5]], zombie=[False, True])
    info = world.step(stay(world))
    assert not world.is_zombie[0]
    assert info["infections"] == []


def test_no_chain_infection_in_one_step():
    # zombie -- A -- B, each 0.6 apart: only A is in range of the zombie this step
    world = make_world([[10.0, 10.0], [10.6, 10.0], [11.2, 10.0]], zombie=[True, False, False])
    info = world.step(stay(world))
    assert info["infections"] == [(0, 1)]
    info = world.step(stay(world))
    assert info["infections"] == [(1, 2)]  # A, now a zombie, gets credit for B


def test_credit_goes_to_nearest_zombie():
    world = make_world([[10.0, 10.0], [10.7, 10.0], [10.2, 10.0]], zombie=[False, True, True])
    info = world.step(stay(world))
    assert info["infections"] == [(2, 0)]


def test_terminated_when_all_infected():
    world = make_world([[10.0, 10.0], [10.5, 10.0]], zombie=[False, True])
    info = world.step(stay(world))
    assert info["terminated"] and not info["truncated"]


def test_truncated_at_max_steps():
    world = make_world([[5.5, 5.5], [30.5, 30.5]], zombie=[False, True], max_steps=5)
    for _ in range(4):
        assert not world.step(stay(world))["truncated"]
    info = world.step(stay(world))
    assert info["truncated"] and not info["terminated"]


def test_full_random_episode():
    world = World(Config(max_steps=300)).reset(0)
    rng = np.random.default_rng(0)
    n_zombies = world.is_zombie.sum()
    while True:
        info = world.step(rng.uniform(-1, 1, size=world.pos.shape))
        assert world.is_zombie.sum() >= n_zombies  # zombies never turn back
        assert world.is_zombie.sum() == n_zombies + len(info["infections"])
        n_zombies = world.is_zombie.sum()
        if info["terminated"] or info["truncated"]:
            break
    assert ((world.turned_at >= 0) == world.is_zombie).all()