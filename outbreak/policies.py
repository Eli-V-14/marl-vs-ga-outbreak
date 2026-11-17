"""Scripted (non-learning) behaviors. Each policy takes a World and returns (N, 2) directions;
rows for agents the policy doesn't control are zero. Combine them with scripted_actions()."""
import numpy as np


def _nearest(src, dst):
    """For each point in src, the vector to and distance of the nearest point in dst."""
    diff = dst[None, :, :] - src[:, None, :]          # (S, D, 2)
    dist = np.linalg.norm(diff, axis=2)                # (S, D)
    j = dist.argmin(axis=1)
    rows = np.arange(len(src))
    return diff[rows, j], dist[rows, j]


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


def greedy_zombies(world):
    """Every zombie walks straight toward the nearest human (no pathfinding)."""
    dirs = np.zeros_like(world.pos)
    z = world.is_zombie
    if z.any() and (~z).any():
        vec, _ = _nearest(world.pos[z], world.pos[~z])
        dirs[z] = _unit(vec)
    return dirs


def flee_humans(world, vision=10.0):
    """Baseline humans: run in a straight line directly away from the nearest zombie
    if one is within `vision` tiles, otherwise stand still."""
    dirs = np.zeros_like(world.pos)
    h = ~world.is_zombie
    if h.any() and world.is_zombie.any():
        vec, dist = _nearest(world.pos[h], world.pos[world.is_zombie])
        away = -_unit(vec)
        away[dist > vision] = 0.0
        dirs[h] = away
    return dirs


def random_walk(world):
    """Uniform random directions for every agent (useful for testing)."""
    return world.rng.uniform(-1, 1, size=world.pos.shape)


def scripted_actions(world, human_policy=flee_humans, zombie_policy=greedy_zombies):
    """Merge a human policy and a zombie policy into one (N, 2) action array."""
    h = human_policy(world)
    z = zombie_policy(world)
    return np.where(world.is_zombie[:, None], z, h)