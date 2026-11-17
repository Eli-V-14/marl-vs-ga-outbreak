from dataclasses import dataclass
import numpy as np

from outbreak.maps import load_blueprints, generate_map

WALL = 1


@dataclass
class Config:
    # map
    map_w: int = 60
    map_h: int = 40
    n_buildings: int = 8
    building_margin: int = 2
    # agents
    n_humans: int = 20
    n_zombies: int = 3
    zombie_spawn_min_dist: float = 8.0
    # used in 1c / 1d
    human_speed: float = 0.25
    zombie_speed: float = 0.20
    infect_radius: float = 0.75
    max_steps: int = 1000

    def __post_init__(self):
        assert self.human_speed < 1 and self.zombie_speed < 1, "speeds must be < 1 tile/step"
        assert self.infect_radius < 1, "infect_radius must be < 1 so infection can't pass through walls"


class World:
    """Simulation state. Coordinates are in tiles: pos[:, 0] = x (column), pos[:, 1] = y (row).
    Tile lookup is grid[int(y), int(x)]."""

    def __init__(self, config=None, blueprints=None):
        self.cfg = config or Config()
        self.blueprints = blueprints or load_blueprints()

    def reset(self, seed):
        cfg = self.cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        map_seed = int(self.rng.integers(2**31))
        self.grid, self.buildings = generate_map(
            self.blueprints, cfg.n_buildings, cfg.building_margin,
            cfg.map_w, cfg.map_h, map_seed, fill=-1,
        )
        self.walkable = self.grid != WALL

        n = cfg.n_humans + cfg.n_zombies
        self.pos = np.zeros((n, 2), dtype=np.float64)
        self.is_zombie = np.zeros(n, dtype=bool)
        self.is_zombie[cfg.n_humans:] = True

        self.pos[:cfg.n_humans] = self._spawn(cfg.n_humans)
        self.pos[cfg.n_humans:] = self._spawn(
            cfg.n_zombies, avoid=self.pos[:cfg.n_humans], min_dist=cfg.zombie_spawn_min_dist
        )
        # step at which each agent became a zombie: 0 for starting zombies, -1 = still human
        self.turned_at = np.where(self.is_zombie, 0, -1)
        self.t = 0
        return self

    def _spawn(self, k, avoid=None, min_dist=0.0):
        ys, xs = np.nonzero(self.walkable)
        centers = np.stack([xs + 0.5, ys + 0.5], axis=1)
        if avoid is not None and len(avoid) and min_dist > 0:
            d = np.linalg.norm(centers[:, None, :] - avoid[None, :, :], axis=2).min(axis=1)
            centers = centers[d >= min_dist]
        if len(centers) < k:
            raise RuntimeError(f"only {len(centers)} free spawn tiles for {k} agents")
        idx = self.rng.choice(len(centers), size=k, replace=False)
        return centers[idx]

    def walkable_at(self, pos):
        """Bool per position: inside the map and not on a wall."""
        pos = np.atleast_2d(pos)
        ix = np.floor(pos[:, 0]).astype(int)
        iy = np.floor(pos[:, 1]).astype(int)
        inside = (ix >= 0) & (ix < self.cfg.map_w) & (iy >= 0) & (iy < self.cfg.map_h)
        out = np.zeros(len(pos), dtype=bool)
        out[inside] = self.walkable[iy[inside], ix[inside]]
        return out

    @property
    def n_humans_left(self):
        return int((~self.is_zombie).sum())


    # ---------- 1c: movement ----------

    def max_speed(self):
        """Per-agent speed cap (tiles per step), depends on whether the agent is a zombie."""
        return np.where(self.is_zombie, self.cfg.zombie_speed, self.cfg.human_speed)

    def step(self, dirs):
        """Advance one tick. dirs: (N, 2) desired directions; length 1 = full speed,
        longer vectors are clipped to length 1, shorter ones move proportionally slower."""
        dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 2)
        norm = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = np.where(norm > 1.0, dirs / np.maximum(norm, 1e-9), dirs)
        vel = dirs * self.max_speed()[:, None]

        self._move(vel)
        self.t += 1
        infections = self._infect()

        terminated = self.n_humans_left == 0
        truncated = (not terminated) and self.t >= self.cfg.max_steps
        return {
            "infections": infections,   # list of (zombie_id, human_id)
            "terminated": terminated,   # every human infected
            "truncated": truncated,     # hit max_steps
        }

    def _move(self, vel):
        """Move x then y separately so agents slide along walls instead of sticking.
        Blocked agents are snapped flush against the wall they hit."""
        eps = 1e-6
        limits = (self.cfg.map_w, self.cfg.map_h)
        for axis in (0, 1):
            new = self.pos.copy()
            new[:, axis] = np.clip(new[:, axis] + vel[:, axis], 0.0, limits[axis] - eps)
            blocked = ~self.walkable_at(new)
            if blocked.any():
                cur_tile = np.floor(self.pos[blocked, axis])
                v = vel[blocked, axis]
                new[blocked, axis] = np.where(v > 0, cur_tile + 1 - eps, cur_tile)
            self.pos[:, axis] = new[:, axis]

    # ---------- 1d: infection ----------

    def _infect(self):
        """Every human within infect_radius of a zombie turns. Credit goes to the nearest zombie.
        Only zombies from before this tick can infect, so infection doesn't chain within one step."""
        zombies = np.flatnonzero(self.is_zombie)
        humans = np.flatnonzero(~self.is_zombie)
        if len(zombies) == 0 or len(humans) == 0:
            return []
        d = np.linalg.norm(self.pos[humans, None, :] - self.pos[None, zombies, :], axis=2)  # (H, Z)
        caught = d.min(axis=1) <= self.cfg.infect_radius
        h_ids = humans[caught]
        z_ids = zombies[d[caught].argmin(axis=1)]
        self.is_zombie[h_ids] = True
        self.turned_at[h_ids] = self.t
        return list(zip(z_ids.tolist(), h_ids.tolist()))