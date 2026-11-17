"""Record episodes to .npz files and load them back for viewing or analysis (heatmaps, reels).

Usage while running any episode (scripted, training, evaluation):
    rec = EpisodeRecorder(world)      # call right after world.reset(seed)
    info = world.step(actions)
    rec.record(info)                  # after every step
    rec.save("runs/replays/ep.npz")
"""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np


class EpisodeRecorder:
    def __init__(self, world, meta=None):
        self.world = world
        self.meta = dict(meta or {})          # anything extra: checkpoint name, generation, ...
        self.grid = world.grid.copy()
        self.pos = [world.pos.astype(np.float32)]
        self.is_zombie = [world.is_zombie.copy()]
        self.infections = []                  # rows of (t, zombie_id, human_id)

    def record(self, info):
        w = self.world
        self.pos.append(w.pos.astype(np.float32))
        self.is_zombie.append(w.is_zombie.copy())
        self.infections += [(w.t, z, h) for z, h in info["infections"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        w = self.world
        np.savez_compressed(
            path,
            seed=w.seed,
            config=json.dumps(asdict(w.cfg)),
            meta=json.dumps(self.meta),
            grid=self.grid.astype(np.int8),
            pos=np.stack(self.pos),                       # (T+1, N, 2)
            is_zombie=np.stack(self.is_zombie),           # (T+1, N)
            turned_at=w.turned_at,                        # (N,)
            infections=np.array(self.infections, dtype=np.int32).reshape(-1, 3),
        )
        return path


class Frame:
    """One moment of a replay. Has the same fields the Renderer reads from a World."""
    def __init__(self, ep, t):
        self.grid = ep.grid
        self.seed = ep.seed
        self.t = t
        self.pos = ep.pos[t]
        self.is_zombie = ep.is_zombie[t]

    @property
    def n_humans_left(self):
        return int((~self.is_zombie).sum())


class Episode:
    def __init__(self, path):
        with np.load(path) as d:
            self.seed = int(d["seed"])
            self.config = json.loads(str(d["config"]))
            self.meta = json.loads(str(d["meta"]))
            self.grid = d["grid"].astype(int)
            self.pos = d["pos"]
            self.is_zombie = d["is_zombie"]
            self.turned_at = d["turned_at"]
            self.infections = d["infections"]

    def __len__(self):
        return len(self.pos)                  # number of frames = steps + 1

    def frame(self, t):
        return Frame(self, t)


def load_episode(path):
    return Episode(path)