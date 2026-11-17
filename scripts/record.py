"""Run scripted episodes headless and save replays.

    python -m scripts.record --episodes 5 --seed 0 --out runs/replays
"""
import argparse
from pathlib import Path

from outbreak.sim import World, Config
from outbreak.policies import scripted_actions
from outbreak.replay import EpisodeRecorder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/replays")
    args = p.parse_args()

    world = World(Config())
    for seed in range(args.seed, args.seed + args.episodes):
        world.reset(seed)
        rec = EpisodeRecorder(world, meta={"zombies": "greedy", "humans": "flee"})
        while True:
            info = world.step(scripted_actions(world))
            rec.record(info)
            if info["terminated"] or info["truncated"]:
                break
        path = rec.save(Path(args.out) / f"scripted_seed{seed}.npz")
        print(f"{path}  steps={world.t}  humans left={world.n_humans_left}")


if __name__ == "__main__":
    main()