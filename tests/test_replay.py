import os
import numpy as np
from outbreak.sim import World, Config
from outbreak.policies import scripted_actions
from outbreak.replay import EpisodeRecorder, load_episode


def run_and_record(seed, max_steps=300):
    world = World(Config(max_steps=max_steps)).reset(seed)
    rec = EpisodeRecorder(world, meta={"note": "test"})
    positions = [world.pos.copy()]
    while True:
        info = world.step(scripted_actions(world))
        rec.record(info)
        positions.append(world.pos.copy())
        if info["terminated"] or info["truncated"]:
            break
    return world, rec, np.stack(positions)


def test_roundtrip(tmp_path):
    world, rec, positions = run_and_record(seed=3)
    ep = load_episode(rec.save(tmp_path / "ep.npz"))

    assert len(ep) == world.t + 1
    assert ep.seed == 3
    assert ep.meta == {"note": "test"}
    assert ep.config["max_steps"] == 300
    assert (ep.grid == world.grid).all()
    assert np.allclose(ep.pos, positions, atol=1e-4)        # float32 storage
    assert (ep.is_zombie[-1] == world.is_zombie).all()
    assert (ep.turned_at == world.turned_at).all()


def test_infection_log_matches_frames(tmp_path):
    _, rec, _ = run_and_record(seed=1)
    ep = load_episode(rec.save(tmp_path / "ep.npz"))
    for t, z, h in ep.infections:
        assert not ep.is_zombie[t - 1, h] and ep.is_zombie[t, h]   # turned exactly at step t
        assert ep.is_zombie[t - 1, z]                              # infector was already a zombie
        assert ep.turned_at[h] == t
    n_start = ep.is_zombie[0].sum()
    assert ep.is_zombie[-1].sum() == n_start + len(ep.infections)


def test_replay_matches_resimulation(tmp_path):
    # same seed + same deterministic policies => identical episode
    _, rec, _ = run_and_record(seed=7)
    ep = load_episode(rec.save(tmp_path / "ep.npz"))
    _, _, positions = run_and_record(seed=7)
    assert np.allclose(ep.pos, positions, atol=1e-4)


def test_replay_frames_render_headless(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from outbreak.render import Renderer
    _, rec, _ = run_and_record(seed=0, max_steps=20)
    ep = load_episode(rec.save(tmp_path / "ep.npz"))
    renderer = Renderer(ep.frame(0))
    for t in range(len(ep)):
        renderer.draw(ep.frame(t), "test")
    renderer.close()