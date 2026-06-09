"""Headless sanity checks for CorridorNavEnv (no training)."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import math
import numpy as np
from corridor_env import CorridorNavEnv


def basic_api():
    env = CorridorNavEnv(seed=0, randomize_layout=False, layout="corridor")
    obs, info = env.reset()
    assert env.observation_space.contains(obs), "reset obs out of bounds"
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs), "step obs out of bounds"
        if term or trunc:
            break
    print(f"[api] obs_dim={obs.shape[0]} action_n={env.action_space.n} OK")
    env.close()


def sensors_and_goal():
    for layout in ("corridor", "corridor_obstacles", "corridor_turn"):
        env = CorridorNavEnv(seed=1, randomize_layout=False, layout=layout)
        obs, info = env.reset()
        env._cast_rays()
        left = env.last_ray_distances[env.left_ray_idx]
        right = env.last_ray_distances[env.right_ray_idx]
        reachable = env._is_reachable(env.robot_pos, env.goal_pos)
        sep = float(np.linalg.norm(env.goal_pos - env.robot_pos))
        print(f"[{layout}] start={env.robot_pos.round(0)} goal={env.goal_pos.round(0)} "
              f"sep={sep:.0f}px reachable={reachable} sideL={left:.0f} sideR={right:.0f}")
        assert reachable, f"goal not reachable on {layout}"
        env.close()


def _goal_angle_offset(env):
    g = env.goal_pos - env.robot_pos
    rel = env._wrap_angle(math.atan2(g[1], g[0]) - env.robot_heading)
    offset, _ = env._corridor_offset()
    return rel, offset


def centered_controller(env):
    rel, offset = _goal_angle_offset(env)
    if rel > 0.15:
        return 2
    if rel < -0.15:
        return 1
    if offset > 0.12:
        return 2
    if offset < -0.12:
        return 1
    return 0


def hug_controller(env):
    """Deliberately drive along one wall (what we are trying to AVOID)."""
    rel, offset = _goal_angle_offset(env)
    # try to keep the +90 (right) wall close -> offset strongly negative
    target = -0.7
    if rel > 0.45:
        return 2
    if rel < -0.45:
        return 1
    if offset > target + 0.1:
        return 2
    if offset < target - 0.1:
        return 1
    return 0


def run(policy, layout, seed):
    env = CorridorNavEnv(seed=seed, randomize_layout=False, layout=layout)
    obs, info = env.reset()
    total, steps = 0.0, 0
    reached = collided = False
    while True:
        a = policy(env)
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        reached = info["goal_reached"]
        collided = info["collision"]
        if term or trunc:
            break
    return dict(reward=total, steps=steps, reached=reached,
                collided=collided, mean_off=info["mean_abs_offset"])


def compare_centered_vs_hug():
    print("\n[reward check] centred controller should beat wall-hugging:")
    for layout in ("corridor", "corridor_obstacles"):
        c = run(centered_controller, layout, seed=3)
        h = run(hug_controller, layout, seed=3)
        print(f"  {layout:20s} centred: reward={c['reward']:7.1f} reached={c['reached']} "
              f"mean|off|={c['mean_off']:.2f} | hug: reward={h['reward']:7.1f} "
              f"reached={h['reached']} mean|off|={h['mean_off']:.2f}")
        assert c["reward"] > h["reward"], f"centring not rewarded on {layout}!"
        assert c["reached"], f"centred controller failed to reach goal on {layout}"
    print("  -> centring is correctly rewarded and the task is solvable.")


def render_check():
    env = CorridorNavEnv(seed=0, randomize_layout=False, layout="corridor",
                         render_mode="rgb_array")
    env.reset()
    frame = env.render(info={"layout": "corridor"}, reward=0.0)
    assert frame is not None and frame.shape == (env.map_height, env.map_width, 3)
    print(f"[render] rgb_array frame {frame.shape} OK")
    env.close()


if __name__ == "__main__":
    basic_api()
    sensors_and_goal()
    compare_centered_vs_hug()
    render_check()
    print("\nALL SMOKE TESTS PASSED")
