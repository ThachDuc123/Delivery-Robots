# nav2d — Sensor-only 2D corridor navigation (PPO, pure RL)

A lightweight **2D, numpy-only** sandbox where a robot learns to **navigate using
only a LiDAR fan + relative goal bearing — no map, no global planner**. Every
episode draws a *new* random corridor (straight / L-turn / S-curve / arc / U-turn
/ niches), so a trained policy must learn to *follow corridors* and therefore
**generalises to unseen maps**. The task is a **round trip**: drive to the goal,
then return to the start.

This is the fast research sandbox that precedes the heavier PyBullet project in
`../delivery_rl/`. Here we get millions of steps in minutes to nail the core
navigation behaviour first.

## Files
- `world2d.py` — procedural corridor generator (centerline → offset walls), LiDAR
  raycast + circle-vs-segment collision, all numpy.
- `nav_env.py` — Gymnasium env. Obs = `24 LiDAR | goal sin/cos/dist | fwd-clear |
  prev_action(2) | phase`. Action = `[forward, turn]`. Round-trip reward.
- `train_ppo.py` — PPO + **VecFrameStack(4)** (short-term memory) +
  **SubprocVecEnv** (parallel) + **VecNormalize** + TensorBoard.
- `eval_ppo.py` — deterministic eval on unseen seeds; path grid + GIFs.
- `render2d.py` — top-down matplotlib drawing + GIF recorder.

## Run
```bash
python nav2d/train_ppo.py --timesteps 1500000 --n-envs 8 --subproc
python nav2d/eval_ppo.py --episodes 20 --gifs
tensorboard --logdir nav2d/runs/tb
```

## Result (1.5M steps, ~23 min CPU; eval on 20 unseen maps per style)

| Style | reach goal | round-trip | collision |
|-------|:---:|:---:|:---:|
| straight, L_turn, S_curve, arc, niches | 100% | **100%** | 0% |
| U_turn | 100% | 85% | 15% |
| **ALL** | **100%** | **98%** | **2%** |

Artifacts: `runs/ppo_nav2d.zip` (+ `_vecnorm.pkl`), `results/paths_grid.png`,
`results/gifs/nav2d_<style>.gif`.

## Why PPO learns this (and a plain controller doesn't)
A hand-written follower fails on curves (0/6 on L/U/S/arc in our smoke test).
PPO with frame-stacking learns to read the *shape* of the LiDAR returns and turn
ahead of bends — reaching 100% on the same curves it never saw during training.

## Next steps (planned)
- **RecurrentPPO (LSTM)** for longer memory → nail U-turn round-trip and longer
  dead-end mazes.
- Harder maps: junctions (T/+), multi-door rooms, dynamic obstacles.
- Port the learned navigation behaviour back into the PyBullet `delivery_rl` env.
