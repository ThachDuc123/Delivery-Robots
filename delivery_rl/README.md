# delivery_rl — Sensor-only Apartment-Corridor Delivery Robot (RL)

Base code for training an autonomous delivery robot to drive down an apartment
corridor and deliver parcels to lockers, using **sensors only (no camera / no
vision)**. Trains and compares **PPO, SAC and TD3** (Stable-Baselines3) on a
single shared Gymnasium + PyBullet environment, with YAML configs and
TensorBoard logging.

> Runs out of the box with **no assets**: if a URDF / `scene_map.json` is not
> found, the corridor, lockers, dock and robot are generated from PyBullet
> primitives. See *Plugging in assets* below.

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on Linux/Mac
pip install -r delivery_rl/requirements.txt
```

Python 3.10+ required.

## Project layout

```
delivery_rl/
  envs/
    corridor_delivery_env.py   # Gymnasium env (ties everything together)
    robot.py                   # mecanum kinematics + delivery mechanism (+ URDF hook)
    sensors.py                 # LiDAR / ToF / IMU / odometry via raycast (no vision)
    world.py                   # corridor + alcoves + lockers + dock (scene_map.json or primitives)
    pedestrians.py             # moving residents (curriculum L4 stub)
  tasks/delivery_task.py       # manifest, reward, termination, curriculum
  configs/                     # default.yaml + ppo/sac/td3.yaml (inherit default)
  train.py                     # --algo {ppo,sac,td3}, TensorBoard, checkpoints -> runs/<algo>/
  eval.py                      # delivery rate, collisions/ep, avg time, avg reward
  notebooks/experiments.ipynb  # smoke test + short train + compare 3 algos (headless)
  requirements.txt
  assets/                      # drop scene_map.json + robot/ URDF here (optional)
```

## Observation (sensor-only, flat vector in [-1, 1])

`36 LiDAR | 4 ToF (F/L/B/R) | 7 IMU (accel xyz, gyro xyz, yaw) | 6 odometry
(x,y,yaw,vx,vy,omega) | 3 relative pose to current target locker | N_lockers
remaining-mask | 1 parcels-carried | 5 mechanism (arm_lift, tray_extend,
carousel, bumper, battery)`.

## Action

`Box(3,) = [vx, vy, omega]` mecanum base velocities (scaled by configured maxima
and converted to 4 wheel speeds via `mecanum_wheel_speeds`). The delivery macro
(lift → extend → auth → release → retract) auto-triggers inside a locker dock
zone — a hook left to be learned by RL later.

## Train

```bash
python delivery_rl/train.py --algo ppo
python delivery_rl/train.py --algo sac --timesteps 50000 --level 0
python delivery_rl/train.py --algo td3
tensorboard --logdir delivery_rl/runs
```

## Evaluate

```bash
python delivery_rl/eval.py --algo ppo --model delivery_rl/runs/ppo/ppo_final.zip --episodes 20
```

## Notebook

`notebooks/experiments.ipynb` runs **headless PyBullet (DIRECT)** and:
1. checks versions, 2. smoke-tests the env (≈300 random steps, prints
obs/action shapes), 3. short-trains PPO/SAC/TD3 on the same env/seed,
4. evaluates all three (table + reward-vs-step plot), 5. concludes which model
is best per metric.

## Curriculum (set `env.curriculum.level` in YAML, or `--level`)

| Level | Scenario |
|------:|----------|
| L0 | 1 parcel, nearest locker to dock |
| L1 | 1 parcel, random locker |
| L2 | multiple parcels, self-routing (target = nearest remaining) |
| L3 | + static obstacles |
| L4 | + moving residents (stub) |
| L5 | + domain randomization (mass / friction / sensor noise / spawn) |

## Plugging in assets (later)

- **Scene map:** drop `assets/scene_map.json` with locker/dock coordinates
  (same schema as the auto-emitted `assets/scene_map.generated.json`). It is
  loaded automatically if present.  TODO marker: `envs/world.py`.
- **Robot URDF/USD:** drop `assets/robot/delivery_robot.urdf`; it is loaded
  instead of the primitive robot. Replace the holonomic shortcut in
  `envs/robot.py::apply_velocity` with wheel-velocity motor control once the
  URDF has mecanum rollers.  TODO markers: `envs/robot.py`.

## Execution-time layers (smoothing + pedestrian safety shield)

These run **on top of the trained policy at execution time** — they change how
the policy is *executed*, not how it was trained, so no retraining is needed.
All are config-gated under `env.control` / `env.safety` in `configs/default.yaml`.

- **Action smoothing** (`env.control.action_smoothing`): a low-pass filter on the
  commanded `[vx, vy, omega]` to reduce jerk / vibration and give steadier motion.
- **Reactive pedestrian-avoidance shield** (`envs/safety_shield.py`): watches the
  nearest resident in the robot's path and overrides the command per these rules:
  - approaching head-on, room to pass → **sidestep** into the gap;
  - approaching head-on, no room → **yield** (stop and wait);
  - catching up behind someone → **slow-follow**, then **overtake** once a gap opens;
  - directly blocked, can't pass → **stop and beep**.
  It has hysteresis (commits to one pass-side) to avoid left/right oscillation.
- **Target marker**: a **red dot** hovers over the locker the robot is currently
  routing to (drawn in the 3D scene; visible in the GIFs / overhead camera).

Tune everything in `configs/default.yaml` (`concern_distance`, `slow_distance`,
`stop_distance`, `overtake_speed`, `commit_steps`, ...). Verified across PPO/SAC/TD3
with 2–3 moving residents: the robot reaches its goal while exhibiting all four
behaviours and recording zero terminal collisions.

## GIF gallery

`python delivery_rl/make_gallery.py` records, for every trained model and each
scenario (plain navigation, static obstacles, moving pedestrians), one annotated
GIF that runs **until the robot reaches its target**. Output: `results/gifs/*.gif`
(+ `manifest.json`). The notebook displays these inline (Section C4).

## Notes / scope

- This is the **sensor-only navigation phase**: the base is driven holonomically
  (kinematic) and collisions are resolved via PyBullet contact queries; full
  wheel/roller dynamics are deferred to the URDF phase.
- No camera, no vision, no features beyond the spec.
