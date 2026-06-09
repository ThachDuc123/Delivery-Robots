# Phân tích toàn bộ — Robot giao hàng RL (sensor-only)

Tài liệu này phân tích 3 thứ, **dựa trên code thật** trong dự án (không nói chung chung):

1. **Vì sao PPO / SAC / TD3 chạy khác nhau đến vậy** — do thuật toán hay do yếu tố khác?
2. **Các thuật toán cảm biến** dùng để "thấy" tường / người / vật cản (không camera).
3. **Phân tích đầy đủ con robot** đang dùng trong RL.

Số liệu trích từ lần train cuối (warm-start, pool hành lang chính), lưu ở
[results/SUMMARY.md](results/SUMMARY.md) và [results/train_log.txt](results/train_log.txt).

---

## PHẦN 1 — Vì sao 3 mô hình chạy khác nhau?

### 1.1. Kết quả thực tế (cùng map, cùng env, cùng pool tủ hành lang)

| Model | Reach (tới đúng tủ) | Reward TB | Va chạm/ep | Tủ #0 | #1 | #2 | #3 | #4 | #5 | Thời gian train |
|-------|:---:|:---:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| **PPO** | **100% (6/6)** | 130 | 0.33 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **4 phút** (200k) |
| **TD3** | 67% (4/6) | 69 | 0.33 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | 25 phút (120k) |
| **SAC** | 50% (3/6) | 55 | 0.25 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 39 phút (120k) |

Quy luật rõ: cả 3 đều đi được **tủ gần/giữa** (#1–#3), nhưng **chỉ PPO đi được tủ xa hai đầu** (#0, #5). Đây không phải ngẫu nhiên — nó phản ánh đúng bản chất 3 thuật toán.

### 1.2. Khác biệt CỐT LÕI: on-policy vs off-policy

Đây là nguyên nhân lớn nhất, không phải "model nào thông minh hơn".

- **PPO** ([configs/ppo.yaml](configs/ppo.yaml)) là **on-policy**: mỗi vòng nó thu thập `n_steps=2048` bước **bằng chính policy hiện tại**, cập nhật `n_epochs=10` lần rồi **vứt dữ liệu đi**. Vì luôn học từ trải nghiệm mới nhất, nó ổn định và rất nhanh hội tụ cho bài toán "đi tới đích" có reward dày (dense). → **4 phút đạt 100%.**

- **SAC / TD3** ([configs/sac.yaml](configs/sac.yaml), [configs/td3.yaml](configs/td3.yaml)) là **off-policy**: lưu mọi bước vào **replay buffer** (`buffer_size=300000`) và học lại từ dữ liệu cũ (`train_freq=1, gradient_steps=1`). Ưu điểm là tận dụng lại dữ liệu (sample-efficient về số bước môi trường), nhưng:
  - Mỗi bước môi trường kèm 1 bước gradient → **chậm hơn nhiều theo thời gian thực** (SAC 39', TD3 25' so với PPO 4').
  - Với mục tiêu thay đổi liên tục (mỗi episode 1 tủ khác), replay buffer chứa nhiều "kinh nghiệm cũ ứng với tủ khác" → tín hiệu học bị nhiễu, khó tổng quát tới tủ xa.

> **Kết luận 1:** Khác biệt lớn **chủ yếu do loại thuật toán** (on-policy vs off-policy) tương tác với đặc thù bài toán (reward dày, mục tiêu đổi mỗi episode). Bài toán này **thiên vị PPO**.

### 1.3. Khác biệt về EXPLORATION (cách thăm dò)

| | Cơ chế thăm dò | Hệ quả trên map dài 36m |
|--|--|--|
| **PPO** | Policy ngẫu nhiên (stochastic), `ent_coef=0.0` | Khám phá đủ nhờ phương sai action; đi xa tốt |
| **SAC** | **Entropy tối đa** (`ent_coef="auto"`) — tự thưởng cho hành vi ngẫu nhiên | Hay "lưỡng lự", dừng/đổi hướng giữa đường → **kẹt trước khi tới tủ xa** (#0,#4,#5) |
| **TD3** | Policy **tất định** + nhiễu Gauss nhỏ (`action_noise_std=0.1`) | Thăm dò yếu nhất; nếu chưa từng "lỡ" đi xa thì khó học được đường xa |

SAC bị entropy kéo về hành vi ngẫu nhiên → trên hành lang dài nó dễ tới tủ giữa nhưng "ngập ngừng" ở tủ xa. TD3 thăm dò ít nên chỉ giỏi vùng đã quen.

### 1.4. Khác biệt về tính ỔN ĐỊNH khi học

- **TD3** = "Twin Delayed DDPG": dùng **2 critic** lấy min (twin) để tránh đánh giá quá cao, và **trì hoãn cập nhật actor** (`policy_delay=2`), cộng **target policy smoothing** (`target_policy_noise=0.2, target_noise_clip=0.5`). → ổn định hơn DDPG, nhưng vẫn nhạy với siêu tham số.
- **SAC** tự điều chỉnh nhiệt độ entropy → bền hơn TD3 về va chạm (0.25 vs 0.33/ep) nhưng đánh đổi bằng việc tới đích kém hơn.
- **PPO** dùng **clip** (`clip_range=0.2`) giới hạn mỗi bước cập nhật → rất khó "học hỏng", phù hợp khi reward đã được shaping tốt.

### 1.5. Các yếu tố KHÁC thuật toán cũng góp phần

Để công bằng, khác biệt **không chỉ do thuật toán**:

1. **Ngân sách train khác nhau:** PPO 200k bước, SAC/TD3 120k (xem `TIMESTEPS` trong [train_corridor.py](train_corridor.py)). PPO được train nhiều hơn 1.67×.
2. **Warm-start giống nhau:** cả 3 đều nạp từ model L0 trong `_L0newmap_backup/` → xuất phát công bằng.
3. **Reward shaping chung** (xem [tasks/delivery_task.py](tasks/delivery_task.py) `compute_reward`): dense progress + thưởng giao đúng. Reward dày **ưu ái PPO** (on-policy tận dụng tín hiệu tức thời tốt).
4. **Cùng kiến trúc mạng** `[256,256]` MLP, cùng `gamma=0.99`, cùng seed → loại trừ khác biệt do mạng/seed.

> **Kết luận tổng (Phần 1):**
> Khác biệt đến **~70% từ loại thuật toán** (on-policy PPO hợp bài toán reward-dày, mục-tiêu-đổi hơn off-policy), **~30% từ yếu tố thực nghiệm** (PPO được train nhiều bước hơn, exploration của SAC/TD3 không hợp map dài). **Không phải** vì cảm biến hay vật lý khác nhau — cả 3 dùng **chung một env, chung cảm biến, chung robot**.

---

## PHẦN 2 — Thuật toán cảm biến (làm sao "thấy" tường, người, vật cản mà không có camera)

Toàn bộ ở [envs/sensors.py](envs/sensors.py). **Không dùng camera/ảnh** — chỉ hình học + động học.

### 2.1. LiDAR 2D 360° + ToF — thuật toán **Ray casting (raycasting)**

Đây là cách robot "thấy" tường/người/vật cản.

- **Thuật toán:** bắn tia (ray) từ tâm robot theo nhiều góc, tìm giao điểm đầu tiên với vật thể. Trong code dùng `pybullet.rayTestBatch` — phóng đồng thời cả chùm tia (batch) để nhanh.
- **LiDAR:** `num_rays=36` tia trải đều 360° (`np.linspace(0, 2π, 36)`), tầm `max_range=6.0 m`.
- **ToF/siêu âm:** 4 tia ở 4 hướng Trước/Trái/Sau/Phải (`[0, π/2, π, -π/2]`), tầm 2 m — mô phỏng cảm biến khoảng cách điểm.
- **Đầu ra chuẩn hóa:** mỗi tia trả về `fraction` ∈ [0,1] = tỉ lệ quãng đường tới điểm va (1.0 = không trúng gì). Đây chính là khoảng cách tới **tường / người / vật cản** đã chuẩn hóa.
- **Phân biệt vật thể?** LiDAR/ToF **không phân loại** đâu là tường hay người — nó chỉ cho **khoảng cách**. Robot học từ *mẫu hình khoảng cách* (tường = bề mặt dài liên tục; người = cụm gần di chuyển). Việc nhận biết "người" để xử lý theo luật (dừng/né/bíp) được làm riêng ở **safety shield** (xem mục 2.5), nơi lấy được vị trí + vận tốc người.

```
read_lidar(pos, yaw):
    với mỗi góc a trong 36 góc:
        tia từ (x, y, h=0.20) tới (x + cos·R, y + sin·R)
    rayTestBatch → fraction[i] ∈ [0,1]
    + nhiễu Gauss(0, lidar_noise) → mô phỏng nhiễu cảm biến thật
```

### 2.2. IMU — thuật toán **sai phân hữu hạn (finite difference)** cho gia tốc

- Gia tốc tuyến tính không đo trực tiếp mà **tính từ thay đổi vận tốc**: `a = (v_t − v_{t−1}) / dt`, rồi xoay về hệ thân robot (body frame) bằng ma trận quay yaw (xem [robot.py](envs/robot.py) `apply_velocity`).
- Vận tốc góc `omega` và `yaw` lấy trực tiếp từ trạng thái.
- IMU 7 chiều: `[ax, ay, az, ωx, ωy, ωz, yaw]`. Trong mô phỏng 2D thì az, ωx, ωy ≈ 0.

### 2.3. Odometry — tích phân động học bánh xe (dead reckoning)

- `[x, y, yaw, vx, vy, omega]` lấy từ vị trí tích lũy của robot — tương đương **đếm vòng quay bánh (encoder)** trên robot thật.
- Trên robot thật odometry sẽ trôi (drift); ở đây ta thêm nhiễu Gauss để mô phỏng.

### 2.4. Mô phỏng NHIỄU cảm biến (đặc tả trong YAML)

Mọi cảm biến đều cộng nhiễu Gauss, cường độ cấu hình trong
[configs/default.yaml](configs/default.yaml) mục `env.sensors`:
`lidar.noise_std`, `tof.noise_std`, `imu.*_noise_std`, `odometry.*_noise_std`.
Ở curriculum L5 (domain randomization) còn nhân thêm `noise_scale=1.5` để robot
học bền với cảm biến nhiễu.

### 2.5. Phát hiện va chạm — **circle-vs-rectangle** (không phải raycast)

Tách bạch với cảm biến: việc *đã đụng hay chưa* tính bằng hình học chính xác trong
[robot.py](envs/robot.py) `_detect_collision` → `pybullet.getContactPoints` (truy vấn tiếp xúc vật lý), kiểm tra độ xuyên (penetration). Nếu nước đi làm xuyên vật thể, robot **hủy nước đi** (đứng yên) → mô phỏng "đụng tường thì không đi xuyên được".

### 2.6. "Thấy người" để xử lý theo luật — **reactive safety shield**

Ở [envs/safety_shield.py](envs/safety_shield.py). Đây là lớp phản ứng *trên* policy RL, mô phỏng một **people-tracker dựa LiDAR**: nó lấy vị trí + vận tốc người ([envs/pedestrians.py](envs/pedestrians.py) `get_states`) rồi quyết định theo thuật toán hình học:

- Chiếu vị trí người lên trục **dọc đường đi** (along) và **ngang** (perp) của robot.
- Nếu người trong dải đường đi (`path_halfwidth`) và trong tầm quan tâm (`concern_distance`):
  - Tới ngược chiều, đủ chỗ → **lách** (sidestep) vào khe trống (có kiểm tra `wall_margin` để không quẹt tường).
  - Tới ngược chiều, hết chỗ → **dừng đợi** (yield).
  - Đi sau người → **đi chậm theo**, có khe đủ lớn → **vượt** (overtake).
  - Bị chặn hẳn → **dừng + bíp** (beep).
- Có **hysteresis** (cam kết một bên để khỏi đảo trái/phải liên tục).

> **Tóm tắt Phần 2:** cảm biến tường/vật cản = **raycasting** (LiDAR/ToF). Va chạm = **hình học circle-vs-rect + truy vấn tiếp xúc PyBullet**. Định vị = **odometry + IMU (sai phân)**. "Hiểu" người để né = **safety shield hình học** trên dữ liệu vị trí/vận tốc người. Tất cả **không dùng ảnh/camera**.

---

## PHẦN 3 — Phân tích đầy đủ con robot

Toàn bộ ở [envs/robot.py](envs/robot.py), tham số trong [configs/default.yaml](configs/default.yaml) mục `env.robot` và `env.mechanism`.

### 3.1. Khung gầm & truyền động: **đế Mecanum 4 bánh (holonomic)**

- Kích thước đế `0.50×0.50×0.25 m`, khối lượng `18 kg`, bán kính bánh `0.05 m`.
- **Động học nghịch Mecanum** (`mecanum_wheel_speeds`): chuyển lệnh `(vx, vy, ω)` thành 4 tốc độ bánh:
  ```
  FL = (vx − vy − (lx+ly)·ω) / r
  FR = (vx + vy + (lx+ly)·ω) / r
  RL = (vx + vy − (lx+ly)·ω) / r
  RR = (vx − vy + (lx+ly)·ω) / r
  ```
  với `lx=ly=0.20 m` (nửa khoảng cách bánh), `r=0.05 m`. Đặc tính Mecanum: **đi ngang (vy ≠ 0) không cần xoay** → linh hoạt trong hành lang hẹp.
- **Lưu ý phạm vi (quan trọng):** ở giai đoạn sensor-only này, đế được điều khiển **holonomic kinematic** — tích phân trực tiếp `(vx,vy,ω)` rồi giải va chạm bằng truy vấn tiếp xúc; **chưa** mô phỏng mô-men xoắn từng bánh/con lăn. `mecanum_wheel_speeds` vẫn được tính để dùng cho phạt năng lượng và để sau này cắm URDF có con lăn thật (xem TODO trong file).

### 3.2. Giới hạn động học

- `max_linear_speed = 0.8 m/s` (cho cả vx, vy), `max_yaw_rate = 1.5 rad/s`.
- Điều khiển ở `control_hz = 20 Hz` (mỗi bước RL = 0.05 s); vật lý PyBullet ở `sim_dt = 0.01 s`.

### 3.3. Không gian quan sát (observation) — **78 chiều, sensor-only**

Ghép trong [corridor_delivery_env.py](envs/corridor_delivery_env.py) `_build_obs`, mọi giá trị chuẩn hóa về [−1, 1]:

| Khối | Số chiều | Nguồn |
|------|:---:|------|
| LiDAR 360° | 36 | raycast |
| ToF (F/L/B/R) | 4 | raycast |
| IMU (ax,ay,az,ωx,ωy,ωz,yaw) | 7 | sai phân + trạng thái |
| Odometry (x,y,yaw,vx,vy,ω) | 6 | tích phân động học |
| Pose tương đối tới **waypoint** đích | 3 | planner + map (đã biết tọa độ tủ) |
| Mặt nạ tủ còn phải giao | 16 | nhiệm vụ |
| Số bưu kiện đang mang | 1 | nhiệm vụ |
| Cơ cấu (arm_lift, tray, carousel, bumper, pin) | 5 | robot |
| **Tổng** | **78** | |

### 3.4. Không gian hành động (action) — `Box(3,) = [vx, vy, ω]`

3 số liên tục ∈ [−1,1], nhân với giới hạn tốc độ → lệnh đế Mecanum. **Đây là lý do bài toán cần thuật toán cho hành động liên tục** — cả PPO, SAC, TD3 đều hỗ trợ. (Cơ cấu giao hàng *không* nằm trong action mà chạy bằng macro tự kích hoạt, xem 3.6.)

### 3.5. Lớp thực thi (execution layer) — làm mượt + an toàn

Chạy *trên* output của policy, cấu hình ở `env.control` / `env.safety`:
- **Action smoothing** (`action_smoothing=0.6`): lọc thông thấp `a = α·a_prev + (1−α)·a_raw` → giảm giật/rung.
- **Safety shield** (Phần 2.6): né người.
Hai lớp này đổi *cách thực thi*, **không** đổi cách policy được train → không cần train lại.

### 3.6. Cơ cấu giao hàng — **macro tham số 2 bậc tự do**

- `carousel` 6 ngăn chứa bưu kiện; cơ cấu `arm_lift` (nâng 0–0.5 m) + `tray_extend` (đẩy khay 0–0.35 m) + `probe_auth` (mô phỏng mở khóa tủ 0–0.08 m).
- Khi robot vào **vùng dock** của đúng tủ đích (`dock_zone_radius=0.65 m`), một **macro** tự chạy theo trình tự: nâng → đẩy khay → mở khóa → nhả bưu kiện → thu về (`macro_steps=24` bước). Xem `step_macro` trong [robot.py](envs/robot.py).
- Đây là **hook để học bằng RL sau** — hiện cố định để tập trung vào điều hướng.

### 3.7. Pin & các phạt vật lý

- Pin (`battery_capacity=1.0`) hao theo quãng đường (`battery_drain_per_m`) + hao đứng yên (`battery_drain_idle`) → khuyến khích đường ngắn.
- Reward phạt: va chạm, va người (nặng hơn), thời gian, năng lượng, **rung lắc** (biến thiên ω), **nghiêng/lật** (`tilt_threshold_deg=35°`). Xem `compute_reward` / `check_termination` trong [tasks/delivery_task.py](tasks/delivery_task.py).

### 3.8. Định vị tới đích — **planner toàn cục + policy cục bộ**

Vì policy chỉ thấy LiDAR + hướng-tới-waypoint (không có bản đồ toàn cục trong đầu), robot dùng kiến trúc chuẩn ngành:
- **Global planner** ([envs/planner.py](envs/planner.py)): **BFS trên lưới chiếm dụng** (occupancy grid) từ vị trí robot tới dock của tủ đích, rồi rút gọn bằng **string-pulling (line-of-sight)** thành các **waypoint**. Tọa độ tủ là **đã biết từ map** (`force_locker_id` / `scene_map`).
- **Local controller** (chính là policy RL): bám waypoint kế tiếp, tự né tường/vật cản bằng LiDAR.
→ Nhờ vậy robot đi được quanh phòng/khúc cua mà không cần "nhìn xuyên tường".

---

## Tóm tắt 1 dòng cho mỗi câu hỏi

1. **Vì sao 3 model khác nhau?** Chủ yếu do **on-policy (PPO) vs off-policy (SAC/TD3)** kết hợp với bài toán reward-dày + mục-tiêu-đổi-mỗi-episode (ưu ái PPO), cộng exploration của SAC/TD3 không hợp hành lang dài và PPO được train nhiều bước hơn. Không phải do cảm biến/vật lý — cả 3 dùng chung env.
2. **Cảm biến thấy tường/người bằng gì?** **Raycasting** (LiDAR 36 tia + 4 ToF) cho khoảng cách; **circle-vs-rect + contact query** cho va chạm; **odometry + IMU (sai phân)** cho định vị; **safety shield hình học** để hiểu/né người. Không camera.
3. **Robot là gì?** Đế **Mecanum 4 bánh holonomic** (điều khiển kinematic ở pha này), obs **78 chiều sensor-only**, action `[vx,vy,ω]` liên tục, cơ cấu giao 2-bậc bằng macro, có planner BFS + waypoint dẫn đường, kèm lớp làm mượt + safety shield khi chạy.
