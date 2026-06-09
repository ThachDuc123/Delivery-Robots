# -*- coding: utf-8 -*-
"""Builds TECHNICAL_REPORT.ipynb — báo cáo kỹ thuật Hybrid Navigation (render LaTeX).
Dùng raw-string để giữ nguyên ký hiệu LaTeX; nbconvert -> HTML render bằng MathJax.
"""
import json, os
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TECHNICAL_REPORT.ipynb")
cells = []
def md(s): cells.append(s)

md(r"""# BÁO CÁO KỸ THUẬT TOÀN DIỆN
## Hệ thống Điều hướng Robot Lai (Hybrid Navigation System) trong môi trường hành lang hẹp

*Lead AI/Robotics Engineer — rà soát toàn bộ codebase (Giai đoạn 1 → 5). Mọi công thức khớp đúng hằng số trong mã nguồn (`multistop_env.py`, `hybrid_controller.py`, `safety_shield.py`, `train_ms_mixed_robust.py`).*""")

md(r"""## PHẦN 1 — INTRODUCTION: CẢI TIẾN & KẾ HOẠCH

### 1.1. Vì sao phải mô phỏng trước khi triển khai thực tế
- **An toàn & chi phí:** policy RL chưa hội tụ cho hành vi hỗn loạn (đâm tường, xoay tại chỗ). Thử trên robot thật gây hư phần cứng, nguy hiểm cho người. Mô phỏng cho phép "phá" hàng triệu lần với chi phí $\approx 0$.
- **Tốc độ thu thập dữ liệu:** RL cần $10^6$–$10^7$ bước tương tác; mô phỏng nhanh hơn thời gian thực hàng trăm lần và song song hoá ($8$ env `SubprocVecEnv`).
- **Sinh dữ liệu chủ động:** tạo **1000 bản đồ** đa dạng (Domain Randomization) — bất khả thi nếu xây 1000 toà nhà thật.
- **Đo lường có kiểm soát:** mọi đại lượng (va chạm, độ mượt, độ phủ) đo chính xác.

### 1.2. Cải tiến của kiến trúc Hybrid — phân tích *do đâu* phải cải tiến

Bảng tổng hợp (chi tiết từng mục bên dưới):

| # | Vấn đề (triệu chứng) | Nguyên nhân GỐC | Cải tiến (code cụ thể) |
|---|---|---|---|
| 1 | Kẹt / "xoay compa" ở ngõ cụt, S/U-curve | POMDP aliasing + thiếu đường toàn cục + thiếu bộ nhớ | SLAM + A\* + **LSTM** + cho lùi |
| 2 | Đi sát mép tường, đánh võng | A\* thô cắt cua sát tường; RL không bị phạt rung lái | **Inflation 0.3m** + reward $r_{wall},r_{smooth},r_{straight}$ + lookahead $0.7$ m |
| 3 | Va chạm người động, học không vào | Một mạng học cả tĩnh+động = phương sai cao; người không có trong map tĩnh | Tách **Safety Shield** (reactive) |
| 4 | Zero-shot loạn trên map lạ | **Distribution shift** (overfit địa hình train) | **Domain Randomization 1000 map** + Early-Stopping |
| 5 | Bế tắc / trạm không tới được | Kiến trúc tĩnh không xử lý ngõ cụt bất ngờ | **Mission Controller** + **Dynamic Rerouting** |
| 6 | Báo "đã giao" sai (false positive) | Đếm theo cạn-waypoint thay vì tới-thật | Chỉ tính khi $\lVert p-\text{tgt}\rVert<0.5$ + replan từ vị trí thật |

---

#### Cải tiến 1 — Trị dứt "khựng lại / xoay compa" ở ngõ cụt & đường cong

- **Triệu chứng (GĐ1, RL thuần PPO):** tới ngõ cụt hoặc giữa S-curve/U-curve, robot dừng, quay tới quay lui như "kim la bàn", không thoát.
- **Nguyên nhân gốc (3 lớp):**
  1. **POMDP aliasing:** quan sát ego-centric (LiDAR + bearing) tại nhiều điểm trong hành lang **trông giống hệt nhau** → policy không phân biệt được "đang ở đâu", ra quyết định mâu thuẫn.
  2. **Thiếu định hướng toàn cục:** RL thuần chỉ thấy cục bộ, **không biết phía nào dẫn tới đích** → ở ngã ba/ngõ cụt chọn sai rồi dao động.
  3. **Thiếu bộ nhớ:** mạng feed-forward **không nhớ hướng đã đi vào** → tại ngõ cụt không quyết được "phải quay ra hướng ngược lại".
- **Cải tiến:** (a) **SLAM** dựng occupancy grid để **có thể** lập kế hoạch; (b) **A\*** cấp waypoint toàn cục → bearing tới Lookahead luôn chỉ đúng hướng thoát (khử aliasing); (c) đổi sang **RecurrentPPO (LSTM)** mang trạng thái ẩn $h_t$ → nhớ hướng vào; (d) cho **lùi** ($v\ge -0.4v_{max}$) để de ra khỏi ngõ cụt.

#### Cải tiến 2 — Đi "chuẩn form": bám tim đường, hết đánh võng

- **Triệu chứng:** robot men sát tường hoặc lượn sóng quanh đường đi.
- **Nguyên nhân gốc:** (a) A\* trên grid **thô** tìm đường ngắn nhất → ôm sát góc/tường; (b) bộ lái không bị phạt khi **đổi lái liên tục** → cộng hưởng dao động.
- **Cải tiến:** (a) **`inflate_map` bơm phồng tường $0.3$ m** → A\* buộc chạy ở **tim hành lang**, luôn chừa khoảng cách an toàn; (b) Lookahead **ngắn $0.7$ m** (không cắt cua xa); (c) reward $r_{wall}$ (phạt sát tường), $r_{smooth}=2w_{dturn}\Delta\omega$ (phạt rung lái), $r_{straight}$ (thưởng đi thẳng ổn định). *Đo được: độ mượt $\approx 1.4$–$3.0^\circ$/step.*

#### Cải tiến 3 — Tách an toàn động: Safety Shield

- **Triệu chứng:** thêm người đi bộ vào, train RL học **không vào** (reward đi ngang ~$1.43$M bước), va chạm người cao.
- **Nguyên nhân gốc:** ép **một mạng** học đồng thời (tránh tường tĩnh) **và** (né người động) → không gian học bùng nổ, phương sai lớn; hơn nữa **vị trí người không nằm trong map tĩnh** mà A\* dùng.
- **Cải tiến:** **tách module Safety Shield** phản ứng theo luật (proportional steering, Phần 4.3c), chỉ lo người động; LiDAR cấp cho RL **loại bỏ người**. Mỗi module đơn giản, kiểm thử độc lập, robust.

#### Cải tiến 4 — Chống overfit địa hình: Domain Randomization

- **Triệu chứng:** model train trên ít map chạy tốt map quen nhưng **loạn trên map lạ** (apartment_complex: dock hẹp, giếng trời).
- **Nguyên nhân gốc:** **distribution shift** — RL chỉ tổng quát hoá tới địa hình *giống* lúc train; train ít map → **học thuộc lòng** map đó.
- **Cải tiến:** **Procedural Generation $1000$ map** đa dạng (hotel/curved/apartment) + nhiễu cảm biến/động cơ + **Smart Early-Stopping** (dừng khi eval hết cải thiện, tránh train-mù/overfit). *Kết quả: full-trip $87.5\%$, va chạm $1.5\%$ trên 1000 map.*

#### Cải tiến 5 — Xử lý bế tắc: Mission Controller + Dynamic Rerouting

- **Triệu chứng:** toà nhà thật có trạm sau lối quá hẹp, hoặc tắc nghẽn tạm thời → kế hoạch tĩnh thất bại âm thầm.
- **Nguyên nhân gốc:** kiến trúc một-lần-vạch-đường không có cơ chế phát hiện & phản ứng khi đường bất khả thi/kẹt runtime.
- **Cải tiến:** **Mission Controller** — A\* trả `None` ⟹ **hủy trạm** (log "🔴 BỎ QUA Trạm X") rồi tính trạm kế; **Stuck Detector** ⟹ đánh dấu vật cản tạm ⟹ **A\* vẽ đường vòng** ("🧭", Phần 4.3d).

#### Cải tiến 6 — Sửa bug đếm "đã giao" sai (bài học kỹ thuật)

- **Triệu chứng:** hệ báo "giao 8/8" nhưng robot thực tế **kẹt trong dock**.
- **Nguyên nhân gốc:** vòng lặp chặng kết thúc theo **cạn danh sách waypoint**, không theo **tới đích thật** → robot kẹt (con trỏ waypoint vẫn nhảy khi đụng tường) bị tính là đã giao.
- **Cải tiến:** chỉ tính `delivered` khi $\lVert p_{robot}-p_{tgt}\rVert\le 0.5$ m; mỗi chặng **replan A\* từ vị trí THỰC TẾ** (không từ điểm giả định) → hết lỗi dây chuyền.

---

**Triết lý cốt lõi:** *"A\* lo bài toán toàn cục (đi đâu), RL lo cục bộ (lái thế nào), Shield lo an toàn động (né ai)."* Phân tách trách nhiệm → mỗi module đơn giản, kiểm thử được, robust.

> **Ghi chú kỹ sư (trung thực):** đo thực nghiệm — trên tuyến A\* đã vạch, **pure-pursuit** bám tim đường mượt hơn ($\approx 1.4$–$3.0^\circ$/step) so với RL làm tracker ($\approx 25^\circ$/step), vì việc của RL (né cục bộ) trở nên **thừa** khi A\* đã cung cấp đường tim. Sản phẩm cuối dùng pure-pursuit làm bộ lái mặc định; **RL robust là local tracker tuỳ chọn / fallback** cho vùng chưa có đường A\*. RL vẫn được huấn luyện đầy đủ và là chủ thể của Phần 3–4.""")

md(r"""## PHẦN 2 — DỮ LIỆU & MÔ PHỎNG

### 2.1. Mô phỏng Robot — Differential Drive
Robot là **đĩa tròn** bán kính va chạm $r_{robot}=0.22$ m. Pose $(x,y,\theta)$ cập nhật theo động học xe vi sai, bước $\Delta t = 0.1$ s:

$$\begin{aligned}
x_{t+1} &= x_t + v_t \cos(\theta_{t+1})\,\Delta t \\
y_{t+1} &= y_t + v_t \sin(\theta_{t+1})\,\Delta t \\
\theta_{t+1} &= (\theta_t + \omega_t\,\Delta t + \pi)\bmod 2\pi - \pi
\end{aligned}$$

**Trong đó:** $(x_t,y_t)$ — toạ độ robot tại bước $t$ (m); $\theta_t$ — góc hướng mũi xe (rad); $v_t$ — vận tốc tuyến tính (m/s); $\omega_t$ — vận tốc góc / tốc độ xoay (rad/s); $\Delta t=0.1$ s — chu kỳ điều khiển; $\bmod$ — phép chia lấy dư, dùng để gói $\theta$ về khoảng $(-\pi,\pi]$.

### 2.2. Cảm biến
- **2D LiDAR raycasting:** $N=24$ tia đều $\alpha_i = \frac{2\pi i}{24}$, tầm $R_{max}=5.0$ m; chuẩn hoá $\tilde{d}_i = d_i/R_{max}\in[0,1]$.
- **Odometry / AMCL:** pose $(x,y,\theta)$ để A\* tính Lookahead.
- **Tách động/tĩnh:** LiDAR cấp cho RL **chỉ chứa tường tĩnh**; người do Safety Shield xử lý.

### 2.3. Môi trường — Procedural Generation (1000 map)
`generate_training_maps.py` sinh **1000 Occupancy Grid** trộn 3 archetype: procedural/curved ($\sim61\%$), hotel ($\sim29\%$), apartment hẹp $\sim1.2$ m ($\sim10\%$). Mỗi episode bốc ngẫu nhiên 1 map → buộc tổng quát hoá. Đã kiểm chứng 100% điểm tới được.

### 2.4. Sim-to-Real Gap — Domain Randomization
- **Nhiễu LiDAR:** $\tilde{d}_i \leftarrow \mathrm{clip}(\tilde{d}_i + \mathcal{N}(0,\sigma_{lidar}),\,0,\,1)$.
- **Trượt/ma sát động cơ:** nhân hệ số ngẫu nhiên vào $v,\omega$.
- **Người đi bộ:** $0.5$–$1.0$ m/s, ràng buộc trong hành lang (`is_free`) + bật tường.""")

md(r"""## PHẦN 3 — ĐỊNH DẠNG MDP

MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, $\gamma = 0.997$.

### 3.1. State Space — $s \in \mathbb{R}^{32}$
Tổng **32 chiều**, trong đó **24 là LiDAR** (không phải 32 LiDAR thuần). Theo `_obs`, chuẩn hoá $[-1,1]$:

$$s = \big[\tilde{d}_0,\dots,\tilde{d}_{23}\big]\ \Vert\ \big[\sin\beta,\ \cos\beta,\ \min(\tfrac{d_g}{20},1)\big]\ \Vert\ \big[\hat{e}_{xt},\ \tfrac{\omega_{prev}}{\omega_{max}},\ \tilde{d}_0,\ \mathbb{1}_{grace},\ \rho\big]$$

- $\beta$: **bearing** từ mũi xe tới Lookahead A\* (biểu diễn $(\sin\beta,\cos\beta)$ liên tục qua $\pm\pi$) — chính là **toạ độ cực** $(d{=}d_g,\ \theta{=}\beta)$.
- $\hat{e}_{xt}=\mathrm{clip}(e_{xt},0,1)$: sai số bám tuyến (cross-track).
- $\omega_{prev}/\omega_{max}$, $\mathbb{1}_{grace}$ (cờ vừa chuyển trạm), $\rho$ (tỉ lệ trạm còn lại).

*LSTM mang trạng thái ẩn $h_t$ xuyên episode → thực chất là POMDP xử lý bằng bộ nhớ hồi quy.*

### 3.2. Action Space — liên tục
$$a = [a_v, a_w] \in [-1,1]^2 \subset \mathbb{R}^2$$

### 3.3. Action thực tế ($v_{max}=0.9$ m/s, $\omega_{max}=2.2$ rad/s, $\kappa=0.4$)
$$v = \mathrm{clip}\big((0.7\,a_v + 0.3)\,v_{max},\ -0.4\,v_{max},\ v_{max}\big),\qquad \omega = a_w\,\omega_{max}$$

**Trong đó:** $a_v,a_w\in[-1,1]$ — hai đầu ra của mạng (ga & lái, không thứ nguyên); $v$ — vận tốc tuyến tính sau ánh xạ (m/s); $\omega$ — vận tốc góc (rad/s); $v_{max}=0.9$ m/s, $\omega_{max}=2.2$ rad/s — giới hạn vật lý; $\kappa=0.4$ — tỉ lệ lùi tối đa cho phép (nên cận dưới là $-0.4\,v_{max}$); $\mathrm{clip}(z,lo,hi)$ — hàm kẹp $z$ vào đoạn $[lo,hi]$.

> Cho phép **lùi nhẹ** tới $-0.4\,v_{max}$ để thoát ngõ cụt/cua gắt (fix nạn "khựng lại" GĐ1).

### 3.4. Policy — Actor–Critic (Recurrent)
Cài đặt thật là **`MlpLstmPolicy` (RecurrentPPO)**, không phải MLP thuần — vì thoát ngõ cụt và đuổi Lookahead lật $180^\circ$ sau khi giao cần **bộ nhớ**.
- Thân: MLP $[256,256]$ + **LSTM** ẩn $128$.
- **Actor** $\pi_\theta(a\mid s)=\mathcal{N}(\mu_\theta(s),\sigma_\theta)$, bound `tanh` về $[-1,1]$.
- **Critic** $V_\phi(s)$: baseline đánh giá vị trí tốt/xấu, giảm phương sai.""")

md(r"""### 3.5. Reward Function — phân rã (khớp `step`)
$$R_t = r_{prog} - r_{time} - r_{ang} - r_{smooth} - r_{xtrack} - r_{wall} + r_{straight} - r_{stuck} - r_{col} + r_{event}$$

**Trong đó (ký hiệu dùng chung ở các thành phần bên dưới):** $w_{(\cdot)}$ — trọng số mỗi thành phần (hằng số trong code); $\ell_t$ — quãng đường đã đi dọc tuyến A\* tới bước $t$ (m); $a_w$ — lệnh lái ($\in[-1,1]$); $\Delta\omega=|\omega_t-\omega_{t-1}|$ — mức **đổi lái** giữa 2 bước; $e_{xt}$ — sai số lệch ngang khỏi tim đường (m); $\beta$ — góc lệch tới Lookahead (rad); $c_{front}$ — khoảng cách tường gần nhất trong cone $\pm40^\circ$ trước mũi (m); $d_{safe}=0.55$ m — ngưỡng an toàn; $\mathbb{1}[\cdot]$ — hàm chỉ thị (bằng $1$ nếu điều kiện đúng, $0$ nếu sai); $v$ — vận tốc tuyến tính.

**1) Tiến theo tuyến (dense):** thưởng theo độ tiến dọc tuyến A\* (progress), ổn định hơn $1/d$ thô:
$$r_{prog} = w_{prog}\,(\ell_t - \ell_{t-1}),\qquad w_{prog}=1.5$$

**2) Phạt thời gian:** $r_{time}=w_{time}=0.01$.

**3) Phạt vận tốc góc:** $r_{ang}=w_{turn}\,|a_w|,\ w_{turn}=0.03$.

**4) Phạt đổi lái (zigzag):** $\Delta\omega=|\omega_t-\omega_{t-1}|$,
$$r_{smooth}=2\,w_{dturn}\,\Delta\omega,\qquad w_{dturn}=0.05\ (\text{hệ số hiệu dụng }0.1)$$

**5) Phạt lệch tim đường (ân hạn khi cua gắt):**
$$r_{xtrack}=w_{xt}\,\chi\,\min(e_{xt},1),\quad w_{xt}=0.4,\quad \chi=\begin{cases}0.3 & |\beta|>20^\circ\\ 1.0 & \text{ngược lại}\end{cases}$$

**6) Phạt sát tường:** $c_{front}=\min$ LiDAR cone $\pm40^\circ$, $d_{safe}=0.55$:
$$r_{wall}=w_{nw}\,\frac{\max(0,\,d_{safe}-c_{front})}{d_{safe}},\qquad w_{nw}=0.6$$

**7) Thưởng đi thẳng ổn định:**
$$r_{straight}=w_{st}\,\mathbb{1}\big[c_{front}\!\ge\! d_{safe}\,\wedge\,v\!>\!0\,\wedge\,|a_w|\!<\!0.2\,\wedge\,\Delta\omega\!<\!0.2\,\wedge\,|\beta|\!\le\!20^\circ\big],\ w_{st}=0.06$$

**8) Phạt kẹt:** cửa sổ $W=40$ bước, bán kính $0.4$ m:
$$r_{stuck}=w_{stuck}\,\mathbb{1}\big[\mathrm{spread}_{40}<0.4\big],\qquad w_{stuck}=0.4$$

**9) Va chạm:** chạm tường $-w_{col}=-8$/bước; kẹt-chạm $>25$ bước thêm $-w_{jam}=-60$ và **done**.

**10) Sự kiện:** tới trạm $+w_{stop}=50$; về dock (chặng cuối) $+w_{dock}=100$ và **done**.

> **Khác bản lý tưởng:** dùng **progress-based** thay $r_{target}\propto1/d$; phạt va chạm **2 mức** ($-8$/bước + $-60$ khi kẹt-chạm) thay $-100$ cứng — cho "chạm nhẹ rồi sửa" thay vì giết episode, giúp học thoát kẹt.""")

md(r"""## PHẦN 4 — THUẬT TOÁN (CÔNG THỨC)

### 4.1. So sánh sâu PPO / A2C / SAC

Cả ba đều là RL **policy-gradient / actor-critic**, khác nhau ở *cách dùng dữ liệu* và *cơ chế ổn định cập nhật*.

#### (i) A2C — Advantage Actor-Critic
- **Cơ chế:** on-policy, cập nhật đồng bộ nhiều worker; gradient policy:
$$\nabla_\theta J(\theta) = \hat{\mathbb{E}}_t\big[\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,\hat{A}_t\big]$$
- **Ổn định:** **không có "phanh"** giới hạn bước cập nhật → một batch xấu có thể **đẩy policy đi quá xa** (destructive update), reward sụp.
- **Sample efficiency:** thấp (on-policy, vứt dữ liệu sau mỗi update).
- **Khám phá:** chỉ qua entropy phụ trợ; yếu.
- **Continuous / Recurrent:** dùng được nhưng phương sai cao, hội tụ thất thường.
- **Kết luận:** đơn giản, nhẹ, hợp làm baseline — **quá bấp bênh** cho 1000 map đa dạng.

#### (ii) SAC — Soft Actor-Critic
- **Cơ chế:** **off-policy** + **maximum-entropy**; mục tiêu cộng entropy để khám phá:
$$J(\pi) = \sum_t \mathbb{E}\big[r_t + \alpha\,\mathcal{H}\big(\pi(\cdot\mid s_t)\big)\big]$$
dùng **twin critics** (clipped double-Q) chống over-estimate, tự chỉnh nhiệt độ $\alpha$, học từ **replay buffer**.
- **Ổn định:** tốt khi tuning đúng, nhưng **nhạy** $\alpha$, learning-rate, kích thước buffer.
- **Sample efficiency:** **cao nhất** (tái dùng dữ liệu cũ) — lợi thế khi sim **đắt**.
- **Khám phá:** **mạnh** nhờ entropy — tốt cho continuous control.
- **Recurrent:** **khó** — LSTM cần học theo **chuỗi liên tục**, nhưng replay buffer lấy mẫu **rời rạc** → phải lưu/relay hidden-state, dễ bất ổn; hỗ trợ kém trong stack ta dùng.
- **Kết luận:** sample-efficient nhất nhưng **phức tạp + khó ghép LSTM + nhiều hyperparam nhạy**.

#### (iii) PPO — Proximal Policy Optimization
- **Cơ chế:** on-policy với **trust-region mềm** qua clip tỉ số xác suất (Phần 4.3a) — giới hạn mỗi cập nhật trong "vùng tin cậy".
- **Ổn định:** **cao nhất trong 3** — clip ngăn nhảy policy gắt → không "quên thảm hoạ".
- **Sample efficiency:** trung bình (on-policy) — nhưng **bù được** bằng mô phỏng rẻ + song song.
- **Khám phá:** entropy bonus + std học được; đủ dùng.
- **Continuous / Recurrent:** **rất hợp**; PPO+LSTM (`RecurrentPPO`) được hỗ trợ chín muồi, ổn định.
- **Kết luận:** ổn định, dễ tuning, ghép LSTM tốt — **đánh đổi đúng** cho bài toán này.

#### Bảng đối chiếu đa tiêu chí

| Tiêu chí | A2C | SAC | **PPO (chọn)** |
|---|---|---|---|
| Loại dữ liệu | on-policy | off-policy (replay) | on-policy |
| Ổn định cập nhật | Thấp (không trust region) | Trung bình (nhạy $\alpha$) | **Cao (clip)** |
| Sample efficiency | Thấp | **Cao** | Trung bình |
| Khám phá | Yếu | **Mạnh (entropy)** | Khá |
| Continuous action | Có | Có | **Có** |
| Ghép LSTM (recurrent) | Khá | **Khó** | **Tốt** |
| Độ nhạy hyperparam | Trung bình | **Cao** | **Thấp** |
| Hợp sim song song | Có | Hạn chế | **Rất hợp** |

### 4.2. Vì sao PPO là lựa chọn đúng cho hành lang chung cư

1. **Domain Randomization 1000 map cần ổn định:** mỗi episode một map khác → phân phối dữ liệu rất rộng. Trust-region clip giữ policy **không bị một loại map lạ kéo lệch** rồi quên các map đã học (điều A2C/SAC dễ mắc).
2. **Cần bộ nhớ (LSTM) để thoát ngõ cụt:** PPO+LSTM ổn định; SAC+LSTM (off-policy chuỗi) rất khó triển khai đúng.
3. **Action liên tục $(v,\omega)$:** cả ba làm được, nhưng PPO cho quỹ đạo ổn định nhất khi kết hợp reward smoothness.
4. **Sim rẻ + song song $8$ env:** nhược điểm sample-efficiency của on-policy **không còn quan trọng** — ta đổi "giờ máy" lấy "độ ổn định + dễ tuning".
5. **Ít hyperparam nhạy:** rút ngắn vòng lặp phát triển; phù hợp dự án nhiều giai đoạn.

> **Tóm lại:** SAC tiết kiệm mẫu hơn nhưng đánh đổi bằng độ phức tạp và rủi ro bất ổn (nhất là với LSTM); A2C đơn giản nhưng quá bấp bênh. PPO cho **độ ổn định + tương thích LSTM + dễ tuning** — đúng thứ một hệ multi-map, multi-stage cần nhất.

### 4.3. Công thức cốt lõi

**(a) PPO — Clipped Surrogate Objective.** $r_t(\theta)=\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)}$, lợi thế $\hat{A}_t$:
$$L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\big)\Big],\quad \epsilon=0.2$$

**Trong đó:** $\theta$ — tham số mạng policy đang cập nhật; $\theta_{old}$ — tham số policy lúc thu thập dữ liệu; $\pi_\theta(a_t\mid s_t)$ — xác suất policy chọn hành động $a_t$ ở trạng thái $s_t$; $r_t(\theta)$ — **tỉ số xác suất** (policy mới so với cũ); $\hat{A}_t$ — **lợi thế** (hành động $a_t$ tốt hơn mức trung bình bao nhiêu); $\hat{\mathbb{E}}_t[\cdot]$ — trung bình thực nghiệm trên các bước $t$; $\mathrm{clip}(r,1-\epsilon,1+\epsilon)$ — kẹp tỉ số trong $[0.8,1.2]$; $\epsilon=0.2$ — biên "vùng tin cậy". *Ý nghĩa: lấy $\min$ giữa bản gốc và bản đã kẹp → **chặn cập nhật quá gắt**.*

Value loss (Critic):
$$L^{VF}(\phi)=\hat{\mathbb{E}}_t\big[(V_\phi(s_t)-\hat{V}_t^{\,targ})^2\big]$$

**Trong đó:** $\phi$ — tham số mạng Critic; $V_\phi(s_t)$ — giá trị Critic dự đoán cho trạng thái $s_t$ (kỳ vọng tổng thưởng chiết khấu); $\hat{V}_t^{\,targ}$ — giá trị mục tiêu (return thực tính từ rollout). *Ý nghĩa: ép Critic dự đoán sát giá trị thật (sai số bình phương).*

Mục tiêu tổng (entropy $c_2=0.015$):
$$L(\theta,\phi)=\hat{\mathbb{E}}_t\big[L^{CLIP}(\theta)-c_1 L^{VF}(\phi)+c_2\,\mathcal{S}[\pi_\theta](s_t)\big]$$

**Trong đó:** $c_1$ — trọng số value loss; $c_2=0.015$ (`ent_coef`) — trọng số entropy; $\mathcal{S}[\pi_\theta](s_t)$ — **entropy** của phân phối hành động (đo độ "ngẫu nhiên"); cộng entropy để **khuyến khích khám phá**, tránh policy chín ép quá sớm.

GAE — *Generalized Advantage Estimation* ($\gamma=0.997,\ \lambda=0.95$):
$$\hat{A}_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},\qquad \delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$$

**Trong đó:** $\delta_t$ — **TD-error** (sai số chênh lệch thời gian: thưởng tức thời + giá trị tương lai − giá trị hiện tại); $\gamma=0.997$ — **hệ số chiết khấu** (coi trọng thưởng tương lai tới mức nào); $\lambda=0.95$ — hệ số GAE cân bằng **bias–variance**; $l$ — chỉ số bước nhìn về tương lai. *Ý nghĩa: gộp nhiều TD-error tương lai (trọng số giảm dần) để ước lượng lợi thế $\hat{A}_t$ mượt, ít nhiễu.*

*Hyperparam: `n_steps`=1024 (độ dài rollout), `batch`=512, `epochs`=10 (số lần học lại mỗi batch), `max_grad_norm`=0.5 (chặn gradient), lr $3\times10^{-4}\!\to\!1\times10^{-4}$ (tốc độ học giảm dần).*

**Smart Early-Stopping (chống overfit):** dừng nếu reward eval không lập đỉnh mới trong $7$ lần đánh giá:
$$\text{stop} \iff \forall k\in[t-6,t]:\ \bar{R}^{eval}_k \le \max_{j<t-6}\bar{R}^{eval}_j$$

**Trong đó:** $\bar{R}^{eval}_k$ — phần thưởng trung bình khi **đánh giá** (eval) ở lần thứ $k$; $\forall$ — "với mọi"; ý nghĩa: nếu **7 lần eval gần nhất** đều không vượt kỷ lục cũ ⟹ dừng train. (Thực tế dừng ở $2$M bước: full-trip $87.5\%$, va chạm $1.5\%$.)

**(b) A\* Planner.**
$$f(n)=g(n)+h(n)$$
**Trong đó:** $n$ — một ô đang xét trên lưới; $f(n)$ — tổng chi phí ước lượng của đường đi qua $n$; $g(n)$ — **chi phí thực** từ ô xuất phát tới $n$ ($1$ cho bước trực giao, $\sqrt2\approx1.41$ cho bước chéo, 8 hướng); $h(n)$ — **heuristic** (ước lượng chi phí còn lại tới đích), dùng khoảng cách Euclid:
$$h(n)=\sqrt{(n_x-goal_x)^2+(n_y-goal_y)^2}$$
với $(n_x,n_y)$ toạ độ ô $n$, $(goal_x,goal_y)$ toạ độ đích. $h$ **admissible** (không bao giờ ước lượng quá) ⟹ A\* tìm được đường tối ưu.

**Inflation layer** (đẩy đường ra tim hành lang): ô free thành cản nếu có lân cận là tường:
$$\text{free}'(r,c)=\text{free}(r,c)\wedge\bigwedge_{(dr,dc)\in\mathcal{N}_8}\text{free}(r{+}dr,c{+}dc)$$
**Trong đó:** $\text{free}(r,c)$ — ô $(r,c)$ có đi được không (đúng/sai); $\mathcal{N}_8$ — tập $8$ hướng lân cận; $\wedge$ — phép "và" logic; $\bigwedge$ — "và" trên mọi lân cận. *Ý nghĩa: ô chỉ còn free nếu **cả 8 lân cận** đều free → co biên free vào $1$ ô ($\approx0.3$ m).* Không có đường (ngách quá hẹp) ⟹ A\* trả `None` ⟹ Mission Controller **hủy trạm**.

**(c) Safety Shield — Proportional Steering.** Cone $\pm35^\circ$; khoảng cách bề mặt $d=\lVert\vec{p}_{ped}-\vec{p}_{robot}\rVert-r_{ped}$ ($r_{ped}=0.28$); $d_{react}=1.0,\ d_{brake}=0.5$. Thay $1/d$ (phát nổ) bằng *closeness* bị chặn:
$$\sigma(d)=\mathrm{clip}\!\Big(1-\frac{d-d_{brake}}{d_{react}-d_{brake}},\,0,\,1\Big)$$
$$\boxed{\ \omega_{evade}=-\,\mathrm{sign}(s_{side})\,K_p\,\sigma(d),\qquad K_p=1.4\ \text{rad/s}\ }$$
$$v_{scale}=\begin{cases}0 & d<d_{brake}\ (\textbf{phanh gấp})\\ 1-0.7\,\sigma(d) & d_{brake}\le d<d_{react}\ (\text{lách+giảm tốc})\\ 1 & d\ge d_{react}\end{cases}$$

**Trong đó:** $d$ — khoảng cách bề mặt tới người gần nhất (m); $d_{react}=1.0$ m — ngưỡng bắt đầu né; $d_{brake}=0.5$ m — ngưỡng phanh gấp; $\sigma(d)\in[0,1]$ — **độ gần chuẩn hoá** ($0$ khi vừa chạm vùng né, $\to1$ khi sát); $s_{side}$ — vị trí người ở bên trái ($>0$) hay phải ($<0$) robot; $\mathrm{sign}(\cdot)$ — hàm dấu ($\pm1$); $K_p=1.4$ — **hệ số tỉ lệ** (gain) của bộ điều khiển; $\omega_{evade}$ — lượng đánh lái né thêm (rad/s); $v_{scale}$ — hệ số giảm tốc; $\omega_{base},v_{base}$ — lệnh gốc từ bộ lái. *Ý nghĩa: càng gần người ($\sigma\!\to\!1$) lái né càng mạnh và càng chậm; $<0.5$ m thì dừng hẳn.*

Hợp nhất: $\omega\leftarrow\omega_{base}+\omega_{evade},\ v\leftarrow v_{scale}\,v_{base}$. Shield **bỏ qua tường tĩnh**.

**(d) Dynamic Rerouting — Stuck Detector.** $stall$ tăng khi $\lVert\Delta\vec{p}\rVert<0.01$ m/bước.
- $stall\ge15$: **fan-recovery** thử hướng lệch $\{0,\pm0.5,\pm1.0,\pm1.6,\pi\}$ rad quanh waypoint kế.
- $stall\ge45$: **reroute** — đánh dấu ô phía trước thành vật cản tạm, A\* vẽ lại từ vị trí thực:
$$\mathcal{G}_{temp}=\mathcal{G}_{inflate}\setminus B_1(\text{cell}_{ahead}),\qquad \text{path}'=\text{A*}(\vec{p}_{now}\to\vec{p}_{goal};\,\mathcal{G}_{temp})$$

**Trong đó:** $\mathcal{G}_{inflate}$ — lưới bản đồ đã inflate; $B_1(\text{cell}_{ahead})$ — khối $3\times3$ quanh ô **phía trước** robot (chỗ nghi bị chặn); $\setminus$ — phép trừ tập hợp (bỏ vùng đó khỏi free → thành cản tạm); $\mathcal{G}_{temp}$ — lưới tạm sau khi chặn; $\vec{p}_{now},\vec{p}_{goal}$ — vị trí hiện tại & đích; $\text{path}'$ — đường vòng mới. *Ý nghĩa: "bịt" chỗ kẹt rồi bắt A\* tìm đường khác.* $\le4$ lần reroute thất bại ⟹ hủy trạm, chuyển trạm kế.""")

md(r"""## PHẦN 5 — CHÚ THÍCH THUẬT NGỮ VIẾT TẮT & GIẢI THÍCH THUẬT TOÁN

### 5.1. Bảng chú thích viết tắt (mỗi từ: tên đầy đủ — nghĩa — vai trò trong model)

| Viết tắt | Tên đầy đủ | Nghĩa & **vai trò trong hệ thống** |
|---|---|---|
| **RL** | *Reinforcement Learning* (Học tăng cường) | Robot tự học bằng thử–sai để tối đa phần thưởng. **Vai trò:** huấn luyện bộ lái cục bộ (local tracker). |
| **MDP** | *Markov Decision Process* | Khung toán $(\mathcal{S},\mathcal{A},P,R,\gamma)$ mô hình hoá bài toán quyết định. **Vai trò:** định nghĩa hình thức bài toán điều hướng. |
| **POMDP** | *Partially Observable MDP* | MDP mà tác tử **không thấy toàn bộ** trạng thái. **Vai trò:** đúng bản chất của ta (LiDAR cục bộ) → cần LSTM để bù bộ nhớ. |
| **PPO** | *Proximal Policy Optimization* | Thuật toán RL on-policy giới hạn bước cập nhật (clip). **Vai trò:** thuật toán **chính** huấn luyện policy. |
| **A2C** | *Advantage Actor-Critic* | RL on-policy đơn giản, không có trust-region. **Vai trò:** đối chứng (không chọn). |
| **SAC** | *Soft Actor-Critic* | RL off-policy, max-entropy, replay buffer. **Vai trò:** đối chứng (không chọn vì khó ghép LSTM). |
| **GAE** | *Generalized Advantage Estimation* | Ước lượng "lợi thế" $\hat{A}_t$ cân bằng bias–variance qua $\lambda$. **Vai trò:** tính tín hiệu học cho PPO. |
| **Actor–Critic** | — | Hai đầu mạng: **Actor** ra hành động, **Critic** chấm điểm trạng thái. **Vai trò:** kiến trúc của policy. |
| **LSTM** | *Long Short-Term Memory* | Mạng hồi quy có **bộ nhớ** dài hạn. **Vai trò:** nhớ hướng đã đi → thoát ngõ cụt, đuổi Lookahead lật $180^\circ$. |
| **MLP** | *Multi-Layer Perceptron* | Mạng nơ-ron truyền thẳng nhiều lớp. **Vai trò:** thân trích đặc trưng $[256,256]$ trước LSTM. |
| **A\*** | *A-star search* | Tìm đường tối ưu trên lưới bằng heuristic. **Vai trò:** **lập kế hoạch toàn cục**, vạch waypoint tim đường. |
| **SLAM** | *Simultaneous Localization And Mapping* | Vừa định vị vừa dựng bản đồ. **Vai trò:** robot mù tự quét tạo Occupancy Grid. |
| **Frontier Exploration** | — | Chiến lược quét: luôn đi tới biên giữa vùng đã biết/chưa biết. **Vai trò:** thuật toán quét mù trong SLAM. |
| **AMCL** | *Adaptive Monte Carlo Localization* | Định vị bằng lọc hạt (particle filter) thích nghi. **Vai trò:** ước lượng pose $(x,y,\theta)$ (trong sim là pose + nhiễu). |
| **LiDAR** | *Light Detection And Ranging* | Cảm biến laser đo khoảng cách. **Vai trò:** $24$ tia "nhìn" tường → input cho RL & Shield. |
| **Occupancy Grid** | — | Lưới ô đánh dấu free/cản. **Vai trò:** biểu diễn bản đồ cho A\*/inflation. |
| **Inflation Layer** | — | Bơm phồng vật cản thêm $0.3$ m. **Vai trò:** ép A\* đi **tim hành lang**, chừa khoảng cách an toàn. |
| **Lookahead point** | — | Điểm mục tiêu phụ cách mũi xe $0.7$ m trên tuyến A\*. **Vai trò:** mục tiêu bám tức thời cho bộ lái. |
| **Cross-track error** | — | Khoảng cách lệch ngang khỏi đường. **Vai trò:** thành phần state + reward để bám tim đường. |
| **Pure-pursuit** | — | Bộ lái hình học bám điểm Lookahead. **Vai trò:** bộ lái cục bộ **mặc định** (mượt nhất). |
| **TSP** | *Traveling Salesman Problem* | Bài toán sắp thứ tự ghé thăm ngắn nhất. **Vai trò:** Mission Controller xếp thứ tự trạm giao tối ưu. |
| **Safety Shield** | — | Lớp phản ứng né vật cản động. **Vai trò:** né người (chỉ động), ghi đè lệnh lái khi nguy hiểm. |
| **Mission Controller** | — | Bộ điều phối chuyến giao. **Vai trò:** hủy trạm bất khả thi, kích hoạt reroute. |
| **Dynamic Rerouting** | *(Replanning động)* | Vạch lại đường khi kẹt. **Vai trò:** đánh dấu vật cản tạm → A\* vẽ đường vòng. |
| **Domain Randomization** | — | Ngẫu nhiên hoá môi trường/cảm biến khi train. **Vai trò:** chống overfit → robust map lạ. |
| **Differential Drive** | — | Động học xe 2 bánh vi sai $(v,\omega)$. **Vai trò:** mô hình chuyển động robot. |
| **Odometry** | — | Ước lượng vị trí từ chuyển động bánh. **Vai trò:** cấp pose (cùng AMCL). |
| **VecNormalize** | — | Chuẩn hoá running-mean/var của observation & reward. **Vai trò:** ổn định huấn luyện. |
| **SubprocVecEnv** | — | Chạy nhiều môi trường song song ở tiến trình riêng. **Vai trò:** tăng tốc thu thập dữ liệu ($8$ env). |
| **Early Stopping** | — | Dừng train khi eval hết cải thiện. **Vai trò:** chống overfit / train-mù. |
| **Sim-to-Real Gap** | — | Khác biệt sim↔thực. **Vai trò:** lý do tiêm nhiễu (Domain Randomization). |

### 5.2. Giải thích từng thuật toán theo các bước

**A\* (lập kế hoạch toàn cục) — các bước:**
1. Hạ bản đồ về Occupancy Grid, áp **Inflation** (bơm phồng tường $0.3$ m).
2. Duyệt ô theo độ ưu tiên $f(n)=g(n)+h(n)$: $g$ = chi phí đã đi, $h$ = khoảng cách Euclid ước lượng tới đích.
3. Mở rộng ô có $f$ nhỏ nhất (8 hướng, chéo $\times\sqrt2$) đến khi chạm đích → truy vết ra chuỗi waypoint tim đường.
4. Nếu hàng đợi cạn mà chưa tới đích ⟹ trả **`None`** (đường quá hẹp) ⟹ Mission Controller hủy trạm.

**SLAM / Frontier Exploration (quét mù) — các bước:**
1. Robot bắn LiDAR, cập nhật Occupancy Grid (ô thấy được = free/cản, còn lại = chưa biết).
2. Tìm **frontier** = ô free kề ô chưa biết; đi tới frontier gần nhất.
3. Lặp đến khi không còn frontier (đã phủ hết) → lưu bản đồ.

**PPO (huấn luyện policy) — vòng lặp:**
1. Thu thập rollout: chạy policy hiện tại trên $8$ env, lưu $(s,a,r)$.
2. Tính lợi thế $\hat{A}_t$ bằng **GAE**; tính return làm target cho Critic.
3. Cập nhật nhiều epoch theo $L^{CLIP}$ (Actor, **clip** để không nhảy quá xa) + $L^{VF}$ (Critic) + entropy bonus.
4. **Early Stopping** kiểm tra eval định kỳ; dừng khi hết cải thiện.

**Safety Shield (né người) — mỗi bước:**
1. Quét người trong cone $\pm35^\circ$ phía trước; lấy người gần nhất, tính khoảng cách bề mặt $d$.
2. Tính độ-gần $\sigma(d)\in[0,1]$; nếu $d<1.0$ m → đánh lái ra xa $\omega_{evade}=-\mathrm{sign}(s)K_p\sigma(d)$ + giảm tốc; nếu $d<0.5$ m → **phanh**.
3. Hết người gần → trả quyền cho bộ lái. **Bỏ qua tường tĩnh** (A\* lo).

**Mission Controller + Dynamic Rerouting — vòng đời chuyến:**
1. **TSP** xếp thứ tự trạm; với mỗi trạm gọi A\* từ vị trí thực.
2. A\* `None` ⟹ **hủy trạm** ("🔴"), sang trạm kế.
3. Khi chạy: nếu kẹt (dịch chuyển $<0.01$ m/bước kéo dài) ⟹ fan-recovery; vẫn kẹt ⟹ **đánh dấu vật cản tạm + A\* vẽ đường vòng** ("🧭").
4. Giao đủ ⟹ về Dock ("✅"). Chỉ tính "đã giao" khi tới thật ($<0.5$ m).""")

md(r"""## PHẦN 6 — CẢI TIẾN v2: CHUẨN HOÁ REWARD (ổn định Critic)

### 6.1. Điều CHƯA TỐT ở model v1 (`ms_mixed_robust`)
Sau khi train v1 ($\sim2$M bước, early-stop), biểu đồ TensorBoard lộ ra điểm yếu:
- **Explained Variance $\approx 0.14$** (cờ đỏ): mạng Critic $V_\phi(s)$ chỉ giải thích
  $\sim14\%$ phương sai của return → ước lượng giá trị **nhiễu** → lợi thế $\hat{A}_t$
  nhiễu → gradient nhiễu → học chậm, kém ổn định.
- **Eval reward dao động lớn** ($\pm180$–$470$) → kết quả **không đồng đều** giữa các map.
- **Value Loss cao** ($\approx0.16$).
- **$\omega$ bão hoà $\sim45\%$** (đánh lái kịch kim nhiều).

**Nguyên nhân GỐC:** reward biến thiên **quá lớn** trên 1000 map đa dạng — phần thưởng
cuối là spike thưa & to ($w_{dock}=100,\ w_{stop}=50,\ w_{jam}=60$) **lấn át** phần
progress dày & nhỏ; giữa các map (dài/ngắn, 1–3 trạm) return chênh nhau rất nhiều
→ Critic **không fit nổi** → explained variance thấp.

### 6.2. ĐÃ THAY ĐỔI GÌ (Cách 1 — chuẩn hoá / scale lại reward)
Giữ nguyên model v1 làm backup; train model mới `ms_mixed_robust_v2` với reward
cân bằng lại (giảm spike cuối, tăng dense, chặn outlier):

| Trọng số | v1 | **v2** | Lý do |
|---|---|---|---|
| $w_{progress}$ | 1.5 | **2.5** | tăng tín hiệu dense (Critic dự đoán dễ) |
| $w_{dock}$ | 100 | **30** | giảm spike cuối |
| $w_{stop}$ | 50 | **12** | giảm spike cuối |
| $w_{jam}$ | 60 | **30** | giảm spike cuối |
| $w_{collide}$ | 8 | **4** | giảm biên độ phạt |
| `shaping_clip` | — | **3.0** | **clip phần shaping** trước thưởng cuối → chặn outlier penalty cộng dồn |

*Hệ quả:* return giữa các map **đồng đều hơn** → Critic fit tốt → explained variance tăng.

### 6.3. KẾT QUẢ v2 (đo từ TensorBoard)

| Chỉ số | v1 | **v2** | Nhận xét |
|---|---|---|---|
| Explained Variance | 0.14 | **0.88** | Critic fit chuẩn hẳn (mục tiêu chính đạt) |
| Value Loss | 0.16 | **0.009** | chính xác hơn $\sim18\times$ |
| Success Rate (full-trip) | 87.5% | **91%** | giao tốt hơn |
| Sample efficiency | đạt $\sim2$M | **đạt $\sim840$k** | nhanh hơn $3$–$4\times$ |
| Action std | 0.71 | 0.81 | còn khám phá tốt |
| Collision rate | 1.5% | 3.0% | ⚠️ **tăng nhẹ** (đánh đổi) |

(Biểu đồ so sánh: `compare_v1_v2.png`.)

### 6.4. CÒN CHƯA TỐT — hướng cải thiện tiếp
- **Va chạm $1.5\%\to3.0\%$:** do giảm $w_{collide}\,8\to4$. *Cải thiện:* nâng lại
  $w_{collide}\approx6$ (giữ `shaping_clip`) → vừa explained-var cao vừa va chạm thấp.
- **$\omega$ vẫn bão hoà** (RL bản chất là bộ né phản ứng): *cải thiện:* tăng
  $w_{turn}/w_{dturn}$ hoặc thêm **giới hạn tốc độ đổi lái** (rate-limit) trong env.
- **Explained variance còn dao động** (noisy theo batch): *cải thiện:* train lâu hơn /
  batch lớn hơn / thêm `target_kl` để giảm `clip_fraction` ($0.34$ là hơi cao).
- **Eval reward không so sánh chéo** giữa 2 phiên bản reward (khác thang) → chỉ so các
  chỉ số độc-lập-reward (explained-var, success, collision).
- **RL vẫn kém mượt hơn pure-pursuit** trên tuyến A\* → giữ pure-pursuit làm bộ lái mặc
  định; v2 dùng làm fallback / khi chưa có đường A\*.""")

md(r"""## PHẦN 7 — CẬP NHẬT: Bản RL LÁI MƯỢT (`ms_guided_smooth` / `ms_guided_smooth2`)

### 7.1. Dùng model gì
- **Thuật toán:** PPO (*Proximal Policy Optimization*), mạng **MlpPolicy** kiến trúc
  $[256,256]$ (feed-forward, **không LSTM** — vì state đã đủ tính Markov cho bài lái).
- **Quan sát (State) 28 chiều, ego-relative:**
$$s = [\,\underbrace{\tilde d_0\dots\tilde d_{23}}_{24\ \text{LiDAR}}\ |\ e_y\ |\ e_\theta\ |\ v\ |\ \omega\,]$$
**Trong đó:** $\tilde d_i$ — LiDAR chuẩn hoá (cách tường); $e_y$ — **lệch tim đường** (cross-track, m); $e_\theta$ — **lệch hướng** so với tiếp tuyến tuyến đường (rad); $v,\omega$ — vận tốc tuyến tính & góc hiện tại. **Không có toạ độ tuyệt đối** $\Rightarrow$ mạng "mù" về đang ở map nào.
- **Dữ liệu train:** 1000 map (domain randomization) + Smart Early-Stopping.

### 7.2. Cập nhật gì so với bản RL trước
1. **Thiết kế lại State (chống học vẹt map):** observation chỉ chứa đại lượng **tương đối so với robot & tuyến đường**, bỏ mọi thông tin gắn với toạ độ map. Cùng một "trạng thái lái" áp dụng cho mọi map $\Rightarrow$ **zero-shot, không nhớ map**.
2. **Reward tối ưu ĐỘ MƯỢT:**
$$R = R_{nav} + R_{collision} - K\,(\omega_t-\omega_{ref})^2 - w_{d\omega}\,|\omega_t-\omega_{t-1}| - w_\omega\,|\omega_t|$$
**Trong đó:** $R_{nav}$ — thưởng tiến dọc tuyến; $R_{collision}$ — phạt nặng va chạm (LiDAR); $\omega_{ref}$ — **góc lái tham chiếu mượt tính từ hình học tuyến đường**; $K$ — trọng số bám tham chiếu; $w_{d\omega}|\omega_t-\omega_{t-1}|$ — **phạt đổi lái đột ngột** (diệt đánh võng); $w_\omega|\omega_t|$ — phạt lái gắt. Tổng hợp $\Rightarrow$ ép quỹ đạo **thẳng, mượt**.
3. **Hạ tầng mới:** môi trường `track_env.py` (bài bám tuyến), adapter `GuidedTracker` (dựng obs 28-chiều $e_y,e_\theta$ từ đường A\*), và **tích hợp vào GUI** (chọn được trong "Bộ lái cục bộ").

### 7.3. Hai biến thể
| Model | Cấu hình điều hoà lái | Ghi chú |
|---|---|---|
| **`ms_guided_smooth`** | vừa phải ($K=1.0$, $w_{d\omega}=0.3$) | bản chốt |
| **`ms_guided_smooth2`** | mạnh hơn ($K=1.2$, $w_{d\omega}=0.6$) | bản thử mượt hơn |

### 7.4. Kết quả (đo trên Set A 1000-map + Set B map cũ)
| Chỉ số | RL trước | **RL mượt** |
|---|---|---|
| Đánh võng (°/bước) | ~5.56 | **~1.6** (giảm ~3.4×) |
| Tới đích | 85-91% | **99%** |
| Bám tim đường $e_y$ | — | **sub-4cm** |
| Va chạm | — | ~1% |
- `smooth` vs `smooth2`: đánh võng gần như nhau (~1.6°) — tăng điều hoà mạnh hơn (smooth2) **không cải thiện thêm đáng kể** (chạm sàn).
- **Chống học vẹt:** cùng cấu trúc obs đúng trên mọi map $\Rightarrow$ không thuộc lòng map nào.

### 7.5. Cách dùng
Trong GUI, mục **"Bộ lái cục bộ"** chọn **"RL mượt (ms_guided_smooth)"** hoặc **"RL mượt v2 (ms_guided_smooth2)"** rồi chạy giao hàng.

---
## PHẦN 8 — SO SÁNH TẤT CẢ PHIÊN BẢN

### 8.1. Kiến trúc & công thức — cái gì GIỮ NGUYÊN, cái gì ĐỔI

| Phiên bản | State (quan sát) | Reward | Policy | A\* + Shield |
|---|---|---|---|---|
| **GĐ1 — End-to-End RL (gốc)** | LiDAR + hướng, **không A\*** | progress + va chạm cơ bản | PPO/LSTM | **KHÔNG có A\*** → kẹt ngõ cụt |
| **Robust v1** (`ms_mixed_robust`) | 32-chiều | reward đa thành phần ($w_{dock}{=}100$...) | RecurrentPPO (LSTM) | Có |
| **Robust v2** (`ms_mixed_robust_v2`) | 32-chiều — **Y HỆT v1** | **CÙNG công thức**, chỉ đổi **trọng số** ($w_{prog}1.5{\to}2.5$, $w_{dock}100{\to}30$...) + `shaping_clip` | RecurrentPPO — **Y HỆT** | **Y HỆT** |
| **Guided smooth / smooth2** (`ms_guided_smooth*`) | **KHÁC** — 28-chiều ego-relative $[24\text{ LiDAR},e_y,e_\theta,v,\omega]$ | **KHÁC** — $R_{nav}{+}R_{col}{-}K(\omega{-}\omega_{ref})^2{-}w_{d\omega}\lvert\Delta\omega\rvert$ | **KHÁC** — PPO MlpPolicy (không LSTM) | **Y HỆT** |
| **Pure-pursuit** | (không phải RL — bộ lái hình học) | — | — | Có |

→ **v2 vs v1:** chỉ khác **trọng số reward** (cùng mọi công thức). **Guided vs gốc:** khác **State + Reward + Policy** (mô hình lái khác hẳn). **A\* + Safety Shield giữ nguyên** ở mọi phiên bản.

### 8.2. Hiệu suất THỐNG NHẤT (cùng khung MissionController, 7 map, không người)

| Bộ lái | Giao đủ điểm | Đánh võng TB (°/bước) | Số bước TB (thời gian) |
|---|---|---|---|
| **Pure-pursuit** | **47/47 (100%)** | **6.77** | **7523** |
| RL v1 (`ms_mixed_robust`) | 40/47 (85%) | 39.42 | 27558 |
| RL v2 (`ms_mixed_robust_v2`) | 36/47 (77%) | 34.14 | 49100 |
| **RL guided smooth** | **46/47 (98%)** | **9.23** | **7685** |
| RL guided smooth2 | 46/47 (98%) | 9.46 | 7635 |

*(Đánh võng TB bị kéo cao bởi 3 map cong test_c/s/u — võng ~12-16° cho MỌI bộ lái do tuyến cong dài; riêng map căn hộ: guided ~4.6-5.5° vs pure-pursuit ~2.5-3.0°.)*

### 8.3. Nhận xét
- **Pure-pursuit** vẫn vô địch tuyệt đối (giao 100%, mượt nhất, nhanh nhất).
- **Guided (smooth/smooth2)** đã **gần pure-pursuit**: giao **98%**, tốc độ **≈ bằng** ($7685$ vs $7523$ bước), đánh võng giảm **~4×** so RL cũ. Đây là bước nhảy lớn so với RL v1/v2.
- **RL v1/v2 cũ (32-chiều, multistop):** làm bộ lái độc lập **yếu** — giao 77-85%, đánh võng 34-39°, chậm **4-6×**. (Vai trò gốc của chúng là điều hướng đa-điểm có goal-transition, không phải bám tuyến.)
- **v2 vs v1:** v2 mượt hơn chút nhưng giao ít hơn (đánh đổi do giảm $w_{collide}$ — xem Phần 6).
- **smooth vs smooth2:** gần như nhau → tăng điều hoà lái mạnh hơn không cải thiện thêm.

---
## TỔNG KẾT
Phân tách trách nhiệm sạch: **A\*** ($f=g+h$, grid inflate) + **RecurrentPPO** ($L^{CLIP}$ + reward đa thành phần) + **Safety Shield** ($\omega_{evade}=-\mathrm{sign}(s)K_p\sigma(d)$) + **Mission Controller** (điều phối + reroute). Mọi công thức khớp đúng hằng số mã nguồn; các điểm cài đặt thật khác bản lý tưởng (LSTM thay MLP, cho phép lùi, progress-reward, closeness-clamp, va chạm 2 mức) đã nêu rõ kèm lý do kỹ thuật. **Bản v2** chuẩn hoá reward đã nâng explained-variance $0.14\to0.88$ và success $87.5\to91\%$ (Phần 6). **Bản RL lái mượt** (Phần 7, `ms_guided_smooth`/`smooth2`) dùng state ego-relative + reward điều hoà lái: đánh võng $5.56^\circ\to1.6^\circ$, tới đích $99\%$, chống học vẹt map.""")

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)} for s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", NB, "| markdown cells:", len(cells))
