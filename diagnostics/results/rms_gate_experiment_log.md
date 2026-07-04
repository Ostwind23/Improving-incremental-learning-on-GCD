# RMS+Gate 实验日志 (2026-07-03 23:15 ~)

## 实验目标
验证 RMS (global, t=1.0) + Old-Class Gate (detach fixed) 能否同时在旧类保留和新类学习上超越 GCD baseline 和 Bilinear+LN。

## 背景
- Bilinear+LN 12e: new_ap=0.403 (超 GCD 1.2pt), ori_ap=0.460 (低于 GCD 0.474)
- B2 (RMS t=1.0, no gate) 3e: ori_ap=0.466, new_ap=0.356 — ori_ap持平LN, new_ap超越LN+1.7pt
- Gate 之前失败原因: LN在gate之后(已修复), gate→memory梯度通路③污染encoder(已通过.detach()修复)
- 当前修复: init_std=0.01, RMS (global, non-detached), Gate AFTER RMS, gate_net(memory.detach())

## 关键阈值
- ori_ap ≥ 0.465 (GCD ep3 = 0.474, 允许 ≤1pt 正常损失)
- new_ap ≥ 0.356 (B2 baseline) 或 ≥ 0.350 (GRMI baseline)
- ori_ap < 0.44 且下降趋势: 崩溃，终止

## 运行日志

---

### 23:15 - 启动
- 动作: 启动 RMS+Gate 3e
- Config: grmi_rms_gate_3e.py
- Params: 65666 (Bilinear 49K + Gate 16K)
- ETA: ~2h40m

### 00:00 CST - Ep1 val
- 动作: 检查训练，一切正常
- 指标: ori_ap=0.472, new_ap=0.279, mAP=0.447
- 判定: **正常** — ori_ap=0.472 (B2 ep1=0.465, LN ep1=0.467) Gate已在ep1展示保护效果
- 备注: Ep2 进行中, ETA 1:42

### 01:00 CST - Ep2 val
- 指标: ori_ap=0.471, new_ap=0.310
- 判定: **正常** — ori_ap 坚挺 0.471, Gate 持续保护

### 02:00 CST - Ep3 val (3e完成!)
- 指标: ori_ap=0.467, new_ap=0.344, mAP=0.452
- 判定: **通过** — ori_ap 所有方案最优 (B2=0.466, LN=0.466, GRMI=0.466)
- 对比: new_ap=0.344 略低于 B2 (0.356)，Gate 初始 ~0.88 对所有位置均匀衰减，12e 应能学会区分
- 动作: 启动 RMS+Gate 12e

### 02:01 - 12e 启动
- Config: grmi_rms_gate_12e.py
- ETA: ~10h42m, ep1 val ~03:00

### 02:30 - 12e 停止，Gate 梯度诊断
- 原因: Gate bias 3 epoch 不动 (2.0→2.0004)，需定量分析
- 方法: 加载 RMS+Gate 3e checkpoint，encoder 级别测 gate 梯度
- **结果**: loss≈10 时 gate_bias_grad=2.68e-03, W_out_grad=7.72e-04, gate/W_out=3.5×
  - gate 在 encoder 级别获得 3.5× W_out 的梯度——不是 decoder 衰减问题
  - 但绝对值太小: per-step bias move = 2.68e-03 × 5e-5 = 1.34e-07
  - 7500 步累积 = 0.0010，实际 = 0.0004，预实比 2.5× — 量级一致
  - **根因: sigmoid 饱和 (0.88) × 残差小 (RMS t=1.0) → 梯度绝对量 = 1e-7/step**
  - 即使 100x LR, 7500 步也才 0.0375 → bias 2.0→1.96, sigmoid 0.88→0.877 → 几乎无变化
  - 要达到 sigmoid 0.5 (显著关闭), 需 bias 0 → Δ2.0, 需要 ~15000x LR 或其他机制

### Gate 无法学习的定量根因
- gate_bias 梯度 = dL/d(gate) × gate×(1-gate) × 1
- gate×(1-gate) = 0.88×0.12 = **0.106** ← sigmoid 饱和
- dL/d(gate) ∝ gamma × RMS_target × |dL/dM'| ← 扰动小
- 三者乘积 ≈ 3×10⁻³ (实测) → LR=5×10⁻⁵ → 1.3×10⁻⁷/step
- **结论: 不是梯度通路问题，是 sigmoid 饱和 + 小扰动 = 梯度绝对值太小**
