# 核心发现：GCD Decoder 对新类的定位能力全层失效

> 日期：2026-05-30  
> 来源：Phase 0.5 Per-Layer IoU 分布诊断  
> 重要性：★★★★★ — 这是所有 loss/matching 级实验失败的根本原因

---

## 发现

Phase 0.5 对 GCD 12e checkpoint 在 200 张 COCO val 上做了 per-layer 的 matched IoU 分布统计。

**结果：所有 decoder 层（d0-d5）+ encoder 层的新类 IoU 分布几乎完全相同。**

| 层 | 新类 Mean IoU | Median | IoU<0.1 | IoU<0.3 | IoU<0.5 | IoU<0.7 |
|----|-------------|--------|---------|---------|---------|---------|
| d0 | 0.096 | 0.0 | 72.1% | 87.5% | 92.3% | 99.0% |
| d1 | 0.084 | 0.0 | 76.9% | 90.4% | 94.2% | 97.1% |
| d2 | 0.086 | 0.0 | 76.9% | 89.4% | 94.2% | 97.1% |
| d3 | 0.086 | 0.0 | 77.9% | 89.4% | 94.2% | 98.1% |
| d4 | 0.091 | 0.0 | 75.0% | 88.5% | 94.2% | 98.1% |
| d5 (last) | 0.085 | 0.0 | 76.0% | 89.4% | 94.2% | 98.1% |
| encoder | 0.085 | 0.0 | 75.0% | 90.4% | 95.2% | 98.1% |

新类 104 个 matched pair 中，94.2% 的 IoU < 0.5，且 d0 到 d5 无差异。

旧类（~31K per layer，含 pseudo-label）也在所有层保持一致的低 IoU（mean ~0.067）。

## 推翻的假设

1. ~~"早期 decoder 层 IoU 低是正常的 coarse-to-fine"~~ → 所有层一样低，没有 refinement
2. ~~"最后一层的低 IoU 匹配才是 background foregrounding"~~ → 所有层都有同等程度的问题
3. ~~"Per-layer 差异化质量过滤能解决问题"~~ → 没有层间差异可利用

## 核心含义

**GCD 的 decoder 在任何层都无法将 query 定位到新类物体上。** 问题不在 matching 算法、不在 loss 设计、不在 per-layer 调权——而在于 decoder 的 coarse-to-fine refinement 机制对新类完全不工作。

这解释了为什么以下所有实验都没能超过 baseline：
- B3/SM/CG-MQD（给 query 加辅助 loss）→ query 本身就不在新类位置，加 loss 没用
- Quality Filter（过滤低 IoU 匹配）→ 所有匹配都是低 IoU，过滤等于删除所有信号
- E1 text embedding offset → text 侧改善不能修复 query 定位失败
- Prompt 替换 → 同上

## 根因分析（2026-05-30 代码分析确认）

### 完整因果链

```
根因 1: BERT text embedding 新旧类不对称
│  基座训练 (0-69): BERT 未冻结 → text embedding 与视觉特征共同优化
│  增量训练 (70-79): BERT 冻结 → text embedding 是通用语言语义，未针对检测适配
│  → 新类 text-image dot-product 分数极低 (s_gt 0.05-0.10)
│
├→ 根因 2: Language-Guided Query Selection (LGQS) 排斥新类
│  │  LGQS 机制: enc_outputs_class = output_memory @ memory_text.T
│  │  对全图 ~1-2 万个 encoder 位置，选 score.max(-1) 的 top-900 作为 query
│  │  新类 token 分数低 → 新类物体区域排不进 top-900
│  │  → 900 个 query 的 reference_points 全部落在旧类/背景区域
│  │
│  └→ 根因 3: Deformable Attention 的有限感受野
│     │  每层 cross-attention 只在 reference point 附近采样 (偏移量 ∝ box_wh)
│     │  如果 reference 距离 GT 中心 > ~0.2 归一化坐标 → 采样不到目标特征
│     │  → query 特征不含目标信息 → reg head 没有正确梯度方向
│     │  → 6 层 refinement 无法"跳跃"到正确位置
│     │
│     └→ 观测结果: Phase 0.5 显示 d0-d5 IoU 分布完全相同
│        → decoder refinement 对新类完全失效（不是慢收敛，是零收敛）
│
└→ 根因 4: GCD 增量阶段关闭了 DN (dn_cfg=None)
   基座训练: 启用 DN → GT 加噪声作为额外 query → 绕过 LGQS → 保证目标附近有 query
   增量训练: 禁用 DN → 新类完全依赖 LGQS → LGQS 选不到新类 → 新类无法被定位
   → 这是 GCD 作者的主动设计选择，可能是为了简化蒸馏对齐

### 吸引域（Basin of Attraction）分析

Decoder refinement 的工作假设：初始 reference 必须在目标的"可见邻域"内。

| 条件 | IoU 估计 | Refinement 效果 |
|------|---------|----------------|
| Query 在目标附近 | IoU ≳ 0.3-0.5 | 6 层可精炼到 IoU > 0.7 |
| Query 在目标边缘 | IoU ~0.1-0.3 | 部分精炼，不稳定 |
| Query 完全不覆盖目标 | IoU ≈ 0 | 6 层无法 recover（Phase 0.5 实证） |

新类 94.2% 匹配 IoU < 0.5，76% IoU < 0.1 → 绝大多数 query 在吸引域之外。

### GCD 双分支的 query 来源差异

| 分支 | Query 内容 | Reference 位置 | 服务对象 |
|------|-----------|---------------|---------|
| Student 新 text | 可学习 embedding | **Student LGQS top-900**（80 类 text） | 主检测 + 新类 loss |
| Student old text | Teacher query | **Teacher reference**（旧类 text LGQS） | 蒸馏对齐 |
| Teacher | 同上 | Teacher 自己的 LGQS | 伪标签生成 |

**Student 新类路径完全依赖 student 自己的 LGQS。** Teacher 蒸馏分支不能帮助 student 覆盖新类区域。

### 所有历史实验失败的统一解释

所有 loss/matching 修改都发生在 decoder 阶段或更下游：
- B3/SM/CG-MQD: 给 decoder 输出的 query 加辅助 loss → 但 query 不在新类位置
- Quality Filter: 过滤低 IoU 匹配 → 但几乎所有匹配都是低 IoU（因为 query 不在目标附近）
- E1 text offset: 改 text embedding → 不改变 LGQS 的 top-k 排序（offset 太小）
- Textual Prototype CE: 对 query 做原型分类 → query 特征不含目标信息

**问题发生在 LGQS（query initialization），所有 decoder 下游修改都是无效干预。**

### 解决方向

必须在 Query Selection 或更上游干预：

| 方向 | 机制 | 可行性 |
|------|------|--------|
| **重启 DN** | GT 加噪直接作为 query，绕过 LGQS | 高（代码已存在） |
| **新类专属 query slots** | LGQS 中为新类 token 保留一定数量的 slots | 中（需改 pre_decoder） |
| **LoRA 改善 text-image fusion** | 提升新类 text 的 encoder 对齐分数 → LGQS 能选到新类区域 | 中（间接改善） |
| **DyQ-DETR 式独立 query** | 新类用独立 learnable query，不依赖 LGQS | 中高（需改 decoder） |

## 文件索引

| 文件 | 路径 |
|------|------|
| Phase 0.5 脚本 | `复现/path2_quality_matching/phase05_per_layer_iou.py` |
| Per-layer 结果 JSON | PolyU: `/home/yelingfei/projects/GCD/phase05_diagnostics/phase05_per_layer_iou.json` |
| 原始数据 | PolyU: `/home/yelingfei/projects/GCD/phase05_diagnostics/phase05_raw.npz` |
| 运行日志 | PolyU: `/home/yelingfei/projects/GCD/logs/phase05_per_layer.log` |
