# 架构级突破方向调研 — 2026-05-29

> 背景：在 GCD 70+10 baseline (mAP 0.464, ori_ap 0.474, new_ap 0.391) 之上，所有 loss 微调/文本侧微扰尝试（B1-B3, SM, CG-MQD, E1 offset, Prompt 替换, B2 Prototype CE）均未超过 baseline 12e。  
> 诊断：GCD 框架内的 loss 级微扰已触碰天花板。需要架构级或策略级改变。  
> 目标：(A) 提升 new_ap 上限；(B) 提升 ori_ap；(C) 6e 达到 12e 性能（节约 50% 算力）。

---

## 路径 1：Incremental LoRA Adapter — 收敛加速 + 防遗忘

### 目标

6 epoch 达到 GCD 12 epoch 的性能（目标 C），同时通过参数解耦天然防遗忘（目标 A/B）。

### 核心思想

不全量微调 encoder/decoder，而是在 attention 层插入低秩旁路 `W_q' = W_q + A @ B`（A: [d, r], B: [r, d], r=8-16）。原始权重 W_q 冻结保留旧知识，LoRA 旁路学习增量新知识。

### 引用论文

1. **EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer**
   - 会议：CVPR 2026
   - 作者：Munish Monga, Vishal Chudasama, Pankaj Wasnik, C.V. Jawahar
   - arXiv：2602.20985
   - 关键贡献：
     - Incremental LoRA Adapters：双适配器架构（aggregate adapter 累积旧知识 + task-specific adapter 学当前任务）
     - Data-Aware Merging：按任务样本比例合并 LoRA 权重，通过 truncated SVD 低秩压缩
     - Query-Norm Objectness Adapter：归一化 decoder query 解耦语义和幅度
     - Exemplar-free：不需要存储旧数据
   - 实验：Pascal Series + Diverse Weather，FOGS 指标提升 57.24%
   - 代码：未公开（截至 2026-05）

2. **CL-LoRA: Continual Low-Rank Adaptation**
   - 年份：2025
   - arXiv：2505.24816
   - 关键贡献：
     - Dual-adapter 架构：task-shared adapter + task-specific adapter
     - 跨任务知识共享 + 任务特定适配的平衡
     - 在 CIFAR-100, ImageNet-R 等 CIL benchmark 上仅用 0.3% 参数达到 SOTA
   - 适用性：CIL 分类，需要适配到检测

3. **DualLoRA: Dual Low-Rank Adaptation for Continual Learning**
   - 年份：2025
   - 关键贡献：
     - Orthogonal LoRA adapter + Residual LoRA adapter 并行
     - Dynamic memory mechanism 平衡 stability/plasticity
     - Task identity prediction + output calibration
   - 实验：ViT-based models，多个 CL benchmark

4. **KeepLoRA: Subspace-Constrained LoRA for Continual Learning**
   - 年份：2026
   - arXiv：2601.19659
   - 关键贡献：
     - 新任务 LoRA 更新被约束在**与预训练主子空间正交的残差子空间**中
     - 梯度引导的 LoRA 初始化
     - 不需要 task ID，不存储旧数据
     - 在约束子空间中性能接近无约束 LoRA
   - 适用性：VLM 持续学习，与 GCD 的 Grounding DINO 架构高度相关

### 在 GCD 中的实现思路

```
Feature Enhancer (6 层):
  每层的 image self-attn, text self-attn, image-to-text cross-attn, text-to-image cross-attn
  → 每个 attention 的 Q/K/V projection 加 LoRA(r=8)
  → 参数增量: 256 × 8 × 2 × 3 × 4 × 6 ≈ 295K

Decoder (6 层):
  每层的 self-attn, image cross-attn, text cross-attn
  → 每个 attention 的 Q/K/V 加 LoRA(r=8)
  → 参数增量: 256 × 8 × 2 × 3 × 3 × 6 ≈ 221K

总增量: ~516K 参数（GCD 172M 的 0.3%）
```

### 预期效果

- 6e 达到 GCD 12e 的 mAP 0.464
- 通过参数解耦减少 SCM 应力（旧类路径不变，新类走 LoRA）
- 训练成本减半

### 风险

- 低：LoRA 技术成熟，大量论文验证
- 需要确认 mmdet/mmengine 中 LoRA 的实现方式（是否有现成的 PEFT 集成）

---

## 路径 2：Quality-Guided Matching — 修复 Hungarian 强制匹配缺陷

### 目标

提升 new_ap 上限（目标 A），修复 DETR 增量检测中的 background foregrounding 问题。

### 核心思想

标准 Hungarian matching 强制每个 GT 匹配一个 query，即使所有 query 对该 GT 的预测质量都很差。这导致低质量 query 被错误标记为新类正样本，破坏模型表征。Q-MCMF 通过几何质量剪枝，允许"匹配不到好 query 的 GT 放弃匹配"。

### 引用论文

1. **Q-MCMF: Better Matching, Less Forgetting: A Quality-Guided Matcher for Transformer-based Incremental Object Detection**
   - 年份：2026
   - arXiv：2603.01524
   - 关键贡献：
     - 识别 DETR 增量检测中的"background foregrounding"问题
     - 构建 flow graph 并基于几何质量（IoU）剪枝不可靠匹配
     - Min-Cost Max-Flow 优化最终匹配
     - 在 COCO 多种增量设置下一致超越 SOTA
   - 实验：COCO 40+40, 70+10, 40+10×4
   - 代码：未公开

### 在 GCD 中的实现思路

```
当前 GCD:
  cost_matrix = α × cls_cost + β × L1_cost + γ × IoU_cost
  matched = hungarian(cost_matrix)  # 强制每个 GT 匹配一个 query

改进:
  1. 计算 matched pair 的 IoU
  2. IoU < threshold (如 0.1) 的 pair → 标记为 invalid
  3. invalid pair 不产生正样本 loss
  4. 或者用 Q-MCMF 的 flow graph 替代 Hungarian
```

更简单的近似版本（无需实现 MCMF）：

```python
# 在 loss_by_feat_single 中，Hungarian 匹配后增加质量过滤
pos_inds = assign_result.gt_inds > 0
if pos_inds.sum() > 0:
    matched_ious = bbox_overlaps(pred_boxes[pos_inds], gt_boxes[matched_gt_inds], is_aligned=True)
    valid_mask = matched_ious > min_iou_threshold  # 0.05-0.10
    # 只对 valid 的匹配计算 loss
```

### 预期效果

- 消除低质量 query 被强制标记为正样本的问题
- 提升新类训练信号质量 → new_ap ↑
- 间接减少对旧类表征的干扰 → ori_ap 不降或略升

### 风险

- 中：需要修改 assigner 和 loss 函数
- 如果 threshold 设太高，可能导致新类得到的训练信号不足（GT 都被放弃了）
- 需要与 GCD 的伪标签生成兼容

---

## 路径 3：Dynamic Query Expansion — 新旧类 Query 解耦

### 目标

提升 new_ap 同时不伤害 ori_ap（目标 A+B），消除新旧类 query 竞争。

### 核心思想

给新类分配专属 query，旧类 query 从 base 模型继承。新旧 query 通过 attention mask 隔离，各自负责各自的类别。

### 引用论文

1. **DyQ-DETR: Dynamic Object Queries for Transformer-based Incremental Object Detection**
   - 年份：2024
   - arXiv：2407.21687
   - 关键贡献：
     - 每个增量阶段新增一组 learnable object queries
     - Isolated bipartite matching：新旧 query 各自独立匹配各自的 GT
     - Disentangled self-attention：新旧 query 之间无交互
     - Risk-balanced partial calibration for exemplar replay
   - 实验：COCO, 基于 Deformable DETR
   - 代码：未公开

### 在 GCD 中的实现思路

```
当前 GCD:
  900 queries → decoder → 检测 0-79 类

改进:
  900 old queries (从 base checkpoint 继承, 冻结 query_embedding)
  + 100 new queries (新增 learnable embedding)
  ↓
  decoder self-attention 中用 mask 隔离:
    old queries 只 attend to old queries
    new queries 只 attend to new queries
    但所有 queries 都 attend to image features (cross-attention 不隔离)
  ↓
  匹配: old queries 只匹配 old 伪标签, new queries 只匹配 new GT
  ↓
  推理: 合并所有 1000 queries 的输出
```

### 预期效果

- 新类有专属 query → s_gt 天花板不再受旧类竞争限制
- 旧类 query 冻结/轻微微调 → ori_ap 不降
- 可与路径 1 (LoRA) 组合：旧 query 冻结，新 query + LoRA 微调

### 风险

- 高：需要修改 decoder 结构和 Hungarian matching
- 增加约 10-15% 的计算量
- 需要仔细处理 NMS（新旧 query 输出的同类框去重）

---

## 路径 4：Post-Hoc 偏置校正 — 零成本快速验证

### 目标

不重新训练，直接改善 GCD 输出的 new/old 平衡（目标 A+B）。

### 核心思想

增量训练后，模型对新类和旧类存在系统性分数偏置。通过在推理时施加简单的线性校正即可改善。

### 引用论文

1. **BiC: Large Scale Incremental Learning**
   - 会议：CVPR 2019
   - 作者：Yue Wu, Yinpeng Chen, Lijuan Wang, et al.
   - 关键贡献：
     - 发现增量学习后分类器对新类有系统性偏置
     - 训练一个 2 参数的线性校正层 (α, β) 来重新平衡
     - 使用一小部分验证集拟合校正参数
   - 被引用 1000+ 次，经典方法

2. **DPCR: Dual-Projection Shift Estimation and Classifier Reconstruction**
   - 年份：2025
   - arXiv：2503.05423
   - 关键贡献：
     - Dual-Projection 估计 semantic shift
     - Ridge regression 重构分类器，BP-free
     - 用旧类 covariance 和 prototype 重建平衡分类器
   - 实验：EFCIL benchmark

3. **Expandable-RCNN: Toward High-Efficiency Incremental Few-Shot Object Detection**
   - 年份：2024
   - PMC/Springer
   - 关键贡献：
     - GSL (Gradient-based Score Limiter)：用无偏验证集校准新旧类分数
     - 减少新类的 over-activation，降低 false positive
     - 在 RCNN 框架下验证

### 在 GCD 中的实现思路

```python
# Step 1: 用 GCD 12e checkpoint 在 val 子集上收集每类得分分布
#   对每个检测框: (class_id, score, is_correct)

# Step 2: 计算偏置
#   对新类 (70-79): mean_score, median_score, optimal_threshold
#   对旧类 (0-69):  mean_score, median_score, optimal_threshold
#   如果新类 mean_score << 旧类 mean_score → 存在偏置

# Step 3: 拟合校正
#   方法 A (BiC): score' = α × score + β, per old/new group
#   方法 B (per-class): score' = α_c × score + β_c, per class
#   方法 C (temperature): score' = score / τ_c, per class

# Step 4: 校正后重新评估 mAP
```

### 预期效果

- 如果偏置存在，mAP 可能提升 0.5-2pt
- 即使不大，也可以作为诊断工具确认偏置程度
- 零训练成本，几分钟完成

### 风险

- 极低：不修改模型
- 上限有限：只调整分数阈值，不改变检测能力

---

## 路径 5：LoRA 解冻 BERT — 打破 Text Embedding 天花板

### 目标

打破 BERT frozen 造成的 s_gt 天花板（目标 A），从文本编码器侧改善新类表征。

### 核心思想

不是给 text feature 加 offset（已被 Phase 0 证明太弱），而是用 LoRA 改写 BERT 内部 attention 层的投影矩阵，让 BERT 学会在检测语境下编码更有判别力的类别文本表示。

### 引用论文

同路径 1 的 EW-DETR, KeepLoRA，加上：

1. **Textual Inversion for Efficient Adaptation of Open-Vocabulary Object Detectors Without Forgetting**
   - 年份：2025
   - arXiv：2508.05323
   - 关键贡献：
     - 在 GLIP/BERT 的 token embedding 层学习新伪 token
     - 梯度需要流过 BERT（不能只在外面加 adapter）
     - 3 张图就能学新概念，不遗忘旧类
   - 与 GCD 的关系：GLIP 和 Grounding DINO 都用 BERT，架构高度相似

2. **NoIn-Det / COVD: Continual Open-Vocabulary Object Detection with Novel Concept Injection**
   - 年份：2026
   - arXiv：2605.27116
   - 关键贡献：
     - 冻结视觉编码器，只微调文本分支注入新概念
     - RSSD 表征空间稳定性蒸馏：在随机旧概念 + 已注入概念上做语义蒸馏
     - KPD 知识感知参数解耦：抑制对旧知识敏感的参数更新
   - 直接适用于增量检测中的文本分支微调

### 在 GCD 中的实现思路

```
BERT 12 层 Transformer:
  每层 attention: Q/K/V 各 768×768
  LoRA(r=4): 每层 768×4×2×3 = 18K
  12 层: 221K 参数

+ text_feat_map (768→256): 解冻
  196K 参数

总增量: ~420K (GCD 的 0.24%)
```

需要同时加旧类 text feature 蒸馏（RSSD/NoIn-Det 思路）防止旧类文本表征漂移。

### 预期效果

- 突破 s_gt 天花板（从 0.05-0.13 提升到更高）
- 新类 text feature 变得更有判别力
- 配合 CG-MQD，软锚点目标变得更可达

### 风险

- 中：需要同时保护旧类 text feature
- 比路径 1 (encoder/decoder LoRA) 复杂，因为 BERT 和 text_feat_map 的冻结是绑定的

---

## 路径 6：改善伪标签质量 — 提升 ori_ap

### 目标

提升旧类 AP（目标 B），通过更好的伪标签减少旧类遗忘。

### 核心思想

GCD 用 teacher 模型一次性生成伪标签。如果伪标签质量差（漏检高频旧类），蒸馏效果打折。用 prototype-guided filtering 或自迭代可以改善。

### 引用论文

1. **PDP: Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection**
   - 会议：CVPR 2026
   - arXiv：2603.02286
   - 关键贡献：
     - Prototypical Pseudo-Label Generation (PPG)
     - 用 class prototype 的 cosine similarity 二次过滤伪标签
     - 阈值 0.5-0.7 均稳定
   - 实验：COCO, PASCAL VOC

2. **CL-DETR: Continual Detection Transformer for Incremental Object Detection**
   - 会议：CVPR 2023
   - 作者：Yaoyao Liu, et al.
   - arXiv：见论文
   - 关键贡献：
     - Detector Knowledge Distillation (DKD)：只蒸馏高质量预测
     - Calibrated exemplar replay：保持标签分布一致性
   - 实验：COCO 2017
   - 代码：公开

### 在 GCD 中的实现思路

```
当前 GCD 伪标签生成（gdino_head_inc_gcd.py generate_pseudo_label）:
  teacher cls_scores > threshold → 直接作为伪标签

改进:
  teacher cls_scores > threshold
  → 提取 query feature
  → 与该类 visual prototype 计算 cosine
  → cosine > proto_threshold → 保留
  → 否则丢弃（这个伪标签可能是误检）
```

### 预期效果

- 过滤掉 teacher 的误检伪标签 → 减少对 student 的错误监督
- ori_ap 可能提升 0.5-1pt

### 风险

- 低：不改变模型结构
- 需要维护 visual prototype（可从 teacher 特征 EMA 获取）

---

## 路径组合建议

### 最小风险快速验证

```
路径 4 (偏置校正) — 今天就能做，零训练成本
  ↓ 如果发现偏置存在
路径 6 (伪标签改善) — 低成本，3e 快速验证
  ↓ 如果 ori_ap 有提升
路径 1 (LoRA) — 核心方法，6e = 12e
```

### 最强组合（论文主方法）

```
路径 1 (LoRA adapter) + 路径 2 (quality matching) + 路径 5 (LoRA BERT)
= "Incremental LoRA-Augmented Grounding DINO with Quality-Guided Matching"
```

### 论文最小可写版本

```
路径 1 (LoRA adapter) alone
= "LoRA 使 GCD 6e 达到 12e 性能"
+ 消融研究 + 分析
```

---

## 附：新发现的 CVPR 2026 增量检测论文列表

| 论文 | 关键技术 | 与本项目关系 |
|------|---------|-------------|
| P2IOD (CVPR 2026) | Parameterized prompts, MLP bottleneck | Prompt-based IOD，可参考 |
| PDP (CVPR 2026) | Dual-pool prompting, prototype pseudo-labels | 伪标签改善参考 |
| Q-MCMF (2026) | Quality-guided matching | 直接适用于 GCD 的 matching 改善 |
| EW-DETR (CVPR 2026) | Incremental LoRA, Query-Norm | 直接适用 |
| DyQ-DETR (2024) | Dynamic object queries | Query 扩展参考 |
| HNC-DETR (2026) | Neural collapse tree | 分类器结构改善参考 |
| COVD/NoIn-Det (2026) | Text branch fine-tuning, RSSD | BERT LoRA 参考 |
