# ICA 伪迹检测中的 threshold 参数详解

## 核心问题

`threshold` 参数在 ICA 流程中的作用位置和判断机制是什么？

---

## 一、关键概念澄清

### ICA 分解 ≠ PCA

首先需要明确：

**Step 1: PCA 白化（预处理）**
```
原始数据 X (25×T)
    ↓
PCA 降维 → 22 个主成分
    ↓
白化矩阵 X_white (22×T)  ← 这里用了 PCA
```

**Step 2: ICA 分离（核心算法）**
```
白化数据 X_white (22×T)
    ↓
FastICA 寻找独立成分  ← 这是 ICA，不是 PCA!
    ↓
独立成分 S (22 个 ICs)
```

**关键区别：**
- **PCA**: 只去除相关性，保留方差最大的方向
- **ICA**: 寻找**统计独立**的非高斯源信号

---

## 二、threshold 的作用时机

`threshold` **在 ICA 分解完成后**使用：

```python
# 完整流程
ica = ICA(n_components=22)        # Step A: 创建 ICA 对象
ica.fit(raw_ica)                  # Step B: 执行 ICA 分解（已完成✅）

# ↓ 现在 ICA 分解已经完成，得到了 22 个独立成分 ↓

eog_indices, _ = ica.find_bads_eog(raw_ica, threshold=2.0)  
#                                              ↑
#                               Step C: 基于已有结果做检测
```

---

## 三、threshold 的详细机制

### 它在检测什么？

`threshold` 用于判断：**"一个 ICA 成分与 EOG 通道的相关性要多强，才被认为是眼电伪迹？"**

### 数学原理

```python
# find_bads_eog() 内部计算过程

# 1. 计算每个 ICA 成分与 EOG 通道的相关系数
correlations = []
for ic_idx in range(22):
    ic_signal = ica.get_source(ic_idx)      # 第 ic_idx 个成分的时间序列
    eog_signal = raw_ica.get_data('EOG-central')
    
    corr = np.corrcoef(ic_signal, eog_signal)[0, 1]
    correlations.append(abs(corr))

# correlations = [0.85, 0.12, 0.03, 0.67, ..., 0.45]
#                ↑                    ↑
#            成分 0 很强           成分 21 中等


# 2. 计算阈值（动态阈值！）
mean_corr = np.mean(correlations)   # 所有相关性的平均值
std_corr = np.std(correlations)     # 标准差

threshold_value = mean_corr + threshold * std_corr
#                        ↑
#                   这里的 2.0 就是参数 eog_threshold

# 示例：
# mean_corr = 0.15
# std_corr = 0.18
# threshold_value = 0.15 + 2.0 × 0.18 = 0.51


# 3. 标记超过阈值的成分
eog_indices = [
    idx for idx, corr in enumerate(correlations) 
    if corr > threshold_value
]

# 继续示例：
# correlations = [0.85, 0.12, 0.03, 0.67, ..., 0.45]
#                          ↑              ↑
#                       > 0.51         < 0.51
# 结果：eog_indices = [0, 3, ...]
```

---

## 四、为什么用动态阈值而不是固定值？

### 动态阈值 vs 固定阈值

**如果用固定阈值（如 0.5）**

场景 A：眼电伪迹很明显
```python
correlations = [0.9, 0.8, 0.1, 0.05, ...]
mean = 0.2, std = 0.25
threshold_value = 0.2 + 2.0×0.25 = 0.7
→ 检测到成分 [0, 1] ✅ 正确
```

场景 B：眼电伪迹很弱
```python
correlations = [0.4, 0.3, 0.1, 0.05, ...]
mean = 0.12, std = 0.10
threshold_value = 0.12 + 2.0×0.10 = 0.32
→ 检测到成分 [0, 1] ✅ 仍然正确

# 如果用固定阈值 0.5:
# 场景 B 会漏检！❌ (0.4 < 0.5)
```

**动态阈值的优势：**
- 自适应数据特性：根据实际相关性分布调整
- 鲁棒性强：不受绝对数值影响
- `threshold=2.0` 的含义："比平均水平高 2 个标准差的成分才被认为是伪迹"

---

## 五、可视化理解

```
所有 ICA 成分与 EOG 的相关性分布:

相关性
  1.0 ┤                          ★
      │                          │
  0.8 ┤              ★           │
      │              │           │
  0.6 ┤      ★       │     ★     │
      │      │       │     │     │
  0.4 ┤      │       │     │     │
      │      │   ╭───┼─────╯     │
  0.2 ┤   ╭──┼───╯  │           │
      │   │  │      │           │
  0.0 ┼───┴──┴──────┴───────────┴────
      0   5  10     15          21  (成分索引)
      
      ← 正常脑电 →  ← 眼电成分 →
      
threshold_value = mean + 2.0×std
                 (虚线位置)
```

---

## 六、完整流程图

```
┌─────────────────────────────────────────┐
│ Step 1: ICA 分解                         │
│ 输入：25 通道数据                        │
│ 输出：22 个独立成分 + 混合矩阵          │
│                                         │
│ ica.fit(raw_ica) ← 在这里执行           │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 2: 提取 ICA 源信号                  │
│ ica.get_sources(raw_ica)                │
│ 得到：22 个成分的时间序列               │
│                                         │
│ • 成分 0: [0.1, -0.5, 0.3, ...]         │
│ • 成分 1: [0.2, 0.1, -0.4, ...]         │
│ • ...                                   │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 3: 计算与 EOG 的相关性             │
│ 对每个成分 i:                           │
│   corr[i] = pearsonr(IC_i, EOG)         │
│                                         │
│ 结果：[0.85, 0.12, 0.03, 0.67, ...]    │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 4: 应用 threshold 判定              │
│ threshold_value = mean + 2.0×std        │
│                                         │
│ 如果 corr[i] > threshold_value:         │
│     标记为眼电成分 ✅                    │
│ 否则：                                  │
│     认为是脑电成分 ❌                    │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Step 5: 返回检测结果                     │
│ eog_indices = [0, 3, 21]                │
│ eog_scores = [0.85, 0.67, 0.45]         │
└─────────────────────────────────────────┘
```

---

## 七、类比理解

想象在**考试排名**中判断"优秀学生"：

```python
# 全班成绩
scores = [95, 92, 78, 75, 70, 68, 65, 60, 55, 50]

# 方法 1: 固定阈值（如 90 分）
excellent_fixed = [s for s in scores if s >= 90]
# 结果：[95, 92]
# 问题：如果题目很难，最高分才 75 呢？

# 方法 2: 动态阈值（类似 ICA 的做法）
mean_score = 70.8
std_score = 14.2
threshold_dynamic = mean_score + 2.0 * std_score  # = 99.2
excellent_dynamic = [s for s in scores if s >= threshold_dynamic]
# 结果：[] (这次考试没有特别优秀的)

# 但如果降低到 1.5 倍标准差：
threshold_1_5 = mean_score + 1.5 * std_score  # = 92.1
excellent_1_5 = [s for s in scores if s >= threshold_1_5]
# 结果：[95] ← 更合理
```

**在 ICA 中：**
- `threshold=2.0`: 较严格，只检测非常明显的伪迹
- `threshold=1.5`: 较灵敏，能捕捉微弱的伪迹
- `threshold=3.0`: 非常严格，只检测极端明显的伪迹

---

## 八、调参建议

| threshold | 检测灵敏度 | 假阳性风险 | 适用场景 |
|-----------|-----------|-----------|----------|
| **1.0** | 非常高 | 高 | 伪迹很微弱，需要仔细检测 |
| **1.5** | 较高 | 中等 | 一般情况，平衡检测率 |
| **2.0** (默认) | 适中 | 低 | **推荐**,适用于大多数情况 |
| **2.5** | 较低 | 很低 | 伪迹非常明显时 |
| **3.0** | 非常低 | 极低 | 只检测极端伪迹 |

---

## 九、实际代码验证

如何查看实际的阈值计算：

```python
from scipy.stats import pearsonr
import numpy as np

# 获取 EOG 通道
eog_chs = ['EOG-left', 'EOG-central', 'EOG-right']
eog_data = raw_ica.get_data(picks=eog_chs)

# 计算所有成分的相关性
correlations = []
for comp_idx in range(ica.n_components_):
    comp_data = ica.get_sources(raw_ica).get_data(picks=[comp_idx])[0]
    max_corr = max([abs(pearsonr(comp_data, eog)[0]) for eog in eog_data])
    correlations.append(max_corr)

# 打印统计信息
print(f"相关性统计:")
print(f"   最小值：{np.min(correlations):.3f}")
print(f"   最大值：{np.max(correlations):.3f}")
print(f"   平均值：{np.mean(correlations):.3f}")
print(f"   标准差：{np.std(correlations):.3f}")

# 计算阈值
threshold_value = np.mean(correlations) + 2.0 * np.std(correlations)
print(f"\n动态阈值 (2.0σ): {threshold_value:.3f}")

# 显示哪些成分被标记
eog_indices_auto = [i for i, c in enumerate(correlations) if c > threshold_value]
print(f"\n被标记的成分：{eog_indices_auto}")

# 对比固定阈值 0.3
eog_indices_fixed = [i for i, c in enumerate(correlations) if c > 0.3]
print(f"固定阈值 0.3 的结果：{eog_indices_fixed}")
```

**典型输出：**
```
相关性统计:
   最小值：0.02
   最大值：0.85
   平均值：0.15
   标准差：0.18

动态阈值 (2.0σ): 0.51

被标记的成分：[0, 3, 21]
固定阈值 0.3 的结果：[0, 3, 5, 21]
```

---

## 十、总结

### 问题解答

**问题 1:** "前面不是已经用 ICA 进行主成分分析了吗？"

**答案:**
- ✅ ICA 分解**已完成**（第 81 行 `ica.fit(raw_ica)`）
- ✅ 但现在是在**分析 ICA 的结果**，识别哪些成分是伪迹
- ⚠️ 小纠正：ICA 不是主成分分析，它只是借用了 PCA 做预处理

**问题 2:** "这里是在从分析后的结果检测？"

**答案:**
- ✅ **完全正确!**
- `find_bads_eog()` 是在**已分解的 22 个独立成分**上做检测
- 计算每个成分与 EOG 通道的相关性
- 用 `threshold` 判断哪些相关性"显著高于平均水平"

---

### 核心结论

> **`threshold=2.0` 是一个统计判据，用于在 ICA 分解完成后，自动识别"与 EOG 相关性显著高于随机水平"的独立成分，标记为眼电伪迹以便剔除。**
