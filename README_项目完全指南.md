# EEG 运动想象分类项目——完全理解指南

> 目标读者：会高数、线代、傅里叶变换，对小波变换和 BCI 基本零基础。
> 读完本文你将能：看懂每一行代码的意义、理解每个参数的作用、独立进行调参优化。

---

## 目录

1. [你需要补的预备知识](#1-预备知识)
2. [项目做了什么（一句话）](#2-项目做了什么)
3. [数据集背景](#3-数据集bcic-iv-2a)
4. [代码阅读顺序](#4-代码阅读顺序)
5. [完整流水线详解](#5-完整流水线详解)
6. [所有可调参数速查表](#6-所有可调参数速查表)
7. [文件与函数索引](#7-文件与函数索引)
8. [如何运行](#8-如何运行)
9. [调参指南](#9-调参指南)
10. [学习资源推荐](#10-学习资源推荐)

---

## 1. 预备知识

按重要程度排序。**加粗的是你必须补的**，其余可以在遇到时再查。

### 1.1 **信号滤波（你已有基础）**

你懂傅里叶变换，只需理解两个操作：
- **带通滤波**：只保留某频段的信号。比如 8-30 Hz 带通 = 去掉 <8 Hz 和 >30 Hz 的成分
- **零相位滤波（zero-phase）**：正向滤一遍 + 反向滤一遍，不引入时间延迟
- **FIR vs IIR**：FIR 滤波器天然稳定，MNE 默认使用；IIR（Butterworth）更快，FBCSP 用到

> 📖 补充阅读：项目 `知识补充/EEG 预处理中的 FIR 滤波器详解.md`

### 1.2 **EEG 基本概念**

| 概念 | 解释 |
|------|------|
| 采样率 (sfreq) | 每秒采集多少个数据点，本数据集 250 Hz |
| 通道 (channel) | 每个电极对应一个通道，本数据集 22 个 EEG + 3 个 EOG |
| μ 节律 (mu) | 8-13 Hz，运动/想象运动时会**减弱**（称为 ERD） |
| β 节律 (beta) | 13-30 Hz，运动结束后会**增强**（称为 ERS） |
| EOG | 眼电信号，眼球运动/眨眼产生的伪迹 |
| 伪迹 (artifact) | 非脑电来源的干扰信号（眼动、肌电、心电等） |

> **核心直觉**：当你想象移动左手时，大脑右侧运动区（C4 电极附近）的 8-30 Hz 脑电会发生可检测的变化。不同类型的运动想象产生不同的空间分布模式，这就是我们分类的依据。

> 📖 推荐阅读：项目 `知识补充/EEG 运动想象分类基础概念详解.md`

### 1.3 **ICA（独立成分分析）**

- **目的**：把混合信号分解为互相独立的信号源。EEG 中最常用于**去除眼电伪迹**
- **原理类比**：鸡尾酒会问题——多个麦克风录到多人混合说话声，ICA 可以分离出每个人的声音
- **本项目用法**：分解 → 找到与 EOG 通道高度相关的成分 → 剔除这些成分 → 重建干净信号

> 📖 推荐阅读：项目 `知识补充/ICA 去除眼电伪迹：EOG 通道的重要性与使用指南.md`

### 1.4 **CSP（公共空间模式）—— 本项目核心算法**

这是你**最需要理解**的算法，只需要线性代数知识：

**目标**：找到一组空间滤波器 W，使得一类信号经 W 投影后方差最大，另一类方差最小。

**数学过程**（你能看懂的版本）：
1. 对每类分别计算协方差矩阵：$R_i = \frac{1}{N}\sum X_i X_i^T$，再做迹归一化 $R_i \leftarrow R_i / \text{tr}(R_i)$
2. 计算总协方差 $R = R_1 + R_2$
3. 对 $R$ 做特征值分解，构造白化矩阵 $W = D^{-1/2} V^T$（让总协方差变为单位阵）
4. 白化后的类别协方差 $P_i = W R_i W^T$ 可以同时对角化
5. 取 $P_1$ 特征值最大和最小的若干方向，即为 CSP 滤波器
6. 投影后取**对数方差** $\log(\text{var}(W_{\text{csp}} \cdot X))$ 作为特征

**直觉**：CSP 相当于找到最能区分两类脑电空间分布的"观察角度"。

> 📖 推荐书：Blankertz et al., "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis" (2008)
> 📖 推荐视频：B站搜索 "CSP 公共空间模式 BCI" 有中文讲解

### 1.5 **小波变换（DWT）—— 你需要补的知识**

你懂傅里叶变换（全局频率信息），小波变换则是**局部时频分析**：

| | 傅里叶变换 | 小波变换 |
|---|----------|---------|
| 信息 | 只有频率 | 频率 + 时间 |
| 基函数 | 正弦/余弦（无限长） | 小波（有限长、可伸缩） |
| 结果 | 一组频率系数 | 多层分解系数 |

**DWT（离散小波变换）过程**：
```
原始信号 → 低通+下采样 → A1（近似，低频）
         → 高通+下采样 → D1（细节，高频）
A1       → 低通+下采样 → A2（更低频）
         → 高通+下采样 → D2
...重复 level 次
```

**本项目用法**：
- 小波基：`db4`（Daubechies 4），与 EEG 形态匹配好
- 分解 4 层 → 得到 A4, D4, D3, D2, D1 共 5 组系数
- 对每组系数计算**能量** $E = \sum c_i^2$
- 每个通道产生 5 个能量值，22 通道 × 5 = **110 维特征**

> 📖 推荐入门书：《小波十讲》（Ingrid Daubechies 著），先看前 3 讲建立直觉
> 📖 更易读：Mallat《信号处理的小波导引》第 7 章
> 📖 视频：3Blue1Brown 没出小波，但 B 站搜 "小波变换 直觉" 有好的中文教程

### 1.6 SVM（支持向量机）

- **目标**：在特征空间找一个超平面，最大化两类数据的间隔
- **RBF 核**：$K(x_i, x_j) = \exp(-\gamma\|x_i - x_j\|^2)$，将数据映射到高维空间使非线性可分
- **关键参数**：C（惩罚系数，越大越严格拟合）、gamma（RBF 核宽度）

> 📖 推荐：周志华《机器学习》第 6 章，或 StatQuest 的 SVM 视频（YouTube/B站有搬运）

---

## 2. 项目做了什么

**一句话**：读取 9 个被试的 EEG 数据 → 预处理去噪 → 提取特征（CSP/小波/融合/FBCSP）→ SVM 分类 4 种运动想象任务（左手/右手/双脚/舌头），10 折交叉验证评估准确率。

---

## 3. 数据集：BCIC IV-2a

| 属性 | 值 |
|------|-----|
| 被试数 | 9 人 (A01-A09) |
| 每人会话 | 训练集 T + 评估集 E |
| 每会话试次 | 288 个（每类 72 个） |
| 任务类型 | 左手(769)、右手(770)、双脚(771)、舌头(772) |
| EEG 通道 | 22 个（10-20 标准位置） |
| EOG 通道 | 3 个（用于伪迹检测后丢弃） |
| 采样率 | 250 Hz |
| 单试次时序 | 0s 开始 → 2s 出现 cue → 2-6s 运动想象 → 6s 结束 |

**关键区别**：
- **T 文件**（如 A01T.gdf）：GDF 中包含类别事件 769-772，可直接提取标签
- **E 文件**（如 A01E.gdf）：GDF 中**不含**类别标签，只有 768（trial 开始），真实标签在 `true_labels/A01E.mat` 中

---

## 4. 代码阅读顺序

```
推荐按此顺序阅读，从全局到细节：

① config.py                    ← 先看所有参数和常量定义
② general_process.py           ← 单被试完整流程（主线剧情）
③ pretreatment/eeg_analysis.py ← 数据加载和通道映射
④ pretreatment/complete_preprocessing.py  ← 8 步预处理详解
⑤ feature_extraction/csp_feature.py      ← CSP 算法实现
⑥ feature_extraction/wavelet_feature.py  ← 小波特征提取
⑦ feature_extraction/eeg_transformers.py ← sklearn 兼容的 Transformer 封装 + FBCSP
⑧ classification/svm_classifier.py       ← Pipeline 构建 + SVM 训练
⑨ test_on_evaluation_set.py   ← 独立测试集评估（含标签对齐）
⑩ batch_processing.py         ← 批量处理所有被试
```

---

## 5. 完整流水线详解

### 5.1 预处理阶段（8 步）

文件：`pretreatment/complete_preprocessing.py` → `complete_preprocessing_pipeline()`

```
原始 GDF 文件
    │
    ▼
Step 1: get_modified_raw_data()
    加载 GDF → 22 个 EEG 通道重命名为标准 10-20 名称
    （如 EEG-0 → Fz, EEG-7 → C3, EEG-9 → Cz, EEG-11 → C4）
    设置 EOG 通道类型 → 设置电极位置
    │
    ▼
Step 2: filter_for_ica()
    FIR 零相位带通滤波 1-40 Hz
    目的：去掉低频漂移和高频噪声，但保留足够宽的频段供 ICA 识别伪迹
    ⚠️ 为什么不直接滤 8-30 Hz？因为眼电伪迹主要在低频，
       如果先窄带滤波就把伪迹信息丢了，ICA 就没法识别了
    │
    ▼
Step 3: fit_ica()
    FastICA 算法，保留 99% 方差的成分
    将混合信号分解为独立成分
    │
    ▼
Step 4: detect_and_remove_artifacts()
    两级 EOG 检测：
      ① 自动阈值检测（2.0 倍标准差）
      ② 若失败 → 手动 Pearson 相关分析（阈值 0.3）
    标记需剔除的伪迹成分（通常 1-3 个）
    │
    ▼
Step 5: apply_ica()
    从滤波数据中去除标记的伪迹成分，重建干净信号
    │
    ▼
Step 6: filter_for_task()
    FIR 零相位带通 8-30 Hz（聚焦运动想象频段：μ + β 节律）
    ⚠️ 这不是"重复滤波"——Step 2 的 1-40 Hz 是为 ICA 服务的，
       现在 ICA 完成了，才缩窄到任务相关频段
    │
    ▼
Step 7: set_reference()
    平均参考（average reference）：每个时间点减去所有通道的平均值
    消除共模噪声
    │
    ▼
Step 8: create_epochs_with_artifact_removal_mne()
    ① 以 769-772 事件为时间零点，截取 [0, 4] 秒时段
    ② 查找 1023 事件（伪迹标记），剔除包含伪迹的试次
    ③ 只保留 22 个 EEG 通道（丢弃 EOG）
    结果：约 270-288 个 Epochs，形状 (n_trials, 22, 1001)
```

### 5.2 标签准备

```python
y = epochs_events_to_class_labels(epochs)
```

**工作原理**：
- GDF 中的注解字符串 '769'-'772' 被 MNE 映射为内部整数 ID（如 7-10）
- 通过 `epochs.event_id` 字典（如 `{'769': 7, '770': 8, ...}`）反查
- 将内部 ID 动态映射为类别标签 1-4

### 5.3 特征提取 + 分类阶段

5 种特征方案，全部封装在 sklearn Pipeline 中防止数据泄漏：

```
Pipeline 结构（以 Fused 为例）:
┌─────────────────────────────────────────────────┐
│ EpochsToArray → FeatureUnion → StandardScaler → SVM │
│                  ├─ CSP (4 维)                       │
│                  └─ Wavelet (110 维)                 │
│                  = 114 维特征                        │
└─────────────────────────────────────────────────┘
```

| 方案 | 特征维度 | 说明 |
|------|---------|------|
| CSP | 4 | 4 个 CSP 成分的对数方差 |
| Wavelet | 110 | 22 通道 × 5 层小波能量 |
| Fused | 114 | CSP(4) + Wavelet(110) 拼接 |
| Fused-Motor | 19 | CSP(4) + Wavelet(15)，小波只用 C3/Cz/C4 三通道 |
| FBCSP | 20 | 5 个频段 × 4 个 CSP 成分 |

**FBCSP 详解**：
```
输入 Epochs (n, 22, 1001)
    │
    ├─ 8-12 Hz 带通 → CSP → 4 维
    ├─ 12-16 Hz 带通 → CSP → 4 维
    ├─ 16-20 Hz 带通 → CSP → 4 维
    ├─ 20-24 Hz 带通 → CSP → 4 维
    └─ 24-30 Hz 带通 → CSP → 4 维
    │
    ▼
    拼接 → 20 维 → StandardScaler → SVM
```

### 5.4 评估

- **交叉验证**：10 折分层 K-Fold（每折保持类别比例），Pipeline 在每折内独立拟合（无泄漏）
- **独立测试集**：在 T 文件上训练，在 E 文件上预测，标签从 .mat 文件读取

---

## 6. 所有可调参数速查表

所有参数集中定义在 `code/config.py` → `EEGPipelineConfig`：

### 预处理参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `ica_l_freq` | 1.0 Hz | ICA 前高通截止 | 0.5-2.0，太低保留漂移噪声，太高丢失伪迹低频特征 |
| `ica_h_freq` | 40.0 Hz | ICA 前低通截止 | 一般不改 |
| `task_l_freq` | **8.0 Hz** | 任务滤波高通 | **关键参数**，6-10 Hz 可试。降到 4 Hz 可引入 θ 节律 |
| `task_h_freq` | **30.0 Hz** | 任务滤波低通 | **关键参数**，25-40 Hz 可试。升到 40 Hz 可引入 γ 节律 |
| `epoch_tmin` | 0.0 s | Epoch 起始（相对于 cue） | -0.5 到 0.5 可试，负值包含 cue 前基线 |
| `epoch_tmax` | 4.0 s | Epoch 结束（相对于 cue） | 3.0-6.0 可试，更长包含更多信息但也更多噪声 |

### 特征提取参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `csp_components` | **4** | CSP 成分数 | **关键参数**，2-8 可试。4 = 2 对滤波器 |
| `wavelet` | `'db4'` | 小波基类型 | `db2`-`db8` 可试，`db4` 是 EEG 领域公认最佳 |
| `wavelet_level` | 4 | 小波分解层数 | 3-5 可试，层数越多频率分辨率越高 |

### 分类参数

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `svm_kernel` | `'rbf'` | SVM 核函数 | `'linear'` 高维特征可能更好，`'rbf'` 通用 |
| `cv_folds` | 10 | 交叉验证折数 | 5-10，少数据用 5 避免测试集太小 |
| `random_state` | 42 | 随机种子 | 改变种子看结果稳定性 |

### FBCSP 专属参数（在 `train_eeg_svm_pipeline` 调用时传入）

| 参数 | 默认值 | 含义 | 调参建议 |
|------|--------|------|---------|
| `freq_bands` | `[(8,12),(12,16),(16,20),(20,24),(24,30)]` | 频段划分 | 可试 `[(4,8),(8,12),(12,16),(16,20),(20,30)]` 引入 θ |

---

## 7. 文件与函数索引

### 核心流程文件

| 文件 | 核心函数 | 作用 |
|------|---------|------|
| `config.py` | `EEGPipelineConfig`, `epochs_events_to_class_labels()` | 全局配置 + 事件映射 |
| `general_process.py` | `single_subject_pipeline()` | 单被试：预处理→5种特征→分类→汇总 |
| `test_on_evaluation_set.py` | `train_and_evaluate()` | 训练集训练+评估集独立测试 |
| `batch_processing.py` | `batch_process_subjects()` | 批量处理 A01-A09 |

### 预处理模块 (`pretreatment/`)

| 文件 | 核心函数 | 作用 |
|------|---------|------|
| `eeg_analysis.py` | `get_modified_raw_data()` | 加载 GDF + 通道重命名 + 设置电极位置 |
| `complete_preprocessing.py` | `complete_preprocessing_pipeline()` | 编排 8 步预处理流程 |
| 同上 | `filter_for_ica()` | 1-40 Hz FIR 滤波 |
| 同上 | `fit_ica()` | FastICA 分解 |
| 同上 | `detect_and_remove_artifacts()` | 自动+手动 EOG 检测 |
| 同上 | `apply_ica()` | 应用 ICA 去噪 |
| 同上 | `filter_for_task()` | 8-30 Hz FIR 滤波 |
| 同上 | `set_reference()` | 平均重参考 |
| 同上 | `create_epochs_with_artifact_removal_mne()` | 分段 + 1023 伪迹剔除 |

### 特征提取模块 (`feature_extraction/`)

| 文件 | 核心类/函数 | 作用 |
|------|------------|------|
| `csp_feature.py` | `MNECSPTransformer` | sklearn 兼容的 CSP 封装（MNE 实现） |
| 同上 | `extract_csp_features()` | 手写 CSP 实现（教学用，Pipeline 未使用） |
| `wavelet_feature.py` | `WaveletEnergyTransformer` | sklearn 兼容的小波能量提取器 |
| 同上 | `extract_wavelet_energy_features()` | 独立的小波特征提取函数 |
| `eeg_transformers.py` | `FilterBankCSP` | FBCSP：多频段 CSP（scipy 高效滤波） |
| 同上 | `EpochsToArray` | Epochs → numpy 数组转换器 |
| 同上 | `make_feature_union()` | 构建 CSP/Wavelet/Fused 的 FeatureUnion |
| 同上 | `select_motor_channels()` | 选取 C3(idx=6), Cz(8), C4(10) |

### 分类模块 (`classification/`)

| 文件 | 核心函数 | 作用 |
|------|---------|------|
| `svm_classifier.py` | `make_eeg_svm_pipeline()` | 构建完整 Pipeline |
| 同上 | `train_eeg_svm_pipeline()` | **核心训练函数**：构建 Pipeline + CV + 全量拟合 |
| 同上 | `train_svm_classifier()` | 低层 SVM 训练（已弃用，用于向后兼容） |
| 同上 | `optimize_svm_hyperparameters()` | 网格搜索 C/gamma |
| 同上 | `evaluate_model()` | 计算 accuracy/kappa/混淆矩阵 |
| 同上 | `plot_confusion_matrix()` | 混淆矩阵热力图可视化 |

### CLI 入口

| 文件 | 命令 |
|------|------|
| `train.py` | `python code/train.py --subject A01T` |
| `evaluate.py` | `python code/evaluate.py --train-subject A01T --eval-subject A01E` |
| `batch_preprocess.py` | `python code/batch_preprocess.py --subjects A01 A02` |

---

## 8. 如何运行

### 环境要求
- Python ≥ 3.11
- 依赖：`mne`, `numpy`, `scipy`, `scikit-learn`, `pywavelets`, `matplotlib`, `pandas`

### 安装
```bash
cd "f:\Graduation Design"
pip install -e .   # 或 uv sync
```

### 单被试训练（最快看到结果）
```bash
python code/train.py --subject A01T
```
输出：5 种特征方案的 10 折 CV 准确率 + 混淆矩阵

### 独立测试集评估
```bash
python code/evaluate.py --train-subject A01T --eval-subject A01E
```

### 批量处理
```bash
python code/batch_preprocess.py
```

### Jupyter 交互式
```bash
jupyter notebook jupyter/general_process.ipynb
```

---

## 9. 调参指南

### 优先级排序（影响从大到小）

**第一梯队（改 1 个可能涨 2-5%）：**

1. **`task_l_freq` / `task_h_freq`**（任务滤波频段）
   - 当前 8-30 Hz，可试 6-35 Hz 或 4-40 Hz
   - 低频端：4 Hz 引入 θ 节律（有些被试有用）
   - 高频端：35-40 Hz 可能捕获低 γ 成分

2. **`csp_components`**（CSP 成分数）
   - 当前 4（2 对），可试 6 或 8
   - 成分越多特征越丰富但也越容易过拟合
   - 4 类问题建议 ≥4

3. **选择哪种特征方案**
   - 不同被试最优方案不同，可以用 `general_process.py` 跑一遍比较

**第二梯队（微调 1-2%）：**

4. **SVM 超参数 C 和 gamma**
   - 用 `optimize_svm_hyperparameters()` 网格搜索
   - C: [0.1, 1, 10, 100]，gamma: ['scale', 0.001, 0.01, 0.1]

5. **FBCSP 频段划分**
   - 更细的频段（如 2 Hz 一段）可能更精确但也更容易过拟合
   - 尝试包含 θ 频段 (4-8 Hz)

6. **`epoch_tmin` / `epoch_tmax`**
   - [0.5, 3.5] 可能比 [0, 4] 更好（去掉反应延迟期和末尾噪声）

**第三梯队（通常不需要改）：**

7. `wavelet` / `wavelet_level`：db4 + 4 层在 EEG 领域是标准配置
8. `cv_folds`：10 折是标准
9. ICA 参数：99% 方差保留 + 自动阈值检测通常足够

---

## 10. 学习资源推荐

### 入门必读（按阅读顺序）

| 主题 | 资源 | 说明 |
|------|------|------|
| EEG 基础 | 本项目 `知识补充/EEG 运动想象分类基础概念详解.md` | 最贴合本项目的基础概念 |
| BCI 竞赛数据 | 本项目 `知识补充/BCIC_IV_2a 数据集详解.md` | 数据集每个字段的含义 |
| CSP 算法 | Blankertz et al. (2008), "Optimizing Spatial Filters" | CSP 原始论文，数学清晰 |
| 小波变换 | 《小波十讲》前 3 讲 或 B站搜"小波变换入门" | 建立小波直觉 |
| SVM | 周志华《机器学习》第 6 章 | 中文最好的 SVM 讲解 |

### 进阶提升

| 主题 | 资源 |
|------|------|
| MNE-Python 官方教程 | https://mne.tools/stable/auto_tutorials/ |
| FBCSP 原始论文 | Ang et al. (2008), "Filter Bank Common Spatial Pattern" |
| BCI 综述 | Lotte et al. (2018), "A review of classification algorithms for EEG-based BCI" |
| sklearn Pipeline | https://scikit-learn.org/stable/modules/compose.html |
| 交叉验证与数据泄漏 | https://scikit-learn.org/stable/common_pitfalls.html |

### 视频推荐

| 主题 | 平台 | 搜索关键词 |
|------|------|-----------|
| 线性代数复习 | B站/YouTube | 3Blue1Brown 线性代数的本质 |
| 傅里叶→小波 | B站 | "从傅里叶变换到小波变换" |
| CSP 直觉 | B站 | "CSP 公共空间模式 BCI 脑机接口" |
| ICA 原理 | YouTube | "Independent Component Analysis StatQuest" |
| SVM | YouTube/B站 | "StatQuest SVM" |

---

## 附：项目目录结构

```
Graduation Design/
├── BCICIV_2a_gdf/          # 原始数据（GDF 格式）
│   ├── A01T.gdf ~ A09T.gdf  # 训练集（含类别标签）
│   └── A01E.gdf ~ A09E.gdf  # 评估集（不含标签）
├── true_labels/            # 评估集真实标签（.mat 格式）
├── code/                   # 核心代码
│   ├── config.py           # 全局配置
│   ├── general_process.py  # 单被试完整流程
│   ├── test_on_evaluation_set.py  # 独立测试评估
│   ├── batch_processing.py # 批量处理
│   ├── train.py            # CLI: 训练
│   ├── evaluate.py         # CLI: 评估
│   ├── batch_preprocess.py # CLI: 批量预处理
│   ├── pretreatment/       # 预处理模块
│   │   ├── eeg_analysis.py
│   │   └── complete_preprocessing.py
│   ├── feature_extraction/ # 特征提取模块
│   │   ├── csp_feature.py
│   │   ├── wavelet_feature.py
│   │   └── eeg_transformers.py
│   └── classification/     # 分类模块
│       └── svm_classifier.py
├── jupyter/                # 交互式 Notebook
│   └── general_process.ipynb
├── results/                # 输出结果
│   └── {subject_id}/
│       ├── figures/        # 混淆矩阵等图片
│       ├── models/         # 保存的模型
│       └── metrics/        # 评估指标
└── 知识补充/               # 背景知识文档
```
