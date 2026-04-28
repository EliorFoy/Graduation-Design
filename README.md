# EEG 运动想象分类流程说明与逻辑评估

本项目面向 BCI Competition IV 2a 数据集，完成四类运动想象 EEG 分类任务：左手、右手、双脚、舌头。当前代码主流程是离线实验流程，重点目标是保证实验逻辑可信、避免数据泄漏，并在此基础上继续提升 Accuracy、Kappa 等关键指标。

## 1. 项目与数据结构

推荐目录结构如下：

```text
F:\Graduation Design
├── BCICIV_2a_gdf/          # BCI Competition IV 2a GDF 数据
│   ├── A01T.gdf            # 训练集，含 769-772 类别 cue 事件
│   └── A01E.gdf            # 评估集，通常只含 trial start 事件 768
├── true_labels/            # 官方评估集真实标签 .mat
│   └── A01E.mat
├── code/                   # 主要代码
│   ├── config.py           # 全局配置、标签映射、结果目录
│   ├── train.py            # 单被试训练入口
│   ├── evaluate.py         # 训练集 T → 评估集 E 独立测试入口
│   ├── batch_preprocess.py # 批量预处理入口
│   ├── pretreatment/       # 数据加载、通道处理、滤波、ICA、Epoch
│   ├── feature_extraction/ # CSP、小波、融合特征 Transformer
│   └── classification/     # SVM 与 sklearn Pipeline
└── results/                # 运行后保存图像、模型、指标
```

核心入口：

- `code/train.py`：单个训练集 session 的训练和交叉验证。
- `code/evaluate.py`：使用训练集 `T` 训练，在独立评估集 `E` 上测试。
- `code/batch_preprocess.py`：批量预处理多个被试或 session。

## 2. 当前处理流程

当前主链路可以概括为：

```text
GDF 读取
→ 通道重命名与类型设置
→ 1-40 Hz 轻度滤波用于 ICA
→ ICA 去除 EOG/ECG 等伪迹成分
→ 8-30 Hz 任务频段滤波
→ 平均参考
→ Epoch 分段与伪迹 trial 剔除
→ 标签映射
→ CSP / 小波能量 / 融合特征
→ StandardScaler
→ SVM 分类器
→ 训练集交叉验证 / 独立评估集测试
```

### 2.1 数据读取与通道处理

`pretreatment/eeg_analysis.py` 负责读取 GDF 文件，并将 BCI IV 2a 的 EEG/EOG 通道设置为更标准的名称和类型。随后设置 `standard_1020` montage，以便后续 CSP 拓扑或空间模式解释。

数学意义：这一步不改变分类标签，只是将原始数据整理为 MNE 可处理的标准 Raw 对象。

### 2.2 滤波与 ICA

当前预处理在 `pretreatment/complete_preprocessing.py` 中执行：

1. 使用 `1-40 Hz` 轻度滤波数据拟合 ICA。
2. 自动检测 EOG/ECG 相关 ICA 成分。
3. 将 ICA 应用于原始 Raw。
4. 再使用 `8-30 Hz` 任务频段滤波，聚焦 mu/beta 节律。
5. 设置平均参考。

数学意义：

- 运动想象 EEG 常关注 mu 节律约 `8-13 Hz` 与 beta 节律约 `13-30 Hz`。
- ICA 用于降低眼动、心电等非脑源伪迹影响。
- 当前流程适合离线分析；如果目标是实时系统，需要进一步考虑因果滤波、在线 ICA 或无需 ICA 的在线方案。

### 2.3 Epoch 分段与标签逻辑

训练集 `T` 与评估集 `E` 的事件逻辑不同，因此代码分别处理：

- 训练集 `AxxT.gdf`：直接使用 cue 事件 `769, 770, 771, 772` 分段。
- 评估集 `AxxE.gdf`：通常没有类别 cue 标签，使用 trial start 事件 `768` 分段，再加载 `true_labels/AxxE.mat` 中的真实标签。

当前默认时间窗：

- 训练集：以 cue onset 为 `t=0`，取 `[0, 4]s`。
- 评估集：以 trial start `768` 为 `t=0`，cue 约在 2 秒后出现，因此取 `[2, 6]s`。

两者在数学上等价，都是 cue 后 4 秒运动想象窗口。

### 2.4 标签映射

BCI IV 2a 训练集中的事件 ID 与 `.mat` 标签并不直接同值：

| GDF 事件 | 类别标签 | 类别含义 |
|---|---:|---|
| `769` | `1` | left hand |
| `770` | `2` | right hand |
| `771` | `3` | feet |
| `772` | `4` | tongue |

当前代码通过 `code/config.py` 中的 `events_to_class_labels()` 将训练集事件映射为 `1-4`。这一步非常关键，因为评估集 `.mat` 文件中的 `classlabel` 本身就是 `1-4`。如果不做这个映射，训练标签和测试标签会处在不同编码空间，测试指标会失真。

### 2.5 特征提取与分类

当前代码使用 sklearn Pipeline 封装特征与分类：

```text
Epochs/ndarray
→ EpochsToArray
→ FeatureUnion(CSP, WaveletEnergy)
→ StandardScaler
→ SVC
```

支持三种特征：

- `CSP`：使用 `mne.decoding.CSP`，提取空间滤波后的 log-variance 特征。
- `Wavelet`：对每个 trial、每个通道做离散小波分解，提取各层能量。
- `Fused`：拼接 CSP 与小波能量特征。

重要逻辑：CSP、Wavelet、StandardScaler、SVM 都在 Pipeline 内部，因此交叉验证时每个 fold 都只在训练折上 `fit`，再在验证折上 `transform/predict`。这避免了“先用全量数据提 CSP 或标准化，再做交叉验证”的数据泄漏。

## 3. 数学逻辑是否正确

总体结论：当前主流程在离线 EEG 分类实验中是合理的，数学主链路基本正确。

### 3.1 已经正确的部分

- **训练/测试标签空间一致**：训练集 `769-772` 已映射到 `1-4`，与评估集 `.mat` 标签一致。
- **训练与评估分离**：评估集只调用训练好的 Pipeline 做 `predict`，不重新拟合 CSP 或 scaler。
- **交叉验证无明显特征泄漏**：CSP 和 StandardScaler 被放入 Pipeline，在 CV 内部拟合。
- **训练集与评估集时间窗对齐**：训练集 cue 后 `[0, 4]s` 等价于评估集 trial start 后 `[2, 6]s`。
- **指标选择合理**：Accuracy、Kappa、混淆矩阵都适合四分类 EEG 评估，其中 Kappa 更适合补充说明多分类一致性。

### 3.2 仍需注意的问题

- **当前是离线流程，不是实时流程**：滤波和 ICA 在整段连续数据上处理，适合毕业设计离线实验；实时 BCI 需另行设计在线滤波和在线预测。
- **ICA 自动成分剔除可能不稳定**：不同被试 EOG/ECG 成分识别质量不同，可能影响准确率。
- **单一 `8-30 Hz` 频段可能不是最优**：不同被试的 mu/beta 节律峰值不同，固定频段可能限制性能。
- **默认 epoch `[0, 4]s` 不一定最优**：cue 刚出现后的早期时间可能包含视觉诱发和准备活动，未必是最强运动想象区间。
- **小波能量特征可能受幅值尺度影响**：原始能量可能被高幅值通道或噪声主导，建议改为 log-energy 或相对能量。

## 4. 提高 Accuracy / Kappa 的重点改进建议

下面按优先级列出建议。建议每次只改一个因素，并记录实验结果，避免多个变量同时变化导致无法解释。

### 4.1 优先优化滤波策略

当前默认频段是 `8-30 Hz`。建议比较：

- `8-30 Hz`：当前 baseline。
- `8-32 Hz`：保留更高 beta 频段。
- `7-35 Hz`：覆盖更宽 mu/beta 活动。
- Filter Bank CSP：将频段拆成多个子频带，例如：
  - `8-12 Hz`
  - `12-16 Hz`
  - `16-20 Hz`
  - `20-24 Hz`
  - `24-30 Hz`

推荐原因：运动想象的判别性频段具有明显个体差异，Filter Bank CSP 通常比单个宽频 CSP 更稳健。

### 4.2 优先优化 epoch 时间窗

当前默认取 cue 后 `[0, 4]s`。建议比较：

- `[0.0, 4.0]s`：当前 baseline。
- `[0.5, 3.5]s`：去掉 cue 后早期视觉/准备期。
- `[0.5, 4.0]s`：保留较长运动想象后段。
- `[1.0, 4.0]s`：更聚焦稳定运动想象期。

推荐原因：CSP 对时间窗敏感，不同被试最佳运动想象时段可能不同。

### 4.3 CSP 与 SVM 超参数搜索

建议系统搜索：

```text
CSP n_components: 4, 6, 8
SVM C: 0.1, 1, 10, 100
SVM gamma: scale, 0.01, 0.001
SVM kernel: rbf, linear
```

推荐主指标：

- 训练集内部：Stratified CV mean accuracy、CV Kappa。
- 独立评估集：test accuracy、test Kappa、每类 recall。

注意：超参数搜索也必须放在 Pipeline 内部完成，不能先用全量数据提 CSP 后再调参。

### 4.4 小波特征改进

当前小波特征是各层能量。建议尝试：

- `log(energy + eps)`：减弱极端能量值影响。
- 相对能量：每层能量除以总能量。
- 只保留与 `8-30 Hz` 更相关的小波层。
- 融合后加入 `SelectKBest`、PCA 或 L1 选择，减少冗余特征。

推荐原因：小波特征维度较高，直接拼接可能带来噪声和过拟合。

### 4.5 更严谨的实验协议

建议 README 和论文中明确区分两类结果：

1. **训练集内部交叉验证**：用于模型选择和调参。
2. **训练集 T → 评估集 E 独立测试**：用于报告泛化能力。

最终报告建议至少包含：

- Accuracy
- Cohen's Kappa
- 混淆矩阵
- 每类 precision / recall / F1
- 不同被试的平均值和标准差

## 5. 运行方式

建议在项目根目录运行以下命令。

### 5.1 环境检查

```powershell
.\.venv\Scripts\python.exe -m compileall code
```

### 5.2 单被试训练与交叉验证

```powershell
.\.venv\Scripts\python.exe code\train.py --subject A01T
```

### 5.3 独立评估集测试

```powershell
.\.venv\Scripts\python.exe code\evaluate.py --train-subject A01T --eval-subject A01E
```

### 5.4 批量预处理

```powershell
.\.venv\Scripts\python.exe code\batch_preprocess.py --subjects A01 A02 A03 --sessions T E
```

如果数据不在默认 `BCICIV_2a_gdf/`，可以指定：

```powershell
.\.venv\Scripts\python.exe code\evaluate.py --train-subject A01T --eval-subject A01E --data-root "F:\path\to\BCICIV_2a_gdf"
```

## 6. 指标解释

### Accuracy

Accuracy 是预测正确 trial 数占总 trial 数的比例：

```text
Accuracy = 正确预测数 / 总 trial 数
```

优点是直观；缺点是当类别不均衡时可能过于乐观。

### Cohen's Kappa

Kappa 衡量预测结果与真实标签的一致性，并扣除了随机猜测的一致性。四分类 EEG 中，Kappa 通常比单纯 Accuracy 更适合作为补充指标。

一般理解：

- `Kappa < 0.2`：较弱一致性。
- `0.2-0.4`：一般。
- `0.4-0.6`：中等。
- `0.6-0.8`：较好。
- `>0.8`：优秀。

### Confusion Matrix

混淆矩阵用于观察哪些类别容易混淆。例如左手和右手混淆较多，可能说明左右半球运动皮层模式不够明显；双脚和舌头混淆较多，可能说明空间特征或频段选择需要调整。

## 7. 建议实验记录表

| 实验编号 | 被试 | 频段 | 时间窗 | 特征 | CSP 维度 | SVM 参数 | CV Accuracy | Test Accuracy | Test Kappa | 备注 |
|---|---|---|---|---|---:|---|---:|---:|---:|---|
| baseline-01 | A01 | 8-30Hz | 0-4s | Fused | 4 | RBF, default |  |  |  | 当前 baseline |
| exp-01 | A01 | 8-32Hz | 0-4s | Fused | 4 | RBF, default |  |  |  | 频段扩展 |
| exp-02 | A01 | 8-30Hz | 0.5-3.5s | Fused | 4 | RBF, default |  |  |  | 时间窗优化 |
| exp-03 | A01 | FBCSP | 0.5-4s | CSP | 6 | grid search |  |  |  | Filter Bank CSP |

## 8. 当前结论

当前代码的主逻辑已经具备可信的离线分类实验基础：标签映射正确、训练/测试分离正确、交叉验证中的特征拟合逻辑正确。后续提升指标的关键不再是修正明显错误，而是系统调参和特征工程优化，尤其是滤波频段、时间窗、FBCSP、CSP 维度、SVM 超参数和特征选择。

建议下一阶段先建立 baseline 结果表，然后逐项开展单因素实验，最终选择在独立评估集 Kappa 和 Accuracy 上都稳定提升的方案。
