# Code 模块说明

本目录包含 EEG 运动想象信号处理的完整代码实现。

## 📁 目录结构

```
code/
├── classification/          # 分类器模块
│   ├── __init__.py
│   └── svm_classifier.py   # SVM 分类器（CSP/小波/融合特征）
│
├── feature_extraction/      # 特征提取模块
│   ├── __init__.py
│   ├── csp_feature.py      # CSP 空间滤波特征
│   ├── eeg_transformers.py # sklearn Transformer 封装
│   └── wavelet_feature.py  # 小波能量特征
│
├── pretreatment/            # 预处理模块
│   ├── __init__.py
│   ├── complete_preprocessing.py  # 完整预处理流程
│   └── eeg_analysis.py     # 通道映射与电极设置
│
├── generate_img/            # PPT 图片生成（独立工具）
│   ├── output/             # 生成的图片输出目录
│   └── generate_ppt_images.py
│
├── config.py               # 全局配置
├── general_process.py      # 单被试完整处理流程
├── train.py                # CLI 训练入口
├── test_on_evaluation_set.py  # 测试集评估
├── evaluate.py             # 模型评估工具
├── batch_preprocess.py     # 批量预处理（可选）
└── batch_processing.py     # 批量处理（可选）
```

## 🔗 依赖关系图

### 核心训练流程依赖链

```
train.py
  │
  └─> code.general_process (single_subject_pipeline)
       │
       ├─> code.pretreatment.complete_preprocessing
       │    │
       │    └─> code.pretreatment.eeg_analysis
       │         └─> code.config
       │
       ├─> code.classification.svm_classifier
       │    │
       │    ├─> code.feature_extraction.eeg_transformers
       │    │    │
       │    │    ├─> code.feature_extraction.csp_feature
       │    │    │    └─> (mne, numpy, sklearn)
       │    │    │
       │    │    └─> code.feature_extraction.wavelet_feature
       │    │         └─> (pywt, numpy, sklearn)
       │    │
       │    └─> (sklearn, matplotlib, numpy)
       │
       └─> code.config
```

### 模块职责说明

| 模块 | 职责 | 关键功能 |
|------|------|----------|
| **pretreatment** | EEG 信号预处理 | 通道映射、滤波、ICA 去噪、分段、伪迹剔除 |
| **feature_extraction** | 特征提取 | CSP 空间滤波、小波能量、sklearn Transformer 封装 |
| **classification** | 分类器训练 | SVM 分类、交叉验证、防数据泄漏 Pipeline |
| **config** | 全局配置 | 超参数、事件 ID、路径配置 |
| **general_process** | 流程编排 | 整合预处理→特征→分类的完整 pipeline |
| **generate_img** | 可视化展示 | 生成 PPT 所需图表（独立工具，不参与训练） |

## 🚀 快速开始

### 1. 训练单个被试

```bash
python code/train.py --subject A01T
```

### 2. 使用 Python API

```python
from code.general_process import single_subject_pipeline

# 处理 A01T 被试
results = single_subject_pipeline(subject_id='A01T')

# 查看结果
print(f"准确率 (融合特征): {results['metrics']['fused']['cv_mean']:.4f}")
```

### 3. 生成 PPT 图片

```bash
python code/generate_img/generate_ppt_images.py
```

## 📊 数据处理流程

### Step 1: 预处理 (`pretreatment/`)
1. **通道映射**: GDF 原始通道 → 标准 10-20 系统
2. **轻度滤波**: 1-40 Hz（为 ICA 准备）
3. **ICA 去噪**: 自动识别并剔除眼电/心电成分
4. **任务滤波**: 8-30 Hz（μ + β 节律）
5. **重参考**: 平均参考
6. **分段**: 提取 trial，剔除伪迹试次

### Step 2: 特征提取 (`feature_extraction/`)
- **CSP 特征**: 共空间模式，提取最具判别性的空间滤波
- **小波特征**: db4 小波分解，提取各频段能量
- **融合特征**: CSP + 小波特征拼接

### Step 3: 分类 (`classification/`)
- **SVM 分类器**: RBF 核支持向量机
- **防泄漏 CV**: 使用 sklearn Pipeline 确保特征提取在 fold 内进行
- **10 折交叉验证**: 评估模型泛化能力

## ⚙️ 配置说明

主要配置项在 `config.py` 中：

```python
DEFAULT_CONFIG = EEGPipelineConfig(
    # 预处理参数
    ica_l_freq=1.0,        # ICA 前高通
    ica_h_freq=40.0,       # ICA 前低通
    task_l_freq=8.0,       # 任务滤波高通
    task_h_freq=30.0,      # 任务滤波低通
    
    # Epochs 参数
    epoch_tmin=0.0,        # 起始时间
    epoch_tmax=4.0,        # 结束时间
    
    # 特征提取参数
    csp_components=4,      # CSP 成分数
    wavelet='db4',         # 小波基
    wavelet_level=4,       # 分解层数
    
    # 分类参数
    cv_folds=10,           # 交叉验证折数
    random_state=42,       # 随机种子
)
```

## 🔑 关键设计原则

### 1. 防止数据泄漏
- 使用 `sklearn.Pipeline` 将特征提取和分类器封装
- 交叉验证时，每折独立拟合并转换特征
- 避免在 CV 前全局标准化或提取特征

### 2. 模块化设计
- 预处理、特征提取、分类完全解耦
- 每个模块可独立测试和替换
- 便于实验不同特征组合

### 3. 符合竞赛规范
- 最终 Epochs 仅包含 22 个 EEG 通道
- EOG 通道不参与分类（防止"偷看"眼动信息）
- 严格遵循 BCIC IV-2a 官方推荐流程

## 📝 注意事项

1. **`generate_img/` 是独立工具**
   - 用于生成 PPT 展示图片
   - 不被训练流程依赖
   - 可以单独运行或删除（不影响训练）

2. **批量处理脚本**
   - `batch_preprocess.py` 和 `batch_processing.py` 为可选工具
   - 如需处理所有被试，可自行扩展

3. **数据路径**
   - 默认从项目根目录的 `BCICIV_2a_gdf/` 读取
   - 可通过 `--data-root` 参数指定其他路径

## 🛠️ 常见问题

### Q: 如何修改特征提取方法？
A: 编辑 `feature_extraction/eeg_transformers.py` 中的 `make_feature_union()` 函数

### Q: 如何更换分类器？
A: 修改 `classification/svm_classifier.py` 中的 `make_eeg_svm_pipeline()` 函数

### Q: 预处理参数在哪里调整？
A: 修改 `config.py` 中的 `EEGPipelineConfig` 配置类

### Q: 为什么不用 `generate_img` 模块？
A: 该模块仅用于生成展示图片，训练流程通过 `general_process.py` 自行绘制混淆矩阵

## 📚 相关文档

- 项目根目录 `README.md`: 整体项目介绍
- `知识补充/`: EEG 理论基础和技术详解
- `进度规划/`: 开发进度和计划

---

**最后更新**: 2026-04-24  
**维护者**: 毕业设计项目组
