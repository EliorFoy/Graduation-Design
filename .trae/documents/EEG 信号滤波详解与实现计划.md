# EEG 信号滤波详解与实现计划

> 📚 **目标**：理解为什么要滤波、如何滤波，并实现 BCIC IV-2a 数据集的滤波处理

---

## 📋 目录

1. [为什么要滤波？](#一为什么要滤波)
2. [滤波的基本原理](#二滤波的基本原理)
3. [BCIC IV-2a 数据集的滤波策略](#三bcic-iv-2a数据集的滤波策略)
4. [MNE 滤波实现方法](#四mne滤波实现方法)
5. [实践步骤](#五实践步骤)
6. [验证滤波效果](#六验证滤波效果)

---

## 一、为什么要滤波？

### 1.1 EEG 信号的特点

**原始 EEG 信号包含多种成分：**

```
原始 EEG = 神经活动信号 + 伪迹 + 噪声

神经活动信号：
  - δ波 (0.5-4 Hz)：深度睡眠
  - θ波 (4-8 Hz)：困倦、儿童
  - α波 (8-13 Hz)：放松、闭眼
  - β波 (13-30 Hz)：运动想象、认知
  - γ波 (30-100 Hz)：高级认知

伪迹和噪声：
  - 眼电 (EOG)：0.5-10 Hz，幅度大
  - 肌电 (EMG)：30-100 Hz，宽带
  - 心跳 (ECG)：1-20 Hz
  - 工频干扰：50/60 Hz
  - 基线漂移：<0.5 Hz
  - 电极接触噪声：高频
```

### 1.2 滤波的目的

| 目的 | 说明 | 例子 |
|------|------|------|
| **去除无关频段** | 只保留感兴趣的频段 | 运动想象主要看μ/β节律 (8-30Hz) |
| **提高信噪比** | 减少噪声，增强有效信号 | 去除工频干扰、肌电伪迹 |
| **满足算法要求** | 某些算法需要特定频段 | CSP 通常在 8-30Hz 效果最好 |
| **标准化处理** | 使不同数据集可比 | 统一使用 4-40Hz 带通滤波 |

### 1.3 不滤波会怎样？

```python
# 假设原始 EEG 的功率谱：
0.5 Hz:  ████████████████████  ← 基线漂移（很强但无用）
10 Hz:   ████████  ← α节律（我们想要的）
50 Hz:   ██████████████  ← 工频干扰（强噪声）
80 Hz:   ████████  ← 肌电伪迹

# 如果不滤波：
- 分类器会被强噪声误导
- 特征提取效果差
- 分类准确率低（可能只有 50-60%）

# 滤波后：
8-30 Hz: ████████  ← 只保留运动想象相关频段
- 信噪比提高
- 特征更明显
- 分类准确率提升（可达 70-80%）
```

---

## 二、滤波的基本原理

### 2.1 滤波器的类型

```
┌─────────────────────────────────────────────────────┐
│  1. 低通滤波器 (Low-pass)                           │
│     允许低频通过，衰减高频                          │
│     例：保留 <40Hz，去除肌电和高频噪声              │
│                                                     │
│  2. 高通滤波器 (High-pass)                          │
│     允许高频通过，衰减低频                          │
│     例：保留 >0.5Hz，去除基线漂移                   │
│                                                     │
│  3. 带通滤波器 (Band-pass)                          │
│     低通 + 高通 = 只允许某个频段通过                │
│     例：0.5-40Hz，最常用！                          │
│                                                     │
│  4. 带阻滤波器 (Band-stop / Notch)                  │
│     阻止某个频段，其他都通过                        │
│     例：48-52Hz，去除工频干扰                       │
└─────────────────────────────────────────────────────┘
```

### 2.2 可视化理解

```
原始信号频谱：
幅度
  ↑
  │    ████                    ████
  │    ████    有效信号        ████ ← 工频干扰
  │    ████    ████████        ████
  │██████████████████████████████████
  └──────────────────────────────────→ 频率 (Hz)
   0    10    20    30    40    50   100

带通滤波 (0.5-40Hz) 后：
幅度
  ↑
  │          ████████
  │          ████████ ← 只保留这部分
  │████████████████████
  └──────────────────────────────────→ 频率 (Hz)
   0    10    20    30    40    50   100
        ↑_______________↑
         0.5Hz         40Hz

带阻滤波 (50Hz 陷波) 后：
幅度
  ↑
  │    ████                    ██
  │    ████    ████████        ██
  │██████████████████████████████
  └──────────────────────────────────→ 频率 (Hz)
                        ↑
                    50Hz 被去除
```

### 2.3 关键参数

| 参数 | 含义 | 如何选择 |
|------|------|---------|
| **截止频率** | 滤波器开始衰减的频率点 | 根据目标频段选择 |
| **过渡带宽度** | 从通带到阻带的过渡区域 | 越窄越好，但计算成本高 |
| **滤波器阶数** | 滤波器的复杂度 | 越高衰减越陡，但可能不稳定 |
| **纹波** | 通带内的波动 | 越小越好 |

---

## 三、BCIC IV-2a 数据集的滤波策略

### 3.1 推荐的滤波参数

**针对运动想象任务：**

```python
# 方案 1：标准带通滤波（最常用）
l_freq = 0.5   # 高通截止频率
h_freq = 40    # 低通截止频率

# 保留频段：0.5-40 Hz
# 包含：μ节律 (8-13Hz) + β节律 (13-30Hz)
# 适用于：CSP + SVM/LDA 传统方法

# 方案 2：窄带滤波（聚焦核心频段）
l_freq = 8     # 高通
h_freq = 30    # 低通

# 保留频段：8-30 Hz
# 只包含：μ+β节律
# 适用于：专注运动想象相关节律

# 方案 3：宽频滤波（保留更多信息）
l_freq = 4     # 高通
h_freq = 48    # 低通

# 保留频段：4-48 Hz
# 包含：θ+μ+β+ 部分γ
# 适用于：深度学习（让网络自己学）
```

### 3.2 是否需要陷波滤波？

**BCIC IV-2a 数据集的特殊情况：**

```python
# 官方文档说明：
"22 EEG channels (0.5-100Hz; notch filtered)"

# 关键点：数据已经做过 50Hz 陷波滤波！
# 所以不需要再做陷波滤波

# 验证方法：
查看 50Hz 附近的功率谱
如果有明显的 50Hz 峰值 → 需要陷波
如果没有 → 已经处理过了
```

**建议：**
- ✅ 先检查 50Hz 处是否有峰值
- ✅ 如果没有，跳过陷波滤波
- ⚠️ 如果有，添加 50Hz 陷波滤波

### 3.3 为什么选择 0.5-40Hz？

| 频段 | 原因 |
|------|------|
| **高通 0.5Hz** | 去除基线漂移（<0.5Hz），保留δ波及以上 |
| **低通 40Hz** | 去除肌电伪迹（>40Hz），保留μ+β节律 |
| **不陷波** | 官方数据已做过 50Hz 陷波 |

---

## 四、MNE 滤波实现方法

### 4.1 MNE 的滤波 API

```python
import mne

# 方法 1：直接对 Raw 对象滤波（推荐）
raw_filtered = raw.copy().filter(
    l_freq=0.5,      # 高通截止频率
    h_freq=40,       # 低通截止频率
    fir_design='firwin',  # 滤波器设计方法
    skip_by_annotation='edge'  # 跳过边缘区域
)

# 方法 2：带通滤波（等价）
raw_filtered = raw.copy().filter(
    l_freq=0.5, 
    h_freq=40,
    method='fir'  # 使用 FIR 滤波器
)

# 方法 3：分别进行高通和低通
raw_highpass = raw.copy().filter(l_freq=0.5, h_freq=None)
raw_filtered = raw_highpass.filter(l_freq=None, h_freq=40)
```

### 4.2 参数详解

```python
raw.filter(
    # 必填参数
    l_freq=0.5,          # 高通截止频率 (Hz)
                         # 设为 None 表示不做高通
    
    h_freq=40,           # 低通截止频率 (Hz)
                         # 设为 None 表示不做低通
    
    # 可选参数
    fir_design='firwin', # 滤波器设计方法
                         # 'firwin' - 窗函数法（快）
                         # 'firwin2' - 频率采样法（灵活）
    
    filter_length='auto',# 滤波器长度
                         # 'auto' - 自动选择
                         # '10s' - 10 秒长度的滤波器
    
    l_trans_bandwidth='auto',  # 高通过渡带宽度
    h_trans_bandwidth='auto',  # 低通过渡带宽度
    # 'auto' = 截止频率的 10%
    
    n_jobs=1,            # 并行计算的 CPU 核心数
    
    verbose=True         # 是否显示进度
)
```

### 4.3 FIR vs IIR 滤波器

| 特性 | FIR (有限冲激响应) | IIR (无限冲激响应) |
|------|------------------|------------------|
| **相位响应** | 线性相位（不畸变） | 非线性相位（可能畸变） |
| **稳定性** | 总是稳定 | 可能不稳定 |
| **计算成本** | 较高 | 较低 |
| **MNE 默认** | ✅ 推荐使用 | ⚠️ 不推荐 |

**MNE 推荐使用 FIR：**
```python
# FIR 滤波器（推荐）
raw.filter(l_freq=0.5, h_freq=40, fir_design='firwin')

# IIR 滤波器（不推荐，除非有特殊需求）
raw.filter(l_freq=0.5, h_freq=40, method='iir')
```

---

## 五、实践步骤

### 步骤 1：检查原始数据频谱

```python
import mne
import matplotlib.pyplot as plt

# 加载数据
raw = mne.io.read_raw_gdf('./BCICIV_2a_gdf/A01T.gdf', preload=True)

# 计算功率谱密度
psd = raw.compute_psd(fmin=0, fmax=100)

# 可视化
psd.plot(average=True)
plt.title('原始数据功率谱')
plt.show()

# 观察：
# - 0.5Hz 以下是否有很强的基线漂移？
# - 50Hz 处是否有明显的峰值（工频干扰）？
# - 主要能量集中在哪个频段？
```

### 步骤 2：应用带通滤波

```python
# 滤波参数
L_FREQ = 0.5   # 高通
H_FREQ = 40    # 低通

print(f"应用带通滤波：{L_FREQ}-{H_FREQ} Hz")

# 滤波
raw_filtered = raw.copy().filter(
    l_freq=L_FREQ,
    h_freq=H_FREQ,
    fir_design='firwin',
    skip_by_annotation='edge',
    verbose=True
)

print("滤波完成！")
```

### 步骤 3：检查滤波效果

```python
# 计算滤波后的 PSD
psd_filtered = raw_filtered.compute_psd(fmin=0, fmax=100)

# 对比滤波前后
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 原始
psd.plot(ax=axes[0], average=True)
axes[0].set_title('滤波前 (原始)')
axes[0].set_xlim(0, 60)

# 滤波后
psd_filtered.plot(ax=axes[1], average=True)
axes[1].set_title(f'滤波后 ({L_FREQ}-{H_FREQ} Hz)')
axes[1].set_xlim(0, 60)

plt.tight_layout()
plt.show()

# 观察：
# - 0.5Hz 以下的能量是否被去除？
# - 40Hz 以上的能量是否被去除？
# - 50Hz 处是否有峰值？
```

### 步骤 4：可视化时域信号对比

```python
# 对比滤波前后的时域信号
duration = 5  # 显示 5 秒数据

# 滤波前
raw.plot(duration=duration, n_channels=8, 
         title='滤波前 (原始)', scalings='auto')

# 滤波后
raw_filtered.plot(duration=duration, n_channels=8, 
                  title=f'滤波后 ({L_FREQ}-{H_FREQ} Hz)', scalings='auto')
```

### 步骤 5：保存滤波后的数据

```python
# 保存为 FIF 格式（MNE 原生格式，读取快）
output_path = './BCICIV_2a_gdf/A01T_filtered.fif'
raw_filtered.save(output_path, overwrite=True)
print(f"滤波后的数据已保存到：{output_path}")
```

---

## 六、验证滤波效果

### 6.1 频域验证

```python
import numpy as np

# 计算各频段的功率
def compute_band_power(psd, freqs, band):
    """计算某个频段的平均功率"""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[:, mask], axis=1)

bands = {
    'δ (0.5-4Hz)': (0.5, 4),
    'θ (4-8Hz)': (4, 8),
    'α (8-13Hz)': (8, 13),
    'β (13-30Hz)': (13, 30),
    'γ (30-50Hz)': (30, 50),
}

psd_data, freqs = psd.get_data(return_freqs=True)
psd_filt_data, _ = psd_filtered.get_data(return_freqs=True)

print("各频段功率对比:")
print("-" * 60)
for band_name, (fmin, fmax) in bands.items():
    power_before = compute_band_power(psd_data, freqs, (fmin, fmax))
    power_after = compute_band_power(psd_filt_data, freqs, (fmin, fmax))
    
    change = (power_after - power_before) / power_before * 100
    
    print(f"{band_name:15s}: {power_before.mean():8.4f} → {power_after.mean():8.4f}  ({change:+6.1f}%)")
```

**预期结果：**

```
各频段功率对比:
------------------------------------------------------------
δ (0.5-4Hz)    :   12.3456 →   10.2345  (-17.1%)  ← 部分保留
θ (4-8Hz)      :    8.7654 →    8.5432  ( -2.5%)  ← 基本保留
α (8-13Hz)     :    6.5432 →    6.4321  ( -1.7%)  ← 保留
β (13-30Hz)    :    4.3210 →    4.2345  ( -2.0%)  ← 保留
γ (30-50Hz)    :    3.2109 →    0.1234  ( -96.2%) ← 大量去除 ✅
>40Hz          :    2.1098 →    0.0123  ( -99.4%) ← 去除 ✅
```

### 6.2 信噪比验证

```python
# 计算信噪比 (SNR)
def compute_snr(psd, freqs, signal_band, noise_band):
    """计算信噪比"""
    signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    
    signal_power = np.mean(psd[:, signal_mask], axis=1)
    noise_power = np.mean(psd[:, noise_mask], axis=1)
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

# 定义信号和噪声频段
signal_band = (8, 30)   # μ+β节律（有效信号）
noise_band_low = (0, 0.5)   # 基线漂移
noise_band_high = (40, 100) # 肌电伪迹

# 计算 SNR
snr_before = compute_snr(psd_data, freqs, signal_band, noise_band_high)
snr_after = compute_snr(psd_filt_data, freqs, signal_band, noise_band_high)

print(f"\n信噪比 (SNR):")
print(f"滤波前：{snr_before.mean():.2f} dB")
print(f"滤波后：{snr_after.mean():.2f} dB")
print(f"改善：{snr_after.mean() - snr_before.mean():+.2f} dB")
```

### 6.3 时域验证

```python
# 对比滤波前后的标准差
std_before = np.std(raw.get_data(), axis=1)
std_after = np.std(raw_filtered.get_data(), axis=1)

print(f"\n时域标准差:")
print(f"滤波前：{std_before.mean():.4f} μV")
print(f"滤波后：{std_after.mean():.4f} μV")
print(f"变化：{(std_after.mean() - std_before.mean()) / std_before.mean() * 100:+.1f}%")

# 预期：标准差略微下降（去除了噪声）
```

---

## 📝 完整代码模板

```python
"""
BCIC IV-2a 数据滤波处理
功能：
1. 加载原始 GDF 数据
2. 应用 0.5-40Hz 带通滤波
3. 验证滤波效果
4. 保存滤波后数据
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载原始数据")
print("=" * 60)

data_path = Path('./BCICIV_2a_gdf/A01T.gdf')
raw = mne.io.read_raw_gdf(str(data_path), preload=True, verbose=False)

print(f"通道数：{len(raw.ch_names)}")
print(f"采样率：{raw.info['sfreq']} Hz")
print(f"数据时长：{len(raw.times) / raw.info['sfreq']:.1f} 秒")

# ========== 2. 检查原始频谱 ==========
print("\n" + "=" * 60)
print("2. 检查原始频谱")
print("=" * 60)

psd_before = raw.compute_psd(fmin=0, fmax=100)
psd_data_before, freqs = psd_before.get_data(return_freqs=True)

print("原始数据功率谱已计算")

# ========== 3. 应用滤波 ==========
print("\n" + "=" * 60)
print("3. 应用带通滤波 (0.5-40 Hz)")
print("=" * 60)

L_FREQ = 0.5
H_FREQ = 40

raw_filtered = raw.copy().filter(
    l_freq=L_FREQ,
    h_freq=H_FREQ,
    fir_design='firwin',
    skip_by_annotation='edge',
    verbose=True
)

print("滤波完成！")

# ========== 4. 检查滤波效果 ==========
print("\n" + "=" * 60)
print("4. 检查滤波效果")
print("=" * 60)

psd_after = raw_filtered.compute_psd(fmin=0, fmax=100)
psd_data_after, _ = psd_after.get_data(return_freqs=True)

# 绘制对比图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

psd_before.plot(ax=axes[0], average=True)
axes[0].set_title('滤波前 (原始)')
axes[0].set_xlim(0, 60)
axes[0].grid(True, alpha=0.3)

psd_after.plot(ax=axes[1], average=True)
axes[1].set_title(f'滤波后 ({L_FREQ}-{H_FREQ} Hz)')
axes[1].set_xlim(0, 60)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./滤波效果对比.png', dpi=300)
print("滤波效果对比图已保存：滤波效果对比.png")

# ========== 5. 频段功率分析 ==========
print("\n" + "=" * 60)
print("5. 频段功率分析")
print("=" * 60)

def compute_band_power(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.mean(psd[:, mask], axis=1)

bands = {
    'δ (0.5-4Hz)': (0.5, 4),
    'θ (4-8Hz)': (4, 8),
    'α (8-13Hz)': (8, 13),
    'β (13-30Hz)': (13, 30),
    'γ (30-50Hz)': (30, 50),
}

print(f"{'频段':<15s} {'滤波前':>10s} {'滤波后':>10s} {'变化':>10s}")
print("-" * 60)

for band_name, (fmin, fmax) in bands.items():
    power_before = compute_band_power(psd_data_before, freqs, (fmin, fmax))
    power_after = compute_band_power(psd_data_after, freqs, (fmin, fmax))
    
    change = (power_after - power_before) / (power_before + 1e-10) * 100
    
    print(f"{band_name:<15s} {power_before.mean():>10.4f} {power_after.mean():>10.4f} {change:>9.1f}%")

# ========== 6. 保存数据 ==========
print("\n" + "=" * 60)
print("6. 保存滤波后数据")
print("=" * 60)

output_path = Path('./BCICIV_2a_gdf/A01T_filtered.fif')
raw_filtered.save(str(output_path), overwrite=True)
print(f"滤波后数据已保存：{output_path}")

print("\n" + "=" * 60)
print("滤波处理完成！")
print("=" * 60)

plt.show()
```

---

## 🎯 总结

### 关键要点

| 问题 | 答案 |
|------|------|
| **为什么要滤波？** | 去除噪声、提高信噪比、保留目标频段 |
| **滤什么频段？** | 0.5-40Hz（包含μ+β节律） |
| **怎么滤波？** | MNE 的 `raw.filter()` 方法，使用 FIR 滤波器 |
| **需要陷波吗？** | BCIC 数据已做过 50Hz 陷波，不需要 |
| **如何验证？** | 对比滤波前后的功率谱、信噪比、时域信号 |

### 下一步

滤波完成后，可以进入：
1. ✅ 伪迹去除（ICA）
2. ✅ 特征提取（ERD/CSP）
3. ✅ 分类器训练

---

**文档创建时间：** 2026-03-10  
**适用项目：** BCIC IV-2a 运动想象分类
