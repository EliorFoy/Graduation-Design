"""
小波特征提取模块

使用离散小波变换 (DWT) 对 EEG 信号进行时频分析
提取各频段能量作为分类特征
"""

import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import platform

# 设置中文字体（兼容 Linux/Windows/macOS）
system_name = platform.system()

if system_name == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif system_name == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
else:  # Linux
    # 尝试使用 Linux 常见的中文字体
    plt.rcParams['font.sans-serif'] = [
        'WenQuanYi Zen Hei',      # 文泉驿正黑
        'WenQuanYi Micro Hei',    # 文泉驿微米黑
        'Noto Sans CJK SC',       # Google Noto 字体
        'Droid Sans Fallback',
        'DejaVu Sans'             # fallback：英文字体
    ]

plt.rcParams['axes.unicode_minus'] = False


def _wavelet_energy_from_array(data, wavelet='db4', level=4):
    """Extract wavelet energy features from (trials, channels, times) data."""

    features_list = []
    for trial_data in data:
        trial_features = []
        for channel_signal in trial_data:
            coeffs = pywt.wavedec(channel_signal, wavelet, level=level)
            trial_features.extend(float(np.sum(coeff ** 2)) for coeff in coeffs)
        features_list.append(trial_features)
    return np.asarray(features_list)


class WaveletEnergyTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wavelet energy feature extractor."""

    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _wavelet_energy_from_array(np.asarray(X), self.wavelet, self.level)


def extract_wavelet_energy_features(epochs, wavelet='db4', level=4):
    """
    提取小波分解各层的能量特征（推荐方法）

    Args:
        epochs: Epochs 数据 (n_trials, n_channels, n_times)
        wavelet: 小波基类型，默认'db4'（最适合运动想象）
        level: 分解层数，默认 4 层

    Returns:
        X_wavelet: 特征矩阵 (n_trials, n_channels × (level+1))
                   例如：(288, 22×5) = (288, 110)
    """
    print("\n" + "=" * 60)
    print("小波能量特征提取")
    print("=" * 60)
    
    data = epochs.get_data()
    n_trials, n_channels, n_times = data.shape
    print(f"   - 输入数据形状：{data.shape}")
    print(f"   - 试次数：{n_trials}")
    print(f"   - 通道数：{n_channels}")
    print(f"   - 小波基：{wavelet}")
    print(f"   - 分解层数：{level}")
    
    # 计算每个系数的频率范围
    sfreq = epochs.info['sfreq']
    print("\n   各层对应频段:")
    freq_bands = compute_wavelet_freq_bands(sfreq, level)
    for i, (band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
        print(f"   - Level {i+1} ({band_name}): {fmin:.1f}-{fmax:.1f} Hz")
    
    X_wavelet = _wavelet_energy_from_array(data, wavelet=wavelet, level=level)
    
    print("\n✅ 小波能量特征提取完成")
    print(f"   - 特征形状：{X_wavelet.shape}")
    print(f"   - 每个试次特征数：{X_wavelet.shape[1]}")
    
    return X_wavelet


def extract_band_power_features(epochs, bands=None, wavelet='cmor1.5-1.0'):
    """
    提取标准频段的功率特征（使用连续小波变换）
    
    Args:
        epochs: Epochs 数据
        bands: 频段定义字典
               默认：{'mu': (8, 13), 'beta': (13, 30)}
        wavelet: 复数小波基，默认'cmor1.5-1.0'（Morlet 小波）
    
    Returns:
        X_band: 特征矩阵 (n_trials, n_channels × n_bands)
    """
    print("\n" + "=" * 60)
    print("频段功率特征提取")
    print("=" * 60)
    
    if bands is None:
        bands = {
            'mu': (8, 13),
            'beta': (13, 30)
        }
    
    data = epochs.get_data()
    n_trials, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']
    
    print(f"   - 输入数据形状：{data.shape}")
    print(f"   - 采样率：{sfreq} Hz")
    print(f"   - 频段定义：{list(bands.keys())}")
    
    n_bands = len(bands)
    features_list = []
    
    for trial_idx in range(n_trials):
        trial_features = []
        
        for ch_idx in range(n_channels):
            signal = data[trial_idx, ch_idx, :]
            
            # 对每个频段计算功率
            for band_name, (fmin, fmax) in bands.items():
                # 使用小波变换提取频段
                freqs = np.linspace(fmin, fmax, 10)
                power = 0
                
                for freq in freqs:
                    # 连续小波变换
                    scales = pywt.frequency2scale(wavelet, freq / sfreq)
                    coeffs = pywt.cwt(signal, scales, wavelet)[0]
                    power += np.mean(np.abs(coeffs) ** 2)
                
                trial_features.append(power / len(freqs))
        
        features_list.append(trial_features)
    
    X_band = np.array(features_list)
    
    print("\n✅ 频段功率特征提取完成")
    print(f"   - 特征形状：{X_band.shape}")
    
    return X_band


def normalize_features(X_train, X_test=None):
    """
    特征标准化（Z-score 归一化）
    
    Args:
        X_train: 训练集特征 (n_samples_train, n_features)
        X_test: 测试集特征 (n_samples_test, n_features)，可选
    
    Returns:
        X_train_norm: 归一化后的训练特征
        X_test_norm: 归一化后的测试特征（如果有）
        scaler: 标准化器（保存用于后续预测）
    """
    print("\n" + "=" * 60)
    print("特征标准化 (Z-score)")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    
    print(f"   - 训练集形状：{X_train.shape} → {X_train_norm.shape}")
    print(f"   - 均值：{np.mean(X_train_norm):.6f} (应接近 0)")
    print(f"   - 标准差：{np.std(X_train_norm):.6f} (应接近 1)")
    
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
        print(f"   - 测试集形状：{X_test.shape} → {X_test_norm.shape}")
        print("✅ 训练集和测试集标准化完成")
        return X_train_norm, X_test_norm, scaler
    
    print("✅ 训练集标准化完成")
    return X_train_norm, scaler


def compute_wavelet_freq_bands(sfreq, level):
    """
    计算小波分解各层对应的频率范围
    
    Args:
        sfreq: 采样率 (Hz)
        level: 分解层数
    
    Returns:
        freq_bands: 有序字典，包含各层名称和频率范围
    """
    # Nyquist 频率
    nyquist = sfreq / 2.0
    
    # 初始化频段字典
    freq_bands = {}
    
    # 计算各层频段（从高频到低频）
    f_high = nyquist
    for i in range(1, level + 1):
        f_low = f_high / 2.0
        
        if i == 1:
            name = 'gamma'
        elif i == 2:
            name = 'beta_high'
        elif i == 3:
            name = 'beta_low+mu'
        elif i == 4:
            name = 'mu+alpha'
        else:
            name = f'level{i}'
        
        freq_bands[f'D{i}'] = (f_low, f_high)
        f_high = f_low
    
    # 近似系数（最低频）
    freq_bands['A'] = (0, f_high)
    
    # 反转顺序，从低频到高频
    freq_bands = dict(reversed(list(freq_bands.items())))
    
    return freq_bands


def plot_wavelet_decomposition(signal, sfreq, wavelet='db4', level=4, save_path='./output_img/wavelet_decomposition.png'):
    """
    可视化小波分解结果（已修复重构系数对齐 Bug）
    
    Args:
        signal: 原始信号
        sfreq: 采样率
        wavelet: 小波基
        level: 分解层数
        save_path: 保存路径
    """
    print("\n绘制小波分解图...")
    
    # 小波分解 (coeffs = [cA_n, cD_n, cD_n-1, ..., cD_1])
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    freq_bands = compute_wavelet_freq_bands(sfreq, level)
    
    # 创建图形 (原始信号 + 近似系数A + 各层细节系数D)
    n_plots = level + 2 
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    times = np.arange(len(signal)) / sfreq
    
    # 1. 绘制原始信号
    axes[0].plot(times, signal, 'k-', linewidth=0.8)
    axes[0].set_title('Original Signal', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    
    # 2. 绘制近似系数 (cA_n) - 最低频
    cA_rec = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(signal))
    fmin, fmax = freq_bands['A']
    axes[1].plot(times, cA_rec, 'r-', linewidth=0.8)
    axes[1].set_title(f'Approximation A{level} ({fmin:.1f}-{fmax:.1f} Hz)', fontsize=10)
    axes[1].set_ylabel('Amplitude')
    
    # 3. 绘制细节系数 (cD_n 到 cD_1)
    # coeffs[1] 对应 cD_level, coeffs[2] 对应 cD_level-1 ... coeffs[-1] 对应 cD_1
    for i in range(1, level + 1):
        current_level = level - i + 1
        cD_rec = pywt.upcoef('d', coeffs[i], wavelet, level=current_level, take=len(signal))
        
        ax_idx = i + 1
        fmin, fmax = freq_bands[f'D{current_level}']
        
        axes[ax_idx].plot(times, cD_rec, 'b-', linewidth=0.8)
        axes[ax_idx].set_title(f'Detail D{current_level} ({fmin:.1f}-{fmax:.1f} Hz)', fontsize=10)
        axes[ax_idx].set_ylabel('Amplitude')
    
    # 底部设置 X 轴标签
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 小波分解图已保存：{save_path}")
    plt.close()


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("小波特征提取模块")
    print("\n使用方法:")
    print("from feature_extraction import extract_wavelet_energy_features, normalize_features")
    print("X_wavelet = extract_wavelet_energy_features(epochs, wavelet='db4', level=4)")
    print("X_wavelet_norm, scaler = normalize_features(X_wavelet)")
    print("\n推荐配置:")
    print("  - 小波基：db4 (Daubechies 4)")
    print("  - 分解层数：4 层")
    print("  - 特征类型：能量特征")
    print("  - 归一化：Z-score 标准化")
    
    # 可视化测试代码（实际运行）
    print("\n" + "=" * 80)
    print("运行可视化测试...")
    print("=" * 80)
    
    try:
        print("\n--- 生成测试信号 ---")
        fs = 250
        t = np.linspace(0, 1, fs)
        # 生成包含 5Hz, 15Hz, 40Hz 的混合测试信号
        test_signal = np.sin(2 * np.pi * 5 * t) + \
                      0.5 * np.sin(2 * np.pi * 15 * t) + \
                      0.2 * np.sin(2 * np.pi * 40 * t)
        
        print(f"✅ 测试信号生成完成：{len(test_signal)} 个采样点")
        print(f"   - 包含频率成分：5Hz, 15Hz, 40Hz")
        print(f"   - 采样率：{fs} Hz")
        print(f"   - 时长：1 秒")
        
        print("\n--- 绘制小波分解图 ---")
        plot_wavelet_decomposition(
            test_signal, 
            sfreq=fs, 
            wavelet='db4', 
            level=4, 
            save_path='./test_wavelet.png'
        )
        
        print("\n✅ 测试完成！查看生成的图片：test_wavelet.png")
        print("   图片将显示:")
        print("   - 原始测试信号")
        print("   - A4 近似系数 (0-7.8 Hz)")
        print("   - D4 细节系数 (7.8-15.6 Hz)")
        print("   - D3 细节系数 (15.6-31.2 Hz)")
        print("   - D2 细节系数 (31.2-62.5 Hz)")
        print("   - D1 细节系数 (62.5-125 Hz)")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
