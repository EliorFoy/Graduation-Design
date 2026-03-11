"""
小波特征提取模块

使用离散小波变换 (DWT) 对 EEG 信号进行时频分析
提取各频段能量作为分类特征
"""

import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler


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
    n_trials, n_channels, n_times= data.shape
  print(f"   - 输入数据形状：{data.shape}")
  print(f"   - 试次数：{n_trials}")
  print(f"   - 通道数：{n_channels}")
  print(f"   - 小波基：{wavelet}")
  print(f"   - 分解层数：{level}")
    
    # 计算每个系数的频率范围
    sfreq = epochs.info['sfreq']
  print(f"\n   各层对应频段:")
   freq_bands = compute_wavelet_freq_bands(sfreq, level)
    for i, (band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
      print(f"   - Level {i+1} ({band_name}): {fmin:.1f}-{fmax:.1f} Hz")
    
    features_list = []
    
    for trial_idx in range(n_trials):
        trial_features = []
        
        for ch_idx in range(n_channels):
            signal = data[trial_idx, ch_idx, :]
            
            # 进行多层小波分解
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            # coeffs = [cA_level, cD_level, cD_level-1, ..., cD_1]
            # 例如 level=4: [cA4, D4, D3, D2, D1]
            
            # 计算每层的能量
            for coeff in coeffs:
                energy = np.sum(coeff ** 2)
                trial_features.append(energy)
        
        features_list.append(trial_features)
    
    X_wavelet = np.array(features_list)
    
  print(f"\n✅ 小波能量特征提取完成")
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
    n_trials, n_channels, n_times= data.shape
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
    
  print(f"\n✅ 频段功率特征提取完成")
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
      print(f"✅ 训练集和测试集标准化完成")
        return X_train_norm, X_test_norm, scaler
    
  print(f"✅ 训练集标准化完成")
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


def plot_wavelet_decomposition(signal, sfreq, wavelet='db4', level=4, save_path='./wavelet_decomposition.png'):
    """
   可视化小波分解结果
    
   Args:
        signal: 原始信号
        sfreq: 采样率
        wavelet: 小波基
        level: 分解层数
        save_path: 保存路径
    """
   from matplotlib import pyplot as plt
    
  print(f"\n绘制小波分解图...")
    
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 计算各层对应的频率范围
    freq_bands = compute_wavelet_freq_bands(sfreq, level)
    
    # 创建图形
   fig, axes = plt.subplots(len(coeffs), 1, figsize=(12, 2*len(coeffs)))
    
    if len(coeffs) == 1:
        axes= [axes]
    
    # 绘制原始信号
    times = np.arange(len(signal)) / sfreq
    axes[0].plot(times, signal, 'k-', linewidth=0.5)
    axes[0].set_title('Original Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (μV)')
    
    # 绘制各层细节系数
    band_names = list(freq_bands.keys())
    for i, coeff in enumerate(coeffs[1:], 1):
        # 上采样系数以匹配时间轴
        coeff_upsampled = pywt.upcoef('d', coeff, wavelet, level=i)
        
        # 截取与原始信号相同长度
        if len(coeff_upsampled) > len(signal):
            offset = (len(coeff_upsampled) - len(signal)) // 2
            coeff_upsampled = coeff_upsampled[offset:offset+len(signal)]
        
        times_coeff = np.arange(len(coeff_upsampled)) / sfreq
        axes[i].plot(times_coeff, coeff_upsampled, 'b-', linewidth=0.5)
        
        band_name = band_names[i] if i < len(band_names) else f'D{i}'
        fmin, fmax = freq_bands[band_name]
        axes[i].set_title(f'{band_name} ({fmin:.1f}-{fmax:.1f} Hz)')
        axes[i].set_ylabel('Coeff')
    
    # 隐藏最后一个子图的 x 轴标签
    axes[-1].set_xlabel('Time (s)')
    
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
