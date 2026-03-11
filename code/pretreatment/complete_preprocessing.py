"""
BCIC IV-2a 数据完整预处理流程
遵循 MNE 官方推荐：轻度滤波 → ICA → 任务滤波
"""

import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from eeg_analysis import get_modified_raw_data  # 导入之前的处理函数

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def filter_for_ica(raw, l_freq=1.0, h_freq=40.0):
    """
    ICA 前的轻度滤波（关键！）
    
    Args:
        raw: mne.io.Raw 对象
        l_freq: 高通截止频率 (默认 1Hz)
        h_freq: 低通截止频率 (默认 40Hz)
    
    Returns:
        filtered_raw: 滤波后的 Raw 对象
    """
    print(f"\n应用轻度滤波：{l_freq}-{h_freq} Hz")
    
    raw_filtered = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        phase='zero',
        verbose=False # 静默模式，不打印任何中间日志
    )
    
    print(f"✅ 轻度滤波完成：{l_freq}-{h_freq} Hz")
    print(f"   - 去除 <{l_freq}Hz 的低频漂移")
    print(f"   - 去除 >{h_freq}Hz 的高频噪声")
    
    return raw_filtered


def fit_ica(raw_ica, n_components=None, method='fastica', max_iter=800):
    """
    ICA 分解
    
    Args:
        raw_ica: 用于 ICA 分解的数据（轻度滤波）
        n_components: 保留的独立成分数
                     - None: 使用所有通道数（推荐，能更好分离伪迹）
                     - int: 固定成分数
                     - float: 保留的方差比例（如 0.99）
        method: ICA 算法 ('fastica', 'picard', 'infomax')
        max_iter: 最大迭代次数
    
    Returns:
        ica: 训练好的 ICA 对象
    """
    print("\n进行 ICA 分解...")
    
    # 默认使用 EEG 通道数，确保能充分分离伪迹成分
    if n_components is None:
        # 只计算 EEG 通道（排除 EOG）
        eeg_picks = mne.pick_types(raw_ica.info, eeg=True, exclude=[])
        n_components = len(eeg_picks)
        print(f"   - 使用 EEG 通道数作为成分数：{n_components}")
    
    # 创建 ICA 对象
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=42,  # 固定随机种子确保可重复性
        max_iter=max_iter
    )
    
    # 拟合 ICA
    ica.fit(raw_ica, verbose=False)
    
    print(f"✅ ICA 分解完成")
    print(f"   - 独立成分数：{ica.n_components_}")
    print(f"   - 算法：{method}")
    print(f"   - 迭代次数：{ica.n_iter_}")
    
    return ica


def detect_and_remove_artifacts(ica, raw_ica, eog_threshold=2.0, ecg_method='correlation'):
    """
    自动识别并去除伪迹成分
    
    Args:
        ica: 训练好的 ICA 对象
        raw_ica: 用于检测的轻度滤波数据
        eog_threshold: EOG 检测阈值（标准差倍数，降低到 2.0 提高灵敏度）
        ecg_method: ECG 检测方法
    
    Returns:
        ica: 更新了 exclude 的 ICA 对象
    """
    print("\n自动识别伪迹成分...")
    
    # 检查 EOG 通道是否存在
    eog_chs = [ch for ch in raw_ica.ch_names if 'EOG' in ch.upper()]
    print(f"   - 检测到 EOG 通道：{eog_chs}")
    
    # 检测眼电成分（降低阈值提高灵敏度）
    eog_indices, eog_scores = ica.find_bads_eog(raw_ica, threshold=eog_threshold)
    
    # 如果没有自动检测到，手动检查 EOG 通道与 ICA 成分的相关性
    if len(eog_indices) == 0 and len(eog_chs) > 0:
        print("   ⚠️  自动检测未找到 EOG 成分，尝试手动相关性分析...")
        from scipy.stats import pearsonr
        
        # 获取 EOG 通道数据
        eog_data = raw_ica.get_data(picks=eog_chs)
        
        # 计算每个 ICA 成分与 EOG 通道的相关性
        eog_components = []
        component_correlations = {}
        
        for comp_idx in range(ica.n_components_):
            # 获取该成分的时间序列
            comp_data = ica.get_sources(raw_ica).get_data(picks=[comp_idx])[0]
            
            # 计算与每个 EOG 通道的相关性
            max_corr_for_comp = 0
            for eog_ch_data in eog_data:
                corr, _ = pearsonr(comp_data, eog_ch_data)
                if abs(corr) > max_corr_for_comp:
                    max_corr_for_comp = abs(corr)
            
            # 记录相关性
            component_correlations[comp_idx] = max_corr_for_comp
            
            # 如果相关性超过阈值，标记为 EOG 成分
            if max_corr_for_comp > 0.3:
                eog_components.append((comp_idx, max_corr_for_comp))
        
        # 按相关性排序，取最相关的几个成分
        eog_components.sort(key=lambda x: x[1], reverse=True)
        
        # 取前几个强相关的成分（通常眨眼 + 水平眼动 = 1-3 个成分）
        if len(eog_components) > 0:
            # 只取相关性>0.3 的成分
            significant_components = [comp for comp, corr in eog_components if corr > 0.3]
            if len(significant_components) > 0:
                eog_indices = significant_components
                print(f"   - 手动检测到 {len(eog_indices)} 个 EOG 相关成分：{eog_indices}")
                for comp_idx, corr in eog_components[:3]:  # 显示前 3 个
                    print(f"     * 成分 {comp_idx}: 相关性 = {corr:.3f}")
            else:
                print(f"   - 未检测到明显的 EOG 成分（最高相关性：{eog_components[0][1]:.3f}）")
        else:
            max_corr = max(component_correlations.values()) if component_correlations else 0
            print(f"   - 未检测到明显的 EOG 成分（最大相关性：{max_corr:.3f}）")
    
    # 检测心电成分（可选）
    try:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_ica, method=ecg_method)
    except:
        ecg_indices = []
        print("   ⚠️  未检测到心电成分")
    
    # 合并要剔除的成分
    ica.exclude = list(set(eog_indices + ecg_indices))
    
    print(f"✅ 伪迹成分识别完成")
    print(f"   - 眼电成分：{eog_indices}")
    print(f"   - 心电成分：{ecg_indices}")
    print(f"   - 剔除成分总数：{len(ica.exclude)}")
    
    # 可视化检测到的 EOG 成分
    if len(eog_indices) > 0:
        try:
            ica.plot_components(picks=eog_indices, show=False)
            plt.suptitle('检测到的 EOG 成分')
            plt.savefig('./ica_eog_components.png', dpi=300, bbox_inches='tight')
            print("   📊 EOG 成分图已保存：ica_eog_components.png")
        except RuntimeError as e:
            print(f"   ⚠️  无法绘制 ICA 成分图：{e}")
            print("   （这不影响 ICA 去噪效果，只是可视化问题）")
    
    return ica


def apply_ica(ica, raw_original):
    """
    应用 ICA 去噪
    
    Args:
        ica: 训练好的 ICA 对象
        raw_original: 原始数据（用于应用 ICA）
    
    Returns:
        raw_clean: 去噪后的 Raw 对象
    """
    print("\n应用 ICA 去噪...")
    
    # 在原始数据上应用 ICA（保留完整频段）
    raw_clean = ica.apply(raw_original.copy())
    
    print(f"✅ ICA 去噪完成")
    print(f"   - 剔除了 {len(ica.exclude)} 个伪迹成分")
    
    return raw_clean


def filter_for_task(raw_clean, l_freq=8.0, h_freq=30.0):
    """
    ICA 后的任务定制滤波
    
    Args:
        raw_clean: ICA 去噪后的数据
        l_freq: 高通截止频率 (默认 8Hz，μ节律起始)
        h_freq: 低通截止频率 (默认 30Hz，β节律结束)
    
    Returns:
        raw_final: 最终滤波后的 Raw 对象
    """
    print(f"\n应用任务定制滤波：{l_freq}-{h_freq} Hz")
    
    raw_final = raw_clean.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        phase='zero',
        verbose=False
    )
    
    print(f"✅ 任务滤波完成：{l_freq}-{h_freq} Hz")
    print(f"   - 保留 μ节律 (8-13 Hz)")
    print(f"   - 保留 β节律 (13-30 Hz)")
    
    return raw_final


def set_reference(raw_final, ref='average'):
    """
    重参考
    
    Args:
        raw_final: 最终滤波后的数据
        ref: 参考方式，默认为 'average'（平均参考）
    
    Returns:
        raw_final: 重参考后的 Raw 对象
    """
    print(f"\n应用 {ref} 参考...")
    
    raw_final.set_eeg_reference(ref, projection=False)
    
    print(f"✅ 重参考完成：{ref}")
    
    return raw_final


def create_epochs(raw_final, event_id=None, tmin=0, tmax=4, baseline=None):
    """
    分段
    
    Args:
        raw_final: 最终处理后的数据
        event_id: 事件 ID 字典
        tmin: 起始时间（相对于事件）
        tmax: 结束时间
        baseline: 基线校正（None 表示不做）
    
    Returns:
        epochs: 分段后的 Epochs 对象
    """
    print("\n创建分段...")
    
    # 默认事件 ID（BCIC IV-2a）
    if event_id is None:
        event_id = {
            'left': 769,
            'right': 770,
            'feet': 771,
            'tongue': 772
        }
    
    # 从 annotations 中提取事件（BCIC IV-2a 数据集使用 annotations 存储事件）
    events, event_dict = mne.events_from_annotations(raw_final)
    
    print(f"   - 从 annotations 中提取到 {len(events)} 个事件")
    print(f"   - 事件类型：{list(event_dict.keys())}")
    
    # 创建 Epochs
    epochs = mne.Epochs(
        raw_final, 
        events, 
        event_id=event_dict,  # 使用从 annotations 提取的事件 ID
        tmin=tmin, 
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
        event_repeated='drop'  # 处理重复事件
    )
    
    print(f"✅ 分段完成")
    print(f"   - 事件总数：{len(events)}")
    print(f"   - Epochs 数：{len(epochs)}")
    print(f"   - 时间窗口：[{tmin}, {tmax}] s")
    
    return epochs


def drop_artifact_epochs(epochs, events, artifact_codes=[1023]):
    """
    剔除官方标记的伪迹试次
    
    Args:
        epochs: 分段后的数据
        events: 事件数组
        artifact_codes: 伪迹事件代码列表，默认 [1023]
    
    Returns:
        epochs: 剔除伪迹后的 Epochs 对象
    """
    print("\n剔除伪迹试次...")
    
    total_artifacts = 0
    for artifact_code in artifact_codes:
        # 找到伪迹事件
        artifact_mask = events[:, 2] == artifact_code
        n_artifacts = np.sum(artifact_mask)
        
        if n_artifacts > 0:
            # 剔除包含伪迹的 epochs
            epochs.drop_events(artifact_mask)
            print(f"   - 剔除了 {n_artifacts} 个事件代码为 {artifact_code} 的试次")
            total_artifacts += n_artifacts
    
    if total_artifacts > 0:
        print(f"✅ 总共剔除了 {total_artifacts} 个伪迹试次")
        print(f"   - 剩余 Epochs 数：{len(epochs)}")
    else:
        print(f"ℹ️  未检测到伪迹试次标记")
    
    return epochs


def plot_preprocessing_comparison(raw_original, raw_ica_filtered, raw_clean, raw_final, save_path='./preprocessing_comparison.png'):
    """
    可视化预处理前后对比
    
    Args:
        raw_original: 原始数据
        raw_ica_filtered: ICA 前滤波数据
        raw_clean: ICA 去噪后数据
        raw_final: 最终滤波数据
        save_path: 保存路径
    """
    print("\n可视化预处理效果...")
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    # 1. 原始数据 PSD
    psd_orig = raw_original.compute_psd(fmin=0, fmax=50)
    psd_data_orig, freqs_orig = psd_orig.get_data(return_freqs=True)
    axes[0, 0].plot(freqs_orig, np.mean(psd_data_orig, axis=0), linewidth=1)
    axes[0, 0].set_title('原始数据 PSD')
    axes[0, 0].set_xlim(0, 50)
    axes[0, 0].set_xlabel('频率 (Hz)')
    axes[0, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ICA 前滤波 PSD
    psd_ica = raw_ica_filtered.compute_psd(fmin=0, fmax=50)
    psd_data_ica, freqs_ica = psd_ica.get_data(return_freqs=True)
    axes[1, 0].plot(freqs_ica, np.mean(psd_data_ica, axis=0), linewidth=1, color='orange')
    axes[1, 0].set_title('ICA 前滤波 PSD (1-40Hz)')
    axes[1, 0].set_xlim(0, 50)
    axes[1, 0].set_xlabel('频率 (Hz)')
    axes[1, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. ICA 去噪后 PSD
    psd_clean = raw_clean.compute_psd(fmin=0, fmax=50)
    psd_data_clean, freqs_clean = psd_clean.get_data(return_freqs=True)
    axes[2, 0].plot(freqs_clean, np.mean(psd_data_clean, axis=0), linewidth=1, color='green')
    axes[2, 0].set_title('ICA 去噪后 PSD')
    axes[2, 0].set_xlim(0, 50)
    axes[2, 0].set_xlabel('频率 (Hz)')
    axes[2, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 4. 最终滤波 PSD
    psd_final = raw_final.compute_psd(fmin=0, fmax=50)
    psd_data_final, freqs_final = psd_final.get_data(return_freqs=True)
    axes[3, 0].plot(freqs_final, np.mean(psd_data_final, axis=0), linewidth=1, color='red')
    axes[3, 0].set_title('最终滤波 PSD (8-30Hz)')
    axes[3, 0].set_xlim(0, 50)
    axes[3, 0].set_xlabel('频率 (Hz)')
    axes[3, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[3, 0].grid(True, alpha=0.3)
    
    # 5. 时域信号对比
    duration = 2  # 显示 2 秒
    start_time = 10  # 从 10 秒开始
    start_sample = int(raw_original.info['sfreq'] * start_time)
    end_sample = start_sample + int(duration * raw_original.info['sfreq'])
    
    times = np.linspace(start_time, start_time + duration, end_sample - start_sample) * 1000  # ms
    
    axes[0, 1].plot(times, raw_original.get_data()[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5)
    axes[0, 1].set_title('原始信号 (通道 1)')
    axes[0, 1].set_xlabel('时间 (ms)')
    axes[0, 1].set_ylabel('幅度 (μV)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(times, raw_ica_filtered.get_data()[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5, color='orange')
    axes[1, 1].set_title('ICA 前滤波信号 (通道 1)')
    axes[1, 1].set_xlabel('时间 (ms)')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(times, raw_clean.get_data()[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5, color='green')
    axes[2, 1].set_title('ICA 去噪后信号 (通道 1)')
    axes[2, 1].set_xlabel('时间 (ms)')
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[3, 1].plot(times, raw_final.get_data()[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5, color='red')
    axes[3, 1].set_title('最终滤波信号 (8-30Hz, 通道 1)')
    axes[3, 1].set_xlabel('时间 (ms)')
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存：{save_path}")
    
    return fig


def complete_preprocessing_pipeline():
    """
    完整预处理流程
    
    Returns:
        epochs_final: 预处理后的 Epochs 对象
        ica: 训练好的 ICA 对象
    """
    print("=" * 60)
    print("🚀 运行完整预处理流程")
    print("遵循 MNE 官方推荐：轻度滤波 → ICA → 任务滤波")
    print("=" * 60)
    
    # 1. 获取已处理的原始数据（通道映射已完成）
    print("Step 1: 获取已映射通道的原始数据")
    raw = get_modified_raw_data()
    print(f"✅ 原始数据加载完成，通道数：{len(raw.ch_names)}")
    
    # 2. ICA 前的轻度滤波（保留足够频段供 ICA 识别伪迹）
    print("\n" + "=" * 40)
    print("Step 2: ICA 前轻度滤波")
    print("=" * 40)
    raw_ica_filtered = filter_for_ica(raw, l_freq=1.0, h_freq=40.0)
    
    # 3. ICA 分解
    print("\n" + "=" * 40)
    print("Step 3: ICA 分解")
    print("=" * 40)
    ica = fit_ica(raw_ica_filtered)
    
    # 4. 检测并去除伪迹
    print("\n" + "=" * 40)
    print("Step 4: 检测并去除伪迹")
    print("=" * 40)
    ica = detect_and_remove_artifacts(ica, raw_ica_filtered)
    
    # 5. 应用 ICA 去噪
    print("\n" + "=" * 40)
    print("Step 5: 应用 ICA 去噪")
    print("=" * 40)
    raw_clean = apply_ica(ica, raw)
    
    # 6. 任务定制滤波（聚焦运动想象相关频段）
    print("\n" + "=" * 40)
    print("Step 6: 任务定制滤波")
    print("=" * 40)
    raw_final = filter_for_task(raw_clean, l_freq=8.0, h_freq=30.0)
    
    # 7. 重参考
    print("\n" + "=" * 40)
    print("Step 7: 重参考")
    print("=" * 40)
    raw_final = set_reference(raw_final, ref='average')
    
    # 8. 分段
    print("\n" + "=" * 40)
    print("Step 8: 分段")
    print("=" * 40)
    epochs = create_epochs(raw_final, tmin=0, tmax=4)
    
    # 9. 剔除伪迹试次
    print("\n" + "=" * 40)
    print("Step 9: 剔除伪迹试次")
    print("=" * 40)
    events, _ = mne.events_from_annotations(raw_final)
    epochs_final = drop_artifact_epochs(epochs, events)
    
    # 10. 可视化对比
    print("\n" + "=" * 40)
    print("Step 10: 可视化预处理效果")
    print("=" * 40)
    plot_preprocessing_comparison(raw, raw_ica_filtered, raw_clean, raw_final)
    
    print("\n" + "=" * 60)
    print("🎉 完整预处理流程完成！")
    print("=" * 60)
    print(f"📊 最终 Epochs 信息：")
    print(f"   - 试次数：{len(epochs_final)}")
    print(f"   - 通道数：{len(epochs_final.ch_names)}")
    print(f"   - 时间点数：{epochs_final.get_data().shape[2]}")
    print(f"   - 频段：8-30 Hz (运动想象相关)")
    
    return epochs_final, ica


def save_processed_data(epochs, output_path='./A01T_preprocessed-epo.fif'):
    """
    保存预处理后的数据
    
    Args:
        epochs: 预处理后的 Epochs 对象
        output_path: 输出路径
    """
    print(f"\n保存预处理结果到：{output_path}")
    
    epochs.save(output_path, overwrite=True)
    print(f"✅ 预处理后的 Epochs 已保存：{output_path}")


if __name__ == "__main__":
    # 运行完整预处理流程
    epochs_final, ica_model = complete_preprocessing_pipeline()
    
    # 保存结果
    save_processed_data(epochs_final, output_path='./A01T_preprocessed-epo.fif')
    
    # 显示一些统计信息
    print(f"\n📈 预处理统计：")
    print(f"   - 剔除的 ICA 成分数：{len(ica_model.exclude)}")
    print(f"   - 保留的独立成分：{len(ica_model.mixing_matrix_)}")
    print(f"   - 最终试次数：{len(epochs_final)}")
    
    # 显示预处理效果
    plt.show()