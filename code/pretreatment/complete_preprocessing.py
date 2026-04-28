"""
BCIC IV-2a 数据完整预处理流程
遵循 MNE 官方推荐：轻度滤波 → ICA → 任务滤波
"""

import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .eeg_analysis import get_modified_raw_data  # 导入之前的处理函数
try:
    from code.config import DEFAULT_CONFIG, EEGPipelineConfig
except ModuleNotFoundError:
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from code.config import DEFAULT_CONFIG, EEGPipelineConfig

# 设置中文字体（兼容 Linux/Windows/macOS）
import platform
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
                     - None: 使用方差保留策略（推荐）
                     - int: 固定成分数
                     - float: 保留的方差比例（如 0.99）
        method: ICA 算法 ('fastica', 'picard', 'infomax')
        max_iter: 最大迭代次数
    
    Returns:
        ica: 训练好的 ICA 对象
    """
    print("\n进行 ICA 分解...")
    
    # 默认使用方差保留策略，确保能充分分离伪迹成分
    if n_components is None:
        n_components = 0.99  # 保留 99% 方差的成分
        print(f"   - 将使用方差保留策略 (n_components=0.99) 自动选择成分数")
    
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


def detect_and_remove_artifacts(ica, raw_ica, eog_threshold=2.0, ecg_method='correlation', eog_corr_threshold=0.3, save_path='./output_img/ica_eog_components.png'):
    """
    自动识别并去除伪迹成分
    
    Args:
        ica: 训练好的 ICA 对象
        raw_ica: 用于检测的轻度滤波数据
        eog_threshold: EOG 检测阈值（标准差倍数，降低到 2.0 提高灵敏度）
        ecg_method: ECG 检测方法
        eog_corr_threshold: EOG 手动相关性检测阈值（默认 0.3）
        save_path: EOG 成分图保存路径
    
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
            if max_corr_for_comp > eog_corr_threshold:
                eog_components.append((comp_idx, max_corr_for_comp))
        
        # 按相关性排序，取最相关的几个成分
        eog_components.sort(key=lambda x: x[1], reverse=True)
        
        # 取前几个强相关的成分（通常眨眼 + 水平眼动 = 1-3 个成分）
        if len(eog_components) > 0:
            # 只取相关性>eog_corr_threshold 的成分
            significant_components = [comp for comp, corr in eog_components if corr > eog_corr_threshold]
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
            # 确保输出目录存在
            output_dir = Path(save_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            ica.plot_components(picks=eog_indices, show=False)
            plt.suptitle('检测到的 EOG 成分')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 EOG 成分图已保存：{save_path}")
        except RuntimeError as e:
            print(f"   ⚠️  无法绘制 ICA 成分图：{e}")
            print("   （这不影响 ICA 去噪效果，只是可视化问题）")
    
    return ica


def apply_ica(ica, raw_filtered_for_ica):
    """
    应用 ICA 去噪
    
    Args:
        ica: 训练好的 ICA 对象
        raw_filtered_for_ica: 用于拟合 ICA 的滤波后数据（必须与拟合时相同）
    
    Returns:
        raw_clean: 去噪后的 Raw 对象
    """
    print("\n应用 ICA 去噪...")
    
    # 【关键修复】在相同的滤波数据上应用 ICA，确保一致性
    raw_clean = ica.apply(raw_filtered_for_ica.copy())
    
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


def create_epochs_with_artifact_removal_mne(raw_final, tmin=0, tmax=4):
    """
    使用 MNE 创建 epochs 并剔除伪迹试次
    不依赖 BioSig，通过 trial 时间窗口匹配 1023 事件
    
    Args:
        raw_final: 预处理后的 Raw 对象
        tmin, tmax: epoch 时间窗口
    
    Returns:
        epochs: 剔除伪迹后的 Epochs 对象
    """
    print("\n创建分段（含伪迹剔除）...")
    
    # 1. 从 annotations 提取所有事件
    events, event_dict = mne.events_from_annotations(raw_final)
    
    print(f"   📊 事件字典：{event_dict}")
    
    # 2. 检查是否有 1023 事件
    event_1023_id = event_dict.get('1023', None)
    has_artifact_events = event_1023_id is not None
    
    if has_artifact_events:
        artifact_mask = events[:, 2] == event_1023_id
        artifact_events = events[artifact_mask]
        print(f"   - 检测到 {len(artifact_events)} 个 1023 伪迹事件")
    else:
        print(f"   ⚠️  MNE 未读取到 1023 事件")
        artifact_events = np.array([])
    
    # 3. 找到 trial start (768) 事件或 cue 事件
    event_768_id = event_dict.get('768', None)
    
    # 获取所有 cue 事件 (769-772)
    cue_ids = [event_dict.get(str(k), None) for k in [769, 770, 771, 772]]
    cue_ids = [x for x in cue_ids if x is not None]
    cue_mask_all = np.isin(events[:, 2], cue_ids)
    cue_events = events[cue_mask_all]
    cue_events = cue_events[np.argsort(cue_events[:, 0])]
    
    if event_768_id is not None:
        # 有 768 事件，使用 768 作为 trial 起始
        trial_start_mask = events[:, 2] == event_768_id
        trial_start_events = events[trial_start_mask]
        trial_start_events = trial_start_events[np.argsort(trial_start_events[:, 0])]
        print(f"   - 检测到 {len(trial_start_events)} 个 trial start (768)")
        print(f"   - 检测到 {len(cue_events)} 个 cue 事件 (769-772)")
        
        # 验证数量一致性
        if len(trial_start_events) != len(cue_events):
            print(f"   ⚠️  警告：768 事件数 ({len(trial_start_events)}) 与 cue 事件数 ({len(cue_events)}) 不一致")
            print(f"      将使用较短的数量进行对齐")
            min_len = min(len(trial_start_events), len(cue_events))
            trial_start_events = trial_start_events[:min_len]
            cue_events = cue_events[:min_len]
    else:
        # fallback：直接使用 cue 事件作为 trial 起始
        print(f"   ⚠️  MNE 未读取到 768 事件，使用 cue 事件作为 trial 起始")
        trial_start_events = cue_events.copy()
        print(f"   - 使用 {len(trial_start_events)} 个 cue 事件")
    
    n_trials = len(trial_start_events)
    print(f"   📊 总 trial 数：{n_trials}")
    
    if n_trials == 0:
        raise ValueError("没有有效的 trial 事件，无法创建 Epochs。请检查数据格式。")
    
    # 4. 标记哪些 trial 包含伪迹
    keep_mask = np.ones(n_trials, dtype=bool)
    sfreq = raw_final.info['sfreq']
    
    if len(artifact_events) > 0:
        for i, trial_event in enumerate(trial_start_events):
            trial_start_time = trial_event[0]
            
            # 【关键修复】根据事件类型确定正确的伪迹检测窗口
            event_id = trial_event[2]
            
            # 检查是否是 768 事件（trial start）
            if event_768_id is not None and event_id == event_768_id:
                # 768 是 trial 起始 (t=0)，trial 范围是 [0, 6] 秒
                trial_window_start = trial_start_time
                trial_window_end = trial_start_time + 6 * sfreq
            else:
                # cue 事件 (769-772)，cue 在 t=2s 出现，trial 范围是 [-2, 4] 相对于 cue
                # 即 [cue-2s, cue+4s]
                trial_window_start = trial_start_time - 2 * sfreq
                trial_window_end = trial_start_time + 4 * sfreq
            
            # 检查是否有 1023 在这个 trial 窗口内
            for artifact_event in artifact_events:
                artifact_time = artifact_event[0]
                if trial_window_start <= artifact_time <= trial_window_end:
                    keep_mask[i] = False
                    print(f"      - Trial {i} 包含伪迹 (1023 时间：{artifact_time/sfreq:.1f}s)")
                    break
        
        n_removed = np.sum(~keep_mask)
        print(f"   ✅ 标记剔除 {n_removed} 个伪迹 trial")
    else:
        print(f"   ℹ️  无 1023 事件，保留所有 trial")
    
    # 5. 【关键修复】直接使用 keep_mask 过滤 cue 事件，避免事件匹配错位
    clean_cue_events = cue_events[keep_mask]
    final_events = clean_cue_events
    
    # 【关键检查】确保有有效事件
    if len(final_events) == 0:
        raise ValueError("没有有效的 trial 事件，无法创建 Epochs。请检查数据或伪迹剔除是否过于严格。")
    
    print(f"   ✅ 最终有效 cue 事件：{len(final_events)}")
    
    # 6. 创建 epochs
    event_id_final = {k: event_dict[k] for k in ['769', '770', '771', '772'] if k in event_dict}
    
    # 【关键检查】确保 event_id_final 不为空
    if not event_id_final:
        raise ValueError(f"未找到任何任务事件 (769-772)。可用事件：{list(event_dict.keys())}")
    
    print(f"   📋 使用的事件 ID 映射：{event_id_final}")
    
    epochs = mne.Epochs(
        raw_final, 
        final_events, 
        event_id=event_id_final,
        tmin=tmin, 
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
        event_repeated='drop'
    )
    
    # 【关键修复】只保留 EEG 通道，剔除 EOG 通道，防止模型"偷看"眼动信息
    epochs.pick_types(eeg=True, exclude=[])
    
    print(f"✅ 分段完成：{len(epochs)} 个 epochs")
    print(f"   - 通道类型：仅 EEG ({len(epochs.ch_names)} 个通道)")
    
    return epochs




def plot_preprocessing_comparison(raw_original, raw_ica_filtered, raw_clean, raw_final, save_path='./output_img/preprocessing_comparison.png'):
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
    
    # 确保输出目录存在
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    # 1. 原始数据 PSD（仅 EEG 通道，避免 EOG 低频能量扭曲频谱）
    psd_orig = raw_original.compute_psd(fmin=0, fmax=50, picks='eeg')
    psd_data_orig, freqs_orig = psd_orig.get_data(return_freqs=True)
    axes[0, 0].plot(freqs_orig, np.mean(psd_data_orig, axis=0), linewidth=1)
    axes[0, 0].set_title('原始数据 PSD (EEG)')
    axes[0, 0].set_xlim(0, 50)
    axes[0, 0].set_xlabel('频率 (Hz)')
    axes[0, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ICA 前滤波 PSD
    psd_ica = raw_ica_filtered.compute_psd(fmin=0, fmax=50, picks='eeg')
    psd_data_ica, freqs_ica = psd_ica.get_data(return_freqs=True)
    axes[1, 0].plot(freqs_ica, np.mean(psd_data_ica, axis=0), linewidth=1, color='orange')
    axes[1, 0].set_title('ICA 前滤波 PSD (1-40Hz, EEG)')
    axes[1, 0].set_xlim(0, 50)
    axes[1, 0].set_xlabel('频率 (Hz)')
    axes[1, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. ICA 去噪后 PSD
    psd_clean = raw_clean.compute_psd(fmin=0, fmax=50, picks='eeg')
    psd_data_clean, freqs_clean = psd_clean.get_data(return_freqs=True)
    axes[2, 0].plot(freqs_clean, np.mean(psd_data_clean, axis=0), linewidth=1, color='green')
    axes[2, 0].set_title('ICA 去噪后 PSD (EEG)')
    axes[2, 0].set_xlim(0, 50)
    axes[2, 0].set_xlabel('频率 (Hz)')
    axes[2, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 4. 最终滤波 PSD
    psd_final = raw_final.compute_psd(fmin=0, fmax=50, picks='eeg')
    psd_data_final, freqs_final = psd_final.get_data(return_freqs=True)
    axes[3, 0].plot(freqs_final, np.mean(psd_data_final, axis=0), linewidth=1, color='red')
    axes[3, 0].set_title('最终滤波 PSD (8-30Hz, EEG)')
    axes[3, 0].set_xlim(0, 50)
    axes[3, 0].set_xlabel('频率 (Hz)')
    axes[3, 0].set_ylabel('功率谱密度 (μV²/Hz)')
    axes[3, 0].grid(True, alpha=0.3)
    
    # 5. 时域信号对比（自适应选择时间段）
    duration = 2  # 显示 2 秒
    total_duration = raw_original.times[-1]
    start_time = max(10, total_duration / 3)  # 至少从 10s 开始，或在数据的 1/3 处
    start_time = min(start_time, total_duration - duration)  # 确保不会超出数据范围
    
    start_sample = int(raw_original.info['sfreq'] * start_time)
    end_sample = start_sample + int(duration * raw_original.info['sfreq'])
    
    times = np.linspace(start_time, start_time + duration, end_sample - start_sample) * 1000  # ms
    
    # 【关键修复】确保提取的是 EEG 通道，避免索引 0 可能是 EOG
    eeg_data_orig = raw_original.copy().pick_types(eeg=True).get_data()
    eeg_data_ica = raw_ica_filtered.copy().pick_types(eeg=True).get_data()
    eeg_data_clean = raw_clean.copy().pick_types(eeg=True).get_data()
    eeg_data_final = raw_final.copy().pick_types(eeg=True).get_data()
    
    axes[0, 1].plot(times, eeg_data_orig[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5)
    axes[0, 1].set_title('原始信号 (通道 1)')
    axes[0, 1].set_xlabel('时间 (ms)')
    axes[0, 1].set_ylabel('幅度 (μV)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(times, eeg_data_ica[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5, color='orange')
    axes[1, 1].set_title('ICA 前滤波信号 (通道 1)')
    axes[1, 1].set_xlabel('时间 (ms)')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 1].plot(times, eeg_data_clean[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5, color='green')
    axes[2, 1].set_title('ICA 去噪后信号 (通道 1)')
    axes[2, 1].set_xlabel('时间 (ms)')
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[3, 1].plot(times, eeg_data_final[0, start_sample:end_sample] * 1e6, 
                     alpha=0.7, linewidth=0.5, color='red')
    axes[3, 1].set_title('最终滤波信号 (8-30Hz, 通道 1)')
    axes[3, 1].set_xlabel('时间 (ms)')
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存：{save_path}")
    
    return fig


def complete_preprocessing_pipeline(
    subject='A01T',
    data_root=None,
    raw=None,
    config: EEGPipelineConfig = DEFAULT_CONFIG,
    plot_comparison=True,
):
    """
    完整预处理流程
    
    Args:
        subject: 被试 ID，如 'A01T'
        data_root: GDF 数据根目录，默认为项目下 BCICIV_2a_gdf
        raw: 已加载 Raw 对象；提供时不再从磁盘读取
        config: 统一配置对象
        plot_comparison: 是否绘制预处理对比图
    
    Returns:
        epochs_final: 预处理后的 Epochs 对象
        ica: 训练好的 ICA 对象
    """
    print("=" * 60)
    print("运行完整预处理流程")
    print("遵循 MNE 官方推荐：轻度滤波 → ICA → 任务滤波")
    print("=" * 60)
    
    # 1. 获取已处理的原始数据（通道映射已完成）
    print("Step 1: 获取已映射通道的原始数据")
    raw = get_modified_raw_data(subject=subject, data_root=data_root, raw=raw)
    print(f"✅ 原始数据加载完成，通道数：{len(raw.ch_names)}")
    
    # 🔧 关键新增：无需 BioSig，直接使用 MNE 方法
    print("\n" + "=" * 40)
    print("Step 1.5: 准备伪迹剔除信息")
    print("=" * 40)
    print("   使用 MNE 方法通过 1023 事件匹配剔除伪迹试次")
    
    # 2. ICA 前的轻度滤波（保留足够频段供 ICA 识别伪迹）
    print("\n" + "=" * 40)
    print("Step 2: ICA 前轻度滤波")
    print("=" * 40)
    raw_ica_filtered = filter_for_ica(raw, l_freq=config.ica_l_freq, h_freq=config.ica_h_freq)
    
    # 3. ICA 分解
    print("\n" + "=" * 40)
    print("Step 3: ICA 分解")
    print("=" * 40)
    ica = fit_ica(raw_ica_filtered)
    
    # 4. 检测并去除伪迹
    print("\n" + "=" * 40)
    print("Step 4: 检测并去除伪迹")
    print("=" * 40)
    ica = detect_and_remove_artifacts(ica, raw_ica_filtered, save_path='./output_img/ica_eog_components.png')
    
    # 5. 应用 ICA 去噪（在相同的滤波数据上应用）
    print("\n" + "=" * 40)
    print("Step 5: 应用 ICA 去噪")
    print("=" * 40)
    raw_clean = apply_ica(ica, raw_ica_filtered)
    
    # 6. 任务定制滤波（聚焦运动想象相关频段）
    print("\n" + "=" * 40)
    print("Step 6: 任务定制滤波")
    print("=" * 40)
    raw_final = filter_for_task(raw_clean, l_freq=config.task_l_freq, h_freq=config.task_h_freq)
    
    # 7. 重参考
    print("\n" + "=" * 40)
    print("Step 7: 重参考")
    print("=" * 40)
    raw_final = set_reference(raw_final, ref='average')
    
    # 8. 分段 + 伪迹剔除（使用 MNE 方法）
    print("\n" + "=" * 40)
    print("Step 8: 分段 + 伪迹剔除")
    print("=" * 40)
    
    # 使用 MNE 方法创建 epochs 并剔除伪迹
    epochs_final = create_epochs_with_artifact_removal_mne(
        raw_final,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
    )
    
    # 10. 可视化对比
    print("\n" + "=" * 40)
    print("Step 10: 可视化预处理效果")
    print("=" * 40)
    if plot_comparison:
        plot_preprocessing_comparison(raw, raw_ica_filtered, raw_clean, raw_final)
    
    print("\n" + "=" * 60)
    print("🎉 完整预处理流程完成！")
    print("=" * 60)
    print(f"📊 最终 Epochs 信息：")
    print(f"   - 试次数：{len(epochs_final)}")
    print(f"   - 通道数：{len(epochs_final.ch_names)}")
    print(f"   - 时间点数：{epochs_final.get_data().shape[2]}")
    print(f"   - 频段：{config.task_l_freq:g}-{config.task_h_freq:g} Hz (运动想象相关)")
    
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
