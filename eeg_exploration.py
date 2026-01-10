
import mne
import numpy as np
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文字体
# 注意：如果您的系统没有SimHei，请替换为 'Microsoft YaHei' 或其他支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=== 开始脑电数据深度探索 ===")

    # 1. 加载数据
    data_path = ".\BCICIV_2a_gdf\A01T.gdf"
    print(f"正在加载数据: {data_path}")
    try:
        raw = mne.io.read_raw_gdf(data_path, preload=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        return

    print("数据基本信息:")
    print(raw.info)

    # 2. 通道分类与重命名 (为了匹配标准Montage)
    print("\n=== 通道处理 ===")
    
    # 获取原始通道名
    original_names = raw.info['ch_names']
    print(f"原始通道名 (前5个): {original_names[:5]}")
    
    # 创建重命名映射
    # BCICIV 2a 数据集的标准通道顺序 (22 EEG + 3 EOG)
    # 注意：原始文件中可能除了 Fz 外，其他通道被标记为 EEG-0, EEG-1 等
    bciciv_2a_montage = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    ]
    
    rename_mapping = {}
    # 首先处理已知的 EOG 通道 (保持不变或简化)
    for ch in original_names:
        if 'EOG' in ch:
            continue

    # 处理 EEG 通道
    # 假设前22个通道顺序对应标准列表
    eeg_channels = [ch for ch in original_names if 'EOG' not in ch and ch != 'NaN']
    
    if len(eeg_channels) >= 22:
        print("检测到 BCICIV 2a 格式，正在应用标准通道名称映射...")
        for i, std_name in enumerate(bciciv_2a_montage):
            if i < len(eeg_channels):
                rename_mapping[eeg_channels[i]] = std_name
    else:
        # 回退到简单的字符串处理
        for ch in original_names:
            new_name = ch.replace('EEG-', '').strip()
            if new_name != ch:
                rename_mapping[ch] = new_name

    print(f"计划重命名 {len(rename_mapping)} 个通道...")
    if len(rename_mapping) > 0:
        print(f"示例映射: {list(rename_mapping.items())[:3]}")
    
    try:
        raw.rename_channels(rename_mapping)
        print("通道重命名完成。")
    except Exception as e:
        print(f"警告: 通道重命名失败: {e}")

    # 3. 设置电极位置 (Montage)
    print("设置标准 10-20 电极位置...")
    montage = mne.channels.make_standard_montage('standard_1020')
    try:
        raw.set_montage(montage, on_missing='ignore') # 忽略未匹配的通道警告
        print("Montage 设置完成 (未匹配通道已忽略)。")
    except Exception as e:
        print(f"警告: 设置Montage失败: {e}")

    # 4. 实验事件 (Events) 分析
    print("\n=== 实验事件分析 ===")
    try:
        events, event_id = mne.events_from_annotations(raw)
        print(f"检测到的事件类型 (Event IDs): {event_id}")
        print(f"事件总数: {len(events)}")
        
        # 绘制事件分布图
        fig_events = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], 
                                       first_samp=raw.first_samp, show=False)
        plt.title('实验事件分布 (Events)')
    except Exception as e:
        print(f"无法提取事件 (可能是因为没有Annotations): {e}")

    # 5. 信号可视化
    print("\n=== 信号可视化 ===")
    # 5.1 原始 EEG 信号
    try:
        raw.plot(n_channels=10, duration=10, title='原始 EEG 信号 (前10通道)', 
                 show=False, block=False)
    except Exception as e:
        print(f"无法绘制原始信号: {e}")

    # 5.2 功率谱密度 (PSD)
    print("计算功率谱密度...")
    try:
        spectrum = raw.compute_psd(fmax=60)
        # 手动绘制 PSD 以确保显示
        psd_data, freqs = spectrum.get_data(return_freqs=True)
        # psd_data shape: (n_channels, n_freqs)
        mean_psd = np.mean(psd_data, axis=0)
        
        fig_psd = plt.figure(figsize=(10, 6))
        plt.plot(freqs, 10 * np.log10(mean_psd), color='k', linewidth=1.5)
        plt.title('全通道平均功率谱密度')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.grid(True)
        print("PSD 绘制完成。")
    except Exception as e:
        print(f"无法计算或绘制PSD: {e}")
        spectrum = None

    # 6. 空间拓扑分析 (Topomaps)
    if spectrum is not None:
        print("\n=== 空间拓扑分析 ===")
        try:
            fig_topo, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Alpha 波段
            print("绘制 Alpha 波段 (8-13 Hz) 拓扑图...")
            spectrum.plot_topomap(ch_type='eeg', bands={'Alpha (8-13 Hz)': (8, 13)}, 
                                  axes=axes[0], show=False)
            
            # Beta 波段
            print("绘制 Beta 波段 (13-30 Hz) 拓扑图...")
            spectrum.plot_topomap(ch_type='eeg', bands={'Beta (13-30 Hz)': (13, 30)}, 
                                  axes=axes[1], show=False)
            plt.tight_layout()
        except RuntimeError as e:
            if "digitization points" in str(e):
                print(f"错误: 无法绘制拓扑图 - 缺少电极位置信息。")
                print("可能原因: 通道名称不匹配标准 10-20 系统。")
                print(f"当前通道名: {raw.ch_names[:10]}...")
            else:
                print(f"绘制拓扑图时发生错误: {e}")
            plt.close(fig_topo)
        except Exception as e:
            print(f"绘制拓扑图时发生未知错误: {e}")

    # 7. 通道相关性分析
    print("\n=== 通道相关性分析 ===")
    try:
        # 获取数据 (n_channels, n_times)
        picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)
        if len(picks_eeg) > 0:
            data_eeg = raw.get_data(picks=picks_eeg)
            
            # 计算相关系数矩阵
            corr_matrix = np.corrcoef(data_eeg)
            
            # 绘制热力图
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # 设置标签
            ch_names_eeg = [raw.info['ch_names'][i] for i in picks_eeg]
            ax_corr.set_xticks(np.arange(len(ch_names_eeg)))
            ax_corr.set_yticks(np.arange(len(ch_names_eeg)))
            ax_corr.set_xticklabels(ch_names_eeg, rotation=90)
            ax_corr.set_yticklabels(ch_names_eeg)
            plt.colorbar(im, ax=ax_corr, label='Pearson Correlation')
            plt.title('EEG 通道间相关性矩阵')
            plt.tight_layout()
        else:
            print("未找到EEG通道，跳过相关性分析。")
    except Exception as e:
        print(f"相关性分析失败: {e}")

    print("\n分析脚本执行完毕！")
    print("请查看弹出的所有图形窗口进行详细分析。")
    plt.show()

if __name__ == "__main__":
    main()
