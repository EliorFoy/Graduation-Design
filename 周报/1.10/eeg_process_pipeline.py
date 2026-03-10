import mne
import numpy as np
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class EEGProcessor:
    def __init__(self, raw):
        self.raw = raw
        self.raw_filtered = None
        self.raw_clean = None  # ICA处理后的最终数据
        self.eeg_channels = []
        self.eog_channels = []
        self.figs = []  # Store figure references to prevent GC and ensure display
        
    def classify_channels(self):
        """基于通道名称分类EEG和EOG通道"""
        print("\n=== 通道类型分类 ===")
        self.eeg_channels = [ch for ch in self.raw.info['ch_names'] if 'EOG' not in ch]
        self.eog_channels = [ch for ch in self.raw.info['ch_names'] if 'EOG' in ch]
        
        # 设置通道类型
        ch_types = {}
        for ch_name in self.raw.info['ch_names']:
            if 'EOG' in ch_name:
                ch_types[ch_name] = 'eog'
            else:
                ch_types[ch_name] = 'eeg'
        
        self.raw.set_channel_types(ch_types)
        print(f"EEG通道: {self.eeg_channels}")
        print(f"EOG通道: {self.eog_channels}")
        print("通道类型设置完成！")

    def filter_signal(self, l_freq=0.5, h_freq=40):
        """带通滤波"""
        print(f"\n=== 带通滤波 ({l_freq}-{h_freq}Hz) ===")
        # 复制原始数据用于滤波
        self.raw_filtered = self.raw.copy()
        # 应用带通滤波
        self.raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, 
                               fir_design='firwin', skip_by_annotation='edge')
        print("滤波完成！")

    def compare_filter(self, duration=10):
        """对比滤波前后的信号"""
        print("\n=== 对比滤波前后信号 ===")
        # 绘制原始信号
        fig1 = self.raw.plot(duration=duration, n_channels=8, scalings='auto', 
                     title='滤波前 (Original Raw)', block=False)
        self.figs.append(fig1)
        
        # 绘制滤波后信号
        fig2 = self.raw_filtered.plot(duration=duration, n_channels=8, scalings='auto', 
                             title='滤波后 (Filtered)', block=False)
        self.figs.append(fig2)

    def remove_artifacts_ica(self, n_components=20):
        """使用ICA去除伪迹"""
        print("\n=== ICA伪迹去除 ===")
        
        # 确保有滤波后的数据
        if self.raw_filtered is None:
            print("错误：请先执行滤波操作")
            return

        # 同样复制一份进行ICA处理
        self.raw_clean = self.raw_filtered.copy()
        
        # 运行ICA
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        ica.fit(self.raw_clean)
        
        # 1. 自动识别EOG相关成分
        eog_indices, eog_scores = ica.find_bads_eog(self.raw_clean)
        print(f"识别出的EOG伪迹成分: {eog_indices}")
        
        # 2. 识别肌电(EMG)相关成分 (基于高频能量)
        emg_indices = []
        sources = ica.get_sources(self.raw_clean)
        source_data = sources.get_data()
        
        for i in range(ica.n_components_):
            # 使用部分数据计算PSD
            sample_data = source_data[i, :2500]
            psd, freqs = mne.time_frequency.psd_array_multitaper(
                sample_data, sfreq=self.raw_clean.info['sfreq'],
                fmin=30, fmax=50, adaptive=True, n_jobs=1
            )
            if len(psd) > 0 and np.mean(psd) > 1e-10:
                emg_indices.append(i)
        
        print(f"识别出的EMG伪迹成分: {emg_indices}")
        
        # 3. 合并排除
        all_artifact_indices = list(set(eog_indices + emg_indices))
        if all_artifact_indices:
            print(f"剔除总成分: {all_artifact_indices}")
            ica.exclude = all_artifact_indices
            ica.apply(self.raw_clean)
            print("ICA去伪迹完成！")
        else:
            print("未检测到需要剔除的成分")

    def compare_ica(self, duration=10):
        """对比ICA去伪迹前后的信号"""
        print("\n=== 对比ICA去伪迹前后信号 ===")
        # ICA处理前 (即滤波后)
        fig1 = self.raw_filtered.plot(duration=duration, n_channels=8, scalings='auto', 
                             title='ICA处理前 (Filtered)', block=False)
        self.figs.append(fig1)
        
        # ICA处理后
        fig2 = self.raw_clean.plot(duration=duration, n_channels=8, scalings='auto', 
                          title='ICA处理后 (Cleaned)', block=False)
        self.figs.append(fig2)

if __name__ == "__main__":
    # 1. 加载数据
    data_path = ".\BCICIV_2a_gdf\A01T.gdf"
    print(f"正在加载数据: {data_path}")
    try:
        raw = mne.io.read_raw_gdf(data_path, preload=True)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        exit()

    processor = EEGProcessor(raw)
    
    # 2. 运行流程
    processor.classify_channels()
    
    processor.filter_signal()
    processor.compare_filter()  # 生成对比图
    
    processor.remove_artifacts_ica()
    processor.compare_ica()     # 生成对比图
    
    print("\n所有处理完成，正在显示图像...")
    plt.show(block=True)
    input()
