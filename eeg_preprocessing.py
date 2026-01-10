import mne
import numpy as np
import matplotlib.pyplot as plt

# 设置Matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class EEGPreprocessor:
    """脑电信号预处理类"""
    
    def __init__(self, raw):
        """初始化预处理器
        
        Args:
            raw: mne.io.Raw对象
        """
        self.raw = raw
        self.raw_clean = None
        self.eeg_channels = []
        self.eog_channels = []
    
    def classify_channels(self):
        """基于通道名称分类EEG和EOG通道"""
        print("\n=== 通道类型分类 ===")
        self.eeg_channels = [ch for ch in self.raw.info['ch_names'] if 'EOG' not in ch]
        self.eog_channels = [ch for ch in self.raw.info['ch_names'] if 'EOG' in ch]
        print(f"EEG通道数: {len(self.eeg_channels)}")
        print(f"EOG通道数: {len(self.eog_channels)}")
        print(f"EEG通道: {self.eeg_channels}")
        print(f"EOG通道: {self.eog_channels}")
        
        # 设置通道类型
        print("\n=== 设置通道类型 ===")
        ch_types = {}
        for ch_name in self.raw.info['ch_names']:
            if 'EOG' in ch_name:
                ch_types[ch_name] = 'eog'
            else:
                ch_types[ch_name] = 'eeg'
        
        # 应用通道类型设置
        self.raw.set_channel_types(ch_types)
        print("通道类型设置完成！")
    
    def filter_signal(self, l_freq=0.5, h_freq=40):
        """带通滤波
        
        Args:
            l_freq: 低通频率
            h_freq: 高通频率
        """
        print(f"\n=== 带通滤波 ({l_freq}-{h_freq}Hz) ===")
        # 复制原始数据以避免修改
        self.raw_clean = self.raw.copy()
        # 应用带通滤波，使用FIR设计，避免边缘伪影
        self.raw_clean = self.raw_clean.filter(l_freq=l_freq, h_freq=h_freq, 
                                              fir_design='firwin', skip_by_annotation='edge')
        print("滤波完成！")
    
    def detect_eog_artifacts(self):
        """基于EOG通道检测伪迹
        
        Note:
            此函数仅用于评估数据质量和可视化伪迹分布（统计伪迹样本占比）。
            其检测结果不会传递给后续的 ICA 去伪迹步骤。
            ICA 步骤会使用独立的相关性算法自动识别眼电成分。
        """
        print("\n=== 伪迹检测 ===")
        # 滤波后提取EOG数据
        raw_eog = self.raw_clean.copy().pick(picks=self.eog_channels)
        eog_data = raw_eog.get_data()
        
        # 设置阈值（3倍标准差）
        threshold = 3 * np.std(eog_data)
        eog_artifacts = np.any(np.abs(eog_data) > threshold, axis=0)
        artifact_ratio = np.sum(eog_artifacts) / len(eog_artifacts)
        print(f"EOG伪迹样本占比: {artifact_ratio:.2%}")
        
        # 可视化伪迹分布
        print("绘制伪迹时间分布...")
        time_axis = np.arange(len(eog_artifacts)) / self.raw_clean.info['sfreq']
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, eog_artifacts.astype(int), 'r-', alpha=0.6)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('伪迹检测')
        ax.set_title('EOG伪迹时间分布')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['无伪迹', '有伪迹'])
        ax.grid(True, alpha=0.3)
        
        return eog_artifacts
    
    def detect_emg_artifacts(self):
        """检测肌电伪迹
        
        Note:
            与 detect_eog_artifacts 类似，此函数仅用于评估数据质量（统计肌电伪迹样本占比）。
            其检测结果（emg_artifacts）不参与后续的 ICA 去伪迹。
        """
        print("\n=== 肌电伪迹检测 ===")
        
        # 提取EEG数据
        eeg_data = self.raw_clean.get_data()
        
        # 1. 时域检测：基于标准差的阈值法
        threshold = 4 * np.std(eeg_data)  # 肌电伪迹阈值可适当提高
        emg_artifacts = np.any(np.abs(eeg_data) > threshold, axis=0)
        artifact_ratio = np.sum(emg_artifacts) / len(emg_artifacts)
        print(f"肌电伪迹样本占比: {artifact_ratio:.2%}")
        
        # 2. 频域检测：检查高频能量
        print("分析高频能量分布...")
        # 计算每个通道的功率谱密度
        psd, freqs = mne.time_frequency.psd_array_welch(
            eeg_data, sfreq=self.raw_clean.info['sfreq'],
            fmin=30, fmax=50, n_fft=2048, n_per_seg=2048,
            n_jobs=1, verbose=False
        )
        # 计算高频能量
        high_freq_power = np.mean(psd, axis=1)
        print(f"高频(30-50Hz)平均能量: {np.mean(high_freq_power):.6f}")
        
        return emg_artifacts
    
    def remove_artifacts_ica(self, n_components=20, visualize=False):
        """使用ICA去除伪迹
        
        Note:
            这是实际执行“去伪迹”操作的步骤。
            它独立于前面的 detect_eog/emg_artifacts 函数，使用 ICA 算法
            重新识别并去除眼电和肌电成分。
            
        Args:
            n_components: ICA成分数量
            visualize: 是否可视化ICA成分（默认False，避免阻塞）
        """
        print("\n=== ICA伪迹去除 ===")
        # 运行ICA
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        ica.fit(self.raw_clean.copy())
        print("ICA拟合完成！")
        
        # 1. 自动识别与EOG相关的成分
        print("识别EOG伪迹成分...")
        eog_indices, eog_scores = ica.find_bads_eog(self.raw_clean)
        print(f"识别出的EOG伪迹成分: {eog_indices}")
        
        # 2. 识别与EMG相关的成分
        print("识别肌电伪迹成分...")
        emg_indices = []
        
        # 获取ICA源信号
        sources = ica.get_sources(self.raw_clean)
        source_data = sources.get_data()
        
        # 分析每个成分的频谱特征（优化版本）
        for i in range(ica.n_components_):
            # 只使用数据的一部分进行PSD计算，减少计算量
            # 选择前10秒的数据（2500个采样点）
            sample_data = source_data[i, :2500]  # 只使用前10秒数据
            
            # 计算成分的功率谱密度（使用更少的时间点）
            psd, freqs = mne.time_frequency.psd_array_multitaper(
                sample_data, sfreq=self.raw_clean.info['sfreq'],
                fmin=30, fmax=50, adaptive=True,
                n_jobs=1  # 确保单线程执行，避免阻塞
            )
            
            # 计算高频能量
            if len(psd) > 0:
                # 简化判断：如果高频能量超过阈值，认为是肌电成分
                mean_power = np.mean(psd)
                if mean_power > 1e-10:  # 设置一个合理的阈值
                    emg_indices.append(i)
        
        print(f"识别出的肌电伪迹成分: {emg_indices}")
        
        # 3. 合并所有伪迹成分
        all_artifact_indices = list(set(eog_indices + emg_indices))
        print(f"总共识别出的伪迹成分: {all_artifact_indices}")
        
        if all_artifact_indices:
            # 可视化伪迹成分（可选）
            if visualize:
                print("可视化ICA成分...")
                if eog_indices:
                    ica.plot_scores(eog_scores, show=False)
                    ica.plot_sources(self.raw_clean, picks=eog_indices, show=False, title='ICA EOG Sources')
                if emg_indices:
                    ica.plot_sources(self.raw_clean, picks=emg_indices, show=False, title='ICA EMG Sources')
            
            # 去除伪迹成分
            ica.exclude = all_artifact_indices
            self.raw_clean = ica.apply(self.raw_clean.copy())
            print("ICA伪迹去除完成！")
        else:
            print("未检测到伪迹成分。")
    
    def normalize_signal(self, method='zscore'):
        """数据标准化
        
        Note:
            此步骤主要用于为后续的机器学习或深度学习模型准备数据（消除量纲差异，加速收敛）。
            若仅进行信号分析或可视化（需要保留微伏等物理单位），或者后续不通过此文件直接对接模型，
            此步骤可选。
        
        Args:
            method: 标准化方法 ('zscore', 'minmax', 'range')
        """
        print(f"\n=== 数据标准化 ({method}) ===")
        
        # 提取EEG数据
        eeg_data = self.raw_clean.get_data()
        
        if method == 'zscore':
            # Z-score标准化 (均值为0，标准差为1)
            mean = np.mean(eeg_data, axis=1, keepdims=True)
            std = np.std(eeg_data, axis=1, keepdims=True)
            std[std == 0] = 1  # 避免除零错误
            normalized_data = (eeg_data - mean) / std
            print(f"Z-score标准化: 均值={np.mean(normalized_data):.6f}, 标准差={np.std(normalized_data):.6f}")
            
        elif method == 'minmax':
            # 最小-最大标准化 (0-1范围)
            min_val = np.min(eeg_data, axis=1, keepdims=True)
            max_val = np.max(eeg_data, axis=1, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # 避免除零错误
            normalized_data = (eeg_data - min_val) / range_val
            print(f"Min-Max标准化: 最小值={np.min(normalized_data):.6f}, 最大值={np.max(normalized_data):.6f}")
            
        elif method == 'range':
            # 幅度范围标准化 (-1到1)
            max_abs = np.max(np.abs(eeg_data), axis=1, keepdims=True)
            max_abs[max_abs == 0] = 1  # 避免除零错误
            normalized_data = eeg_data / max_abs
            print(f"幅度范围标准化: 最小值={np.min(normalized_data):.6f}, 最大值={np.max(normalized_data):.6f}")
        
        else:
            raise ValueError("未知的标准化方法，请选择 'zscore', 'minmax' 或 'range'")
        
        # 创建新的Raw对象
        info = self.raw_clean.info.copy()
        self.raw_clean = mne.io.RawArray(normalized_data, info)
        print("数据标准化完成！")
    
    def compare_raw_clean(self, duration=10):
        """比较原始信号和清洗后的信号"""
        print("\n=== 原始信号与清洗信号对比 ===")
        
        # 绘制原始信号
        print("绘制原始信号...")
        fig_raw = self.raw.plot(duration=duration, n_channels=8, scalings='auto', title='原始EEG信号', show=False)
        
        # 绘制清洗后的信号
        print("绘制清洗后的信号...")
        fig_clean = self.raw_clean.plot(duration=duration, n_channels=8, scalings='auto', title='清洗后的EEG信号', show=False)
    
    def run_preprocessing_pipeline(self, visualize=False):
        """运行完整的预处理流程
        
        Args:
            visualize: 是否进行可视化（默认False，避免阻塞）
        """
        print("\n=== 开始脑电信号预处理流程 ===")
        
        # 1. 通道分类
        self.classify_channels()
        
        # 2. 带通滤波
        self.filter_signal(l_freq=0.5, h_freq=40)
        
        # 3. 伪迹检测
        if visualize:
            self.detect_eog_artifacts()
            self.detect_emg_artifacts()
        else:
            print("\n=== 伪迹检测 ===")
            # 仅计算伪迹比例，不可视化
            raw_eog = self.raw_clean.copy().pick(picks=self.eog_channels)
            eog_data = raw_eog.get_data()
            threshold = 3 * np.std(eog_data)
            eog_artifacts = np.any(np.abs(eog_data) > threshold, axis=0)
            artifact_ratio = np.sum(eog_artifacts) / len(eog_artifacts)
            print(f"EOG伪迹样本占比: {artifact_ratio:.2%}")
            
            # 肌电伪迹检测
            print("\n=== 肌电伪迹检测 ===")
            eeg_data = self.raw_clean.get_data()
            threshold = 4 * np.std(eeg_data)
            emg_artifacts = np.any(np.abs(eeg_data) > threshold, axis=0)
            emg_artifact_ratio = np.sum(emg_artifacts) / len(emg_artifacts)
            print(f"肌电伪迹样本占比: {emg_artifact_ratio:.2%}")
        
        # 4. ICA伪迹去除
        self.remove_artifacts_ica(n_components=20, visualize=visualize)
        
        # 5. 数据标准化
        self.normalize_signal(method='zscore')
        
        # 6. 对比原始信号和清洗信号（可选）
        if visualize:
            self.compare_raw_clean(duration=10)
        
        print("\n=== 预处理流程完成！ ===")
        return self.raw_clean

# 主函数
if __name__ == "__main__":
    print("正在加载数据集...")
    # 读取GDF文件
    raw = mne.io.read_raw_gdf(".\BCICIV_2a_gdf\A01T.gdf", preload=True)
    print("数据集加载完成！")
    
    # 初始化预处理器
    preprocessor = EEGPreprocessor(raw)
    
    # 运行预处理流程（默认不可视化，避免阻塞）
    raw_clean = preprocessor.run_preprocessing_pipeline(visualize=True)
    
    # 保存预处理后的数据
    output_path = ".\BCICIV_2a_gdf\A01T_cleaned.fif"
    print(f"\n保存预处理后的数据到: {output_path}")
    raw_clean.save(output_path, overwrite=True)
    print("数据保存完成！")
    
    print("\n所有预处理步骤已完成。请查看弹出的图形窗口。")
    plt.show()