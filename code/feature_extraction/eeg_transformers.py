"""sklearn-compatible EEG feature transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from scipy import signal

from .csp_feature import MNECSPTransformer
from .wavelet_feature import WaveletEnergyTransformer


from sklearn.preprocessing import FunctionTransformer


def select_motor_channels(X):
    """
    选择运动想象相关通道（C3, Cz, C4）
    
    Args:
        X: 三维数组 (n_trials, n_channels, n_times)
    
    Returns:
        X_selected: 选择后的数组 (n_trials, 3, n_times)
    
    Note:
        BCIC IV-2a 数据集的 22 EEG 通道顺序：
        FCz=0, FC3=1, FC1=2, FC2=3, FC4=4,
        C5=5, C3=6, C1=7, Cz=8, C2=9, C4=10, C6=11,
        CP3=12, CP1=13, CP2=14, CP4=15,
        CPz=16, Pz=17, POz=18, Oz=19
        
        所以 C3=6, Cz=8, C4=10（从0开始索引）
    """
    # 【修正】BCIC IV-2a 数据集中 C3, Cz, C4 的实际索引
    motor_idx = [6, 8, 10]  # C3, Cz, C4
    return X[:, motor_idx, :]


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    滤波器组 CSP (FBCSP) 特征提取器（使用 scipy 高效滤波）
    
    对多个频段分别滤波并提取 CSP 特征，最后拼接所有频段的特征。
    使用 scipy.signal.butter + sosfiltfilt 实现零相位 IIR 滤波，
    比 mne.filter.filter_data 快 10-30 倍。
    """
    
    def __init__(self, freq_bands, sfreq, n_components=4, reg=None, log=True, norm_trace=False):
        """
        Args:
            freq_bands: list of (l_freq, h_freq) 频带列表，如 [(8,12), (12,16), ...]
            sfreq: 采样率 (Hz)
            n_components: 每个频段的 CSP 成分数
            reg: CSP 正则化参数
            log: 是否使用对数方差
            norm_trace: 是否归一化协方差矩阵的迹
        """
        self.freq_bands = freq_bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self.csp_list_ = []
        self._sos = {}  # 缓存滤波器系数

    def _get_sos(self, l_freq, h_freq):
        """
        设计并缓存 IIR 带通滤波器（二阶节形式）
        
        使用 6 阶 Butterworth 滤波器，零相位滤波
        """
        key = (l_freq, h_freq)
        if key not in self._sos:
            nyq = self.sfreq / 2.0
            low = l_freq / nyq
            high = h_freq / nyq
            # 6 阶 Butterworth 带通，零相位滤波
            self._sos[key] = signal.butter(6, [low, high], btype='band', output='sos')
        return self._sos[key]

    def _filter_band(self, X, l_freq, h_freq):
        """
        对三维数组 (trials, channels, time) 进行带通滤波（向量化）
        
        Args:
            X: 三维数组 (n_trials, n_channels, n_times)
            l_freq: 低频截止
            h_freq: 高频截止
        
        Returns:
            X_filt: 滤波后的数组
        """
        sos = self._get_sos(l_freq, h_freq)
        # axis=2 表示沿时间维度滤波，一次性处理所有 trial 和通道
        return signal.sosfiltfilt(sos, X, axis=2)

    def fit(self, X, y):
        """
        对每个频段训练独立的 CSP 滤波器
        
        Args:
            X: 三维数组 (n_trials, n_channels, n_times)
            y: 标签向量 (n_trials,)
        """
        self.csp_list_ = []
        
        for l_f, h_f in self.freq_bands:
            # 【优化】使用 scipy 快速滤波
            X_filt = self._filter_band(X, l_f, h_f)
            
            # 训练 CSP
            csp = MNECSPTransformer(
                n_components=self.n_components,
                reg=self.reg,
                log=self.log,
                norm_trace=self.norm_trace
            )
            csp.fit(X_filt, y)
            self.csp_list_.append(csp)
        
        return self

    def transform(self, X):
        """
        提取所有频段的 CSP 特征并拼接
        
        Args:
            X: 三维数组 (n_trials, n_channels, n_times)
        
        Returns:
            features: 二维数组 (n_trials, n_bands * n_components)
        """
        features = []
        
        for (l_f, h_f), csp in zip(self.freq_bands, self.csp_list_):
            # 【优化】使用 scipy 快速滤波
            X_filt = self._filter_band(X, l_f, h_f)
            
            # 提取 CSP 特征
            feat = csp.transform(X_filt)  # (n_trials, n_components)
            features.append(feat)
        
        return np.hstack(features)


class EpochsToArray(BaseEstimator, TransformerMixin):
    """Convert MNE Epochs or array-like input to a 3D ndarray."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "get_data"):
            return X.get_data()
        return np.asarray(X)


def make_feature_union(
    feature_set="fused",
    n_csp_components=4,
    wavelet="db4",
    wavelet_level=4,
    motor_channels_only=False,  # 【新增】是否只使用运动区通道
):
    """
    构建 EEG 特征联合提取器（CSP、小波或融合）
    
    Args:
        feature_set: 特征集类型 ('csp', 'wavelet', 'fused')
        n_csp_components: CSP 成分数
        wavelet: 小波基类型
        wavelet_level: 小波分解层数
        motor_channels_only: 是否只使用运动区通道 (C3, Cz, C4)
    
    Returns:
        FeatureUnion: sklearn FeatureUnion 对象
    
    Note:
        - MNECSPTransformer 和 WaveletEnergyTransformer 内部已通过 np.asarray() 处理 Epochs 转换
        - 如需显式转换，可在 Pipeline 中添加 EpochsToArray() 作为第一步
        - CSP 和小波特征维度可能差异较大（如 4 vs 110），建议在后续 Pipeline 中添加标准化
        - 启用 motor_channels_only 可将小波特征从 110 维降至 15 维
    """

    transformers = []
    if feature_set in {"csp", "fused"}:
        transformers.append(("csp", MNECSPTransformer(n_components=n_csp_components)))
    if feature_set in {"wavelet", "fused"}:
        # 【优化】直接通过 picks 参数选择通道，避免 FunctionTransformer 嵌套问题
        if motor_channels_only:
            # BCIC IV-2a 数据集中 C3=6, Cz=8, C4=10
            picks = [6, 8, 10]
        else:
            picks = None
        
        transformers.append((
            "wavelet", 
            WaveletEnergyTransformer(wavelet=wavelet, level=wavelet_level, picks=picks)
        ))
    if not transformers:
        raise ValueError("feature_set must be one of: 'csp', 'wavelet', 'fused'")
    return FeatureUnion(transformers)

