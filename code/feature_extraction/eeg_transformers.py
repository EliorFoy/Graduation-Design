"""sklearn-compatible EEG feature transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import mne

from .csp_feature import MNECSPTransformer
from .wavelet_feature import WaveletEnergyTransformer


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    滤波器组 CSP (FBCSP) 特征提取器
    
    对多个频段分别滤波并提取 CSP 特征，最后拼接所有频段的特征。
    这是 BCIC IV-2a 竞赛中表现最佳的方法之一。
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

    def fit(self, X, y):
        """
        对每个频段训练独立的 CSP 滤波器
        
        Args:
            X: 三维数组 (n_trials, n_channels, n_times)
            y: 标签向量 (n_trials,)
        """
        self.csp_list_ = []
        
        for l_f, h_f in self.freq_bands:
            # 对每个试次进行带通滤波
            X_filt = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_filt[i] = mne.filter.filter_data(
                    X[i], sfreq=self.sfreq,
                    l_freq=l_f, h_freq=h_f,
                    method='fir', verbose=False
                )
            
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
            # 滤波
            X_filt = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_filt[i] = mne.filter.filter_data(
                    X[i], sfreq=self.sfreq,
                    l_freq=l_f, h_freq=h_f,
                    method='fir', verbose=False
                )
            
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
):
    """
    构建 EEG 特征联合提取器（CSP、小波或融合）
    
    Args:
        feature_set: 特征集类型 ('csp', 'wavelet', 'fused')
        n_csp_components: CSP 成分数
        wavelet: 小波基类型
        wavelet_level: 小波分解层数
    
    Returns:
        FeatureUnion: sklearn FeatureUnion 对象
    
    Note:
        - MNECSPTransformer 和 WaveletEnergyTransformer 内部已通过 np.asarray() 处理 Epochs 转换
        - 如需显式转换，可在 Pipeline 中添加 EpochsToArray() 作为第一步
        - CSP 和小波特征维度可能差异较大（如 4 vs 110），建议在后续 Pipeline 中添加标准化
    """

    transformers = []
    if feature_set in {"csp", "fused"}:
        transformers.append(("csp", MNECSPTransformer(n_components=n_csp_components)))
    if feature_set in {"wavelet", "fused"}:
        transformers.append(("wavelet", WaveletEnergyTransformer(wavelet=wavelet, level=wavelet_level)))
    if not transformers:
        raise ValueError("feature_set must be one of: 'csp', 'wavelet', 'fused'")
    return FeatureUnion(transformers)

