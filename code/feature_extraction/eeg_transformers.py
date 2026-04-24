"""sklearn-compatible EEG feature transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

from .csp_feature import MNECSPTransformer
from .wavelet_feature import WaveletEnergyTransformer


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

