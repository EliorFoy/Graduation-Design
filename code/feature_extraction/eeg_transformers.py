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
    """Build a feature union for CSP, wavelet, or fused EEG features."""

    transformers = []
    if feature_set in {"csp", "fused"}:
        transformers.append(("csp", MNECSPTransformer(n_components=n_csp_components)))
    if feature_set in {"wavelet", "fused"}:
        transformers.append(("wavelet", WaveletEnergyTransformer(wavelet=wavelet, level=wavelet_level)))
    if not transformers:
        raise ValueError("feature_set must be one of: 'csp', 'wavelet', 'fused'")
    return FeatureUnion(transformers)

