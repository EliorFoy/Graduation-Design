"""Feature extraction module for EEG signals"""

from .csp_feature import MNECSPTransformer, extract_csp_features, visualize_csp_topo
from .eeg_transformers import EpochsToArray, make_feature_union
from .wavelet_feature import (
    WaveletEnergyTransformer,
    extract_wavelet_energy_features,
    extract_band_power_features,
    normalize_features
)

__all__ = [
    'MNECSPTransformer',
    'EpochsToArray',
    'make_feature_union',
    'WaveletEnergyTransformer',
    'extract_csp_features',
    'visualize_csp_topo',
    'extract_wavelet_energy_features',
    'extract_band_power_features',
    'normalize_features'
]
