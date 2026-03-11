"""Feature extraction module for EEG signals"""

from .csp_feature import extract_csp_features, visualize_csp_topo
from .wavelet_feature import (
    extract_wavelet_energy_features,
    extract_band_power_features,
    normalize_features
)

__all__ = [
    'extract_csp_features',
    'visualize_csp_topo',
    'extract_wavelet_energy_features',
    'extract_band_power_features',
    'normalize_features'
]
