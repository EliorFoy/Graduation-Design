"""Classification module for EEG signal processing"""

from .svm_classifier import (
    make_eeg_svm_pipeline,
    train_eeg_svm_pipeline,
    train_svm_classifier,
    optimize_svm_hyperparameters,
    evaluate_model
)

__all__ = [
    'make_eeg_svm_pipeline',
    'train_eeg_svm_pipeline',
    'train_svm_classifier',
    'optimize_svm_hyperparameters',
    'evaluate_model'
]
