"""Classification module for EEG signal processing"""

from .svm_classifier import (
    train_svm_classifier,
    optimize_svm_hyperparameters,
    evaluate_model
)

__all__ = [
    'train_svm_classifier',
    'optimize_svm_hyperparameters',
    'evaluate_model'
]
