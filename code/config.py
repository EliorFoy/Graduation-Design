"""Shared configuration for the EEG motor imagery pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EEGPipelineConfig:
    """Centralized defaults used by preprocessing, training, and evaluation."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir_name: str = "BCICIV_2a_gdf"
    true_labels_dir_name: str = "true_labels"
    results_dir_name: str = "results"
    random_state: int = 42
    cv_folds: int = 10
    ica_l_freq: float = 1.0
    ica_h_freq: float = 40.0
    task_l_freq: float = 8.0
    task_h_freq: float = 30.0
    epoch_tmin: float = 0.0
    epoch_tmax: float = 4.0
    eval_trial_start_offset: float = 2.0
    csp_components: int = 4
    wavelet: str = "db4"
    wavelet_level: int = 4
    svm_kernel: str = "rbf"

    @property
    def data_root(self) -> Path:
        return self.project_root / self.data_dir_name

    @property
    def true_labels_root(self) -> Path:
        return self.project_root / self.true_labels_dir_name

    @property
    def results_root(self) -> Path:
        return self.project_root / self.results_dir_name

    def subject_results_dir(self, subject_id: str) -> Path:
        return self.results_root / subject_id


DEFAULT_CONFIG = EEGPipelineConfig()


TASK_EVENT_IDS = (769, 770, 771, 772)
TASK_CLASS_IDS = (1, 2, 3, 4)
EVENT_TO_CLASS = dict(zip(TASK_EVENT_IDS, TASK_CLASS_IDS))
TASK_CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]


def events_to_class_labels(events):
    """Map BCI IV 2a event ids 769-772 to class labels 1-4."""

    import numpy as np

    events = np.asarray(events)
    labels = np.empty(events.shape, dtype=int)
    for event_id, class_id in EVENT_TO_CLASS.items():
        labels[events == event_id] = class_id
    unknown_mask = ~np.isin(events, TASK_EVENT_IDS)
    if np.any(unknown_mask):
        raise ValueError(f"Cannot map non-task event ids: {np.unique(events[unknown_mask])}")
    return labels

def resolve_data_path(subject: str, data_root=None, config: EEGPipelineConfig = DEFAULT_CONFIG) -> Path:
    """Resolve a BCI IV 2a GDF path from a subject/session id."""

    root = Path(data_root) if data_root is not None else config.data_root
    return root / f"{subject}.gdf"


def ensure_result_dirs(subject_id: str, config: EEGPipelineConfig = DEFAULT_CONFIG) -> dict[str, Path]:
    """Create and return standard result subdirectories for a subject/session."""

    base = config.subject_results_dir(subject_id)
    dirs = {
        "base": base,
        "figures": base / "figures",
        "models": base / "models",
        "metrics": base / "metrics",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

