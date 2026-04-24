"""EEG 运动想象流水线的共享配置。"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EEGPipelineConfig:
    """预处理、训练和评估使用的集中化默认配置。"""

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
        """获取指定被试的结果目录路径。"""
        return self.results_root / subject_id


DEFAULT_CONFIG = EEGPipelineConfig()


TASK_EVENT_ANNOTATIONS = ("769", "770", "771", "772")
"""GDF 文件中运动想象 cue 事件的原始注解字符串。"""

TASK_CLASS_IDS = (1, 2, 3, 4)
TASK_CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]

# 注解字符串 → 类别标签的静态映射
_ANNOTATION_TO_CLASS = dict(zip(TASK_EVENT_ANNOTATIONS, TASK_CLASS_IDS))


def epochs_events_to_class_labels(epochs):
    """
    利用 epochs.event_id 将 MNE 内部事件 ID 动态映射为类别标签 (1-4)。

    MNE 的 events_from_annotations 会将 GDF 注解字符串 '769'-'772'
    映射为连续整数（具体值取决于文件中注解的排序），不能假设为固定值。
    本函数通过 epochs.event_id 获取实际映射关系，确保始终正确。
    """
    import numpy as np

    event_id = epochs.event_id  # e.g. {'769': 7, '770': 8, ...}
    mne_id_to_class = {}
    for annotation, mne_id in event_id.items():
        if annotation in _ANNOTATION_TO_CLASS:
            mne_id_to_class[mne_id] = _ANNOTATION_TO_CLASS[annotation]

    raw_ids = epochs.events[:, 2]
    y = np.array([mne_id_to_class.get(eid, 0) for eid in raw_ids])

    invalid_mask = y == 0
    if np.any(invalid_mask):
        bad_ids = np.unique(raw_ids[invalid_mask])
        raise ValueError(
            f"Cannot map non-task MNE event ids: {bad_ids}. "
            f"epochs.event_id = {event_id}"
        )
    return y

def resolve_data_path(subject: str, data_root=None, config: EEGPipelineConfig = DEFAULT_CONFIG) -> Path:
    """根据被试/会话 ID 解析 BCI IV 2a GDF 文件路径。"""

    root = Path(data_root) if data_root is not None else config.data_root
    return root / f"{subject}.gdf"


def ensure_result_dirs(subject_id: str, config: EEGPipelineConfig = DEFAULT_CONFIG) -> dict[str, Path]:
    """创建并返回被试/会话的标准结果子目录。"""

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

