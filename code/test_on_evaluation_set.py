"""
在独立测试集上评估模型泛化能力（修正版）

关键修正：
  - 评估集 GDF 文件不含类别标签（769-772），只有 trial 起始标记（768）
  - 真实标签需从 true_labels/*.mat 文件中单独读取
  - 使用 768 事件进行 Epoching，再与 .mat 标签对齐

训练集：A01T.gdf（含类别标签，288 个试次）
测试集：A01E.gdf（不含类别标签） + true_labels/A01E.mat（真实标签）
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, cohen_kappa_score,
)
import mne

# 添加项目路径

from classification.svm_classifier import train_eeg_svm_pipeline, plot_confusion_matrix
from code.config import DEFAULT_CONFIG, TASK_CLASS_IDS, TASK_CLASS_NAMES, TASK_EVENT_IDS, ensure_result_dirs, events_to_class_labels
from pretreatment.complete_preprocessing import (
    complete_preprocessing_pipeline,
    filter_for_ica,
    fit_ica,
    detect_and_remove_artifacts,
    apply_ica,
    filter_for_task,
    set_reference,
)
from pretreatment.eeg_analysis import get_modified_raw_data


# =========================================================================
#  核心新增函数：加载评估集真实标签
# =========================================================================

def load_true_labels(subject_id: str) -> np.ndarray:
    """
    从 true_labels/*.mat 文件中读取评估集的真实类别标签

    Args:
        subject_id: 被试 ID，如 'A01E'

    Returns:
        labels: 形状 (n_trials,) 的标签数组，值为 1/2/3/4
    """
    project_root = Path(__file__).parent.parent
    mat_path = project_root / "true_labels" / f"{subject_id}.mat"

    if not mat_path.exists():
        raise FileNotFoundError(f"标签文件不存在：{mat_path}")

    data = loadmat(str(mat_path))
    labels = data["classlabel"].flatten()  # shape: (288,)

    print(f"✅ 加载真实标签：{mat_path.name}")
    print(f"   - 标签数量：{len(labels)}")
    print(f"   - 唯一值：{np.unique(labels)}")
    print(f"   - 各类数量：{dict(zip(*np.unique(labels, return_counts=True)))}")

    return labels


# =========================================================================
#  核心新增函数：评估集 Epoching（使用 768 事件）
# =========================================================================

def create_epochs_for_evaluation(raw_final, tmin=0, tmax=4, baseline=None, trial_start_offset=2.0):
    """
    为评估集创建 Epochs

    评估集 GDF 中没有 769-772 类别标签，只有 768（trial 起始标记）。
    因此使用 768 事件进行分段，之后再与外部 .mat 标签对齐。

    Args:
        raw_final: 预处理后的 Raw 数据
        tmin: 起始时间（相对于 768 事件，cue 在 t=2s 出现，但运动想象在 cue 后开始）
        tmax: 结束时间
        baseline: 基线校正

    Returns:
        epochs: 分段后的 Epochs 对象（不含类别标签，所有事件 ID 相同）
    """
    print("\n创建评估集分段（使用 768 事件）...")

    events, event_dict = mne.events_from_annotations(raw_final)

    print(f"   - 从 annotations 中提取到 {len(events)} 个事件")
    print(f"   - 事件类型：{event_dict}")

    # 找到 768（trial 起始）对应的 MNE 映射值
    trial_start_key = "768"
    if trial_start_key not in event_dict:
        raise ValueError(
            f"未找到 trial 起始标记 '{trial_start_key}'，"
            f"可用事件：{event_dict}"
        )

    trial_start_id = event_dict[trial_start_key]

    # 筛选 768 事件
    trial_mask = events[:, 2] == trial_start_id
    events_trial = events[trial_mask]
    print(f"   - 768 事件数：{len(events_trial)}")

    # 创建 Epochs：所有 trial 使用同一 event_id
    # 注意：cue 在 t=2s 出现，运动想象持续到 t=6s
    # 768 标记是 trial 起始（t=0），所以我们取 [tmin, tmax] 相对于 768 的偏移
    # 这里 tmin=2, tmax=6 对应 cue 之后的运动想象时段
    # 或者和训练集保持一致，取 cue 后的 [0, 4] 秒
    # 由于训练集 create_epochs 使用 769-772 事件（cue 出现时刻），tmin=0, tmax=4
    # 这里 768 是 trial 开始（t=0），cue 在 t=2 出现
    # 所以用 768 分段时要偏移 2 秒：tmin_768 = tmin + 2, tmax_768 = tmax + 2
    tmin_adjusted = tmin + trial_start_offset  # cue 在 768 之后约 2 秒出现
    tmax_adjusted = tmax + trial_start_offset

    print(f"   - 时间窗口调整：768 + [{tmin_adjusted}, {tmax_adjusted}] s")
    print(f"     (对应 cue 后 [{tmin}, {tmax}] s)")

    event_id_eval = {trial_start_key: trial_start_id}

    epochs = mne.Epochs(
        raw_final,
        events_trial,
        event_id=event_id_eval,
        tmin=tmin_adjusted,
        tmax=tmax_adjusted,
        baseline=baseline,
        preload=True,
        verbose=False,
        event_repeated="drop",
    )

    # 【关键修复】只保留 EEG 通道，剔除 EOG 通道
    epochs.pick_types(eeg=True, exclude=[])

    print(f"\n✅ 评估集分段完成")
    print(f"   - Epochs 数：{len(epochs)}")
    print(f"   - 通道数：{len(epochs.ch_names)} (仅 EEG)")

    epochs.metadata = {
        "trial_index": epochs.selection.astype(int),
        "event_sample": events_trial[epochs.selection, 0].astype(int),
    }

    return epochs


# =========================================================================
#  评估集预处理 Pipeline
# =========================================================================

def preprocess_evaluation_set(subject_id="A01E", data_root=None, config=DEFAULT_CONFIG):
    """
    对评估集运行预处理流程

    与训练集流程基本一致，区别在于 Epoching 使用 768 事件而非 769-772

    Args:
        subject_id: 被试 ID，如 'A01E'

    Returns:
        epochs: 预处理后的 Epochs
        ica: ICA 对象
    """
    print("=" * 60)
    print(f"🚀 评估集预处理：{subject_id}")
    print("=" * 60)

    # 1. 加载数据
    print("\nStep 1: 加载评估集原始数据")
    raw = get_modified_raw_data(subject=subject_id, data_root=data_root)
    print(f"✅ 数据加载完成，通道数：{len(raw.ch_names)}")

    # 2. ICA 前轻度滤波
    print("\nStep 2: ICA 前轻度滤波")
    raw_ica_filtered = filter_for_ica(raw, l_freq=config.ica_l_freq, h_freq=config.ica_h_freq)

    # 3. ICA 分解
    print("\nStep 3: ICA 分解")
    ica = fit_ica(raw_ica_filtered)

    # 4. 检测并去除伪迹
    print("\nStep 4: 检测并去除伪迹")
    ica = detect_and_remove_artifacts(ica, raw_ica_filtered, save_path='./output_img/ica_eog_components.png')

    # 5. 应用 ICA 去噪
    print("\nStep 5: 应用 ICA 去噪")
    raw_clean = apply_ica(ica, raw)

    # 6. 任务定制滤波
    print(f"\nStep 6: 任务定制滤波 ({config.task_l_freq:g}-{config.task_h_freq:g} Hz)")
    raw_final = filter_for_task(raw_clean, l_freq=config.task_l_freq, h_freq=config.task_h_freq)

    # 7. 重参考
    print("\nStep 7: 平均参考")
    raw_final = set_reference(raw_final, ref="average")

    # 8. 分段（使用 768 而非 769-772）
    print("\nStep 8: 分段（768 事件）")
    epochs = create_epochs_for_evaluation(
        raw_final,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
        trial_start_offset=config.eval_trial_start_offset,
    )

    print(f"\n{'=' * 60}")
    print(f"🎉 评估集预处理完成：{subject_id}")
    print(f"   - 试次数：{len(epochs)}")
    print(f"   - 通道数：{len(epochs.ch_names)}")
    print(f"{'=' * 60}")

    return epochs, ica


# =========================================================================
#  对齐标签与 Epochs
# =========================================================================

def align_labels_with_epochs(epochs, true_labels):
    """
    将 .mat 中的 288 个真实标签与（可能被丢弃部分 trial 的）Epochs 对齐

    MNE 在创建 Epochs 时可能会因为数据长度不足等原因自动丢弃部分 trial，
    需要通过 epochs.selection 获取实际保留的 trial 索引。

    Args:
        epochs: MNE Epochs 对象
        true_labels: (288,) 的标签数组

    Returns:
        aligned_labels: 与 epochs 长度一致的标签数组
    """
    if epochs.metadata is not None and "trial_index" in epochs.metadata:
        kept_indices = epochs.metadata["trial_index"].to_numpy(dtype=int)
    else:
        kept_indices = epochs.selection.astype(int)
    n_epochs = len(epochs)
    n_labels = len(true_labels)

    print(f"\n标签对齐：")
    print(f"   - 原始标签数：{n_labels}")
    print(f"   - Epochs 数：{n_epochs}")
    print(f"   - 保留索引范围：[{kept_indices.min()}, {kept_indices.max()}]")

    if kept_indices.max() >= n_labels:
        print(f"   ⚠️  索引超出标签范围，使用前 {n_epochs} 个标签")
        aligned_labels = true_labels[:n_epochs]
    else:
        aligned_labels = true_labels[kept_indices]

    print(f"   - 对齐后标签数：{len(aligned_labels)}")
    print(f"   - 标签分布：{dict(zip(*np.unique(aligned_labels, return_counts=True)))}")

    return aligned_labels


# =========================================================================
#  主函数：训练 + 评估
# =========================================================================

def train_and_evaluate(train_subject="A01T", eval_subject="A01E", data_root=None, config=DEFAULT_CONFIG):
    """
    Train on a labelled training session and evaluate on an independent E session.

    Training labels come from GDF events 769-772. Evaluation labels come from
    true_labels/*.mat and are aligned through the retained trial indices.
    """
    print("\n" + "=" * 80)
    print("Independent evaluation with leakage-safe sklearn pipelines")
    print(f"Train: {train_subject} | Eval: {eval_subject}")
    print("=" * 80)

    dirs = ensure_result_dirs(eval_subject, config=config)

    print("\n" + "=" * 80)
    print(f"Step 1: preprocess training set {train_subject}")
    print("=" * 80)
    epochs_train, ica_train = complete_preprocessing_pipeline(
        subject=train_subject,
        data_root=data_root,
        config=config,
    )

    y_train_events = epochs_train.events[:, 2]
    valid_mask = np.isin(y_train_events, TASK_EVENT_IDS)
    if not np.all(valid_mask):
        print(f"Warning: dropped {np.sum(~valid_mask)} non-task training epochs")
        epochs_train = epochs_train[valid_mask]
        y_train_events = y_train_events[valid_mask]
    y_train = events_to_class_labels(y_train_events)

    print("\nTraining set ready")
    print(f"   - epochs: {len(epochs_train)}")
    print(f"   - channels: {len(epochs_train.ch_names)}")
    print(f"   - labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    print("\n" + "=" * 80)
    print("Step 2: train leakage-safe EEG SVM pipelines")
    print("=" * 80)
    classifiers = {}
    cv_results = {}
    for feature_set, display_name in [("csp", "CSP"), ("wavelet", "Wavelet"), ("fused", "Fused")]:
        clf, scores, mean_acc = train_eeg_svm_pipeline(
            epochs_train,
            y_train,
            feature_set=feature_set,
            cv_folds=config.cv_folds,
            n_csp_components=config.csp_components,
            wavelet=config.wavelet,
            wavelet_level=config.wavelet_level,
            kernel=config.svm_kernel,
            random_state=config.random_state,
        )
        classifiers[display_name] = clf
        cv_results[display_name] = {"scores": scores, "mean": mean_acc, "std": scores.std()}

    print("\n" + "=" * 80)
    print(f"Step 3: preprocess evaluation set {eval_subject}")
    print("=" * 80)
    epochs_test, ica_test = preprocess_evaluation_set(eval_subject, data_root=data_root, config=config)

    print("\n" + "=" * 80)
    print("Step 4: load and align evaluation labels")
    print("=" * 80)
    y_test_all = load_true_labels(eval_subject)
    y_test = align_labels_with_epochs(epochs_test, y_test_all)

    print("\n" + "=" * 80)
    print("Step 5: predict and evaluate")
    print("=" * 80)
    results = {}
    for name, clf in classifiers.items():
        print(f"\n--- {name} features ---")
        y_pred = clf.predict(epochs_test.get_data())
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=TASK_CLASS_IDS)

        print(f"   Accuracy: {acc:.4f}")
        print(f"   Kappa: {kappa:.4f}")
        print(f"\n{classification_report(y_test, y_pred, labels=TASK_CLASS_IDS, target_names=TASK_CLASS_NAMES, zero_division=0)}")

        results[name] = {
            "accuracy": acc,
            "kappa": kappa,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

    cm_path = dirs["figures"] / f"{eval_subject}_test_confusion_matrix.png"
    plot_confusion_matrix(results["Fused"]["confusion_matrix"], class_names=TASK_CLASS_NAMES, save_path=str(cm_path))

    print("\n" + "=" * 80)
    print("Step 6: summary")
    print("=" * 80)
    for name in ("CSP", "Wavelet", "Fused"):
        print(
            f"{name:7s} CV: {cv_results[name]['mean']:.4f} +/- {cv_results[name]['std']:.4f} | "
            f"Eval: {results[name]['accuracy']:.4f} | Kappa: {results[name]['kappa']:.4f}"
        )

    gap = cv_results["Fused"]["mean"] - results["Fused"]["accuracy"]
    if abs(gap) < 0.05:
        print(f"Generalization looks strong; CV/eval gap is small ({gap:+.4f})")
    elif abs(gap) < 0.10:
        print(f"Generalization is acceptable; mild gap detected ({gap:+.4f})")
    else:
        print(f"Potential overfitting; CV/eval gap is large ({gap:+.4f})")

    save_data = {
        "config": config,
        "train": {
            "subject": train_subject,
            "n_epochs": len(epochs_train),
            "cv_results": cv_results,
        },
        "test": {
            "subject": eval_subject,
            "n_epochs": len(epochs_test),
            "results": {
                key: {"accuracy": value["accuracy"], "kappa": value["kappa"]}
                for key, value in results.items()
            },
        },
        "models": classifiers,
        "predictions": {"y_test": y_test, "y_pred": results["Fused"]["y_pred"]},
    }

    results_path = dirs["metrics"] / f"evaluation_results_{eval_subject}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Saved results: {results_path}")
    print("\nIndependent evaluation complete")
    return save_data


if __name__ == "__main__":
    results = train_and_evaluate(train_subject="A01T", eval_subject="A01E")

    if results is None:
        print("\n❌ 评估过程中断，请检查错误信息")
    else:
        print("\n" + "=" * 80)
        print("最终结果：")
        print("=" * 80)
        test_results = results["test"]["results"]
        for name, metrics in test_results.items():
            print(f"   {name}特征 → 准确率: {metrics['accuracy']:.4f}, Kappa: {metrics['kappa']:.4f}")
