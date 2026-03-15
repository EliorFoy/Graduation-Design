"""
在独立测试集上评估模型泛化能力（修正版）

关键修正：
  - 评估集 GDF 文件不含类别标签（769-772），只有 trial 起始标记（768）
  - 真实标签需从 true_labels/*.mat 文件中单独读取
  - 使用 768 事件进行 Epoching，再与 .mat 标签对齐

训练集：A01T.gdf（含类别标签，288 个试次）
测试集：A01E.gdf（不含类别标签） + true_labels/A01E.mat（真实标签）
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, cohen_kappa_score,
)
import sys
import mne

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from classification.svm_classifier import train_svm_classifier, plot_confusion_matrix
from feature_extraction.wavelet_feature import (
    extract_wavelet_energy_features,
    normalize_features,
)
from feature_extraction.csp_feature import extract_csp_features
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

def create_epochs_for_evaluation(raw_final, tmin=0, tmax=4, baseline=None):
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
    tmin_adjusted = tmin + 2  # cue 在 768 之后 2 秒出现
    tmax_adjusted = tmax + 2

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

    print(f"\n✅ 评估集分段完成")
    print(f"   - Epochs 数：{len(epochs)}")
    print(f"   - 通道数：{len(epochs.ch_names)}")

    return epochs


# =========================================================================
#  评估集预处理 Pipeline
# =========================================================================

def preprocess_evaluation_set(subject_id="A01E"):
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
    raw = get_modified_raw_data(subject=subject_id)
    print(f"✅ 数据加载完成，通道数：{len(raw.ch_names)}")

    # 2. ICA 前轻度滤波
    print("\nStep 2: ICA 前轻度滤波")
    raw_ica_filtered = filter_for_ica(raw, l_freq=1.0, h_freq=40.0)

    # 3. ICA 分解
    print("\nStep 3: ICA 分解")
    ica = fit_ica(raw_ica_filtered)

    # 4. 检测并去除伪迹
    print("\nStep 4: 检测并去除伪迹")
    ica = detect_and_remove_artifacts(ica, raw_ica_filtered)

    # 5. 应用 ICA 去噪
    print("\nStep 5: 应用 ICA 去噪")
    raw_clean = apply_ica(ica, raw)

    # 6. 任务定制滤波
    print("\nStep 6: 任务定制滤波 (8-30 Hz)")
    raw_final = filter_for_task(raw_clean, l_freq=8.0, h_freq=30.0)

    # 7. 重参考
    print("\nStep 7: 平均参考")
    raw_final = set_reference(raw_final, ref="average")

    # 8. 分段（使用 768 而非 769-772）
    print("\nStep 8: 分段（768 事件）")
    epochs = create_epochs_for_evaluation(raw_final, tmin=0, tmax=4)

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
    # epochs.selection 记录了从原始 events 中保留下来的索引
    kept_indices = epochs.selection
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

def train_and_evaluate(train_subject="A01T", eval_subject="A01E"):
    """
    在训练集上训练，在独立评估集上测试

    Args:
        train_subject: 训练集被试 ID
        eval_subject: 评估集被试 ID
    """
    print("\n" + "=" * 80)
    print("独立测试集评估（修正版）")
    print(f"训练集：{train_subject} | 测试集：{eval_subject}")
    print("=" * 80)

    # =====================================================================
    #  Step 1: 处理训练集
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"【Step 1】处理训练集 {train_subject}")
    print("=" * 80)

    epochs_train, ica_train = complete_preprocessing_pipeline()

    print(f"\n✅ 训练集预处理完成")
    print(f"   - 试次数：{len(epochs_train)}")
    print(f"   - 通道数：{len(epochs_train.ch_names)}")

    # =====================================================================
    #  Step 2: 提取训练集特征
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 2】提取训练集特征")
    print("=" * 80)

    # CSP 特征
    X_csp_train, csp_model = extract_csp_features(epochs_train, n_components=4)

    # 小波特征
    X_wavelet_train = extract_wavelet_energy_features(epochs_train)

    # 归一化
    X_csp_train_norm, scaler_csp = normalize_features(X_csp_train)
    X_wavelet_train_norm, scaler_wavelet = normalize_features(X_wavelet_train)

    # 融合特征
    X_fused_train = np.hstack([X_csp_train_norm, X_wavelet_train_norm])
    print(f"   - 融合特征形状：{X_fused_train.shape}")

    # 获取训练集标签（从 epochs.events 中）
    y_train = epochs_train.events[:, 2]

    # 同时加载训练集 .mat 标签用于验证
    y_train_mat = load_true_labels(train_subject)
    y_train_aligned = align_labels_with_epochs(epochs_train, y_train_mat)

    # 验证标签映射关系
    print("\n--- 验证训练集标签映射 ---")
    event_to_class = {}
    for evt_val, cls_val in zip(y_train, y_train_aligned):
        if evt_val not in event_to_class:
            event_to_class[evt_val] = cls_val
    print(f"   - events 值 → class 映射：{event_to_class}")

    # =====================================================================
    #  Step 3: 训练 SVM 分类器
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 3】训练 SVM 分类器（使用 .mat 标签）")
    print("=" * 80)

    # 使用 .mat 的真实标签（1-4）进行训练
    clf_fused, cv_scores, acc_mean = train_svm_classifier(
        X_fused_train, y_train_aligned, cv_folds=10, kernel="rbf"
    )
    print(f"\n✅ 训练集交叉验证准确率：{acc_mean:.4f} ± {cv_scores.std():.4f}")

    # 也训练单独特征的分类器用于对比
    clf_csp, cv_csp, acc_csp = train_svm_classifier(
        X_csp_train_norm, y_train_aligned, cv_folds=10
    )
    clf_wavelet, cv_wav, acc_wav = train_svm_classifier(
        X_wavelet_train_norm, y_train_aligned, cv_folds=10
    )

    # =====================================================================
    #  Step 4: 处理评估集
    # =====================================================================
    print("\n" + "=" * 80)
    print(f"【Step 4】处理评估集 {eval_subject}")
    print("=" * 80)

    epochs_test, ica_test = preprocess_evaluation_set(eval_subject)

    # =====================================================================
    #  Step 5: 加载评估集真实标签
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 5】加载评估集真实标签")
    print("=" * 80)

    y_test_all = load_true_labels(eval_subject)
    y_test = align_labels_with_epochs(epochs_test, y_test_all)

    # =====================================================================
    #  Step 6: 提取评估集特征
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 6】提取评估集特征")
    print("=" * 80)

    # CSP 特征
    X_csp_test, _ = extract_csp_features(epochs_test, n_components=4)

    # 小波特征
    X_wavelet_test = extract_wavelet_energy_features(epochs_test)

    # 使用训练集的归一化器
    X_csp_test_norm = scaler_csp.transform(X_csp_test)
    X_wavelet_test_norm = scaler_wavelet.transform(X_wavelet_test)

    # 融合特征
    X_fused_test = np.hstack([X_csp_test_norm, X_wavelet_test_norm])
    print(f"   - 融合测试特征形状：{X_fused_test.shape}")

    # =====================================================================
    #  Step 7: 在评估集上预测与评估
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 7】在评估集上预测与评估")
    print("=" * 80)

    target_names = ["左手", "右手", "双脚", "舌头"]

    results = {}

    for name, clf, X_test in [
        ("CSP", clf_csp, X_csp_test_norm),
        ("小波", clf_wavelet, X_wavelet_test_norm),
        ("融合", clf_fused, X_fused_test),
    ]:
        print(f"\n--- {name}特征 ---")
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"   准确率：{acc:.4f}")
        print(f"   Kappa：{kappa:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

        results[name] = {
            "accuracy": acc,
            "kappa": kappa,
            "confusion_matrix": cm,
            "y_pred": y_pred,
        }

    # 绘制融合特征的混淆矩阵
    cm_fused = results["融合"]["confusion_matrix"]
    save_path = f"./{eval_subject}_test_confusion_matrix.png"
    plot_confusion_matrix(cm_fused, class_names=target_names, save_path=save_path)

    # =====================================================================
    #  Step 8: 结果汇总
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 8】结果汇总")
    print("=" * 80)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     性能对比总结                              ║
╠══════════════════════════════════════════════════════════════╣
║  训练集 ({train_subject})                                     ║
║    - 试次数：{len(epochs_train):>3d}                                          ║
║    - CSP   交叉验证准确率：{acc_csp:.4f} ± {cv_csp.std():.4f}               ║
║    - 小波  交叉验证准确率：{acc_wav:.4f} ± {cv_wav.std():.4f}               ║
║    - 融合  交叉验证准确率：{acc_mean:.4f} ± {cv_scores.std():.4f}            ║
║                                                              ║
║  评估集 ({eval_subject})                                      ║
║    - 试次数：{len(epochs_test):>3d}                                          ║
║    - CSP   测试准确率：{results['CSP']['accuracy']:.4f}  Kappa: {results['CSP']['kappa']:.4f}         ║
║    - 小波  测试准确率：{results['小波']['accuracy']:.4f}  Kappa: {results['小波']['kappa']:.4f}         ║
║    - 融合  测试准确率：{results['融合']['accuracy']:.4f}  Kappa: {results['融合']['kappa']:.4f}         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 泛化能力分析
    gap = acc_mean - results["融合"]["accuracy"]
    if abs(gap) < 0.05:
        print(f"✅ 泛化能力优秀！训练-测试差异很小 ({gap:+.4f})")
    elif abs(gap) < 0.10:
        print(f"⚠️  泛化能力良好，存在轻微过拟合 ({gap:+.4f})")
    else:
        print(f"❗ 存在过拟合迹象，差异较大 ({gap:+.4f})")

    # =====================================================================
    #  Step 9: 保存结果
    # =====================================================================
    print("\n" + "=" * 80)
    print("【Step 9】保存结果")
    print("=" * 80)

    save_data = {
        "train": {
            "subject": train_subject,
            "n_epochs": len(epochs_train),
            "cv_accuracy_csp": acc_csp,
            "cv_accuracy_wavelet": acc_wav,
            "cv_accuracy_fused": acc_mean,
        },
        "test": {
            "subject": eval_subject,
            "n_epochs": len(epochs_test),
            "results": {
                k: {"accuracy": v["accuracy"], "kappa": v["kappa"]}
                for k, v in results.items()
            },
        },
        "model": clf_fused,
        "scalers": {"csp": scaler_csp, "wavelet": scaler_wavelet},
        "predictions": {"y_test": y_test, "y_pred": results["融合"]["y_pred"]},
    }

    results_path = f"./evaluation_results_{eval_subject}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"✅ 结果已保存：{results_path}")

    print("\n" + "=" * 80)
    print("✅ 独立测试集评估完成！")
    print("=" * 80)

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
