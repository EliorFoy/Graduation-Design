"""
单被试完整处理流程

从原始 GDF 数据到分类评估的完整 pipeline
包括：预处理 → 特征提取 → 分类 → 评估
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from classification.svm_classifier import train_eeg_svm_pipeline, plot_confusion_matrix
from pretreatment.complete_preprocessing import complete_preprocessing_pipeline
from code.config import DEFAULT_CONFIG, TASK_CLASS_IDS, TASK_CLASS_NAMES, TASK_EVENT_ANNOTATIONS, ensure_result_dirs, epochs_events_to_class_labels, resolve_data_path
import numpy as np

# 添加项目路径


def single_subject_pipeline(subject_id="A01T", data_root=None):
    """
      单被试完整处理流程

    Args:
          subject_id: 被试 ID，如 'A01T' (训练集) 或 'A01E' (测试集)
          data_root: 数据根目录（为 None 则自动查找）

      Returns:
          results: 包含所有结果的字典
                  {
                      'epochs': 预处理后的 Epochs,
                      'y': 标签，
                      'clf_csp': CSP 分类器，
                      'clf_wavelet': 小波分类器，
                      'clf_fused': 融合分类器，
                      'metrics': 评估指标
                  }
    """
    print("\n" + "=" * 80)
    print(f"单被试完整处理流程 - 被试：{subject_id}")
    print("=" * 80)

    # ========== Step 1: 数据准备 ==========
    print("\n【Step 1】数据准备")

    data_path = resolve_data_path(subject_id, data_root=data_root, config=DEFAULT_CONFIG)

    print(f"   - 数据文件：{data_path}")

    if not data_path.exists():
        print(f"❌ 错误：数据文件不存在！")
        return None

    # ========== Step 2: 预处理 ==========
    print("\n【Step 2】EEG 预处理")

    epochs, ica = complete_preprocessing_pipeline(subject=subject_id, data_root=data_root)

    print(f"✅ 预处理完成")
    print(f"   - Epochs 形状：{epochs.get_data().shape}")
    print(f"   - 试次数：{len(epochs)}")
    print(f"   - 通道数：{len(epochs.ch_names)}")

    # ========== Step 3: 获取标签 ==========
    print("\n【Step 3】准备标签")

    # 使用动态映射：通过 epochs.event_id 将 MNE 内部事件 ID 转换为类别标签 (1-4)
    y = epochs_events_to_class_labels(epochs)
    
    # 显示标签分布
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"\n📊 类别标签分布:")
    for cls, cnt in zip(unique_classes, class_counts):
        cls_name = TASK_CLASS_NAMES[cls - 1] if cls <= len(TASK_CLASS_NAMES) else '未知'
        print(f"   - 类别 {cls} ({cls_name}): {cnt} 个")
    
    print(f"\n✅ 最终标签形状：{y.shape}")

    # ========== Step 4: 分类器训练与评估 ==========
    print("\n【Step 4】SVM 分类器训练与评估")

    # 6.1 仅使用 CSP 特征
    print("\n--- 使用 CSP 特征 ---")
    clf_csp, cv_scores_csp, acc_csp = train_eeg_svm_pipeline(
        epochs, y, feature_set="csp", cv_folds=DEFAULT_CONFIG.cv_folds,
        n_csp_components=DEFAULT_CONFIG.csp_components,
        kernel=DEFAULT_CONFIG.svm_kernel,
        random_state=DEFAULT_CONFIG.random_state,
    )

    # 6.2 仅使用小波特征
    print("\n--- 使用小波特征 ---")
    clf_wavelet, cv_scores_wavelet, acc_wavelet = train_eeg_svm_pipeline(
        epochs, y, feature_set="wavelet", cv_folds=DEFAULT_CONFIG.cv_folds,
        wavelet=DEFAULT_CONFIG.wavelet, wavelet_level=DEFAULT_CONFIG.wavelet_level,
        kernel=DEFAULT_CONFIG.svm_kernel,
        random_state=DEFAULT_CONFIG.random_state,
    )

    # 6.3 使用融合特征
    print("\n--- 使用融合特征（全通道） ---")
    clf_fused, cv_scores_fused, acc_fused = train_eeg_svm_pipeline(
        epochs, y, feature_set="fused", cv_folds=DEFAULT_CONFIG.cv_folds,
        n_csp_components=DEFAULT_CONFIG.csp_components,
        wavelet=DEFAULT_CONFIG.wavelet, wavelet_level=DEFAULT_CONFIG.wavelet_level,
        motor_channels_only=False,
        kernel=DEFAULT_CONFIG.svm_kernel,
        random_state=DEFAULT_CONFIG.random_state,
    )
    
    # 6.3b 使用融合特征（仅运动区通道）
    print("\n--- 使用融合特征（运动区通道 C3/Cz/C4） ---")
    clf_fused_motor, cv_scores_fused_motor, acc_fused_motor = train_eeg_svm_pipeline(
        epochs, y, feature_set="fused", cv_folds=DEFAULT_CONFIG.cv_folds,
        n_csp_components=DEFAULT_CONFIG.csp_components,
        wavelet=DEFAULT_CONFIG.wavelet, wavelet_level=DEFAULT_CONFIG.wavelet_level,
        motor_channels_only=True,
        kernel=DEFAULT_CONFIG.svm_kernel,
        random_state=DEFAULT_CONFIG.random_state,
    )

    # 6.4 使用 FBCSP 特征（滤波器组 CSP）
    print("\n--- 使用 FBCSP 特征（滤波器组 CSP） ---")
    clf_fbcsp, cv_scores_fbcsp, acc_fbcsp = train_eeg_svm_pipeline(
        epochs, y, 
        feature_set='fb_csp',
        cv_folds=DEFAULT_CONFIG.cv_folds,
        n_csp_components=DEFAULT_CONFIG.csp_components,
        freq_bands=[(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)],
        kernel=DEFAULT_CONFIG.svm_kernel,
        random_state=DEFAULT_CONFIG.random_state,
    )

    # ========== Step 5: 结果汇总 ==========
    print("\n【Step 5】结果汇总")

    print("\n" + "=" * 60)
    print("分类性能对比")
    print("=" * 60)
    print(f"CSP 特征准确率：         {acc_csp:.4f} ± {cv_scores_csp.std():.4f}")
    print(f"小波特征准确率：       {acc_wavelet:.4f} ± {cv_scores_wavelet.std():.4f}")
    print(f"融合特征准确率（全通道）： {acc_fused:.4f} ± {cv_scores_fused.std():.4f}")
    print(f"融合特征准确率（运动区）： {acc_fused_motor:.4f} ± {cv_scores_fused_motor.std():.4f}")
    print(f"FBCSP 特征准确率：     {acc_fbcsp:.4f} ± {cv_scores_fbcsp.std():.4f}")
    print("=" * 60)

    # 找出最佳特征
    best_acc = max(acc_csp, acc_wavelet, acc_fused, acc_fused_motor, acc_fbcsp)
    if best_acc == acc_fbcsp:
        print(f"\n🏆 FBCSP 特征表现最佳！")
    elif best_acc == acc_fused_motor:
        print(f"\n🏆 融合特征（运动区）表现最佳！")
    elif best_acc == acc_fused:
        print(f"\n🏆 融合特征（全通道）表现最佳！")
    elif best_acc == acc_csp:
        print(f"\n🏆 CSP 特征表现最佳！")
    else:
        print(f"\n🏆 小波特征表现最佳！")

    # ========== Step 6: 保存结果 ==========
    print("\n【Step 6】保存结果")

    results = {
        "subject_id": subject_id,
        "epochs": epochs,
        "ica": ica,
        "y": y,
        "clf_csp": clf_csp,
        "clf_wavelet": clf_wavelet,
        "clf_fused": clf_fused,
        "clf_fused_motor": clf_fused_motor,  # 【新增】
        "clf_fbcsp": clf_fbcsp,
        "metrics": {
            "csp": {
                "accuracy": acc_csp,
                "cv_scores": cv_scores_csp,
                "cv_mean": acc_csp,
                "cv_std": cv_scores_csp.std(),
            },
            "wavelet": {
                "accuracy": acc_wavelet,
                "cv_scores": cv_scores_wavelet,
                "cv_mean": acc_wavelet,
                "cv_std": cv_scores_wavelet.std(),
            },
            "fused": {
                "accuracy": acc_fused,
                "cv_scores": cv_scores_fused,
                "cv_mean": acc_fused,
                "cv_std": cv_scores_fused.std(),
            },
            "fused_motor": {  # 【新增】
                "accuracy": acc_fused_motor,
                "cv_scores": cv_scores_fused_motor,
                "cv_mean": acc_fused_motor,
                "cv_std": cv_scores_fused_motor.std(),
            },
            "fbcsp": {
                "accuracy": acc_fbcsp,
                "cv_scores": cv_scores_fbcsp,
                "cv_mean": acc_fbcsp,
                "cv_std": cv_scores_fbcsp.std(),
            },
        },
    }

    print(f"✅ 结果已保存到 results 字典")

    # ========== Step 7: 可视化（可选） ==========
    print("\n【Step 7】可视化（可选）")

    # 绘制混淆矩阵（使用融合特征）
    try:
        from sklearn.metrics import confusion_matrix
        
        y_pred = clf_fused.predict(epochs.get_data())
        class_names = TASK_CLASS_NAMES
        
        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred, labels=TASK_CLASS_IDS)
        
        # 绘制混淆矩阵图
        dirs = ensure_result_dirs(subject_id, config=DEFAULT_CONFIG)
        cm_path = dirs["figures"] / f"{subject_id}_confusion_matrix.png"
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            save_path=str(cm_path),
        )
        print(f"✅ 混淆矩阵已保存：{cm_path}")
    except Exception as e:
        print(f"⚠️  无法绘制混淆矩阵：{e}")

    print("\n" + "=" * 80)
    print(f"✅ 单被试 {subject_id} 完整流程处理完成！")
    print("=" * 80)

    return results


def demo_usage():
    """
    演示如何使用完整流程
    """
    print("=" * 80)
    print("EEG 信号处理完整流程演示")
    print("=" * 80)
    print("\n本脚本演示了从原始 EEG 数据到分类的完整流程:")
    print("1. 数据加载与预处理（滤波、ICA、分段）")
    print("2. 特征提取（CSP + 小波）")
    print("3. 特征融合与归一化")
    print("4. SVM 分类器训练与评估")
    print("5. 结果可视化")
    print("\n使用方法:")
    print("from general_process import single_subject_pipeline")
    print("results = single_subject_pipeline('A01T')")
    print("\n输出:")
    print("- 预处理后的 Epochs 数据")
    print("- CSP 和小波特征")
    print("- 三个 SVM 分类器（分别基于 CSP、小波、融合特征）")
    print("- 分类准确率和混淆矩阵")
    print("=" * 80)


if __name__ == "__main__":
    # 方式 1: 只显示使用说明（快速查看）
    demo_usage()
    
    # 方式 2: 实际运行完整流程（需要数据文件）
    print("\n" + "=" * 80)
    print("开始实际运行...")
    print("=" * 80)
    
    try:
        # 处理 A01T 被试（训练数据）
        results = single_subject_pipeline('A01T')
        
        if results is not None:
            print("\n" + "=" * 80)
            print("✅ 流程执行完成！结果汇总:")
            print("=" * 80)
            
            # 显示基本信息
            print(f"\n📊 被试：{results['subject_id']}")
            print(f"📈 最终试次数：{len(results['epochs'])}")
            print(f"🔌 通道数：{len(results['epochs'].ch_names)}")
            
            # 显示分类性能
            print(f"\n分类准确率:")
            print(f"  - CSP 特征：{results['metrics']['csp']['cv_mean']:.4f} ± {results['metrics']['csp']['cv_std']:.4f}")
            print(f"  - 小波特征：{results['metrics']['wavelet']['cv_mean']:.4f} ± {results['metrics']['wavelet']['cv_std']:.4f}")
            print(f"  - 融合特征：{results['metrics']['fused']['cv_mean']:.4f} ± {results['metrics']['fused']['cv_std']:.4f}")
            
            # 找出最佳方法
            best_method = max(
                results['metrics'].items(),
                key=lambda x: x[1]['cv_mean']
            )
            print(f"\n🏆 最佳方法：{best_method[0].upper()} (准确率：{best_method[1]['cv_mean']:.4f})")
            
            # 询问是否保存结果
            save = input("\n是否保存结果到文件？(y/n): ")
            if save.lower() == 'y':
                import pickle
                save_path = f"./results_{results['subject_id']}.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"✅ 结果已保存到：{save_path}")
                
        else:
            print("\n❌ 流程执行失败，请检查数据文件是否存在")
            
    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
        print("\n请确认:")
        print("  1. 数据文件在正确位置：BCICIV_2a_gdf/A01T.gdf")
        print("  2. 已安装所有依赖库：mne, numpy, scipy, scikit-learn, pywt")
        
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()
