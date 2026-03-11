"""
单被试完整处理流程

从原始 GDF 数据到分类评估的完整 pipeline
包括：预处理 → 特征提取 → 分类 → 评估
"""

from classification.svm_classifier import (
    train_svm_classifier,
    evaluate_model,
    plot_confusion_matrix,
)
from feature_extraction.wavelet_feature import (
    extract_wavelet_energy_features,
    normalize_features,
)
from feature_extraction.csp_feature import extract_csp_features, visualize_csp_topo
from pretreatment.complete_preprocessing import complete_preprocessing_pipeline
import numpy as np
import mne
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))


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
                      'X_csp': CSP 特征，
                      'X_wavelet': 小波特征，
                      'X_fused': 融合特征，
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

    if data_root is None:
        # 自动查找数据路径
        project_root = Path(__file__).parent.parent
        data_path = project_root / "BCICIV_2a_gdf" / f"{subject_id}.gdf"
    else:
        data_path = Path(data_root) / f"{subject_id}.gdf"

    print(f"   - 数据文件：{data_path}")

    if not data_path.exists():
        print(f"❌ 错误：数据文件不存在！")
        return None

    # ========== Step 2: 预处理 ==========
    print("\n【Step 2】EEG 预处理")

    epochs, ica = complete_preprocessing_pipeline()

    print(f"✅ 预处理完成")
    print(f"   - Epochs 形状：{epochs.get_data().shape}")
    print(f"   - 试次数：{len(epochs)}")
    print(f"   - 通道数：{len(epochs.ch_names)}")

    # ========== Step 3: 特征提取 ==========
    print("\n【Step 3】特征提取")

    # 3.1 CSP 特征
    X_csp, csp = extract_csp_features(epochs, n_components=4)

    # 可视化 CSP 拓扑图
    try:
        visualize_csp_topo(
            csp, epochs, save_path=f"./{subject_id}_csp_topo.png")
    except Exception as e:
        print(f"⚠️  无法绘制 CSP 拓扑图：{e}")

    # 3.2 小波特征
    X_wavelet = extract_wavelet_energy_features(epochs, wavelet="db4", level=4)

    # 3.3 特征归一化
    X_csp_norm, scaler_csp = normalize_features(X_csp)
    X_wavelet_norm, scaler_wavelet = normalize_features(X_wavelet)

    # ========== Step 4: 特征融合 ==========
    print("\n【Step 4】特征融合")

    # 简单拼接融合
    X_fused = np.hstack([X_csp_norm, X_wavelet_norm])
    print(f"   - CSP 特征：{X_csp_norm.shape}")
    print(f"   - 小波特征：{X_wavelet_norm.shape}")
    print(f"   - 融合特征：{X_fused.shape}")

    # ========== Step 5: 获取标签 ==========
    print("\n【Step 5】准备标签")

    y = epochs.events[:, 2]
    print(f"   - 标签形状：{y.shape}")
    print(f"   - 类别分布：{np.bincount(y)}")

    # ========== Step 6: 分类器训练与评估 ==========
    print("\n【Step 6】SVM 分类器训练与评估")

    # 6.1 仅使用 CSP 特征
    print("\n--- 使用 CSP 特征 ---")
    clf_csp, cv_scores_csp, acc_csp = train_svm_classifier(X_csp_norm, y)

    # 6.2 仅使用小波特征
    print("\n--- 使用小波特征 ---")
    clf_wavelet, cv_scores_wavelet, acc_wavelet = train_svm_classifier(
        X_wavelet_norm, y
    )

    # 6.3 使用融合特征
    print("\n--- 使用融合特征 ---")
    clf_fused, cv_scores_fused, acc_fused = train_svm_classifier(X_fused, y)

    # ========== Step 7: 结果汇总 ==========
    print("\n【Step 7】结果汇总")

    print("\n" + "=" * 60)
    print("分类性能对比")
    print("=" * 60)
    print(f"CSP 特征准确率：     {acc_csp:.4f} ± {cv_scores_csp.std():.4f}")
    print(f"小波特征准确率：   {acc_wavelet:.4f} ± {cv_scores_wavelet.std():.4f}")
    print(f"融合特征准确率：   {acc_fused:.4f} ± {cv_scores_fused.std():.4f}")
    print("=" * 60)

    # 找出最佳特征
    best_acc = max(acc_csp, acc_wavelet, acc_fused)
    if best_acc == acc_fused:
        print(f"\n🏆 融合特征表现最佳！")
    elif best_acc == acc_csp:
        print(f"\n🏆 CSP 特征表现最佳！")
    else:
        print(f"\n🏆 小波特征表现最佳！")

    # ========== Step 8: 保存结果 ==========
    print("\n【Step 8】保存结果")

    results = {
        "subject_id": subject_id,
        "epochs": epochs,
        "ica": ica,
        "X_csp": X_csp,
        "X_wavelet": X_wavelet,
        "X_fused": X_fused,
        "y": y,
        "scaler_csp": scaler_csp,
        "scaler_wavelet": scaler_wavelet,
        "clf_csp": clf_csp,
        "clf_wavelet": clf_wavelet,
        "clf_fused": clf_fused,
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
        },
    }

    print(f"✅ 结果已保存到 results 字典")

    # ========== Step 9: 可视化（可选） ==========
    print("\n【Step 9】可视化（可选）")

    # 绘制混淆矩阵（使用融合特征）
    try:
        from sklearn.metrics import confusion_matrix
        
        y_pred = clf_fused.predict(X_fused)
        class_names = ["左手", "右手", "双脚", "舌头"]
        
        # 计算混淆矩阵
        cm = confusion_matrix(y, y_pred)
        
        # 绘制混淆矩阵图
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            save_path=f"./{subject_id}_confusion_matrix.png",
        )
        print(f"✅ 混淆矩阵已保存：./{subject_id}_confusion_matrix.png")
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
            
            # 显示特征形状
            print(f"\n特征维度:")
            print(f"  - CSP 特征：{results['X_csp'].shape}")
            print(f"  - 小波特征：{results['X_wavelet'].shape}")
            print(f"  - 融合特征：{results['X_fused'].shape}")
            
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
