"""
在独立测试集上评估模型泛化能力

训练集：A01T.gdf（288 个试次）
测试集：A01E.gdf（288 个试次）

流程：
1. 在 A01T 上训练模型（使用 10 折交叉验证评估）
2. 在 A01E 上测试模型（评估泛化能力）
3. 生成测试集混淆矩阵和性能报告
"""

from classification.svm_classifier import train_svm_classifier, plot_confusion_matrix
from feature_extraction.wavelet_feature import extract_wavelet_energy_features, normalize_features
from feature_extraction.csp_feature import extract_csp_features
from pretreatment.complete_preprocessing import complete_preprocessing_pipeline
import numpy as np
import pickle
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))


def train_and_evaluate():
    """
    在训练集上训练，在测试集上评估
    """
    print("\n" + "=" * 80)
    print("独立测试集评估")
    print("训练集：A01T | 测试集：A01E")
    print("=" * 80)
    
    # ========== Step 1: 处理训练集 ==========
    print("\n【Step 1】处理训练集 A01T")
    print("=" * 80)
    
    try:
        # 运行完整预处理流程
        epochs_train, ica_train = complete_preprocessing_pipeline()
        
        print(f"\n✅ 训练集预处理完成")
        print(f"   - 试次数：{len(epochs_train)}")
        print(f"   - 通道数：{len(epochs_train.ch_names)}")
        
    except Exception as e:
        print(f"\n❌ 训练集处理失败：{e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== Step 2: 提取训练集特征 ==========
    print("\n【Step 2】提取训练集特征")
    print("=" * 80)
    
    # CSP 特征
    print("\n--- 提取 CSP 特征 ---")
    X_csp_train, csp_model = extract_csp_features(epochs_train, n_components=4)
    print(f"   - CSP 特征形状：{X_csp_train.shape}")
    
    # 小波特征
    print("\n--- 提取小波特征 ---")
    X_wavelet_train = extract_wavelet_energy_features(epochs_train)
    print(f"   - 小波特征形状：{X_wavelet_train.shape}")
    
    # 归一化
    print("\n--- 特征归一化 ---")
    X_csp_train_norm, scaler_csp = normalize_features(X_csp_train)
    X_wavelet_train_norm, scaler_wavelet = normalize_features(X_wavelet_train)
    
    # 融合特征
    print("\n--- 特征融合 ---")
    X_fused_train = np.hstack([X_csp_train_norm, X_wavelet_train_norm])
    print(f"   - 融合特征形状：{X_fused_train.shape}")
    
    # 获取标签
    print("\n--- 准备标签 ---")
    y_train = epochs_train.events[:, 2]
    print(f"   - 标签形状：{y_train.shape}")
    
    # ========== Step 3: 训练模型 ==========
    print("\n【Step 3】训练 SVM 分类器")
    print("=" * 80)
    
    # 使用融合特征训练
    print("\n--- 使用融合特征训练 ---")
    clf_fused, cv_scores, acc_mean = train_svm_classifier(
        X_fused_train, 
        y_train, 
        cv_folds=10,
        kernel='rbf'
    )
    
    print(f"\n✅ 训练集交叉验证准确率：{acc_mean:.4f} ± {cv_scores.std():.4f}")
    
    # 保存训练好的模型和归一化器
    print("\n【保存模型】")
    model_data = {
        'clf_fused': clf_fused,
        'scaler_csp': scaler_csp,
        'scaler_wavelet': scaler_wavelet,
        'cv_accuracy': acc_mean,
        'cv_std': cv_scores.std()
    }
    
    model_path = './trained_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ 模型已保存到：{model_path}")
    
    # ========== Step 4: 处理测试集 ==========
    print("\n【Step 4】处理测试集 A01E")
    print("=" * 80)
    
    try:
        # 注意：这里需要修改预处理流程，使其不重新训练 ICA
        # 而是使用训练集的 ICA 模型
        # 为简化，我们先独立处理测试集
        
        from pretreatment.eeg_analysis import get_modified_raw_data
        from pretreatment.complete_preprocessing import (
            filter_for_ica, fit_ica, detect_and_remove_artifacts,
            apply_ica, filter_for_task, set_reference, create_epochs,
            drop_artifact_epochs
        )
        
        # 加载测试集原始数据
        print("\n加载测试集数据...")
        raw_test = get_modified_raw_data()
        print(f"✅ 测试集数据加载完成，通道数：{len(raw_test.ch_names)}")
        
        # ICA 前的轻度滤波
        print("\nStep 2: ICA 前轻度滤波")
        raw_ica_filtered = filter_for_ica(raw_test, l_freq=1.0, h_freq=40.0)
        
        # ICA 分解
        print("\nStep 3: ICA 分解")
        ica_test = fit_ica(raw_ica_filtered)
        
        # 检测并去除伪迹
        print("\nStep 4: 检测并去除伪迹")
        ica_test = detect_and_remove_artifacts(ica_test, raw_ica_filtered)
        
        # 应用 ICA 去噪
        print("\nStep 5: 应用 ICA 去噪")
        raw_clean = apply_ica(ica_test, raw_test)
        
        # 任务定制滤波
        print("\nStep 6: 任务定制滤波")
        raw_final = filter_for_task(raw_clean, l_freq=8.0, h_freq=30.0)
        
        # 重参考
        print("\nStep 7: 重参考")
        raw_final = set_reference(raw_final, ref='average')
        
        # 分段
        print("\nStep 8: 分段")
        epochs_test = create_epochs(raw_final, tmin=0, tmax=4)
        
        # 剔除伪迹试次
        print("\nStep 9: 剔除伪迹试次")
        epochs_events = epochs_test.events.copy()
        epochs_test_final = drop_artifact_epochs(epochs_test, epochs_events, artifact_codes=[1023])
        
        print(f"\n✅ 测试集预处理完成")
        print(f"   - 试次数：{len(epochs_test_final)}")
        
    except Exception as e:
        print(f"\n❌ 测试集处理失败：{e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== Step 5: 提取测试集特征 ==========
    print("\n【Step 5】提取测试集特征")
    print("=" * 80)
    
    # CSP 特征（使用相同的参数）
    print("\n--- 提取 CSP 特征 ---")
    X_csp_test, csp_model_test = extract_csp_features(epochs_test_final, n_components=4)
    print(f"   - CSP 特征形状：{X_csp_test.shape}")
    
    # 小波特征
    print("\n--- 提取小波特征 ---")
    X_wavelet_test = extract_wavelet_energy_features(epochs_test_final)
    print(f"   - 小波特征形状：{X_wavelet_test.shape}")
    
    # 使用训练集的归一化器
    print("\n--- 应用训练集的归一化参数 ---")
    X_csp_test_norm = scaler_csp.transform(X_csp_test)
    X_wavelet_test_norm = scaler_wavelet.transform(X_wavelet_test)
    
    # 融合特征
    print("\n--- 特征融合 ---")
    X_fused_test = np.hstack([X_csp_test_norm, X_wavelet_test_norm])
    print(f"   - 融合特征形状：{X_fused_test.shape}")
    
    # 获取标签
    print("\n--- 准备标签 ---")
    y_test = epochs_test_final.events[:, 2]
    print(f"   - 标签形状：{y_test.shape}")
    
    # ========== Step 6: 在测试集上评估 ==========
    print("\n【Step 6】在测试集上评估模型")
    print("=" * 80)
    
    # 使用训练好的模型预测测试集
    print("\n--- 预测测试集 ---")
    y_pred = clf_fused.predict(X_fused_test)
    
    # 计算准确率
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
    
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 测试集准确率：{test_accuracy:.4f}")
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"📈 Cohen's Kappa: {kappa:.4f}")
    
    # 分类报告
    print("\n📋 分类报告:")
    target_names = ['左手', '右手', '双脚', '舌头']
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # 混淆矩阵
    print("\n📊 混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 绘制混淆矩阵
    print("\n--- 绘制混淆矩阵 ---")
    save_path = './A01E_test_confusion_matrix.png'
    plot_confusion_matrix(cm, class_names=target_names, save_path=save_path)
    print(f"✅ 混淆矩阵已保存：{save_path}")
    
    # ========== Step 7: 结果对比 ==========
    print("\n【Step 7】训练集 vs 测试集对比")
    print("=" * 80)
    
    print(f"""
╔════════════════════════════════════════════════════════╗
║                    性能对比总结                        ║
╠════════════════════════════════════════════════════════╣
║  训练集 (A01T)                                         ║
║    - 试次数：{len(epochs_train):>3d}                                       ║
║    - 交叉验证准确率：{acc_mean:>6.4f} ± {cv_scores.std():.4f}                  ║
║                                                        ║
║  测试集 (A01E)                                         ║
║    - 试次数：{len(epochs_test_final):>3d}                                       ║
║    - 测试准确率：{test_accuracy:>6.4f}                                    ║
║    - Kappa 系数：{kappa:>6.4f}                                      ║
║                                                        ║
║  泛化能力分析：                                        ║
╚════════════════════════════════════════════════════════╝
    """)
    
    # 分析泛化能力
    gap = acc_mean - test_accuracy
    if abs(gap) < 0.05:
        print(f"✅ 泛化能力优秀！训练集和测试集差异很小 ({gap:.4f})")
    elif abs(gap) < 0.10:
        print(f"⚠️  泛化能力良好，存在轻微过拟合 ({gap:.4f})")
    else:
        print(f"❗ 泛化能力较差，可能存在过拟合 ({gap:.4f})")
    
    # ========== Step 8: 保存完整结果 ==========
    print("\n【Step 8】保存完整结果")
    print("=" * 80)
    
    results = {
        'train_data': {
            'subject_id': 'A01T',
            'n_epochs': len(epochs_train),
            'cv_accuracy': acc_mean,
            'cv_std': cv_scores.std(),
        },
        'test_data': {
            'subject_id': 'A01E',
            'n_epochs': len(epochs_test_final),
            'test_accuracy': test_accuracy,
            'kappa': kappa,
        },
        'model': clf_fused,
        'scalers': {
            'csp': scaler_csp,
            'wavelet': scaler_wavelet
        },
        'predictions': {
            'y_test': y_test,
            'y_pred': y_pred
        }
    }
    
    results_path = './evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ 完整结果已保存：{results_path}")
    
    print("\n" + "=" * 80)
    print("✅ 独立测试集评估完成！")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = train_and_evaluate()
    
    if results is None:
        print("\n❌ 评估过程中断，请检查错误信息")
    else:
        print("\n" + "=" * 80)
        print("评估完成！查看结果:")
        print("=" * 80)
        print(f"📊 训练集准确率：{results['train_data']['cv_accuracy']:.4f} ± {results['train_data']['cv_std']:.4f}")
        print(f"📈 测试集准确率：{results['test_data']['test_accuracy']:.4f}")
        print(f"🎯 Kappa 系数：{results['test_data']['kappa']:.4f}")
        print(f"📁 结果文件：evaluation_results.pkl")
        print(f"📊 混淆矩阵：A01E_test_confusion_matrix.png")
