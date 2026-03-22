"""
中期答辩 PPT 图片生成脚本
生成所有用于 PPT 展示的可视化图表
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA

from pretreatment.eeg_analysis import get_modified_raw_data
from pretreatment.complete_preprocessing import (
    complete_preprocessing_pipeline, 
    plot_preprocessing_comparison
)
from feature_extraction.wavelet_feature import (
    extract_wavelet_energy_features,
    plot_wavelet_decomposition
)
from feature_extraction.csp_feature import (
    extract_csp_features,
    visualize_csp_topo
)
from classification.svm_classifier import (
    train_svm_classifier,
    evaluate_model,
    plot_confusion_matrix
)


def create_output_dir():
    """创建输出目录"""
    output_dir = project_root / "generate_img" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_all_images(subject='A01T'):
    """生成所有 PPT 所需的图片（共 7 步，生成 6 张图）"""
    
    print("="*60)
    print(f"开始生成被试 {subject} 的 PPT 素材图片")
    print("="*60)
    
    output_dir = create_output_dir()
    
    # ========== 1. 加载和预处理数据 ==========
    print("\n[1/7] 加载和预处理数据...")
    try:
        result = complete_preprocessing_pipeline()
        epochs = result[0] if isinstance(result, tuple) else result
        print(f"✓ 成功加载 {len(epochs)} 个有效试次")
    except Exception as e:
        print(f"✗ 预处理失败：{e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== 2. 生成预处理对比图 ==========
    print("\n[2/7] 生成预处理对比图...")
    try:
        save_path = output_dir / "preprocessing_comparison.png"
        
        # 使用 complete_preprocessing 中生成的对比图
        import shutil
        src_path = project_root.parent / "preprocessing_comparison.png"
        if src_path.exists():
            shutil.copy(src_path, save_path)
            print(f"✓ 预处理对比图已复制：{save_path}")
        else:
            # 如果找不到，就用 epochs 绘制平均诱发电位图
            fig = epochs.plot_average(show=False, title=f'平均诱发电位 - {subject}')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ 预处理对比图已保存：{save_path}")
        
    except Exception as e:
        print(f"✗ 预处理对比图生成失败：{e}")
        import traceback
        traceback.print_exc()
    
    # ========== 3. 生成 PSD 分析图 ==========
    print("\n[3/7] 生成功率谱密度分析图...")
    try:
        save_path = output_dir / "psd_analysis.png"
        
        # 计算 PSD
        psd = epochs.compute_psd(method='welch', fmin=1, fmax=50)
        
        # 获取 PSD 数据
        freqs = psd.freqs
        psd_data = psd.get_data()  # 获取功率谱数据
        
        # 转换为 dB (10 * log10(power))
        psd_data_db = 10 * np.log10(psd_data)
        
        # 平均所有通道和试次
        mean_psd = np.mean(psd_data_db, axis=(0, 1))
        std_psd = np.std(psd_data_db, axis=(0, 1))
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制 PSD 曲线（蓝色）
        ax.plot(freqs, mean_psd, color='#1f77b4', linewidth=2, label='平均 PSD')
        
        # 绘制标准差阴影区域
        ax.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, 
                       alpha=0.3, color='#1f77b4', label='±1 标准差')
        
        # 标注目标频段
        ax.axvspan(8, 13, alpha=0.2, color='green', label='μ频段 (8-13 Hz)')
        ax.axvspan(13, 30, alpha=0.2, color='orange', label='β频段 (13-30 Hz)')
        
        # 标注 50Hz 位置（数据集采集时已通过陷波滤波器去除）
        ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50 Hz (采集时已陷波)')
        
        # 设置标签和标题
        ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('功率谱密度 (dB)', fontsize=12, fontweight='bold')
        ax.set_title('功率谱密度 (PSD) - 预处理后\n(8-30 Hz 为运动想象相关频段，50 Hz 已在采集时去除)', 
                    fontsize=14, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)
        
        # 设置白色背景
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✓ PSD 对比图已保存：{save_path}")
        
    except Exception as e:
        print(f"✗ PSD 对比图生成失败：{e}")
        import traceback
        traceback.print_exc()
    
    # ========== 4. 生成小波分解图 ==========
    print("\n[4/7] 生成小波分解图...")
    try:
        save_path = output_dir / "wavelet_decomposition.png"
        
        # 取第一个试次的 C3 通道数据
        data = epochs.get_data()[0]  # (n_channels, n_times)
        c3_channel = data[epochs.ch_names.index('C3')]  # C3 通道
        sfreq = epochs.info['sfreq']
        
        # 绘制小波分解
        plot_wavelet_decomposition(c3_channel, sfreq, wavelet='db4', level=4, 
                                  save_path=str(save_path))
        print(f"✓ 小波分解图已保存：{save_path}")
        
    except Exception as e:
        print(f"✗ 小波分解图生成失败：{e}")
    
    # ========== 5. 生成 CSP 拓扑图 ==========
    print("\n[5/7] 生成 CSP 拓扑图...")
    try:
        save_path = output_dir / "csp_topomap.png"
        
        # 提取 CSP 特征
        result_csp = extract_csp_features(epochs, n_components=4)
        X_csp = result_csp[0] if isinstance(result_csp, tuple) else result_csp
        csp_model = result_csp[1] if isinstance(result_csp, tuple) and len(result_csp) > 1 else None
        
        # 由于通道位置信息缺失，我们改为绘制 CSP 特征值分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        class_names = ['左手', '右手', '双脚', '舌头']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i in range(4):
            ax = axes[i]
            for j, class_idx in enumerate(range(4)):
                mask = np.array(epochs.events[:, 2]) == (j + 7)  # 7,8,9,10 对应 4 类任务
                ax.hist(X_csp[mask, i], alpha=0.6, color=colors[j], 
                       label=class_names[j], bins=20, edgecolor='black')
            ax.set_xlabel(f'CSP Component {i+1}', fontsize=12)
            ax.set_ylabel('试次数量', fontsize=12)
            ax.set_title(f'CSP 成分 {i+1} - 四类任务分布', fontsize=13, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ CSP 特征分布图已保存：{save_path}")
        
    except Exception as e:
        print(f"✗ CSP 拓扑图生成失败：{e}")
        import traceback
        traceback.print_exc()
    
    # ========== 6. 生成混淆矩阵 ==========
    print("\n[6/7] 生成混淆矩阵...")
    try:
        save_path = output_dir / "confusion_matrix.png"
        
        # 准备训练数据
        result_csp = extract_csp_features(epochs, n_components=4)
        X_csp = result_csp[0] if isinstance(result_csp, tuple) else result_csp
        
        result_wavelet = extract_wavelet_energy_features(epochs, wavelet='db4', level=4)
        X_wavelet = result_wavelet[0] if isinstance(result_wavelet, tuple) else result_wavelet
        
        # 归一化
        from sklearn.preprocessing import StandardScaler
        scaler_csp = StandardScaler()
        scaler_wavelet = StandardScaler()
        
        X_csp_norm = scaler_csp.fit_transform(X_csp)
        X_wavelet_norm = scaler_wavelet.fit_transform(X_wavelet)
        
        # 融合特征
        X_fused = np.hstack([X_csp_norm, X_wavelet_norm])
        
        # 获取标签
        y = np.array(epochs.events[:, 2], dtype=int)
        
        # 训练模型
        clf_result = train_svm_classifier(X_fused, y, kernel='rbf')
        clf = clf_result[0] if isinstance(clf_result, tuple) else clf_result
        
        # 评估并绘制混淆矩阵
        y_pred = clf.predict(X_fused)
        metrics = evaluate_model(y, y_pred)
        cm = metrics['confusion_matrix']
        
        # 自定义类别名称
        class_names_custom = ['左手', '右手', '双脚', '舌头']
        plot_confusion_matrix(cm, class_names=class_names_custom, 
                            save_path=str(save_path))
        print(f"✓ 混淆矩阵已保存：{save_path}")
        
    except Exception as e:
        print(f"✗ 混淆矩阵生成失败：{e}")
    
    # ========== 7. 生成准确率对比柱状图 ==========
    print("\n[7/7] 生成特征方案对比图...")
    try:
        save_path = output_dir / "feature_comparison.png"
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        # 三种特征方案
        methods = ['小波特征', 'CSP 特征', '融合特征']
        accuracies = []
        stds = []
        
        # 创建基础分类器
        from sklearn.svm import SVC
        base_clf = SVC(kernel='rbf', random_state=42)
        
        # 1. 小波特征
        scores = cross_val_score(base_clf, X_wavelet_norm, y, 
                                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
        accuracies.append(scores.mean() * 100)
        stds.append(scores.std() * 100)
        
        # 2. CSP 特征
        scores = cross_val_score(base_clf, X_csp_norm, y, 
                                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
        accuracies.append(scores.mean() * 100)
        stds.append(scores.std() * 100)
        
        # 3. 融合特征
        scores = cross_val_score(base_clf, X_fused, y, 
                                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42))
        accuracies.append(scores.mean() * 100)
        stds.append(scores.std() * 100)
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(methods))
        colors = ['#FF8C00', '#4682B4', '#32CD32']  # 橙色、蓝色、绿色
        
        bars = ax.bar(x_pos, accuracies, yerr=stds, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for i, (acc, std) in enumerate(zip(accuracies, stds)):
            ax.text(i, acc + std + 2, f'{acc:.1f}%\n±{std:.1f}%', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('特征方案', fontsize=12, fontweight='bold')
        ax.set_title(f'不同特征方案的分类性能对比 - {subject}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 特征方案对比图已保存：{save_path}")
        
    except Exception as e:
        print(f"✗ 特征方案对比图生成失败：{e}")
    
    print("\n" + "="*60)
    print("✅ 所有图片生成完成！")
    print(f"📁 输出目录：{output_dir}")
    print("="*60)


if __name__ == "__main__":
    # 可以修改为其他被试
    subject_id = 'A01T'
    generate_all_images(subject_id)
